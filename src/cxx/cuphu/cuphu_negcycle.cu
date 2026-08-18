/**
 * cuphu_negcycle.cu
 *
 * GPU replacement for one pass of SNAPHU's TreeSolve inner loop: given a
 * FIXED per-arc linear cost (poscost/negcost, as computed by
 * cuphu_incrcost.cu's ComputeIncrCostsKernel for the current nflow), find
 * and cancel negative-cost cycles in the residual graph, augmenting by
 * exactly nflow units per cycle, until none remain.
 *
 * This does NOT replace SNAPHU's nonconvex outer nflow-relinearization loop
 * (still on CPU, in cuphu_solver.cpp) -- poscost/negcost are recomputed by
 * the caller before each call to cuphu_gpu_treesolve_pass(). Within one
 * call, costs are fixed, so this is classical cycle-canceling: build a
 * directed residual graph (two directed edges per arc slot: u->v at
 * poscost, v->u at negcost), detect negative cycles via parallel
 * Bellman-Ford relaxation, cancel them, repeat.
 *
 * Capacity note (load-bearing, found by debugging a real hang): poscost/
 * negcost are a snapshot valid for exactly one nflow-unit shift away from
 * the flow this pass started with -- they are never recomputed mid-pass
 * (matches TreeSolve: SetupIncrFlowCosts runs once per nflow value, outside
 * the per-source TreeSolve loop). With genuinely unbounded per-arc
 * capacity, a graph containing any negative cycle under that static cost
 * snapshot is unbounded -- canceling the cycle doesn't change its
 * (unchanged) cost, so the same cycle is immediately negative again and
 * cycle-canceling never terminates. TreeSolve avoids this via its
 * tree-pivot bookkeeping (evolving dual prices from the maintained
 * spanning tree naturally prevent re-selecting the same improving move);
 * this GPU version has no equivalent pricing state, so it enforces the
 * same "each arc's flow shifts at most once per pass" invariant directly:
 * once a directed edge participates in a canceled cycle, that specific
 * (arc, direction) pair is marked used and excluded from all further
 * relaxation for the rest of this cuphu_gpu_treesolve_pass() call
 * (arc_used[], indexed per direction -- 2 entries per arc slot -- reset
 * once per call, not per outer iteration). Retiring only the direction
 * actually used (not the whole arc) matters: the opposite direction of
 * the same arc is a legitimately different move, and an earlier whole-
 * arc-retirement version of this fix passed termination but landed at a
 * measurably worse total cost than CPU TreeSolve -- i.e. it was over-
 * restrictive, not just imprecise. This bounds cycles-per-call to at
 * most 2*narcs.
 *
 * dist/pred packing (fixes a real race, code review 2026-08-17): dist and
 * the predecessor edge that justifies it must update as one atomic unit --
 * two threads relaxing into the same node could otherwise leave dist and
 * pred_edge describing different edges (dist from thread B's update, pred
 * from thread A's, written after A's own CAS succeeded but before A's
 * plain follow-up store executed). That produced a real, confirmed bug:
 * ClaimAndCancelKernel tracing a fabricated cycle from the mismatched
 * pair, likely the cause of the ~0.01%-0.6% cost gaps measured against
 * CPU TreeSolve. Fixed by packing (dist, pred_edge) into one uint64 and
 * updating it with a single atomicMin -- see pack_dist_pred() below.
 * pred_node is no longer stored at all: it's fully determined by
 * pred_edge (which arc, which direction) via classify_arc(), so packing
 * only dist+pred_edge is sufficient and keeps the encoding to 64 bits.
 *
 * Graph layout:
 *   - Interior grid nodes: (nrow-1) x (ncol-1), id = row*(ncol-1) + col.
 *   - Ground (virtual, one per tile): id = (nrow-1)*(ncol-1).
 *   - Arc slots use the same flat layout as the rest of cuPHU (see
 *     cuphu_incrcost.cu): row arcs [0, (nrow-1)*ncol), then col arcs
 *     [(nrow-1)*ncol, (nrow-1)*ncol + nrow*(ncol-1)).
 *   - Per-slot node endpoints and the "dead" (never-traversed) corner
 *     slots are derived exactly from SNAPHU's own NeighborNodeGrid() /
 *     GetArcGrid() (snaphu_solver.c) -- verified directly against that
 *     source, not re-derived from scratch. See classify_arc() below.
 *
 * Milestone 2 scope (see implementation plan): correctness over
 * performance. Dense sweep-every-round Bellman-Ford (no frontier
 * pruning -- that's Milestone 3), full relax-to-convergence + claim/cancel
 * outer loop repeated until a pass finds no negative cycle.
 *
 * Known limitation (see comment at the claim/cancel call site below):
 * cancels exactly one cycle per outer iteration, not many in parallel.
 * A first attempt at parallel multi-cycle claiming (atomicCAS node
 * ownership, abandon-on-block) turned out to be a liveness bug, not just
 * a performance shortfall -- deferred to a real follow-up design rather
 * than shipped half-working. See ClaimAndCancelKernel's serial_mode
 * parameter, which is now the only mode exercised (kept as a parameter,
 * not a dead branch, so the eventual parallel version can reuse this
 * kernel with serial_mode=false once its claiming scheme is fixed).
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>

extern "C" {
#include "snaphu.h"
}

/* ── dist/pred_edge packing ──────────────────────────────────────────────── */
/*
 * uint64 = [ 40-bit biased dist | 24-bit biased edge_desc ]. Updated as one
 * unit via atomicMin -- see the file header note above for why.
 *
 * 24 bits for edge_desc supports narcs up to ~8.39M (edge_desc = +-(idx+1),
 * biased into [0, 2^24-1] with a spare zero-point reserved as the "no
 * predecessor" sentinel). Comfortably covers every crop tested against
 * this kernel so far (largest: ~1.9M arcs); would need widening (or a
 * two-word scheme) before trusting this at full-scene (~90M-node) scale.
 *
 * 40 bits for biased dist supports true dist in roughly +-5.5e11 (path
 * length x max |cost| 32000) -- covers node counts into the tens of
 * millions at this kernel's currently-validated scale with large margin.
 */
static constexpr int           PRED_BITS  = 24;
static constexpr unsigned long long PRED_MASK  = (1ULL << PRED_BITS) - 1ULL;
static constexpr long long     PRED_BIAS  = 1LL << (PRED_BITS - 1);   /* edge_desc=0 -> sentinel */
static constexpr long long     DIST_BIAS  = 1LL << 39;

__device__ __forceinline__ unsigned long long pack_dist_pred(long long dist, int edge_desc)
{
    unsigned long long biased_dist = (unsigned long long)(dist + DIST_BIAS);
    unsigned long long biased_pred = (unsigned long long)((long long)edge_desc + PRED_BIAS);
    return (biased_dist << PRED_BITS) | (biased_pred & PRED_MASK);
}

__device__ __forceinline__ void unpack_dist_pred(
    unsigned long long packed, long long &dist, int &edge_desc)
{
    unsigned long long biased_dist = packed >> PRED_BITS;
    unsigned long long biased_pred = packed & PRED_MASK;
    dist      = (long long)biased_dist - DIST_BIAS;
    edge_desc = (int)((long long)biased_pred - PRED_BIAS);
}

/* ── arc classification ──────────────────────────────────────────────────── */
/*
 * Given a flat arc-slot index, return whether it's a "dead" slot (never
 * traversed -- the 4 degenerate corner col-arc entries InitNetwork already
 * zeroes) and, if not, the canonical (u, v) node-id endpoints such that
 * increasing flows[idx] (the poscost direction) corresponds to u -> v.
 *
 * Derived from and verified against snaphu_solver.c's NeighborNodeGrid()/
 * GetArcGrid(): row-arc arccol==0 is ground->node(arcrow,0) (positive
 * direction ground->node); arccol==ncol-1 is node(arcrow,ncol-2)->ground;
 * interior row-arc columns connect node(arcrow,c-1)->node(arcrow,c).
 * Col-arc r==0/r==nrow-1 are the top/bottom ground connections (excluding
 * the two corner columns, which are the dead slots); interior col-arc rows
 * connect node(r-1,c)->node(r,c).
 */
__device__ __forceinline__ void classify_arc(
    int idx, int nrow, int ncol, int ground_id,
    bool &dead, int &u, int &v)
{
    int nrowarc = (nrow - 1) * ncol;
    dead = false;

    if (idx < nrowarc) {
        int arcrow = idx / ncol;
        int arccol = idx % ncol;
        if (arccol == 0) {
            u = ground_id;
            v = arcrow * (ncol - 1);
        } else if (arccol == ncol - 1) {
            u = arcrow * (ncol - 1) + (ncol - 2);
            v = ground_id;
        } else {
            u = arcrow * (ncol - 1) + (arccol - 1);
            v = arcrow * (ncol - 1) + arccol;
        }
    } else {
        int idx2   = idx - nrowarc;
        int r      = idx2 / (ncol - 1);
        int arccol = idx2 % (ncol - 1);
        if (r == 0) {
            if (arccol == 0 || arccol == ncol - 2) { dead = true; return; }
            u = ground_id;
            v = arccol;  /* node(0, arccol) */
        } else if (r == nrow - 1) {
            if (arccol == 0 || arccol == ncol - 2) { dead = true; return; }
            u = (nrow - 2) * (ncol - 1) + arccol;
            v = ground_id;
        } else {
            u = (r - 1) * (ncol - 1) + arccol;
            v = r       * (ncol - 1) + arccol;
        }
    }
}

/* Predecessor NODE for a packed edge_desc: the node on the "from" side of
 * whichever direction was used to relax into the node that recorded this
 * edge_desc. edge_desc>0 means the poscost (u->v) direction was used, so
 * the predecessor is u; edge_desc<0 means negcost (v->u), predecessor v. */
__device__ __forceinline__ int pred_node_from_edge(
    int edge_desc, int nrow, int ncol, int ground_id)
{
    int idx = (edge_desc > 0) ? (edge_desc - 1) : (-edge_desc - 1);
    bool dead; int u, v;
    classify_arc(idx, nrow, ncol, ground_id, dead, u, v);
    return (edge_desc > 0) ? u : v;
}

/* ── kernels ──────────────────────────────────────────────────────────────── */

__global__ void InitNodeStateKernel(
    unsigned long long *dist_pred, int *claimed, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    dist_pred[i] = pack_dist_pred(0, 0);  /* dist=0, edge_desc=0 sentinel */
    claimed[i]   = -1;
}

/*
 * One thread per directed edge (2 per arc slot: poscost direction u->v,
 * negcost direction v->u). Relaxes dist_pred[j] = min(dist_pred[j],
 * pack(dist[i]+cost, edge_desc)) via a single atomicMin -- dist and the
 * predecessor edge that justifies it update together, atomically, so
 * there is no window where they can describe different edges (see the
 * packing note above).
 */
__global__ void RelaxKernel(
    const short *poscost, const short *negcost,
    const unsigned char *arc_used, int use_limit,
    int nrow, int ncol, int ground_id, long long narcs,
    unsigned long long *dist_pred, int *d_changed)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 2LL * narcs) return;

    int idx = (int)(tid >> 1);
    int dir = (int)(tid & 1);

    if (arc_used[tid] >= use_limit) return;  /* (arc,dir) use-count exhausted */

    bool dead; int u, v;
    classify_arc(idx, nrow, ncol, ground_id, dead, u, v);
    if (dead) return;

    int i, j, cost, edge_desc;
    if (dir == 0) { i = u; j = v; cost = poscost[idx]; edge_desc =  (idx + 1); }
    else          { i = v; j = u; cost = negcost[idx]; edge_desc = -(idx + 1); }

    long long dist_i; int unused_ed;
    unpack_dist_pred(dist_pred[i], dist_i, unused_ed);
    long long nd = dist_i + (long long)cost;

    unsigned long long candidate = pack_dist_pred(nd, edge_desc);
    unsigned long long old = atomicMin(&dist_pred[j], candidate);
    if (candidate < old)
        atomicExch(d_changed, 1);
}

/*
 * After relaxation reaches a fixpoint (or max_rounds is hit), snapshot
 * which nodes' dist changed in the *final* extra round (the round after
 * which d_changed was still true) -- those are candidates on (or
 * reachable from) a negative cycle.
 */
__global__ void MarkChangedKernel(
    const unsigned long long *dist_pred_before,
    const unsigned long long *dist_pred_after,
    unsigned char *unsettled, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    long long d_before, d_after; int unused;
    unpack_dist_pred(dist_pred_before[i], d_before, unused);
    unpack_dist_pred(dist_pred_after[i],  d_after,  unused);
    unsettled[i] = (d_after < d_before) ? 1 : 0;
}

__global__ void CopyDistPredKernel(
    const unsigned long long *src, unsigned long long *dst, int nnode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnode) dst[i] = src[i];
}

/*
 * For each unsettled node, walk the predecessor chain backward (derived
 * from dist_pred[]'s packed edge_desc via pred_node_from_edge()),
 * attempting to claim each visited node via atomicCAS(claimed[node], -1,
 * start_id). Three outcomes per step:
 *   - claim succeeds: continue walking to the predecessor.
 *   - claim fails because it's already claimed by THIS SAME walk
 *     (claimed[cur] == start_id): a cycle has been found, closing at
 *     `cur`. Re-walk from `cur` (bounded by the now-known cycle length) to
 *     cancel it: for each arc on the cycle, apply nflow's flow adjustment.
 *   - claim fails because it's claimed by a DIFFERENT walk: abandon this
 *     walk without canceling (the owning walk either already handled this
 *     node's cycle, or will finish shortly; this node gets revisited on
 *     the next outer-loop iteration if still unsettled).
 *
 * serial_mode: only thread 0 (start_id == the smallest unsettled node id,
 * passed in as `serial_leader`) is allowed to claim/cancel. Always true at
 * the current call site -- no env var controls it.
 */
__global__ void ClaimAndCancelKernel(
    const unsigned char *unsettled, const unsigned long long *dist_pred,
    int *claimed, short *d_flows, unsigned char *arc_used,
    int nrow, int ncol, int ground_id,
    int nflow, int nnode, int max_walk,
    bool serial_mode, int serial_leader, int *n_canceled)
{
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    if (start >= nnode) return;
    if (!unsettled[start]) return;
    if (serial_mode && start != serial_leader) return;

    int cur = start;
    int steps = 0;
    while (steps < max_walk) {
        int prevOwner = atomicCAS(&claimed[cur], -1, start);
        if (prevOwner == -1) {
            /* claimed cur for this walk; advance to predecessor */
            long long d; int ed;
            unpack_dist_pred(dist_pred[cur], d, ed);
            if (ed == 0) return;  /* sentinel: no predecessor recorded */
            cur = pred_node_from_edge(ed, nrow, ncol, ground_id);
            ++steps;
            continue;
        }
        if (prevOwner == start) {
            /* Closed a cycle: cur is the repeated node. Re-walk from cur,
             * following the predecessor edges, applying each edge's flow
             * adjustment, until we return to cur again. Bounded by the
             * actual cycle length (<= steps already taken). Only the
             * owning thread (this one) ever reaches this point for this
             * specific cycle -- claiming already forbids two concurrently-
             * canceled cycles from sharing a node, hence from sharing an
             * arc -- so plain (non-atomic) read-modify-write on d_flows
             * is safe. */
            int c = cur;
            int guard = 0;
            do {
                long long d; int ed;
                unpack_dist_pred(dist_pred[c], d, ed);
                if (ed == 0) return;  /* inconsistent state; bail */
                int idx = (ed > 0) ? (ed - 1) : (-ed - 1);
                int dir = (ed > 0) ? 0 : 1;
                if (ed > 0) d_flows[idx] = (short)(d_flows[idx] + nflow);
                else        d_flows[idx] = (short)(d_flows[idx] - nflow);
                arc_used[(long long)idx * 2 + dir]++;
                c = pred_node_from_edge(ed, nrow, ncol, ground_id);
                ++guard;
            } while (c != cur && guard < max_walk);
            atomicAdd(n_canceled, 1);
            return;
        }
        /* claimed by a different walk: abandon, no cancellation */
        return;
    }
    /* exceeded max_walk without closing a cycle or hitting a claimed node:
     * treat as no-op for this round (shouldn't happen once dist/pred are
     * self-consistent and max_walk >= nnode, but guard against runaway
     * walks from any transient inconsistency). */
}

/* ── host driver ─────────────────────────────────────────────────────────── */

extern "C"
long cuphu_gpu_treesolve_pass(
    const short *d_poscost, const short *d_negcost,
    short *d_flows,               /* device, IN/OUT, flat arc layout */
    int nrow, int ncol, int nflow,
    int max_rounds,                /* safety cap on relaxation rounds per attempt */
    unsigned long long *d_dist_pred,      /* scratch, size nnode */
    unsigned long long *d_dist_pred_prev, /* scratch, size nnode */
    int *d_claimed,                /* scratch, size nnode */
    unsigned char *d_unsettled,    /* scratch, size nnode */
    unsigned char *d_arc_used,     /* scratch, size 2*narcs (per direction)
                                     * -- reset once per call, NOT per outer
                                     * iteration; see the capacity note in
                                     * the file header */
    int *d_changed,                /* scratch, 1 int */
    int *d_n_canceled              /* scratch, 1 int */
)
{
    const int ninterior = (nrow - 1) * (ncol - 1);
    const int ground_id = ninterior;
    const int nnode      = ninterior + 1;
    const long long narcs = (long long)(nrow - 1) * ncol
                           + (long long)nrow * (ncol - 1);

    const int threads = 256;
    const int node_blocks = (nnode + threads - 1) / threads;
    const long long edge_threads = 2LL * narcs;
    const int edge_blocks = (int)((edge_threads + threads - 1) / threads);

    const bool debug = (std::getenv("CUPHU_DEBUG") != nullptr);
    const int BATCH = 16;
    /* Hardcoded at 1 (each (arc,direction) usable once per pass). No env
     * var controls it -- see KNOWN ISSUE note in the file header. */
    const int use_limit_experiment = 1;

    CUDA_CHECK(cudaMemset(d_arc_used, 0, (size_t)(2 * narcs)));

    long total_canceled = 0;
    int outer_iter = 0;
    const long outer_iter_cap = 4L * nnode + 1000;

    while (true) {
        ++outer_iter;
        if (outer_iter > outer_iter_cap) {
            fprintf(stderr,
                    "[cuphu][negcycle] WARNING: outer_iter cap (%ld) exceeded "
                    "(nnode=%d, nflow=%d); stopping early with %ld cycle(s) "
                    "canceled so far -- this indicates a bug (non-terminating "
                    "cancellation) or a pathologically slow instance\n",
                    outer_iter_cap, nnode, nflow, total_canceled);
            break;
        }

        /* ── relax to convergence (or max_rounds) ────────────────────── */
        InitNodeStateKernel<<<node_blocks, threads>>>(
            d_dist_pred, d_claimed, nnode);

        int h_changed = 1;
        int round = 0;
        while (h_changed && round < max_rounds) {
            int batch_end = std::min(round + BATCH, max_rounds);
            CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
            for (; round < batch_end; ++round) {
                RelaxKernel<<<edge_blocks, threads>>>(
                    d_poscost, d_negcost, d_arc_used, use_limit_experiment,
                    nrow, ncol, ground_id, narcs, d_dist_pred, d_changed);
            }
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        /* One more full round, snapshotting which nodes still change --
         * those are candidates on (or reachable from) a negative cycle. */
        CopyDistPredKernel<<<node_blocks, threads>>>(
            d_dist_pred, d_dist_pred_prev, nnode);
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        RelaxKernel<<<edge_blocks, threads>>>(
            d_poscost, d_negcost, d_arc_used, use_limit_experiment,
            nrow, ncol, ground_id, narcs, d_dist_pred, d_changed);
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (debug) {
            fprintf(stderr,
                    "[cuphu][negcycle] outer=%d: relax rounds=%d "
                    "extra_round_changed=%d\n", outer_iter, round, h_changed);
            fflush(stderr);
        }

        if (!h_changed) {
            break;  /* no negative cycle remains -- this pass is done */
        }

        MarkChangedKernel<<<node_blocks, threads>>>(
            d_dist_pred_prev, d_dist_pred, d_unsettled, nnode);

        /* ── claim and cancel ─────────────────────────────────────────
         *
         * NOTE (Milestone 2 known limitation): this always resolves
         * exactly one cycle per outer iteration (single leader claiming
         * the whole node-id space), not the many-cycles-per-round
         * parallel claiming the design called for. An earlier version
         * attempted that with per-node atomicCAS claims and "abandon on
         * block," but it has a real correctness/liveness bug: when many
         * unsettled nodes' predecessor chains funnel into the same
         * cycle (the common case -- they're all reachable from the same
         * negative cycle by construction), every walk gets blocked by
         * some other walk's claim without any single walk ever
         * revisiting a node it claimed itself, so nothing ever closes.
         * Simplified to go straight to the reliable single-leader path.
         * Genuine multi-cycle-per-round parallelism (e.g. via
         * functional-graph pointer-jumping to find distinct cycle
         * components, matching cuphu_conncomp.cu's JumpLabelsKernel
         * technique) is real Milestone-3-scale follow-up work.
         */
        CUDA_CHECK(cudaMemset(d_claimed, 0xFF, (size_t)nnode * sizeof(int)));
        std::vector<unsigned char> h_unsettled(nnode);
        CUDA_CHECK(cudaMemcpy(h_unsettled.data(), d_unsettled, nnode,
                              cudaMemcpyDeviceToHost));
        int leader = -1;
        for (int i = 0; i < nnode; ++i) {
            if (h_unsettled[i]) { leader = i; break; }
        }
        if (leader < 0) break;  /* nothing unsettled; shouldn't happen here */

        CUDA_CHECK(cudaMemset(d_n_canceled, 0, sizeof(int)));
        ClaimAndCancelKernel<<<node_blocks, threads>>>(
            d_unsettled, d_dist_pred, d_claimed, d_flows, d_arc_used,
            nrow, ncol, ground_id,
            nflow, nnode, nnode, /*serial_mode=*/true, leader, d_n_canceled);

        int h_n_canceled = 0;
        CUDA_CHECK(cudaMemcpy(&h_n_canceled, d_n_canceled, sizeof(int),
                              cudaMemcpyDeviceToHost));
        total_canceled += h_n_canceled;

        if (debug) {
            fprintf(stderr,
                    "[cuphu][negcycle] outer=%d: canceled %d cycle(s) "
                    "(total=%ld)\n", outer_iter, h_n_canceled, total_canceled);
            fflush(stderr);
        }

        if (h_n_canceled == 0) break;  /* truly stuck; stop to avoid infinite loop */

        /* Loop back: costs are unchanged (fixed for this whole pass), so
         * re-running relaxation from a fresh reset will find any negative
         * cycles that remain. */
    }

    return total_canceled;
}
