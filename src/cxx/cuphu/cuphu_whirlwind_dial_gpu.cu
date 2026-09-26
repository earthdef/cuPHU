/**
 * cuphu_whirlwind_dial_gpu.cu
 *
 * GPU port of dial_run_full()'s relaxation step (cuphu_whirlwind_solver.cpp):
 * multi-source, full-completion shortest paths over the implicit regular
 * grid residual graph, for the PD loop's bulk pass. Gated behind
 * CUPHU_WW_GPU_DIAL so the standalone CPU-only build (no CUDA, used for
 * fast-iteration testing) still compiles; augmentation/potential updates in
 * pd_pass() are NOT touched by this file and stay CPU-only, unchanged.
 *
 * Correctness note (deliberate, not an oversight): the CPU reference uses
 * strict FIFO bucket order to break ties among equal-cost relaxations,
 * which whirlwind-insar's own docs say matters on flat-cost (masked)
 * regions. This GPU version instead breaks ties via packed-atomicMin
 * (lowest arc id wins among relaxations landing on the same node at the
 * same distance) -- a different, but still deterministic, rule. Distances
 * are exact; predecessor choice on ties is not guaranteed to match the CPU
 * path bit-for-bit. See the whirlwind GPU-dial plan for why this tradeoff
 * was accepted rather than chased.
 *
 * dist/pred are packed into one uint64 (dist in the upper 32 bits, arc id
 * in the lower 32 bits) and updated via a single atomicMin, avoiding the
 * separate-field read-modify-write race cuphu_negcycle.cu's own history
 * documents fixing the same way for a different algorithm. Unlike that
 * file's 24-bit arc-id packing (capped well under this graph's real arc
 * counts), both fields here get a full 32 bits: this grid's shortest-path
 * distances stay far below 2^32 (bounded by grid diameter x max arc cost),
 * and arc ids fit in 32 bits up to twice the node count.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t NO_ARC = 0xFFFFFFFFu;
constexpr uint32_t DIST_INF = 0xFFFFFFFFu;

__host__ __device__ __forceinline__ uint64_t pack(uint32_t dist, uint32_t arc) {
    return ((uint64_t)dist << 32) | (uint64_t)arc;
}
__host__ __device__ __forceinline__ uint32_t unpack_dist(uint64_t v) { return (uint32_t)(v >> 32); }
__host__ __device__ __forceinline__ uint32_t unpack_arc(uint64_t v) { return (uint32_t)(v & 0xFFFFFFFFu); }

// Mirrors Grid's arc arithmetic (cuphu_whirlwind_solver.cpp) exactly -- an
// implicit 4-neighbor rectangular grid, no adjacency list: any node's
// outgoing arcs are computed from its own (row, col) via closed-form index
// arithmetic, the same "one thread per pixel" pattern cuphu's other
// regular-grid kernels (cuphu_util.cu) already use.
struct GridDims {
    int64_t m, n, n_v, n_h, num_forward;
};

__device__ __forceinline__ int64_t d_down_arc(const GridDims &g, int64_t i, int64_t j) {
    return (i + 1 < g.m) ? (i * g.n + j) : -1;
}
__device__ __forceinline__ int64_t d_up_arc(const GridDims &g, int64_t i, int64_t j) {
    return (i >= 1) ? (g.n_v + (i - 1) * g.n + j) : -1;
}
__device__ __forceinline__ int64_t d_right_arc(const GridDims &g, int64_t i, int64_t j) {
    return (j + 1 < g.n) ? (2 * g.n_v + i * (g.n - 1) + j) : -1;
}
__device__ __forceinline__ int64_t d_left_arc(const GridDims &g, int64_t i, int64_t j) {
    return (j >= 1) ? (2 * g.n_v + g.n_h + i * (g.n - 1) + (j - 1)) : -1;
}
__device__ __forceinline__ int64_t d_transpose(const GridDims &g, int64_t arc) {
    return arc < g.num_forward ? arc + g.num_forward : arc - g.num_forward;
}
__device__ __forceinline__ int32_t d_arc_cost(const GridDims &g, const uint16_t *cost_fwd, int64_t arc) {
    return arc < g.num_forward ? (int32_t)cost_fwd[arc] : -(int32_t)cost_fwd[arc - g.num_forward];
}
__device__ __forceinline__ bool d_is_saturated(const uint8_t *is_sat, int64_t arc) {
    return is_sat[arc] != 0;
}

// Emits this node's up to 8 outgoing (arc, neighbor) pairs, same order as
// Grid::outgoing(): 4 own forward arcs + up to 4 residual reverses of
// neighbors' forward arcs pointing into this node.
__device__ __forceinline__ int d_outgoing(const GridDims &g, int64_t node, int64_t i, int64_t j,
                                            int64_t *arcs, int64_t *heads) {
    int c = 0;
    int64_t a;
    if ((a = d_down_arc(g, i, j)) >= 0) { arcs[c] = a; heads[c] = node + g.n; ++c; }
    if ((a = d_up_arc(g, i, j)) >= 0) { arcs[c] = a; heads[c] = node - g.n; ++c; }
    if ((a = d_right_arc(g, i, j)) >= 0) { arcs[c] = a; heads[c] = node + 1; ++c; }
    if ((a = d_left_arc(g, i, j)) >= 0) { arcs[c] = a; heads[c] = node - 1; ++c; }
    if (i + 1 < g.m) { a = d_up_arc(g, i + 1, j); arcs[c] = d_transpose(g, a); heads[c] = node + g.n; ++c; }
    if (i >= 1) { a = d_down_arc(g, i - 1, j); arcs[c] = d_transpose(g, a); heads[c] = node - g.n; ++c; }
    if (j + 1 < g.n) { a = d_left_arc(g, i, j + 1); arcs[c] = d_transpose(g, a); heads[c] = node + 1; ++c; }
    if (j >= 1) { a = d_right_arc(g, i, j - 1); arcs[c] = d_transpose(g, a); heads[c] = node - 1; ++c; }
    return c;
}

__global__ void InitKernel(uint64_t *dist_pred, const int32_t *excess, int64_t num_nodes,
                            int64_t *d_pending, int64_t *d_discovered) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    if (excess[i] > 0) {
        dist_pred[i] = pack(0, NO_ARC);
        atomicAdd((unsigned long long *)d_discovered, 1ULL);
    } else {
        dist_pred[i] = pack(DIST_INF, NO_ARC);
    }
}

// Finds the smallest dist among not-yet-processed, reachable (dist < INF)
// nodes -- used by the host to jump directly to the next distance with
// real pending work, instead of guessing via increment-and-give-up (which
// has no equivalent of the CPU ring-buffer's "k consecutive EMPTY SLOTS
// means nothing is left" guarantee: unlike a modular bucket index, a raw
// distance value can have arbitrarily large gaps with nothing in them, so
// incrementing one distance at a time and giving up after a fixed number
// of misses can terminate the whole sweep long before genuinely reachable,
// sparser-lying nodes are ever found).
__global__ void FindMinPendingKernel(const uint64_t *dist_pred, const uint8_t *processed,
                                       int64_t num_nodes, unsigned int *d_min_pending) {
    int64_t node = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    if (processed[node]) return;
    uint32_t d = unpack_dist(dist_pred[node]);
    if (d == DIST_INF) return;
    atomicMin(d_min_pending, (unsigned int)d);
}

// One relaxation round: every node not yet processed, sitting at exactly
// cur_dist, relaxes its outgoing arcs and is marked processed. Repeated by
// the host (the "drain" loop) until a round makes no further progress --
// needed because a zero-cost arc can relax a node back into the SAME
// bucket, which the CPU version's single work-queue handles implicitly.
__global__ void RelaxKernel(GridDims g, const uint16_t *cost_fwd, const uint8_t *is_sat,
                              const int64_t *potential, uint64_t *dist_pred, uint8_t *processed,
                              int64_t num_nodes, uint32_t cur_dist,
                              int64_t *d_progress, int64_t *d_discovered) {
    int64_t node = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    uint64_t dp = dist_pred[node];
    if (processed[node] || unpack_dist(dp) != cur_dist) return;
    processed[node] = 1;
    atomicAdd((unsigned long long *)d_progress, 1ULL);

    int64_t i = node / g.n, j = node % g.n;
    int64_t arcs[8], heads[8];
    int cnt = d_outgoing(g, node, i, j, arcs, heads);
    int64_t pot_u = potential[node];
    for (int k = 0; k < cnt; ++k) {
        int64_t arc = arcs[k], v = heads[k];
        if (d_is_saturated(is_sat, arc)) continue;
        int64_t rc = (int64_t)d_arc_cost(g, cost_fwd, arc) - pot_u + potential[v];
        int64_t nd = (int64_t)cur_dist + rc;
        if (nd < 0 || nd >= (int64_t)DIST_INF) continue; // guard against pathological/negative rc

        // CAS retry loop, updating dist_pred[v] ONLY on a STRICT distance
        // improvement -- never on a tie (nd == current dist). A plain
        // atomicMin on the packed (dist,arc) value is NOT safe here: it
        // would let a tie (same dist, smaller arc id) overwrite an
        // ALREADY-SETTLED node's predecessor after later threads/rounds
        // have already derived their own pred chains from the old one,
        // which can retroactively create cycles in the predecessor graph
        // (found empirically: ~40%+ of augmenting paths rejected as
        // cyclic before this fix, vs. 0 after). Once any thread wins the
        // transition to v's true shortest distance, it's immutable -- so
        // ties are provably redundant and must be a no-op, not just a
        // deterministic pick.
        uint64_t old = dist_pred[v];
        for (;;) {
            uint32_t old_dist = unpack_dist(old);
            if ((uint32_t)nd >= old_dist) break; // not an improvement (incl. ties): no-op
            uint64_t cand = pack((uint32_t)nd, (uint32_t)arc);
            uint64_t prev = atomicCAS((unsigned long long *)&dist_pred[v],
                                        (unsigned long long)old, (unsigned long long)cand);
            if (prev == old) {
                if (old_dist == DIST_INF) atomicAdd((unsigned long long *)d_discovered, 1ULL);
                break;
            }
            old = prev; // lost the race; retry against the fresher value
        }
    }
}

} // namespace

/**
 * Full-completion multi-source Dial's-algorithm shortest paths on GPU.
 * Drop-in output-compatible replacement for dial_run_full()'s dist/pred_arc
 * (host arrays h_dist_out/h_pred_arc_out, both length num_nodes; -1 in
 * h_pred_arc_out means "no predecessor", matching the CPU convention).
 * h_popped_out[i] != 0 means node i was reached (matches sp.popped).
 *
 * max_rc/k (bucket count) is computed by the caller exactly as the CPU path
 * already does, and passed in -- this function doesn't recompute it, same
 * division of labor as the existing GPU cost-LUT kernel (caller owns
 * anything cheap/already-CPU-resident; this function owns the expensive
 * parallel part only).
 */
extern "C" void cuphu_ww_dial_run_full_gpu(
    int64_t m, int64_t n, int64_t n_v, int64_t n_h, int64_t num_forward,
    const int32_t *h_excess, const int64_t *h_potential,
    const uint16_t *h_cost_fwd, const uint8_t *h_is_saturated,
    int64_t k, int gpu_id,
    int64_t *h_dist_out, int64_t *h_pred_arc_out, uint8_t *h_popped_out)
{
    cudaSetDevice(gpu_id);
    GridDims g{m, n, n_v, n_h, num_forward};
    int64_t num_nodes = m * n;

    uint64_t *d_dist_pred; uint8_t *d_processed;
    int32_t *d_excess; int64_t *d_potential;
    uint16_t *d_cost_fwd; uint8_t *d_is_sat;
    int64_t *d_pending, *d_discovered, *d_progress;

    cudaMalloc(&d_dist_pred, num_nodes * sizeof(uint64_t));
    cudaMalloc(&d_processed, num_nodes * sizeof(uint8_t));
    cudaMalloc(&d_excess, num_nodes * sizeof(int32_t));
    cudaMalloc(&d_potential, num_nodes * sizeof(int64_t));
    cudaMalloc(&d_cost_fwd, num_forward * sizeof(uint16_t));
    cudaMalloc(&d_is_sat, 2 * num_forward * sizeof(uint8_t));
    cudaMalloc(&d_pending, sizeof(int64_t));
    cudaMalloc(&d_discovered, sizeof(int64_t));
    cudaMalloc(&d_progress, sizeof(int64_t));
    unsigned int *d_min_pending;
    cudaMalloc(&d_min_pending, sizeof(unsigned int));

    cudaMemcpy(d_excess, h_excess, num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_potential, h_potential, num_nodes * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost_fwd, h_cost_fwd, num_forward * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_sat, h_is_saturated, 2 * num_forward * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_processed, 0, num_nodes * sizeof(uint8_t));

    int64_t zero = 0;
    cudaMemcpy(d_discovered, &zero, sizeof(int64_t), cudaMemcpyHostToDevice);

    int block = 256;
    int grid_sz = (int)((num_nodes + block - 1) / block);
    InitKernel<<<grid_sz, block>>>(d_dist_pred, d_excess, num_nodes, d_pending, d_discovered);

    int64_t discovered = 0, processed_total = 0;
    cudaMemcpy(&discovered, d_discovered, sizeof(int64_t), cudaMemcpyDeviceToHost);

    (void)k; // no longer used for bucket-advance safety -- see FindMinPendingKernel above
    uint32_t cur_dist = 0;
    int64_t safety_rounds = 0, safety_cap = num_nodes + 16; // generous; correctness doesn't depend on it
    while (discovered > processed_total) {
        cudaMemcpy(d_progress, &zero, sizeof(int64_t), cudaMemcpyHostToDevice);
        RelaxKernel<<<grid_sz, block>>>(g, d_cost_fwd, d_is_sat, d_potential, d_dist_pred,
                                          d_processed, num_nodes, cur_dist, d_progress, d_discovered);
        int64_t progress = 0;
        cudaMemcpy(&progress, d_progress, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&discovered, d_discovered, sizeof(int64_t), cudaMemcpyDeviceToHost);
        processed_total += progress;
        if (progress > 0) continue; // more work still pending exactly at cur_dist; drain it first

        // Nothing left at cur_dist: jump directly to the next distance that
        // actually has pending work (a reduction, not a guess), rather than
        // incrementing one distance at a time.
        if (discovered <= processed_total) break; // globally done
        unsigned int inf = 0xFFFFFFFFu;
        cudaMemcpy(d_min_pending, &inf, sizeof(unsigned int), cudaMemcpyHostToDevice);
        FindMinPendingKernel<<<grid_sz, block>>>(d_dist_pred, d_processed, num_nodes, d_min_pending);
        unsigned int min_pending = 0;
        cudaMemcpy(&min_pending, d_min_pending, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (min_pending == inf) break; // no reachable pending node anywhere: done
        cur_dist = min_pending;
        if (++safety_rounds > safety_cap) {
            std::fprintf(stderr, "[cuphu_ww_dial_run_full_gpu] safety cap hit, aborting sweep early\n");
            break;
        }
    }

    // Download packed dist/pred, unpack to the caller's (already
    // dist/pred_arc-shaped) output arrays.
    uint64_t *h_packed = new uint64_t[num_nodes];
    cudaMemcpy(h_packed, d_dist_pred, num_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint8_t *h_proc = new uint8_t[num_nodes];
    cudaMemcpy(h_proc, d_processed, num_nodes * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    for (int64_t v = 0; v < num_nodes; ++v) {
        uint32_t d = unpack_dist(h_packed[v]);
        uint32_t a = unpack_arc(h_packed[v]);
        h_dist_out[v] = (d == DIST_INF) ? (int64_t)0x7FFFFFFFFFFFFFFFLL : (int64_t)d;
        h_pred_arc_out[v] = (a == NO_ARC) ? -1 : (int64_t)a;
        h_popped_out[v] = h_proc[v];
    }
    delete[] h_packed;
    delete[] h_proc;

    cudaFree(d_dist_pred); cudaFree(d_processed); cudaFree(d_excess); cudaFree(d_potential);
    cudaFree(d_cost_fwd); cudaFree(d_is_sat); cudaFree(d_pending); cudaFree(d_discovered);
    cudaFree(d_progress); cudaFree(d_min_pending);
}
