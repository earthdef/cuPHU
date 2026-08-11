/**
 * cuphu_conncomp.cu
 *
 * GPU connected-component labeling for the unwrapped phase output.
 *
 * Default algorithm: local min-merge + pointer jumping (Merge/JumpLabelsKernel).
 *   Each round does one local min-merge pass (a pixel pulls in the smallest
 *   label among itself and its valid neighbors, single-buffered via atomicMin)
 *   followed by one pointer-jumping pass (each pixel's label — itself a pixel
 *   index — is replaced by *its* label, i.e. one extra hop of compression).
 *   Alternating these lets label information travel across a component in
 *   O(log diameter) rounds instead of O(diameter): profiling on a 1847x3498
 *   scene showed the old pure-propagation loop needed 5232 iterations and
 *   dominated 98.9% of total GPU kernel time; merge+jump needs O(log(nrow+ncol))
 *   rounds for the same result. Correctness relies on a simple invariant —
 *   labels[x] <= x always, from init (equality) and every update (merge only
 *   ever takes a smaller value; jump only ever follows labels[l] <= l) — so
 *   the benign read/write races from single-buffering (a thread may see a
 *   stale-but-valid neighbor label within the same launch) cost at most an
 *   extra round, never an incorrect result.
 *
 * Legacy algorithm: iterative label propagation (double-buffered, no atomics).
 *   Each pixel takes the minimum label of itself and valid neighbors, one hop
 *   per pass; O(diameter) passes to converge. Kept as a fallback — set
 *   CUPHU_CONNCOMP_LEGACY=1 to force it (e.g. to isolate a suspected
 *   regression in the new algorithm).
 *
 * Two pixels are connected if:
 *   1. Both have coherence above a coarse no-data floor (not masked), AND
 *   2. The arc's incremental cost -- the actual marginal cost of perturbing
 *      the SOLVED MCF flow on that arc by one unit, i.e. min(poscost,negcost)
 *      from CalcCostSmooth()'s formula applied at the converged flow -- is at
 *      least conncompthresh, AND
 *   3. |unw_a - unw_b| < π  (locally consistent unwrapped phase)
 *
 * Condition 2 faithfully mirrors SNAPHU's own GrowConnCompsMask()
 * (snaphu_tile.c): a *large* incremental cost means the solved network is
 * confident about this arc's flow value (moving away from it is expensive);
 * a *small* incremental cost means many flow values are nearly tied there
 * (typical of decorrelated regions, where sigsq is large), so SNAPHU refuses
 * to treat it as reliably connected. This requires the actual solved flow,
 * not just coherence -- an earlier version of this file approximated
 * connectivity from coherence alone (sigsq derived from rho, thresholded
 * directly), which get the qualitative dependence on sigsq backwards
 * relative to incremental cost (incremental cost ~ 1/sigsq at the converged
 * flow) and in practice was even more permissive than the original flat
 * corr_thresh heuristic it replaced. d_poscost/d_negcost below are computed
 * by cuphu_incrcost.cu's ComputeIncrCostsKernel on the actual post-solve
 * flow, in the same flat row-arc-then-col-arc layout as the cost arrays;
 * pass nullptr when no MCF solve exists (e.g. Laplace init), which falls
 * back to phase-agreement + coherence floor only.
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <cstdlib>
#include <vector>

/* device helper: flat arc index for a given pixel's neighbor in direction d
 * (0=up,1=down,2=left,3=right), matching cuphu_cost.cu's row-arc-then-
 * col-arc flat layout: row arcs [0, (nrow-1)*ncol), then col arcs
 * [(nrow-1)*ncol, (nrow-1)*ncol + nrow*(ncol-1)). Returns -1 if out of
 * range (caller already bounds-checks neighbors, so this is just for
 * clarity/safety). */
__device__ __forceinline__ int arc_index(int row, int col, int d,
                                          int nrow, int ncol) {
    int row_arc_count = (nrow - 1) * ncol;
    switch (d) {
        case 0: return (row - 1) * ncol + col;                       /* up: row arc [row-1][col] */
        case 1: return row * ncol + col;                             /* down: row arc [row][col] */
        case 2: return row_arc_count + row * (ncol - 1) + (col - 1); /* left: col arc [row][col-1] */
        case 3: return row_arc_count + row * (ncol - 1) + col;       /* right: col arc [row][col] */
    }
    return -1;
}

/* device helper: is the arc between (row,col) and its direction-d neighbor
 * confidently connected, per SNAPHU's incremental-cost criterion? d_poscost/
 * d_negcost NULL means no MCF solve exists (e.g. Laplace init) -- always
 * confident in that case, matching the original coherence-only behavior. */
__device__ __forceinline__ bool arc_confident(
    const short *d_poscost, const short *d_negcost,
    int row, int col, int d, int nrow, int ncol, long conncompthresh
) {
    if (!d_poscost) return true;
    int aidx = arc_index(row, col, d, nrow, ncol);
    short pc = d_poscost[aidx];
    short nc = d_negcost[aidx];
    short mincost = (pc < nc) ? pc : nc;
    return (long)mincost >= conncompthresh;
}

/* ── kernel: initialize labels ──────────────────────────────────────────── */
__global__ void InitLabelsKernel(uint32_t *labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) labels[idx] = (uint32_t)idx;
}

/* ── kernel: mask invalid pixels with UINT32_MAX ─────────────────────────── */
__global__ void MaskLabelsKernel(
    uint32_t           *labels,
    const float        *corr,
    const unsigned char *mask,
    int                 n,
    float               corr_thresh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool bad = (corr[idx] < corr_thresh);
    if (mask) bad = bad || (mask[idx] == 0);
    if (bad) labels[idx] = 0xFFFFFFFFu;
}

/* ── kernel: one label-propagation pass ─────────────────────────────────── */
/*
 * Each pixel adopts the minimum label among itself and its valid neighbors.
 * Writes new labels into `labels_out` (double-buffered to avoid race).
 * Returns whether any label changed (via d_changed flag).
 */
__global__ void PropagateLabelsKernel(
    const uint32_t *labels_in,
    uint32_t       *labels_out,
    const float    *unw,
    const short    *d_poscost,     /* NULL if no MCF solve (e.g. Laplace) */
    const short    *d_negcost,
    int             nrow,
    int             ncol,
    float           phase_thresh,
    long            conncompthresh,
    int            *d_changed     /* atomically set to 1 if any label changed */
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    int idx = row * ncol + col;
    uint32_t cur = labels_in[idx];

    if (cur == 0xFFFFFFFFu) {         /* masked pixel — stay masked */
        labels_out[idx] = 0xFFFFFFFFu;
        return;
    }

    uint32_t best = cur;

    /* check 4 neighbors */
    int drow[4] = {-1, 1,  0, 0};
    int dcol[4] = { 0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
        int nr = row + drow[d];
        int nc = col + dcol[d];
        if (nr < 0 || nr >= nrow || nc < 0 || nc >= ncol) continue;
        int nidx = nr * ncol + nc;
        uint32_t nlbl = labels_in[nidx];
        if (nlbl == 0xFFFFFFFFu) continue;
        float diff = fabsf(unw[idx] - unw[nidx]);
        if (diff < phase_thresh && nlbl < best &&
            arc_confident(d_poscost, d_negcost, row, col, d, nrow, ncol, conncompthresh))
            best = nlbl;
    }

    labels_out[idx] = best;
    if (best != cur)
        atomicExch(d_changed, 1);
}

/* ── kernel: one local min-merge pass (default path, single-buffer) ──────── */
/*
 * Each valid pixel atomically pulls in the minimum label among itself and
 * its valid neighbors. Single-buffered: safe despite same-launch races
 * because every write only ever decreases labels[idx] (atomicMin), and the
 * labels[x] <= x invariant (see file header) means any value ever read is a
 * valid, if possibly stale, past-or-current state — a race costs at most one
 * extra round, never an incorrect result.
 */
__global__ void MergeLabelsKernel(
    uint32_t    *labels,
    const float *unw,
    const short *d_poscost,     /* NULL if no MCF solve (e.g. Laplace) */
    const short *d_negcost,
    int          nrow,
    int          ncol,
    float        phase_thresh,
    long         conncompthresh,
    int         *d_changed
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    int idx = row * ncol + col;
    uint32_t cur = labels[idx];
    if (cur == 0xFFFFFFFFu) return;   /* masked pixel — stays masked */

    uint32_t best = cur;

    int drow[4] = {-1, 1,  0, 0};
    int dcol[4] = { 0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
        int nr = row + drow[d];
        int nc = col + dcol[d];
        if (nr < 0 || nr >= nrow || nc < 0 || nc >= ncol) continue;
        int nidx = nr * ncol + nc;
        uint32_t nlbl = labels[nidx];
        if (nlbl == 0xFFFFFFFFu) continue;
        float diff = fabsf(unw[idx] - unw[nidx]);
        if (diff < phase_thresh && nlbl < best &&
            arc_confident(d_poscost, d_negcost, row, col, d, nrow, ncol, conncompthresh))
            best = nlbl;
    }

    if (best < cur) {
        atomicMin(&labels[idx], best);
        atomicExch(d_changed, 1);
    }
}

/* ── kernel: pointer jumping / path compression (default path) ───────────── */
/*
 * Every unmasked label is itself a pixel index into `labels`; follow one
 * more hop so long merge chains collapse in O(log depth) rounds instead of
 * needing one MergeLabelsKernel pass per hop of graph distance.
 */
__global__ void JumpLabelsKernel(
    uint32_t *labels,
    int       n,
    int      *d_changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t l = labels[idx];
    if (l == 0xFFFFFFFFu) return;
    uint32_t ll = labels[l];   /* invariant labels[l] <= l, so ll <= l always */
    if (ll < l) {
        labels[idx] = ll;
        atomicExch(d_changed, 1);
    }
}

/* ── kernel: count component sizes ─────────────────────────────────────── */
__global__ void CountComponentsKernel(
    uint32_t       *counts,
    const uint32_t *labels,
    int             n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t root = labels[idx];
    if (root == 0xFFFFFFFFu) return;
    atomicAdd(counts + root, 1u);
}

/* ── kernel: assign final compact labels ────────────────────────────────── */
__global__ void RelabelKernel(
    uint32_t       *labels,
    const uint32_t *map,
    int             n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t root = labels[idx];
    if (root == 0xFFFFFFFFu) {
        labels[idx] = 0u;
        return;
    }
    labels[idx] = map[root];
}

/* ── host helper: compute compact label map on CPU ──────────────────────── */
static void build_label_map(
    const uint32_t *h_counts,
    uint32_t       *h_map,
    int             npix,
    int             min_size,
    int             max_ncomps
) {
    uint32_t next_label = 1;
    for (int i = 0; i < npix && (int)next_label - 1 < max_ncomps; ++i) {
        if (h_counts[i] >= (uint32_t)min_size)
            h_map[i] = next_label++;
        else
            h_map[i] = 0u;
    }
}

/* ── public entry point ──────────────────────────────────────────────────── */
extern "C"
void cuphu_conncomp_gpu(
    const float        *d_unw,
    const float        *d_corr,
    const unsigned char *d_mask,
    int                 nrow,
    int                 ncol,
    const short         *d_poscost,   /* NULL if no MCF solve (e.g. Laplace) */
    const short         *d_negcost,   /* flat row-arc-then-col-arc layout,   */
                                       /* same as d_poscost/cuphu_cost.cu    */
    const CuPhuParams   *params,
    int                 gpu_id,
    uint32_t           *d_labels_out
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    double min_frac       = params->minconncompfrac;
    long   max_ncomps     = params->maxncomps;
    long   conncompthresh = params->conncompthresh;

    int    npix     = nrow * ncol;
    int    threads  = 256;
    int    blocks1d = (npix + threads - 1) / threads;
    dim3   block2d(16, 16);
    dim3   grid2d(
        (ncol + block2d.x - 1) / block2d.x,
        (nrow + block2d.y - 1) / block2d.y
    );

    /* ── initialize labels ────────────────────────────────────────────── */
    InitLabelsKernel<<<blocks1d, threads>>>(d_labels_out, npix);

    /* ── mask invalid pixels: coarse no-data floor only (real connectivity
       gating happens per-arc below via sigsq, matching SNAPHU's incremental-
       cost criterion instead of a flat coherence cutoff) ─────────────────── */
    MaskLabelsKernel<<<blocks1d, threads>>>(
        d_labels_out, d_corr, d_mask, npix, 0.05f);

    /* ── label propagation ────────────────────────────────────────────── */
    int  *d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    float phase_thresh = (float)CUDART_PI_F;

    /* batch every BATCH iterations before a convergence D2H check;
     * this trades BATCH-1 extra kernel launches for ~16× fewer blocking syncs */
    static const int BATCH = 16;
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

    bool legacy = (std::getenv("CUPHU_CONNCOMP_LEGACY") != nullptr);

    if (legacy) {
        /* ── legacy path: double-buffered O(diameter) label propagation ── */
        uint32_t *d_labels_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_labels_tmp, (size_t)npix * sizeof(uint32_t)));

        /* diameter of grid is nrow+ncol; use 2x as safe upper bound */
        int max_iters = 2 * (nrow + ncol);

        uint32_t *src = d_labels_out;
        uint32_t *dst = d_labels_tmp;

        for (int it = 0; it < max_iters; ) {
            int batch_end = std::min(it + BATCH, max_iters);
            for ( ; it < batch_end; ++it) {
                PropagateLabelsKernel<<<grid2d, block2d>>>(
                    src, dst, d_unw, d_poscost, d_negcost, nrow, ncol,
                    phase_thresh, conncompthresh, d_changed);
                uint32_t *tmp = src; src = dst; dst = tmp;
            }

            int h_changed = 0;
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (!h_changed) break;   /* converged within this batch */

            CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        }

        /* after each iteration we swapped src↔dst; final result is in src */
        if (src != d_labels_out)
            CUDA_CHECK(cudaMemcpy(d_labels_out, src,
                                  (size_t)npix * sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaFree(d_labels_tmp));
    } else {
        /* ── default path: local min-merge + pointer jumping, in place ──── */
        /* O(log diameter) rounds; generous safety cap, not expected to hit it */
        int max_iters = 2 * (nrow + ncol);

        for (int it = 0; it < max_iters; ) {
            int batch_end = std::min(it + BATCH, max_iters);
            for ( ; it < batch_end; ++it) {
                MergeLabelsKernel<<<grid2d, block2d>>>(
                    d_labels_out, d_unw, d_poscost, d_negcost, nrow, ncol,
                    phase_thresh, conncompthresh, d_changed);
                JumpLabelsKernel<<<blocks1d, threads>>>(
                    d_labels_out, npix, d_changed);
            }

            int h_changed = 0;
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (!h_changed) break;   /* converged within this batch */

            CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        }
    }

    CUDA_CHECK(cudaFree(d_changed));

    /* ── count sizes ──────────────────────────────────────────────────── */
    uint32_t *d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)npix * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_counts, 0, (size_t)npix * sizeof(uint32_t)));
    CountComponentsKernel<<<blocks1d, threads>>>(d_counts, d_labels_out, npix);

    std::vector<uint32_t> h_counts((size_t)npix);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                          (size_t)npix * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    int min_size = (int)(min_frac * npix);
    if (min_size < 1) min_size = 1;

    std::vector<uint32_t> h_map((size_t)npix, 0u);
    build_label_map(h_counts.data(), h_map.data(), npix,
                    min_size, (int)max_ncomps);

    uint32_t *d_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_map, (size_t)npix * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_map, h_map.data(),
                          (size_t)npix * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    RelabelKernel<<<blocks1d, threads>>>(d_labels_out, d_map, npix);

    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_map));
}
