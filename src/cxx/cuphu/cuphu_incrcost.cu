/**
 * cuphu_incrcost.cu
 *
 * GPU computation of incremental arc costs for SNAPHU smooth-cost mode.
 * Replicates CalcCostSmooth() from snaphu_cost.c in parallel across all arcs.
 *
 * Arc layout (flat):
 *   [0 .. (nrow-1)*ncol - 1]                   : row arcs (azimuth)
 *   [(nrow-1)*ncol .. (nrow-1)*ncol + nrow*(ncol-1) - 1] : col arcs (range)
 *
 * Also performs a parallel min-reduction to find min(poscost) and min(negcost).
 * If both minima are >= 0, the current flow is at a local MCF optimum for
 * nflow=1 and TreeSolve can be safely skipped.
 */

#include "cuphu.h"
#include <cuda_runtime.h>

extern "C" {
#include "snaphu.h"
}

/* ── device helpers ─────────────────────────────────────────────────────────── */

__device__ __forceinline__ long gpu_ceil_div(long a, long b)
{
    /* ceil(a/b) for integer b > 0 */
    return (a > 0L) ? (a + b - 1L) / b : a / b;
}

__device__ __forceinline__ long gpu_floor_div(long a, long b)
{
    /* floor(a/b) for integer b > 0 */
    return (a >= 0L) ? a / b : (a - b + 1L) / b;
}

/* ── kernel: compute poscost/negcost for every arc ──────────────────────────── */
/*
 * Matches CalcCostSmooth() in snaphu_cost.c:
 *
 *   idz1    = |flow * nshortcycle + offset|
 *   idz2pos = |(flow+nflow) * nshortcycle + offset|
 *   idz2neg = |(flow-nflow) * nshortcycle + offset|
 *
 *   cost1       = (idz1²)    / sigsq               (integer division)
 *   poscost_raw = (idz2pos²) / sigsq - cost1
 *   negcost_raw = (idz2neg²) / sigsq - cost1
 *
 *   poscost = ceil (poscost_raw / nflow²)   if poscost_raw > 0
 *           = floor(poscost_raw / nflow²)   otherwise
 *   (same for negcost)
 *
 *   Special case: sigsq == LARGESHORT (32000) → poscost = negcost = 0
 */
__global__ void ComputeIncrCostsKernel(
    const smoothcostT *d_costs,    /* flat arc costs                           */
    const short       *d_flows,    /* flat arc flows (same layout)             */
    int                narcs,
    long               nshortcycle,
    int                nflow,
    short             *d_poscost,  /* output                                   */
    short             *d_negcost   /* output                                   */
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= narcs) return;

    smoothcostT c = d_costs[idx];
    long flow     = d_flows[idx];

    /* LARGESHORT (32000): arc has no cost preference */
    if (c.sigsq == 32000) {
        d_poscost[idx] = 0;
        d_negcost[idx] = 0;
        return;
    }

    long offset  = (long)c.offset;
    long sigsq   = (long)c.sigsq;
    long nfl     = (long)nflow;

    long idz1    = llabs(flow * nshortcycle + offset);
    long idz2pos = llabs((flow + nfl) * nshortcycle + offset);
    long idz2neg = llabs((flow - nfl) * nshortcycle + offset);

    long cost1       = (idz1 * idz1) / sigsq;
    long poscost_raw = (idz2pos * idz2pos) / sigsq - cost1;
    long negcost_raw = (idz2neg * idz2neg) / sigsq - cost1;

    long nfsq = nfl * nfl;
    long pc   = (poscost_raw > 0L) ? gpu_ceil_div (poscost_raw, nfsq)
                                    : gpu_floor_div(poscost_raw, nfsq);
    long nc   = (negcost_raw > 0L) ? gpu_ceil_div (negcost_raw, nfsq)
                                    : gpu_floor_div(negcost_raw, nfsq);

    d_poscost[idx] = (short)max(-32000L, min(32000L, pc));
    d_negcost[idx] = (short)max(-32000L, min(32000L, nc));
}

/* ── kernel: parallel min-reduction ─────────────────────────────────────────── */
/*
 * Finds min(poscost) and min(negcost) across all arcs via shared-memory tree.
 * Uses atomicMin to merge block results into d_min[0] (pos) and d_min[1] (neg).
 * Caller must initialize d_min[0..1] to LARGESHORT (32000) before launch.
 */
__global__ void MinCostReduceKernel(
    const short *d_poscost,
    const short *d_negcost,
    int          narcs,
    int         *d_min          /* [0]=min_pos, [1]=min_neg */
)
{
    extern __shared__ int sdata[];   /* 2 × blockDim.x ints */
    int *spos = sdata;
    int *sneg = sdata + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    spos[tid] = (idx < narcs) ? (int)d_poscost[idx] : 32000;
    sneg[tid] = (idx < narcs) ? (int)d_negcost[idx] : 32000;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if ((unsigned)tid < s) {
            spos[tid] = min(spos[tid], spos[tid + s]);
            sneg[tid] = min(sneg[tid], sneg[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(&d_min[0], spos[0]);
        atomicMin(&d_min[1], sneg[0]);
    }
}

/* ── public entry point ──────────────────────────────────────────────────────── */
/**
 * Compute incremental costs for all arcs on GPU, then check whether all
 * poscosts ≥ 0 AND all negcosts ≥ 0 (i.e., no negative-cost augmentation
 * is possible at flow increment nflow).
 *
 * If true is returned, the current flow is at a local optimum for nflow=1
 * and the CPU TreeSolve can be skipped entirely.
 *
 * @param d_costs_flat   GPU smoothcostT array, flat layout (narcs elements)
 * @param d_flows_flat   GPU short array, same flat layout
 * @param narcs          total arc count = (nrow-1)*ncol + nrow*(ncol-1)
 * @param nshortcycle    from paramT::nshortcycle
 * @param nflow          flow increment (pass 1 for the initial check)
 * @param d_poscost      GPU output: narcs shorts (pre-allocated)
 * @param d_negcost      GPU output: narcs shorts (pre-allocated)
 * @param d_scratch      GPU scratch: 2 ints (pre-allocated)
 */
extern "C"
bool cuphu_incrcost_early_exit(
    const smoothcostT *d_costs_flat,
    const short       *d_flows_flat,
    int                narcs,
    long               nshortcycle,
    int                nflow,
    short             *d_poscost,
    short             *d_negcost,
    int               *d_scratch
)
{
    const int threads = 256;
    const int blocks  = (narcs + threads - 1) / threads;

    /* compute incremental costs in parallel */
    ComputeIncrCostsKernel<<<blocks, threads>>>(
        d_costs_flat, d_flows_flat, narcs, nshortcycle, nflow,
        d_poscost, d_negcost);

    /* initialize min values to LARGESHORT */
    int init_val[2] = { 32000, 32000 };
    CUDA_CHECK(cudaMemcpy(d_scratch, init_val, 2 * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* parallel min-reduction */
    MinCostReduceKernel<<<blocks, threads, 2u * threads * sizeof(int)>>>(
        d_poscost, d_negcost, narcs, d_scratch);

    int h_min[2];
    CUDA_CHECK(cudaMemcpy(h_min, d_scratch, 2 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    /* if min(poscost) >= 0 AND min(negcost) >= 0: no improvement possible */
    return (h_min[0] >= 0 && h_min[1] >= 0);
}
