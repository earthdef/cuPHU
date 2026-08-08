/**
 * cuphu_cost.cu
 *
 * GPU kernels for SNAPHU statistical cost computation.
 *
 * Cost arrays follow SNAPHU's "2-D row/col" layout:
 *   costs[0..nrow-2][0..ncol-1]  → row arcs  (azimuth / along-track)
 *   costs[nrow-1..2*nrow-3][0..ncol-2]  → col arcs  (range / cross-track)
 * stored as a flat device array of smoothcostT (smooth mode) or costT
 * (topo/defo mode).
 *
 * We expose separate kernels for each mode so the caller can template on
 * cost type at compile time, avoiding runtime branches in inner loops.
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <math_constants.h>

/* bring in SNAPHU's cost structures */
extern "C" {
#include "snaphu.h"
}

/* forward declarations from cuphu_util.cu */
extern "C" void cuphu_phase_diffs(const float*, float*, float*, int, int, cudaStream_t);
extern "C" void cuphu_boxcar2d(const float*, float*, int, int, int, int, cudaStream_t);

/* ── device constants (written once from host before kernel launch) ────── */
__constant__ double d_rho0;
__constant__ double d_defocorrthresh;
__constant__ double d_rhopow;
__constant__ double d_sigsqrhoconst;
__constant__ double d_sigsqcorr;
__constant__ double d_nshortcycle;
__constant__ double d_nshortcyclesq;
__constant__ double d_costscale;
__constant__ int    d_sigsqshortmin;

/* ── helper: upload scalar constants to device ────────────────────────── */
static void upload_smooth_constants(const CuPhuParams &p) {
    double rho0 = p.rhosconst1 / p.ncorrlooks + p.rhosconst2;
    double defocorrthresh = p.defothreshfactor * rho0;
    double rhopow = 2.0 * p.cstd1 + p.cstd2 * log(p.ncorrlooks) + p.cstd3 * p.ncorrlooks;
    double sigsqrhoconst = 2.0 / 12.0;
    double nsc = p.nshortcycle;
    double nsc2 = nsc * nsc;
    int    sqmin = (int)p.sigsqshortmin;

    CUDA_CHECK(cudaMemcpyToSymbol(d_rho0,           &rho0,           sizeof(rho0)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_defocorrthresh, &defocorrthresh, sizeof(defocorrthresh)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rhopow,         &rhopow,         sizeof(rhopow)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sigsqrhoconst,  &sigsqrhoconst,  sizeof(sigsqrhoconst)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sigsqcorr,      &p.sigsqcorr,    sizeof(p.sigsqcorr)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nshortcycle,    &nsc,            sizeof(nsc)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nshortcyclesq,  &nsc2,           sizeof(nsc2)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_costscale,      &p.costscale,    sizeof(p.costscale)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sigsqshortmin,  &sqmin,          sizeof(sqmin)));
}

/* ══════════════════════════════════════════════════════════════════════════
 * SMOOTH-mode cost kernels
 * ══════════════════════════════════════════════════════════════════════════
 *
 * smoothcostT { short offset; short sigsq; }
 *
 * For each arc:
 *   rho = (corr[a] + corr[b]) / 2        (a,b = the two endpoint pixels)
 *   if rho < defocorrthresh: rho = 0
 *   sigsqrho = (sigsqrhoconst * (1-rho)^rhopow + sigsqcorr) * nshortcycle²
 *   offset  = nshortcycle * (dpsi - (rho>0 ? avgdpsi : 0.5*avgdpsi))
 *   sigsq   = max(sigsqrhoconst_per_arc, sigsqshortmin)
 *             (divided by costscale * weight)
 */

/* ── column (range) arcs: shape nrow × (ncol-1) ───────────────────────── */
__global__ void BuildSmoothColCostsKernel(
    smoothcostT *colcost,     /* nrow*(ncol-1) output                   */
    const float *dpsi,        /* range diffs, nrow*(ncol-1)             */
    const float *avgdpsi,     /* smoothed range diffs, nrow*(ncol-1)    */
    const float *corr,        /* coherence, nrow*ncol                   */
    const short *colweight,   /* arc weights, nrow*(ncol-1); NULL→all 1 */
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* 0..ncol-2 */
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* 0..nrow-1 */
    if (row >= nrow || col >= ncol - 1) return;

    int arc_idx = row * (ncol - 1) + col;

    /* weight: 0 → masked arc */
    short w = colweight ? colweight[arc_idx] : 1;
    if (w == 0) {
        colcost[arc_idx].offset = NOCOSTSHELF;
        colcost[arc_idx].sigsq  = 0;
        return;
    }

    double rho = 0.5 * ((double)corr[row * ncol + col] +
                        (double)corr[row * ncol + col + 1]);
    if (rho < d_defocorrthresh) rho = 0.0;

    double sigsqrho = (d_sigsqrhoconst * pow(1.0 - rho, d_rhopow) + d_sigsqcorr)
                      * d_nshortcyclesq;

    double dp    = (double)dpsi[arc_idx];
    double avgdp = (double)avgdpsi[arc_idx];
    double offset = d_nshortcycle * (dp - (rho > 0 ? avgdp : 0.5 * avgdp));

    double sigsq_f = sigsqrho / (d_costscale * (double)w);
    int    sigsq_i = (int)sigsq_f;
    if (sigsq_i < d_sigsqshortmin) sigsq_i = d_sigsqshortmin;

    /* clamp to short range */
    int off_i = __double2int_rn(offset);
    off_i = max(off_i, (int)SHRT_MIN);
    off_i = min(off_i, (int)SHRT_MAX);
    sigsq_i = min(sigsq_i, (int)SHRT_MAX);

    colcost[arc_idx].offset = (short)off_i;
    colcost[arc_idx].sigsq  = (short)sigsq_i;
}

/* ── row (azimuth) arcs: shape (nrow-1) × ncol ────────────────────────── */
__global__ void BuildSmoothRowCostsKernel(
    smoothcostT *rowcost,     /* (nrow-1)*ncol output                   */
    const float *dpsi,        /* azimuth diffs, (nrow-1)*ncol           */
    const float *avgdpsi,     /* smoothed az diffs, (nrow-1)*ncol       */
    const float *corr,        /* coherence, nrow*ncol                   */
    const short *rowweight,   /* arc weights, (nrow-1)*ncol; NULL→all 1 */
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow - 1 || col >= ncol) return;

    int arc_idx = row * ncol + col;

    short w = rowweight ? rowweight[arc_idx] : 1;
    if (w == 0) {
        rowcost[arc_idx].offset = NOCOSTSHELF;
        rowcost[arc_idx].sigsq  = 0;
        return;
    }

    double rho = 0.5 * ((double)corr[row * ncol + col] +
                        (double)corr[(row + 1) * ncol + col]);
    if (rho < d_defocorrthresh) rho = 0.0;

    double sigsqrho = (d_sigsqrhoconst * pow(1.0 - rho, d_rhopow) + d_sigsqcorr)
                      * d_nshortcyclesq;

    double dp    = (double)dpsi[arc_idx];
    double avgdp = (double)avgdpsi[arc_idx];
    double offset = d_nshortcycle * (dp - (rho > 0 ? avgdp : 0.5 * avgdp));

    double sigsq_f = sigsqrho / (d_costscale * (double)w);
    int    sigsq_i = (int)sigsq_f;
    if (sigsq_i < d_sigsqshortmin) sigsq_i = d_sigsqshortmin;

    int off_i = __double2int_rn(offset);
    off_i = max(off_i, (int)SHRT_MIN);
    off_i = min(off_i, (int)SHRT_MAX);
    sigsq_i = min(sigsq_i, (int)SHRT_MAX);

    rowcost[arc_idx].offset = (short)off_i;
    rowcost[arc_idx].sigsq  = (short)sigsq_i;
}

/* ══════════════════════════════════════════════════════════════════════════
 * DEFO-mode cost kernels
 * ══════════════════════════════════════════════════════════════════════════
 *
 * costT { short offset; short sigsq; short dzmax; short laycost; }
 *
 * Defo mode is almost identical to smooth mode for the basic sigsq / offset
 * computation.  The dzmax / laycost fields encode the discontinuity shelf.
 * For defo:
 *   sigsqrhoconst = 2.0/12.0   (same as smooth)
 *   nshortcycle   = same
 *   defomax (in cycles) → short: defomax_n = round(nshortcycle * defomax)
 *   cost shelf: laycost determined from defomax and sigsq
 */

/* additional constants for defo mode */
__constant__ double d_defomax;

static void upload_defo_constants(const CuPhuParams &p) {
    upload_smooth_constants(p);
    CUDA_CHECK(cudaMemcpyToSymbol(d_defomax, &p.defomax, sizeof(p.defomax)));
}

__global__ void BuildDefoColCostsKernel(
    costT       *colcost,
    const float *dpsi,
    const float *avgdpsi,
    const float *corr,
    const short *colweight,
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol - 1) return;

    int arc_idx = row * (ncol - 1) + col;

    short w = colweight ? colweight[arc_idx] : 1;
    if (w == 0) {
        colcost[arc_idx].offset  = NOCOSTSHELF;
        colcost[arc_idx].sigsq   = 0;
        colcost[arc_idx].dzmax   = SHRT_MAX;
        colcost[arc_idx].laycost = 0;
        return;
    }

    double rho = 0.5 * ((double)corr[row * ncol + col] +
                        (double)corr[row * ncol + col + 1]);
    if (rho < d_defocorrthresh) rho = 0.0;

    double sigsqrho = (d_sigsqrhoconst * pow(1.0 - rho, d_rhopow) + d_sigsqcorr)
                      * d_nshortcyclesq;

    double dp    = (double)dpsi[arc_idx];
    double avgdp = (double)avgdpsi[arc_idx];
    double offset = d_nshortcycle * (dp - (rho > 0 ? avgdp : 0.5 * avgdp));

    double sigsq_f = sigsqrho / (d_costscale * (double)w);
    int    sigsq_i = (int)sigsq_f;
    if (sigsq_i < d_sigsqshortmin) sigsq_i = d_sigsqshortmin;

    /* dzmax: number of nshortcycle steps before discontinuity shelf */
    int dzmax = __double2int_rn(d_nshortcycle * d_defomax);
    dzmax = max(1, min(dzmax, (int)SHRT_MAX));

    /* laycost: cost of a discontinuity, set proportional to sigsq */
    int laycost = (int)sqrt((double)(dzmax * sigsq_i));
    laycost = min(laycost, (int)SHRT_MAX);

    int off_i = __double2int_rn(offset);
    off_i = max(off_i, (int)SHRT_MIN);
    off_i = min(off_i, (int)SHRT_MAX);
    sigsq_i = min(sigsq_i, (int)SHRT_MAX);

    colcost[arc_idx].offset  = (short)off_i;
    colcost[arc_idx].sigsq   = (short)sigsq_i;
    colcost[arc_idx].dzmax   = (short)dzmax;
    colcost[arc_idx].laycost = (short)laycost;
}

__global__ void BuildDefoRowCostsKernel(
    costT       *rowcost,
    const float *dpsi,
    const float *avgdpsi,
    const float *corr,
    const short *rowweight,
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow - 1 || col >= ncol) return;

    int arc_idx = row * ncol + col;

    short w = rowweight ? rowweight[arc_idx] : 1;
    if (w == 0) {
        rowcost[arc_idx].offset  = NOCOSTSHELF;
        rowcost[arc_idx].sigsq   = 0;
        rowcost[arc_idx].dzmax   = SHRT_MAX;
        rowcost[arc_idx].laycost = 0;
        return;
    }

    double rho = 0.5 * ((double)corr[row * ncol + col] +
                        (double)corr[(row + 1) * ncol + col]);
    if (rho < d_defocorrthresh) rho = 0.0;

    double sigsqrho = (d_sigsqrhoconst * pow(1.0 - rho, d_rhopow) + d_sigsqcorr)
                      * d_nshortcyclesq;

    double dp    = (double)dpsi[arc_idx];
    double avgdp = (double)avgdpsi[arc_idx];
    double offset = d_nshortcycle * (dp - (rho > 0 ? avgdp : 0.5 * avgdp));

    double sigsq_f = sigsqrho / (d_costscale * (double)w);
    int    sigsq_i = (int)sigsq_f;
    if (sigsq_i < d_sigsqshortmin) sigsq_i = d_sigsqshortmin;

    int dzmax = __double2int_rn(d_nshortcycle * d_defomax);
    dzmax = max(1, min(dzmax, (int)SHRT_MAX));
    int laycost = (int)sqrt((double)(dzmax * sigsq_i));
    laycost = min(laycost, (int)SHRT_MAX);

    int off_i = __double2int_rn(offset);
    off_i = max(off_i, (int)SHRT_MIN);
    off_i = min(off_i, (int)SHRT_MAX);
    sigsq_i = min(sigsq_i, (int)SHRT_MAX);

    rowcost[arc_idx].offset  = (short)off_i;
    rowcost[arc_idx].sigsq   = (short)sigsq_i;
    rowcost[arc_idx].dzmax   = (short)dzmax;
    rowcost[arc_idx].laycost = (short)laycost;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Public C++ entry points
 * ══════════════════════════════════════════════════════════════════════════ */

/**
 * Compute smooth-mode cost arrays entirely on GPU.
 *
 * Inputs are device pointers.  Output is a freshly allocated device array
 * of smoothcostT in SNAPHU layout: row costs [(nrow-1)*ncol] followed
 * immediately by col costs [nrow*(ncol-1)].
 *
 * The caller is responsible for freeing *d_costs_out via cudaFree.
 */
extern "C"
void cuphu_build_smooth_costs_gpu(
    const float        *d_phase,   /* nrow*ncol wrapped phase (radians)  */
    const float        *d_corr,    /* nrow*ncol coherence                */
    const short        *d_roww,    /* (nrow-1)*ncol weights; NULL→all 1  */
    const short        *d_colw,    /* nrow*(ncol-1) weights; NULL→all 1  */
    int                 nrow,
    int                 ncol,
    const CuPhuParams *params,
    int                 kperpdpsi,
    int                 kpardpsi,
    smoothcostT       **d_costs_out,  /* [out] allocated here             */
    cudaStream_t        stream
) {
    upload_smooth_constants(*params);

    /* ── phase difference arrays ───────────────────────────────────────── */
    float *d_dpsi_rng, *d_dpsi_az;
    CUDA_CHECK(cudaMallocAsync(&d_dpsi_rng, (size_t)nrow * (ncol - 1) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dpsi_az,  (size_t)(nrow - 1) * ncol  * sizeof(float), stream));

    cuphu_phase_diffs(d_phase, d_dpsi_rng, d_dpsi_az, nrow, ncol, stream);

    /* ── smoothed (averaged) phase difference arrays ────────────────────── */
    float *d_avg_rng, *d_avg_az;
    CUDA_CHECK(cudaMallocAsync(&d_avg_rng, (size_t)nrow * (ncol - 1) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_avg_az,  (size_t)(nrow - 1) * ncol  * sizeof(float), stream));

    /* range diffs: box-car with kperpdpsi rows and kpardpsi cols */
    cuphu_boxcar2d(d_dpsi_rng, d_avg_rng, nrow, ncol - 1,
                      kperpdpsi / 2, kpardpsi / 2, stream);
    /* azimuth diffs: transposed kernel sizes */
    cuphu_boxcar2d(d_dpsi_az,  d_avg_az,  nrow - 1, ncol,
                      kpardpsi / 2, kperpdpsi / 2, stream);

    /* ── allocate output (row costs + col costs contiguous) ─────────────── */
    size_t nrowcost = (size_t)(nrow - 1) * ncol;
    size_t ncolcost = (size_t)nrow * (ncol - 1);
    smoothcostT *d_costs = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_costs,
        (nrowcost + ncolcost) * sizeof(smoothcostT), stream));

    smoothcostT *d_rowcost = d_costs;
    smoothcostT *d_colcost = d_costs + nrowcost;

    dim3 block(16, 16);

    /* col costs */
    {
        dim3 grid(
            (ncol - 1 + block.x - 1) / block.x,
            (nrow     + block.y - 1) / block.y
        );
        BuildSmoothColCostsKernel<<<grid, block, 0, stream>>>(
            d_colcost, d_dpsi_rng, d_avg_rng, d_corr, d_colw, nrow, ncol);
    }

    /* row costs */
    {
        dim3 grid(
            (ncol     + block.x - 1) / block.x,
            (nrow - 1 + block.y - 1) / block.y
        );
        BuildSmoothRowCostsKernel<<<grid, block, 0, stream>>>(
            d_rowcost, d_dpsi_az, d_avg_az, d_corr, d_roww, nrow, ncol);
    }

    CUDA_CHECK(cudaFreeAsync(d_dpsi_rng, stream));
    CUDA_CHECK(cudaFreeAsync(d_dpsi_az,  stream));
    CUDA_CHECK(cudaFreeAsync(d_avg_rng,  stream));
    CUDA_CHECK(cudaFreeAsync(d_avg_az,   stream));

    *d_costs_out = d_costs;
}

/**
 * Compute defo-mode cost arrays on GPU.
 * Layout identical to smooth mode but elements are costT.
 */
extern "C"
void cuphu_build_defo_costs_gpu(
    const float        *d_phase,
    const float        *d_corr,
    const short        *d_roww,
    const short        *d_colw,
    int                 nrow,
    int                 ncol,
    const CuPhuParams *params,
    int                 kperpdpsi,
    int                 kpardpsi,
    costT             **d_costs_out,
    cudaStream_t        stream
) {
    upload_defo_constants(*params);

    float *d_dpsi_rng, *d_dpsi_az, *d_avg_rng, *d_avg_az;
    CUDA_CHECK(cudaMallocAsync(&d_dpsi_rng, (size_t)nrow * (ncol - 1) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dpsi_az,  (size_t)(nrow - 1) * ncol  * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_avg_rng,  (size_t)nrow * (ncol - 1) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_avg_az,   (size_t)(nrow - 1) * ncol  * sizeof(float), stream));

    cuphu_phase_diffs(d_phase, d_dpsi_rng, d_dpsi_az, nrow, ncol, stream);
    cuphu_boxcar2d(d_dpsi_rng, d_avg_rng, nrow, ncol - 1,
                      kperpdpsi / 2, kpardpsi / 2, stream);
    cuphu_boxcar2d(d_dpsi_az,  d_avg_az,  nrow - 1, ncol,
                      kpardpsi / 2, kperpdpsi / 2, stream);

    size_t nrowcost = (size_t)(nrow - 1) * ncol;
    size_t ncolcost = (size_t)nrow * (ncol - 1);
    costT *d_costs = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_costs,
        (nrowcost + ncolcost) * sizeof(costT), stream));

    costT *d_rowcost = d_costs;
    costT *d_colcost = d_costs + nrowcost;

    dim3 block(16, 16);

    {
        dim3 grid(
            (ncol - 1 + block.x - 1) / block.x,
            (nrow     + block.y - 1) / block.y
        );
        BuildDefoColCostsKernel<<<grid, block, 0, stream>>>(
            d_colcost, d_dpsi_rng, d_avg_rng, d_corr, d_colw, nrow, ncol);
    }
    {
        dim3 grid(
            (ncol     + block.x - 1) / block.x,
            (nrow - 1 + block.y - 1) / block.y
        );
        BuildDefoRowCostsKernel<<<grid, block, 0, stream>>>(
            d_rowcost, d_dpsi_az, d_avg_az, d_corr, d_roww, nrow, ncol);
    }

    CUDA_CHECK(cudaFreeAsync(d_dpsi_rng, stream));
    CUDA_CHECK(cudaFreeAsync(d_dpsi_az,  stream));
    CUDA_CHECK(cudaFreeAsync(d_avg_rng,  stream));
    CUDA_CHECK(cudaFreeAsync(d_avg_az,   stream));

    *d_costs_out = d_costs;
}

