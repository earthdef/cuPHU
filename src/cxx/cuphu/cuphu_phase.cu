/**
 * cuphu_phase.cu
 *
 * GPU kernel for integrating arc flows into an unwrapped phase array.
 *
 * The unwrapped phase is recovered by path-integration:
 *
 *   φ[r][c] = φ[0][0]
 *             + Σ_{i=0..r-1} [ W(ψ[i+1][0] − ψ[i][0]) + 2π · k_h[i][0] ]   (down col-0)
 *             + Σ_{j=0..c-1} [ W(ψ[r][j+1] − ψ[r][j]) + 2π · k_v[r][j] ]   (right in row r)
 *
 * where W(x) = x − 2π·round(x/2π) wraps to (−π, π],
 *       k_h[i][c]  = hflows[i*ncol+c]  = integer cycle count on arc (i,c)→(i+1,c),
 *       k_v[r][j]  = vflows[r*(ncol-1)+j] = integer cycle count on arc (r,j)→(r,j+1).
 *
 * The MCF solution guarantees path-independence for a consistent flow; we
 * use the "down column-0, then right along row r" path for simplicity.
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <math_constants.h>

static constexpr float TWOPI_F = 2.0f * CUDART_PI_F;

/* ── device helper: wrap angle to (−π, π] ───────────────────────────────── */
__device__ __forceinline__ float wrap_angle(float x) {
    return x - TWOPI_F * rintf(x / TWOPI_F);
}

/* ── kernel: integrate downward along column 0 ──────────────────────────── */
/*
 * row_offset[r] = φ[r][0] − φ[0][0]
 *               = Σ_{i=0}^{r-1} [ W(ψ[i+1][0] − ψ[i][0]) − 2π · k_h[i][0] ]
 *
 * NOTE the minus: SNAPHU row flows pair with the azimuth up-diff
 * (ψ[r] − ψ[r+1]) convention of CalcWrappedAzDiffs(), so integrating
 * downward subtracts the flow (see SNAPHU IntegratePhase()).
 *
 * Runs as a single thread (sequential cumulative sum, nrow iterations).
 */
__global__ void IntegrateRowOffsetKernel(
    float       *row_offset,    /* nrow output floats                 */
    const float *phase,         /* nrow*ncol wrapped phase (radians)  */
    const short *hflows,        /* (nrow-1)*ncol horizontal arc flows */
    int          nrow,
    int          ncol
) {
    /* only one thread does the work */
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float cumsum = 0.0f;
    row_offset[0] = 0.0f;
    for (int r = 1; r < nrow; ++r) {
        float dpsi = phase[r * ncol] - phase[(r - 1) * ncol];
        cumsum += wrap_angle(dpsi) - TWOPI_F * (float)hflows[(r - 1) * ncol];
        row_offset[r] = cumsum;
    }
}

/* ── kernel: integrate rightward within each row ────────────────────────── */
/*
 * col_offset[r*ncol + c] = φ[r][c] − φ[r][0]
 *               = Σ_{j=0}^{c-1} [ W(ψ[r][j+1] − ψ[r][j]) + 2π · k_v[r][j] ]
 *
 * One block per row, single thread per block.
 */
__global__ void IntegrateColOffsetKernel(
    float       *col_offset,    /* nrow*ncol output floats             */
    const float *phase,         /* nrow*ncol wrapped phase (radians)   */
    const short *vflows,        /* nrow*(ncol-1) vertical arc flows    */
    int          nrow,
    int          ncol
) {
    int row = blockIdx.x;
    if (row >= nrow || threadIdx.x != 0) return;

    float cumsum = 0.0f;
    col_offset[row * ncol] = 0.0f;
    for (int c = 1; c < ncol; ++c) {
        float dpsi = phase[row * ncol + c] - phase[row * ncol + c - 1];
        cumsum += wrap_angle(dpsi) + TWOPI_F * (float)vflows[row * (ncol - 1) + c - 1];
        col_offset[row * ncol + c] = cumsum;
    }
}

/* ── kernel: combine into unwrapped phase ───────────────────────────────── */
/*
 * φ[r][c] = φ[0][0] + row_offset[r] + col_offset[r*ncol+c]
 */
__global__ void ApplyOffsetKernel(
    float       *unw,
    const float *phase,
    const float *row_offset,    /* nrow  */
    const float *col_offset,    /* nrow*ncol */
    float        ref_phase,     /* phase[0][0] */
    int          nrow,
    int          ncol
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n   = nrow * ncol;
    if (idx >= n) return;
    int row = idx / ncol;
    unw[idx] = ref_phase + row_offset[row] + col_offset[idx];
    (void)phase;   /* not needed; ref_phase captures ψ[0][0] */
}

/* ── public entry point ──────────────────────────────────────────────────── */
extern "C"
void cuphu_integrate_phase_gpu(
    const float *d_phase,   /* nrow*ncol wrapped phase (radians) [device] */
    const short *d_hflows,  /* (nrow-1)*ncol azimuth flows       [device] */
    const short *d_vflows,  /* nrow*(ncol-1) range flows         [device] */
    int          nrow,
    int          ncol,
    int          gpu_id,
    float       *d_unw      /* nrow*ncol output (radians)        [device] */
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    float *d_row_off = nullptr, *d_col_off = nullptr;
    size_t npix = (size_t)nrow * ncol;
    CUDA_CHECK(cudaMalloc(&d_row_off, (size_t)nrow * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_off, npix * sizeof(float)));

    /* row offsets: single thread, sequential scan down column 0 */
    IntegrateRowOffsetKernel<<<1, 1>>>(d_row_off, d_phase, d_hflows, nrow, ncol);

    /* col offsets: one block per row, single thread per block */
    IntegrateColOffsetKernel<<<nrow, 1>>>(d_col_off, d_phase, d_vflows, nrow, ncol);

    /* read reference phase ψ[0][0] from device */
    float ref_phase = 0.0f;
    CUDA_CHECK(cudaMemcpy(&ref_phase, d_phase, sizeof(float), cudaMemcpyDeviceToHost));

    /* combine */
    int threads = 256;
    int blocks  = ((int)npix + threads - 1) / threads;
    ApplyOffsetKernel<<<blocks, threads>>>(
        d_unw, d_phase, d_row_off, d_col_off, ref_phase, nrow, ncol);

    CUDA_CHECK(cudaFree(d_row_off));
    CUDA_CHECK(cudaFree(d_col_off));
}
