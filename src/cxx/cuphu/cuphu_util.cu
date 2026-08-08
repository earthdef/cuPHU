/**
 * cuphu_util.cu
 *
 * GPU kernels for pre-processing steps that precede cost computation:
 *   - phase wrapping to [0, 2π)
 *   - wrapped phase difference arrays (range / azimuth)
 *   - 2-D separable box-car averaging (sliding window mean)
 *   - simple amplitude despeckle (median of arms)
 *
 * All arrays are stored row-major (C order), indexed [row * ncol + col].
 * Phase differences follow the SNAPHU convention:
 *   dpsi[row][col] = wrap( phase[row][col+1] − phase[row][col] ) / (2π)
 * so the range-difference array has shape nrow × (ncol-1) and the azimuth-
 * difference array has shape (nrow-1) × ncol.
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <assert.h>

/* ── constants ────────────────────────────────────────────────────────────── */
static constexpr float TWOPI_F = 2.0f * CUDART_PI_F;

/* ── helpers ─────────────────────────────────────────────────────────────── */
/* Wrap a float difference into (-0.5, +0.5] in units of cycles */
__device__ __forceinline__ float wrap_cycles(float d) {
    d -= floorf(d + 0.5f);   /* equivalent to rounding to nearest integer */
    return d;
}

/* Wrap raw phase to [0, 2π) */
__device__ __forceinline__ float wrap_to_2pi(float p) {
    p -= TWOPI_F * floorf(p / TWOPI_F);
    return p;
}

/* ══ kernel: WrapPhaseKernel ══════════════════════════════════════════════ */
/**
 * Wrap all phase values in-place to [0, 2π).
 * phase: nrow*ncol
 */
__global__ void WrapPhaseKernel(float *phase, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) phase[idx] = wrap_to_2pi(phase[idx]);
}

/* ══ kernel: RangeDiffsKernel ════════════════════════════════════════════ */
/**
 * Compute wrapped phase differences in range (column direction).
 * dpsi[row*(ncol-1)+col] = wrap( (phase[row*ncol+col+1] - phase[row*ncol+col]) / 2π )
 * Output shape: nrow × (ncol-1)
 */
__global__ void RangeDiffsKernel(
    float       *dpsi,       /* nrow*(ncol-1) */
    const float *phase,      /* nrow*ncol     */
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* 0..ncol-2 */
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* 0..nrow-1 */
    if (row >= nrow || col >= ncol - 1) return;

    float d = (phase[row * ncol + col + 1] - phase[row * ncol + col]) / TWOPI_F;
    dpsi[row * (ncol - 1) + col] = wrap_cycles(d);
}

/* ══ kernel: AzimuthDiffsKernel ══════════════════════════════════════════ */
/**
 * Compute wrapped phase differences in azimuth (row direction).
 * dpsi[row*ncol+col] = wrap( (phase[row*ncol+col] - phase[(row+1)*ncol+col]) / 2π )
 * Output shape: (nrow-1) × ncol
 *
 * NOTE the sign: SNAPHU's CalcWrappedAzDiffs() defines the azimuth diff as
 * TOP minus BOTTOM (phase[row] - phase[row+1]) — the reverse of the range
 * convention.  The row-arc cost offsets and the network solver's flow-sign
 * convention depend on it; a forward difference here silently negates every
 * row-arc offset and skews the MCF optimum.
 */
__global__ void AzimuthDiffsKernel(
    float       *dpsi,       /* (nrow-1)*ncol */
    const float *phase,      /* nrow*ncol     */
    int          nrow,
    int          ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow - 1 || col >= ncol) return;

    float d = (phase[row * ncol + col] - phase[(row + 1) * ncol + col]) / TWOPI_F;
    dpsi[row * ncol + col] = wrap_cycles(d);
}

/* ══ kernel: BoxcarRowKernel ══════════════════════════════════════════════ */
/**
 * Horizontal pass of a separable 2-D box-car (moving average).
 * Processes 'cols_in' columns of input, produces 'cols_out' = cols_in output
 * columns (mirror-padded input is assumed to have kc extra columns on each side).
 *
 * padded: padded input  shape nrow_in × cols_padded
 * out:    output        shape nrow_in × cols_in
 * kc:     half-kernel width (window = 2*kc+1)
 */
__global__ void BoxcarRowKernel(
    float       *out,
    const float *padded,
    int          nrow,
    int          cols_in,
    int          cols_padded,
    int          kc
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= cols_in) return;

    float sum = 0.0f;
    int   wlen = 2 * kc + 1;
    const float *rowptr = padded + row * cols_padded;
    for (int k = 0; k < wlen; ++k)
        sum += rowptr[col + k];   /* col + k spans [col, col+wlen) over padded */
    out[row * cols_in + col] = sum / wlen;
}

/* ══ kernel: BoxcarColKernel ══════════════════════════════════════════════ */
/**
 * Vertical pass of separable box-car.
 * padded: vertically padded input shape rows_padded × ncol
 * out:    output                  shape nrow × ncol
 * kr:     half-kernel height (window = 2*kr+1)
 */
__global__ void BoxcarColKernel(
    float       *out,
    const float *padded,
    int          nrow,
    int          ncol,
    int          rows_padded,
    int          kr
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    float sum = 0.0f;
    int   wlen = 2 * kr + 1;
    for (int k = 0; k < wlen; ++k)
        sum += padded[(row + k) * ncol + col];
    out[row * ncol + col] = sum / wlen;
}

/* ══ kernel: MirrorPadKernel ══════════════════════════════════════════════ */
/**
 * Fill a mirror-padded array from the source.
 * src:  nrow × ncol
 * dst:  (nrow + 2*kr) × (ncol + 2*kc)
 */
__global__ void MirrorPadKernel(
    float       *dst,
    const float *src,
    int          nrow,
    int          ncol,
    int          kr,
    int          kc
) {
    /* each thread writes one element of dst */
    int dcol_full = ncol + 2 * kc;
    int drow_full = nrow + 2 * kr;
    int dcol = blockIdx.x * blockDim.x + threadIdx.x;
    int drow = blockIdx.y * blockDim.y + threadIdx.y;
    if (drow >= drow_full || dcol >= dcol_full) return;

    /* map padded index back to source, clamp with mirror */
    auto mirror_idx = [] __device__ (int i, int lo, int hi) -> int {
        /* lo and hi are inclusive bounds in dst space */
        if (i < lo)  i = lo + (lo - i);
        if (i > hi)  i = hi - (i - hi);
        return i;
    };
    int srow = mirror_idx(drow - kr, 0, nrow - 1);
    int scol = mirror_idx(dcol - kc, 0, ncol - 1);
    dst[drow * dcol_full + dcol] = src[srow * ncol + scol];
}

/* ── public C++ functions ─────────────────────────────────────────────────── */

extern "C" {

void cuphu_wrap_phase(float *d_phase, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    WrapPhaseKernel<<<blocks, threads, 0, stream>>>(d_phase, n);
}

/**
 * Compute both range and azimuth wrapped-phase difference arrays on GPU.
 *
 * d_phase:    [in]  device array nrow*ncol (already wrapped to [0,2π))
 * d_dpsi_rng: [out] device array nrow*(ncol-1)  range diffs (cycles)
 * d_dpsi_az:  [out] device array (nrow-1)*ncol  azimuth diffs (cycles)
 */
void cuphu_phase_diffs(
    const float *d_phase,
    float       *d_dpsi_rng,
    float       *d_dpsi_az,
    int          nrow,
    int          ncol,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    {
        dim3 grid(
            (ncol - 1 + block.x - 1) / block.x,
            (nrow     + block.y - 1) / block.y
        );
        RangeDiffsKernel<<<grid, block, 0, stream>>>(
            d_dpsi_rng, d_phase, nrow, ncol);
    }
    {
        dim3 grid(
            (ncol     + block.x - 1) / block.x,
            (nrow - 1 + block.y - 1) / block.y
        );
        AzimuthDiffsKernel<<<grid, block, 0, stream>>>(
            d_dpsi_az, d_phase, nrow, ncol);
    }
}

/**
 * 2-D separable box-car average of a device array.
 *
 * d_src:  [in]  device array nrow*ncol
 * d_dst:  [out] device array nrow*ncol (same shape; overwritten)
 * kr:     half-height of kernel (full height = 2*kr+1)
 * kc:     half-width  of kernel (full width  = 2*kc+1)
 *
 * Uses mirror-symmetric padding.  Allocates temporary device memory.
 */
void cuphu_boxcar2d(
    const float *d_src,
    float       *d_dst,
    int          nrow,
    int          ncol,
    int          kr,
    int          kc,
    cudaStream_t stream
) {
    dim3 block(16, 16);

    int pad_ncol = ncol + 2 * kc;
    int pad_nrow = nrow + 2 * kr;

    float *d_pad  = nullptr;
    float *d_tmp  = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_pad, (size_t)pad_nrow * pad_ncol * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_tmp, (size_t)pad_nrow * ncol      * sizeof(float), stream));

    /* mirror-pad the source */
    {
        dim3 grid(
            (pad_ncol + block.x - 1) / block.x,
            (pad_nrow + block.y - 1) / block.y
        );
        MirrorPadKernel<<<grid, block, 0, stream>>>(d_pad, d_src, nrow, ncol, kr, kc);
    }

    /* horizontal pass: pad_nrow × pad_ncol → pad_nrow × ncol */
    {
        dim3 grid(
            (ncol     + block.x - 1) / block.x,
            (pad_nrow + block.y - 1) / block.y
        );
        BoxcarRowKernel<<<grid, block, 0, stream>>>(
            d_tmp, d_pad, pad_nrow, ncol, pad_ncol, kc);
    }

    /* vertical pass: pad_nrow × ncol → nrow × ncol */
    {
        dim3 grid(
            (ncol + block.x - 1) / block.x,
            (nrow + block.y - 1) / block.y
        );
        BoxcarColKernel<<<grid, block, 0, stream>>>(
            d_dst, d_tmp, nrow, ncol, pad_nrow, kr);
    }

    CUDA_CHECK(cudaFreeAsync(d_pad, stream));
    CUDA_CHECK(cudaFreeAsync(d_tmp, stream));
}

} /* extern "C" */
