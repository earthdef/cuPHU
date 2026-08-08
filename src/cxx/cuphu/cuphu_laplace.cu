/**
 * cuphu_laplace.cu
 *
 * GPU Laplacian Conjugate Gradient (PCG) phase unwrapper.
 *
 * Solves the weighted least-squares phase unwrapping problem:
 *
 *     L · u = b
 *
 * where the weighted graph Laplacian L has:
 *   L_aa   =  Σ w_ab + ε       (diagonal; sum over all arcs touching pixel a)
 *   L_ab   = -w_ab             (off-diagonal)
 *   w_ab   =  1/sigsq_ab       (0 if sigsq ≤ 0 or ≥ 32000)
 *
 * The RHS encodes the wrapped phase gradients:
 *   b[a] = -Σ_{a→b} w·g  +  Σ_{c→a} w·g
 *   g_ab = wrap(φ[b] - φ[a])   (wrapped phase difference in radians)
 *
 * Note: we use the raw wrapped phase differences rather than SNAPHU's
 * offset-corrected avgdpsi.  SNAPHU's offset formula branches on rho vs
 * defocorrthresh in a way that degrades results when ncorrlooks is small
 * (all arcs hit the 0.5*avgdp branch).  Standard WLSQ with coherence
 * weights (from sigsq) is always correct.
 *
 * The system is solved with Jacobi-preconditioned CG.  Initialization is the
 * wrapped phase φ (a good warm start: smooth scenes converge in <50 iters).
 *
 * Two refinements beyond plain WLSQ:
 *
 *  1. IRLS outer loop: after each solve the per-arc weights are updated to
 *     w = w0 / sqrt(e² + δ²) where e = (∇u - g) is the arc residual.  This
 *     drives the objective from L2 toward the weighted-L1 norm of Costantini's
 *     MCF formulation, concentrating unwrapping errors at discontinuities
 *     instead of smearing them across the scene.
 *
 *  2. Congruence projection: the final solution is snapped to
 *     u = φ + 2π·round((u - φ)/2π), so the output is exactly congruent with
 *     the wrapped input (zero rewrap residual), matching the convention of
 *     network-flow unwrappers.
 *
 * Replaces the entire MCF/MST → InitNetwork → TreeSolve pipeline.
 * Expected speedup for a 6.46 Mpix InSAR scene: ~100× over TreeSolve.
 */

#include "cuphu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

extern "C" {
#include "snaphu.h"   /* smoothcostT */
}

/* ── constants ─────────────────────────────────────────────────────────────── */
#define LAPLACE_TWOPI     6.28318530717958647692f
#define LAPLACE_INV_TWOPI 0.15915494309189534561f
#define LAPLACE_EPS       1e-12f   /* near-zero: avoids null-space ill-conditioning */
#define LAPLACE_LARGESIGSQ 32000

/* ── helpers ────────────────────────────────────────────────────────────────── */
__device__ __forceinline__ float laplace_arc_weight(short sigsq) {
    /* sigsq = 0: masked arc; sigsq >= LARGESIGSQ: no preference → both ignored */
    if (sigsq <= 0 || sigsq >= LAPLACE_LARGESIGSQ) return 0.0f;
    return 1.0f / (float)sigsq;
}

__device__ __forceinline__ float laplace_wrap_pi(float x) {
    return x - LAPLACE_TWOPI * rintf(x * LAPLACE_INV_TWOPI);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: laplace_assemble
 *
 * One thread per pixel.  Computes RHS b[r,c] and diagonal d[r,c] from the
 * smooth-cost arcs and the current wrapped phase.
 *
 * Arc layout in d_costs (row arcs first, then col arcs):
 *   row_arcs[(nrow-1)*ncol] at offsets [0 .. (nrow-1)*ncol - 1]
 *   col_arcs[ nrow*(ncol-1)] at offsets [(nrow-1)*ncol .. end]
 *
 * Row arc [r,c] connects pixel (r,c) → (r+1,c)  (downward)
 * Col arc [r,c] connects pixel (r,c) → (r,c+1)  (rightward)
 * ═══════════════════════════════════════════════════════════════════════════════ */
__global__ void laplace_assemble(
    const float* __restrict__ d_w,      /* per-arc weights, row arcs then col */
    const float* __restrict__ phi,
    float*       __restrict__ b,
    float*       __restrict__ d_diag,
    int nrow, int ncol
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nrow || c >= ncol) return;

    const float* row_w = d_w;
    const float* col_w = d_w + (size_t)(nrow - 1) * ncol;

    float phi_a = phi[(size_t)r * ncol + c];
    float bval  = 0.0f;
    float wsum  = 0.0f;

    /* ── Down arc (r,c)→(r+1,c): outgoing → b[r,c] -= w·g ── */
    if (r < nrow - 1) {
        float w = row_w[(size_t)r * ncol + c];
        if (w > 0.0f) {
            float dphi = laplace_wrap_pi(phi[(size_t)(r + 1) * ncol + c] - phi_a);
            bval -= w * dphi;
            wsum += w;
        }
    }

    /* ── Up arc (r-1,c)→(r,c): incoming → b[r,c] += w·g ── */
    if (r > 0) {
        float w = row_w[(size_t)(r - 1) * ncol + c];
        if (w > 0.0f) {
            float dphi = laplace_wrap_pi(phi_a - phi[(size_t)(r - 1) * ncol + c]);
            bval += w * dphi;
            wsum += w;
        }
    }

    /* ── Right arc (r,c)→(r,c+1): outgoing → b[r,c] -= w·g ── */
    if (c < ncol - 1) {
        float w = col_w[(size_t)r * (ncol - 1) + c];
        if (w > 0.0f) {
            float dphi = laplace_wrap_pi(phi[(size_t)r * ncol + c + 1] - phi_a);
            bval -= w * dphi;
            wsum += w;
        }
    }

    /* ── Left arc (r,c-1)→(r,c): incoming → b[r,c] += w·g ── */
    if (c > 0) {
        float w = col_w[(size_t)r * (ncol - 1) + (c - 1)];
        if (w > 0.0f) {
            float dphi = laplace_wrap_pi(phi_a - phi[(size_t)r * ncol + c - 1]);
            bval += w * dphi;
            wsum += w;
        }
    }

    b     [(size_t)r * ncol + c] = bval;
    d_diag[(size_t)r * ncol + c] = wsum + LAPLACE_EPS;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: laplace_matvec   y = L · x
 *
 * L is the same weighted graph Laplacian + ε·I:
 *   y[r,c] = Σ_nb w_nb · (x[r,c] - x[nb]) + ε · x[r,c]
 * ═══════════════════════════════════════════════════════════════════════════════ */
__global__ void laplace_matvec(
    const float* __restrict__ d_w,      /* per-arc weights, row arcs then col */
    const float* __restrict__ x,
    float*       __restrict__ y,
    int nrow, int ncol
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nrow || c >= ncol) return;

    const float* row_w = d_w;
    const float* col_w = d_w + (size_t)(nrow - 1) * ncol;

    float xa   = x[(size_t)r * ncol + c];
    float yval = LAPLACE_EPS * xa;

    if (r < nrow - 1) {
        float w = row_w[(size_t)r * ncol + c];
        yval += w * (xa - x[(size_t)(r + 1) * ncol + c]);
    }
    if (r > 0) {
        float w = row_w[(size_t)(r - 1) * ncol + c];
        yval += w * (xa - x[(size_t)(r - 1) * ncol + c]);
    }
    if (c < ncol - 1) {
        float w = col_w[(size_t)r * (ncol - 1) + c];
        yval += w * (xa - x[(size_t)r * ncol + c + 1]);
    }
    if (c > 0) {
        float w = col_w[(size_t)r * (ncol - 1) + (c - 1)];
        yval += w * (xa - x[(size_t)r * ncol + c - 1]);
    }

    y[(size_t)r * ncol + c] = yval;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: laplace_init_weights
 *
 * One thread per arc.  Initial data weights from the smooth-cost sigsq:
 * w0 = 1/sigsq (coherence-derived; 0 for masked/no-preference arcs).
 * ═══════════════════════════════════════════════════════════════════════════════ */
__global__ void laplace_init_weights(
    const smoothcostT* __restrict__ d_costs,
    float*             __restrict__ d_w,
    int narcs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < narcs) d_w[i] = laplace_arc_weight(d_costs[i].sigsq);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: laplace_reweight  (IRLS step toward the weighted-L1 objective)
 *
 * One thread per arc.  Arc residual e = (u_head - u_tail) - g where
 * g = wrap(φ_head - φ_tail); new weight w = w0 / sqrt(e² + δ²).
 * w0 is recomputed from sigsq so reweighting never compounds.
 * ═══════════════════════════════════════════════════════════════════════════════ */
__global__ void laplace_reweight(
    const smoothcostT* __restrict__ d_costs,
    const float*       __restrict__ phi,
    const float*       __restrict__ u,
    float*             __restrict__ d_w,
    float delta,
    int nrow, int ncol
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nrowarcs = (size_t)(nrow - 1) * ncol;
    size_t narcs    = nrowarcs + (size_t)nrow * (ncol - 1);
    if (i >= (int)narcs) return;

    float w0 = laplace_arc_weight(d_costs[i].sigsq);
    if (w0 <= 0.0f) { d_w[i] = 0.0f; return; }

    size_t tail, head;
    if ((size_t)i < nrowarcs) {
        /* row arc [r,c]: (r,c) → (r+1,c) */
        int r = i / ncol, c = i % ncol;
        tail = (size_t)r * ncol + c;
        head = (size_t)(r + 1) * ncol + c;
    } else {
        /* col arc [r,c]: (r,c) → (r,c+1) */
        int j = i - (int)nrowarcs;
        int r = j / (ncol - 1), c = j % (ncol - 1);
        tail = (size_t)r * ncol + c;
        head = tail + 1;
    }

    float g = laplace_wrap_pi(phi[head] - phi[tail]);
    float e = (u[head] - u[tail]) - g;
    d_w[i] = w0 * rsqrtf(e * e + delta * delta);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * KERNEL: laplace_congruence
 *
 * One thread per pixel: u = φ + 2π·round((u - φ)/2π).  Snaps the WLSQ
 * solution onto the lattice congruent with the wrapped input, so the
 * rewrapped output reproduces the wrapped phase exactly.
 * ═══════════════════════════════════════════════════════════════════════════════ */
__global__ void laplace_congruence(
    const float* __restrict__ phi,
    float*       __restrict__ u,
    int npix
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        float d = u[i] - phi[i];
        u[i] = phi[i] + LAPLACE_TWOPI * rintf(d * LAPLACE_INV_TWOPI);
    }
}

/* ── simple 1-D vector kernels ──────────────────────────────────────────────── */

/* y += alpha * x */
__global__ void laplace_axpy(float alpha, const float* __restrict__ x,
                              float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

/* y = alpha * x + beta * y */
__global__ void laplace_axpby(float alpha, const float* __restrict__ x,
                               float beta,  float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + beta * y[i];
}

/* c = a / b  (element-wise) */
__global__ void laplace_vdiv(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] / b[i];
}

/* y += scalar  (broadcast) */
__global__ void laplace_addscalar(float scalar, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += scalar;
}

/* Parallel dot product — atomicAdd to scalar (zero d_out before calling!) */
__global__ void laplace_dot(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ d_out, int n) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float val = 0.0f;
    /* grid-stride accumulate */
    while (idx < n) {
        val += a[idx] * b[idx];
        idx += gridDim.x * blockDim.x;
    }
    shmem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_out, shmem[0]);
}

/* ── helper: synchronous dot product via DevArray ───────────────────────────── */
static float gpu_dot(const float *a, const float *b, int n,
                     float *d_scratch, cudaStream_t stream)
{
    /* d_scratch must be a device scalar pre-zeroed by caller */
    constexpr int BLK = 256;
    int grid = std::min(4096, (n + BLK - 1) / BLK);
    laplace_dot<<<grid, BLK, BLK * sizeof(float), stream>>>(a, b, d_scratch, n);
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_scratch, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * cuphu_laplace_unwrap_gpu
 *
 * Public entry point.  Solves L·u = b on GPU via Jacobi-PCG.
 *
 * Inputs:
 *   d_smooth_costs  GPU: (nrow-1)*ncol + nrow*(ncol-1) smoothcostT elements
 *   d_phase         GPU: nrow*ncol wrapped phase in [0, 2π)
 *   nshortcycle     cost-discretization scale (from CuPhuParams)
 *   max_iter        maximum PCG iterations (e.g. 300)
 *   tol             convergence: stop when sqrt(rz/rz0) < tol (e.g. 1e-3)
 *   verbose         if true, print residual every 20 iterations
 *
 * Output:
 *   d_unw           GPU: nrow*ncol unwrapped phase (caller-allocated)
 * ═══════════════════════════════════════════════════════════════════════════════ */
extern "C"
void cuphu_laplace_unwrap_gpu(
    const smoothcostT *d_smooth_costs,
    const float       *d_phase,
    int                nrow,
    int                ncol,
    double             nshortcycle,
    int                max_iter,
    float              tol,
    int                verbose,
    float             *d_unw,
    cudaStream_t       stream
) {
    int npix = nrow * ncol;
    (void)nshortcycle;   /* kept in API for future offset-corrected mode */

    size_t narcs = (size_t)(nrow - 1) * ncol + (size_t)nrow * (ncol - 1);

    /* IRLS configuration: outer solves after the first drive the objective
     * from L2 toward weighted-L1 (the MCF relaxation, whose LP optimum is
     * integral on grid graphs).  δ is the residual floor in radians; it is
     * annealed downward each iteration so early solves stay well-conditioned
     * while later ones sharpen residuals onto few arcs — this is what lets
     * IRLS recover multi-cycle large-scale trends that plain L2 flattens.
     *
     * N_IRLS=8/δmin=0.05 (the original defaults) converge fine for small/
     * tiled problems but can leave large, isolated high-coherence regions
     * on big single-tile scenes ~50% misassigned even though the *global*
     * disagreement rate already looks converged (measured on a 1847x3498
     * Sentinel-1 scene: 1.03% overall despite one contained region at 50%).
     * A sweep (N_IRLS in {0,2,4,8,16,30}, δmin in {0.05,0.01,0.005}) showed
     * 16/0.01 resolves it (region disagreement 50% -> 0.66%) with no further
     * gain from going higher — so that is the new default.  Overridable via
     * CUPHU_N_IRLS / CUPHU_IRLS_DELTA_MAX / CUPHU_IRLS_DELTA_MIN for tuning. */
    int   N_IRLS         = 16;
    float IRLS_DELTA_MAX = 0.5f;
    float IRLS_DELTA_MIN = 0.01f;
    if (const char *e = std::getenv("CUPHU_N_IRLS"))         N_IRLS         = std::atoi(e);
    if (const char *e = std::getenv("CUPHU_IRLS_DELTA_MAX")) IRLS_DELTA_MAX = std::atof(e);
    if (const char *e = std::getenv("CUPHU_IRLS_DELTA_MIN")) IRLS_DELTA_MIN = std::atof(e);

    /* ── allocate working arrays ─────────────────────────────────────────── */
    DevArray<float> d_w    (narcs);  /* per-arc weights (IRLS-updated) */
    DevArray<float> d_b    (npix);   /* RHS */
    DevArray<float> d_diag (npix);   /* diagonal of L (Jacobi preconditioner) */
    DevArray<float> d_r    (npix);   /* residual r = b - L·u */
    DevArray<float> d_z    (npix);   /* preconditioned residual z = r / diag */
    DevArray<float> d_p    (npix);   /* search direction */
    DevArray<float> d_q    (npix);   /* q = L·p */
    DevArray<float> d_scal (1);      /* device scalar for dot products */

    /* ── kernel launch dimensions ────────────────────────────────────────── */
    dim3 blk2d(32, 8);
    dim3 grd2d((ncol + 31) / 32, (nrow + 7) / 8);
    constexpr int BLK1 = 256;
    int grd1  = (npix + BLK1 - 1) / BLK1;
    int grd1a = ((int)narcs + BLK1 - 1) / BLK1;

    /* initial data weights w0 = 1/sigsq (coherence-derived) */
    laplace_init_weights<<<grd1a, BLK1, 0, stream>>>(
        d_smooth_costs, d_w.get(), (int)narcs);

    for (int outer = 0; outer <= N_IRLS; ++outer) {

        /* ── assemble b and diag for current weights ─────────────────────── */
        laplace_assemble<<<grd2d, blk2d, 0, stream>>>(
            d_w.get(), d_phase,
            d_b.get(), d_diag.get(),
            nrow, ncol);

        /* ── initialize u ───────────────────────────────────────────────────
         *
         * First solve starts from 0: a phi warm-start makes rz0 ≈ EPS²·||phi||²
         * (below float32's useful range) causing spurious convergence.
         * IRLS re-solves warm-start from the previous solution.
         */
        if (outer == 0)
            CUDA_CHECK(cudaMemsetAsync(d_unw, 0, (size_t)npix * sizeof(float), stream));

        /* ── r = b - L·u ────────────────────────────────────────────────── */
        laplace_matvec<<<grd2d, blk2d, 0, stream>>>(
            d_w.get(), d_unw, d_r.get(), nrow, ncol);
        laplace_axpby<<<grd1, BLK1, 0, stream>>>(
            1.0f, d_b.get(), -1.0f, d_r.get(), npix);

        /* ── z = r / diag  (Jacobi preconditioner) ──────────────────────── */
        laplace_vdiv<<<grd1, BLK1, 0, stream>>>(
            d_r.get(), d_diag.get(), d_z.get(), npix);

        /* ── p = z ──────────────────────────────────────────────────────── */
        CUDA_CHECK(cudaMemcpyAsync(d_p.get(), d_z.get(),
                                   (size_t)npix * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        /* ── rz₀ = <r, z> ───────────────────────────────────────────────── */
        CUDA_CHECK(cudaMemsetAsync(d_scal.get(), 0, sizeof(float), stream));
        float rz = gpu_dot(d_r.get(), d_z.get(), npix, d_scal.get(), stream);
        float rz0 = rz;

        if (rz0 > 0.0f) {
            /* ── PCG loop ───────────────────────────────────────────────── */
            int iter = 0;
            for (; iter < max_iter; ++iter) {

                laplace_matvec<<<grd2d, blk2d, 0, stream>>>(
                    d_w.get(), d_p.get(), d_q.get(), nrow, ncol);

                CUDA_CHECK(cudaMemsetAsync(d_scal.get(), 0, sizeof(float), stream));
                float pq = gpu_dot(d_p.get(), d_q.get(), npix, d_scal.get(), stream);
                if (pq <= 0.0f) break;

                float alpha = rz / pq;

                laplace_axpy<<<grd1, BLK1, 0, stream>>>(alpha, d_p.get(), d_unw, npix);
                laplace_axpy<<<grd1, BLK1, 0, stream>>>(-alpha, d_q.get(), d_r.get(), npix);
                laplace_vdiv<<<grd1, BLK1, 0, stream>>>(
                    d_r.get(), d_diag.get(), d_z.get(), npix);

                CUDA_CHECK(cudaMemsetAsync(d_scal.get(), 0, sizeof(float), stream));
                float rz_new = gpu_dot(d_r.get(), d_z.get(), npix, d_scal.get(), stream);

                if (verbose && (iter % 20 == 0 || rz_new < tol * tol * rz0)) {
                    fprintf(stderr, "[laplace-pcg] outer %d iter %3d  rel_res=%.3e\n",
                            outer, iter, (double)sqrtf(rz_new / rz0));
                    fflush(stderr);
                }

                if (rz_new < tol * tol * rz0) { iter++; break; }

                float beta = rz_new / rz;
                rz = rz_new;

                laplace_axpby<<<grd1, BLK1, 0, stream>>>(
                    1.0f, d_z.get(), beta, d_p.get(), npix);
            }

            if (verbose) {
                fprintf(stderr, "[laplace-pcg] outer %d converged in %d iterations\n",
                        outer, iter);
                fflush(stderr);
            }
        }

        /* ── debug: track mean (u-phi)/2pi over a target region per outer iter ── */
        if (const char *reg = std::getenv("CUPHU_DEBUG_REGION")) {
            int rr0, rr1, cc0, cc1;
            if (sscanf(reg, "%d,%d,%d,%d", &rr0, &rr1, &cc0, &cc1) == 4) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<float> h_unw(npix), h_phi(npix);
                CUDA_CHECK(cudaMemcpy(h_unw.data(), d_unw,  npix*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_phi.data(), d_phase, npix*sizeof(float), cudaMemcpyDeviceToHost));
                double sum = 0.0; long cnt = 0;
                for (int r = rr0; r < rr1 && r < nrow; ++r)
                    for (int c = cc0; c < cc1 && c < ncol; ++c) {
                        size_t idx = (size_t)r*ncol + c;
                        sum += (h_unw[idx] - h_phi[idx]) / LAPLACE_TWOPI;
                        cnt++;
                    }
                fprintf(stderr, "[cuphu-region] outer=%d  mean_cycles=%.4f  n=%ld\n",
                        outer, cnt ? sum/cnt : 0.0, cnt);
                fflush(stderr);
            }
        }

        /* ── IRLS reweight for next solve (annealed δ) ──────────────────── */
        if (outer < N_IRLS) {
            float delta = std::max(IRLS_DELTA_MIN,
                                   IRLS_DELTA_MAX * powf(0.5f, (float)outer));
            laplace_reweight<<<grd1a, BLK1, 0, stream>>>(
                d_smooth_costs, d_phase, d_unw,
                d_w.get(), delta, nrow, ncol);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ── Anchor to wrapped phase at pixel (0,0) ─────────────────────────────
     *
     * The minimum-norm WLSQ solution is centered around 0 rather than the
     * correct absolute phase.  We anchor by adding (phi[0,0] - u[0,0]) to
     * every pixel, matching the standard InSAR convention of using the
     * top-left corner as the reference point.
     */
    float h_phi0 = 0.0f, h_u0 = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_phi0, d_phase,      sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_u0,   d_unw,        sizeof(float), cudaMemcpyDeviceToHost));
    float offset = h_phi0 - h_u0;
    if (offset != 0.0f)
        laplace_addscalar<<<grd1, BLK1, 0, stream>>>(offset, d_unw, npix);

    /* ── Congruence projection ──────────────────────────────────────────────
     *
     * Snap onto the lattice congruent with the wrapped input:
     * u = φ + 2π·round((u - φ)/2π).  The rewrapped output then reproduces
     * the wrapped phase exactly, as with network-flow unwrappers; remaining
     * errors are concentrated at discrete 2π jumps (minimized by the IRLS
     * pass) instead of being smeared across the scene.
     */
    laplace_congruence<<<grd1, BLK1, 0, stream>>>(d_phase, d_unw, npix);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
