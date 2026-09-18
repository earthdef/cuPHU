/**
 * cuphu_whirlwind_cost_gpu.cu
 *
 * GPU acceleration for the single most expensive piece of the whirlwind
 * init's cost computation: the per-edge Carballo spline-LUT trilinear
 * lookup (~63% of compute_carballo_costs' own wall time, confirmed by
 * profiling the venezuela crop -- gradients 1.1s, box_filter 0.48s,
 * cost_assembly 3.0s out of a 4.8s cost stage that is itself only ~5% of
 * the whole solve). Gradients and box-filtering are deliberately NOT moved
 * here: cuphu's existing GPU utilities (cuphu_util.cu's RangeDiffsKernel /
 * AzimuthDiffsKernel / MirrorPadKernel) use different unit/sign/padding
 * conventions (cycles not radians, opposite azimuth sign, mirror padding
 * instead of edge-clamp) than the validated CPU path here, and re-deriving
 * matching kernels for a ~1.5s combined saving wasn't judged worth the
 * risk of a subtle parity regression. This kernel's job is narrow and
 * exactly mirrors CarballoSplineLut::cost() (cuphu_whirlwind_solver.cpp)
 * bit-for-bit, arithmetic operation for arithmetic operation.
 */
#include "cuphu.h"
#include "cuphu_whirlwind_carballo_tables.hpp"

#include <cuda_runtime.h>
#include <mutex>

namespace {

__device__ __forceinline__ void bracket(const float *grid, int n, float x, int &lo, float &t) {
    float xc = fminf(fmaxf(x, grid[0]), grid[n - 1]);
    // Linear scan: n <= 31 here, cheaper than a branchy binary search on GPU
    // and matches std::upper_bound's result exactly (grid is sorted, unique).
    int hi = 0;
    while (hi < n && grid[hi] <= xc) ++hi;
    lo = min(max(hi - 1, 0), n - 2);
    t = fminf(fmaxf((xc - grid[lo]) / (grid[lo + 1] - grid[lo]), 0.0f), 1.0f);
}

__device__ __forceinline__ float trilinear(
    const float *vals, int n_corr, int n_nlooks,
    int ia, float ta, int ic, float tc, int il, float tl)
{
    auto val = [&](int a, int c, int l) {
        return vals[(size_t)a * n_corr * n_nlooks + (size_t)c * n_nlooks + l];
    };
    float v000 = val(ia, ic, il),         v001 = val(ia, ic, il + 1);
    float v010 = val(ia, ic + 1, il),     v011 = val(ia, ic + 1, il + 1);
    float v100 = val(ia + 1, ic, il),     v101 = val(ia + 1, ic, il + 1);
    float v110 = val(ia + 1, ic + 1, il), v111 = val(ia + 1, ic + 1, il + 1);
    float v00 = v000 + tl * (v001 - v000);
    float v01 = v010 + tl * (v011 - v010);
    float v10 = v100 + tl * (v101 - v100);
    float v11 = v110 + tl * (v111 - v110);
    float v0 = v00 + tc * (v01 - v00);
    float v1 = v10 + tc * (v11 - v10);
    return v0 + ta * (v1 - v0);
}

// Matches CarballoSplineLut::cost() exactly: round(100 * max(-ln(p1/p0), 0)).
__device__ __forceinline__ int32_t lut_cost(
    const float *phase, int n_phase, const float *corr, int n_corr,
    const float *nlooks_grid, int n_nlooks, const float *p0, const float *p1,
    float alpha, float gamma, float nlooks)
{
    int ia, ic, il; float ta, tc, tl;
    bracket(phase, n_phase, alpha, ia, ta);
    bracket(corr, n_corr, gamma, ic, tc);
    bracket(nlooks_grid, n_nlooks, nlooks, il, tl);
    float p0v = fmaxf(trilinear(p0, n_corr, n_nlooks, ia, ta, ic, tc, il, tl), 1e-30f);
    float p1v = fmaxf(trilinear(p1, n_corr, n_nlooks, ia, ta, ic, tc, il, tl), 1e-30f);
    float raw = -logf(p1v / p0v);
    return (int32_t)(100.0f * fmaxf(raw, 0.0f));
}

__global__ void CostLutKernel(
    const float *alpha, const float *gamma, const uint8_t *zero_mask,
    int64_t n, float nlooks,
    const float *phase, int n_phase, const float *corr, int n_corr,
    const float *nlooks_grid, int n_nlooks, const float *p0, const float *p1,
    int32_t *cost_pos, int32_t *cost_neg)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (zero_mask[i]) {
        cost_pos[i] = 0;
        cost_neg[i] = 0;
        return;
    }
    float a = alpha[i], g = gamma[i];
    cost_pos[i] = lut_cost(phase, n_phase, corr, n_corr, nlooks_grid, n_nlooks, p0, p1, a, g, nlooks);
    cost_neg[i] = lut_cost(phase, n_phase, corr, n_corr, nlooks_grid, n_nlooks, p0, p1, -a, g, nlooks);
}

struct GpuTables {
    float *phase = nullptr, *corr = nullptr, *nlooks = nullptr, *p0 = nullptr, *p1 = nullptr;
};

const GpuTables &get_gpu_tables() {
    static std::once_flag flag;
    static GpuTables t;
    std::call_once(flag, [] {
        using namespace cuphu_ww;
        cudaMalloc(&t.phase, sizeof(g_carballo_grid_phase));
        cudaMalloc(&t.corr, sizeof(g_carballo_grid_corr));
        cudaMalloc(&t.nlooks, sizeof(g_carballo_grid_nlooks));
        cudaMalloc(&t.p0, sizeof(g_carballo_p0));
        cudaMalloc(&t.p1, sizeof(g_carballo_p1));
        cudaMemcpy(t.phase, g_carballo_grid_phase, sizeof(g_carballo_grid_phase), cudaMemcpyHostToDevice);
        cudaMemcpy(t.corr, g_carballo_grid_corr, sizeof(g_carballo_grid_corr), cudaMemcpyHostToDevice);
        cudaMemcpy(t.nlooks, g_carballo_grid_nlooks, sizeof(g_carballo_grid_nlooks), cudaMemcpyHostToDevice);
        cudaMemcpy(t.p0, g_carballo_p0, sizeof(g_carballo_p0), cudaMemcpyHostToDevice);
        cudaMemcpy(t.p1, g_carballo_p1, sizeof(g_carballo_p1), cudaMemcpyHostToDevice);
    });
    return t;
}

} // namespace

/**
 * Evaluate the Carballo spline-LUT cost for `n` edges on the GPU.
 * `zero_mask[i] != 0` forces cost_pos[i] = cost_neg[i] = 0 (both-invalid or
 * slope-guard-fired edges, decided by the caller on CPU beforehand).
 * cost_pos[i] = lut.cost(alpha[i], gamma[i], nlooks); cost_neg[i] = lut.cost(-alpha[i], ...).
 *
 * Creates and owns its own short-lived stream rather than taking one from
 * the caller: cuphu_whirlwind_solver.hpp's public API (and this function's
 * own extern "C" declaration in that .cpp) deliberately carries no CUDA
 * type, so the standalone/CPU-only build of that file (used for
 * fast-iteration testing against Python whirlwind) never needs
 * cuda_runtime.h. A plain (non-default, non-legacy-blocking) stream still
 * gets the concurrency that matters: multiple tiles run on separate CPU
 * threads via run_parallel()/nproc, each calling into this function on ITS
 * OWN thread, and a private stream here means none of them contend on the
 * legacy default stream's process-wide barrier the way an unstreamed
 * cudaMalloc/cudaMemcpy would. The function still blocks its OWN caller
 * (cudaStreamSynchronize before returning) because compute_carballo_costs()
 * uses h_cost_pos/h_cost_neg synchronously right after -- that only blocks
 * the calling tile's own thread, not other tiles' concurrent streams.
 */
extern "C" void cuphu_ww_cost_lut_eval_gpu(
    const float *h_alpha, const float *h_gamma, const uint8_t *h_zero_mask,
    int64_t n, float nlooks, int gpu_id,
    int32_t *h_cost_pos, int32_t *h_cost_neg)
{
    cudaSetDevice(gpu_id);
    const GpuTables &t = get_gpu_tables();
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_alpha, *d_gamma;
    uint8_t *d_zero;
    int32_t *d_cost_pos, *d_cost_neg;
    cudaMallocAsync(&d_alpha, n * sizeof(float), stream);
    cudaMallocAsync(&d_gamma, n * sizeof(float), stream);
    cudaMallocAsync(&d_zero, n * sizeof(uint8_t), stream);
    cudaMallocAsync(&d_cost_pos, n * sizeof(int32_t), stream);
    cudaMallocAsync(&d_cost_neg, n * sizeof(int32_t), stream);

    cudaMemcpyAsync(d_alpha, h_alpha, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gamma, h_gamma, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_zero, h_zero_mask, n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);

    int block = 256;
    int grid = (int)((n + block - 1) / block);
    using namespace cuphu_ww;
    CostLutKernel<<<grid, block, 0, stream>>>(
        d_alpha, d_gamma, d_zero, n, nlooks,
        t.phase, 31, t.corr, 21, t.nlooks, 21, t.p0, t.p1,
        d_cost_pos, d_cost_neg);

    cudaMemcpyAsync(h_cost_pos, d_cost_pos, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_cost_neg, d_cost_neg, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_alpha, stream);
    cudaFreeAsync(d_gamma, stream);
    cudaFreeAsync(d_zero, stream);
    cudaFreeAsync(d_cost_pos, stream);
    cudaFreeAsync(d_cost_neg, stream);

    // The caller (compute_carballo_costs) uses h_cost_pos/h_cost_neg
    // synchronously right after this call returns -- wait for THIS tile's
    // own stream only, which does not block other tiles' concurrent streams.
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
