/**
 * cuphu_whirlwind_solver.cpp
 *
 * See cuphu_whirlwind_solver.hpp for the overview. Ported from whirlwind-insar
 * (crates/whirlwind-core/src, various .rs files); section headers below name
 * the specific source file each block mirrors. Single-threaded C++, CPU-only --
 * no GPU work in this file itself (matches the CPU-first phase of the
 * implementation plan), EXCEPT the cost-LUT-lookup fast path in
 * compute_carballo_costs(), which optionally offloads to
 * cuphu_whirlwind_cost_gpu.cu's kernel (the single most expensive piece of
 * cost computation, per profiling -- see that file's header). Gated behind
 * CUPHU_WW_GPU_COST, defined only by the real CMake/CUDA build, so this file
 * still compiles standalone with plain g++/no CUDA (used for fast-iteration
 * correctness testing against Python whirlwind throughout development).
 */
#include "cuphu_whirlwind_solver.hpp"
#include "cuphu_whirlwind_carballo_tables.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <deque>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef CUPHU_WW_GPU_COST
extern "C" void cuphu_ww_cost_lut_eval_gpu(
    const float *h_alpha, const float *h_gamma, const uint8_t *h_zero_mask,
    int64_t n, float nlooks, int gpu_id,
    int32_t *h_cost_pos, int32_t *h_cost_neg);
#endif

namespace cuphu_ww {

using std::size_t;

constexpr float TAU = 6.283185307179586f;
constexpr float PI_F = 3.141592653589793f;
constexpr double PI_D = 3.14159265358979323846;

// =============================================================================
// grid.rs: rectangular residual graph over the residue grid (m x n nodes,
// m = nrow+1, n = ncol+1). Forward-arc layout:
//   [0, n_v)                  DOWN   (i,j)   -> (i+1,j)
//   [n_v, 2*n_v)               UP    (i+1,j) -> (i,j)   (indexed by (i,j), i in [0,m-1))
//   [2*n_v, 2*n_v+n_h)        RIGHT  (i,j)   -> (i,j+1)
//   [2*n_v+n_h, num_forward)  LEFT   (i,j+1) -> (i,j)
//   [num_forward, 2*num_forward)  reverse partner of forward arc a is a+num_forward
// n_v = (m-1)*n, n_h = m*(n-1).
// =============================================================================
struct Grid {
    int64_t m, n;
    int64_t n_v, n_h;
    int64_t num_forward;

    Grid(int64_t m_, int64_t n_) : m(m_), n(n_) {
        n_v = (m - 1) * n;
        n_h = m * (n - 1);
        num_forward = 2 * n_v + 2 * n_h;
    }

    inline int64_t num_nodes() const { return m * n; }
    inline int64_t num_arcs() const { return 2 * num_forward; }
    inline int64_t node_id(int64_t i, int64_t j) const { return i * n + j; }
    inline void node_ij(int64_t id, int64_t &i, int64_t &j) const { i = id / n; j = id % n; }
    inline int64_t transpose(int64_t arc) const {
        return arc < num_forward ? arc + num_forward : arc - num_forward;
    }

    inline int64_t down_arc(int64_t i, int64_t j) const {
        return (i + 1 < m) ? (i * n + j) : -1;
    }
    inline int64_t up_arc(int64_t i, int64_t j) const {
        return (i >= 1) ? (n_v + (i - 1) * n + j) : -1;
    }
    inline int64_t right_arc(int64_t i, int64_t j) const {
        return (j + 1 < n) ? (2 * n_v + i * (n - 1) + j) : -1;
    }
    inline int64_t left_arc(int64_t i, int64_t j) const {
        return (j >= 1) ? (2 * n_v + n_h + i * (n - 1) + (j - 1)) : -1;
    }

    // (dir, tail_i, tail_j) for a forward arc. dir: 0=down 1=up 2=right 3=left.
    inline void forward_arc_info(int64_t arc, int &dir, int64_t &ti, int64_t &tj) const {
        if (arc < n_v) {
            dir = 0; ti = arc / n; tj = arc % n;
        } else if (arc < 2 * n_v) {
            int64_t a = arc - n_v;
            dir = 1; ti = a / n + 1; tj = a % n;
        } else if (arc < 2 * n_v + n_h) {
            int64_t a = arc - 2 * n_v;
            dir = 2; ti = a / (n - 1); tj = a % (n - 1);
        } else {
            int64_t a = arc - 2 * n_v - n_h;
            dir = 3; ti = a / (n - 1); tj = a % (n - 1) + 1;
        }
    }

    inline void arc_endpoints(int64_t arc, int64_t &tail, int64_t &head) const {
        bool swap = arc >= num_forward;
        int64_t fwd = swap ? arc - num_forward : arc;
        int dir; int64_t ti, tj;
        forward_arc_info(fwd, dir, ti, tj);
        int64_t hi = ti, hj = tj;
        switch (dir) {
            case 0: hi = ti + 1; break;           // down
            case 1: hi = ti - 1; break;            // up
            case 2: hj = tj + 1; break;            // right
            case 3: hj = tj - 1; break;            // left
        }
        int64_t t = node_id(ti, tj), h = node_id(hi, hj);
        if (swap) { tail = h; head = t; } else { tail = t; head = h; }
    }

    // Emit (arc, head) pairs for every outgoing arc from node (i,j), same
    // order as grid.rs's outgoing(): 4 forward + up to 4 residual reverses of
    // forward arcs pointing INTO (i,j).
    inline void outgoing(int64_t node, std::vector<std::pair<int64_t, int64_t>> &out) const {
        out.clear();
        int64_t i, j; node_ij(node, i, j);
        int64_t a;
        if ((a = down_arc(i, j)) >= 0) out.push_back({a, node + n});
        if ((a = up_arc(i, j)) >= 0) out.push_back({a, node - n});
        if ((a = right_arc(i, j)) >= 0) out.push_back({a, node + 1});
        if ((a = left_arc(i, j)) >= 0) out.push_back({a, node - 1});
        if (i + 1 < m) { a = up_arc(i + 1, j); out.push_back({transpose(a), node + n}); }
        if (i >= 1) { a = down_arc(i - 1, j); out.push_back({transpose(a), node - n}); }
        if (j + 1 < n) { a = left_arc(i, j + 1); out.push_back({transpose(a), node + 1}); }
        if (j >= 1) { a = right_arc(i, j - 1); out.push_back({transpose(a), node - 1}); }
    }
};

// =============================================================================
// cost/spline_lut.rs: trilinear interpolation of the embedded Carballo/Lee
// (1994) + Touzi (1999) p0/p1 tables -- the ACTUAL cost `unwrap_linear` (the
// pipeline behind Python's default `whirlwind.unwrap()`) uses, not the
// simpler on-the-fly analytical LUT (`cost::compute_carballo_costs`, used
// only by whirlwind's tiled/reuse paths). Ported byte-for-byte: see
// cuphu_whirlwind_carballo_tables.hpp (generated from the same embedded
// .bin blobs spline_lut.rs's `include_bytes!` loads).
// =============================================================================
struct CarballoSplineLut {
    const float *phase;  int n_phase;
    const float *corr;   int n_corr;
    const float *nlooks; int n_nlooks;
    const float *p0;
    const float *p1;

    static const CarballoSplineLut &get() {
        static const CarballoSplineLut lut{
            g_carballo_grid_phase, 31,
            g_carballo_grid_corr, 21,
            g_carballo_grid_nlooks, 21,
            g_carballo_p0, g_carballo_p1,
        };
        return lut;
    }

    static void bracket(const float *grid, int n, float x, int &lo, float &t) {
        float xc = std::min(std::max(x, grid[0]), grid[n - 1]);
        int hi = (int)(std::upper_bound(grid, grid + n, xc) - grid);
        lo = std::min(std::max(hi - 1, 0), n - 2);
        t = std::min(std::max((xc - grid[lo]) / (grid[lo + 1] - grid[lo]), 0.0f), 1.0f);
    }

    inline float val(const float *vals, int ia, int ic, int il) const {
        return vals[(size_t)ia * n_corr * n_nlooks + (size_t)ic * n_nlooks + il];
    }

    float trilinear(const float *vals, int ia, float ta, int ic, float tc, int il, float tl) const {
        float v000 = val(vals, ia, ic, il),         v001 = val(vals, ia, ic, il + 1);
        float v010 = val(vals, ia, ic + 1, il),     v011 = val(vals, ia, ic + 1, il + 1);
        float v100 = val(vals, ia + 1, ic, il),     v101 = val(vals, ia + 1, ic, il + 1);
        float v110 = val(vals, ia + 1, ic + 1, il), v111 = val(vals, ia + 1, ic + 1, il + 1);
        float v00 = v000 + tl * (v001 - v000);
        float v01 = v010 + tl * (v011 - v010);
        float v10 = v100 + tl * (v101 - v100);
        float v11 = v110 + tl * (v111 - v110);
        float v0 = v00 + tc * (v01 - v00);
        float v1 = v10 + tc * (v11 - v10);
        return v0 + ta * (v1 - v0);
    }

    // Arc cost = round(100 * max(-ln(p1/p0), 0)) (matches spline_lut.rs::cost).
    int32_t cost(float alpha, float gamma, float nlooks_) const {
        int ia, ic, il; float ta, tc, tl;
        bracket(phase, n_phase, alpha, ia, ta);
        bracket(corr, n_corr, gamma, ic, tc);
        bracket(nlooks, n_nlooks, nlooks_, il, tl);
        float p0v = std::max(trilinear(p0, ia, ta, ic, tc, il, tl), 1e-30f);
        float p1v = std::max(trilinear(p1, ia, ta, ic, tc, il, tl), 1e-30f);
        float raw = -std::log(p1v / p0v);
        return (int32_t)(100.0f * std::max(raw, 0.0f));
    }
};

// =============================================================================
// cost/mod.rs: aliased-gradient robustness guard (SlopeGuard). Production
// defaults (threshold_rad=1.0, budget=0.03, mode=ZeroCost): frees at most
// the steepest 3% of valid edges (floored at 1 rad) to zero cost, so a
// shear margin / rupture edge isn't priced as if only one unwrapped-slope
// hypothesis were possible. No env-var override surface here (cuphu has no
// equivalent mechanism); matches whirlwind's un-overridden defaults, which
// is what `ww.unwrap()` uses without any WHIRLWIND_SLOPE_GUARD* env set.
// =============================================================================
constexpr float SLOPE_GUARD_RAD = 1.0f;
constexpr float SLOPE_GUARD_BUDGET = 0.03f;
constexpr int GUARD_HIST_BINS = 1024;

struct ResolvedSlopeGuard {
    float threshold_rad;
    bool enabled() const { return threshold_rad > 0.0f; }
    bool fires(float raw_abs, float gamma, bool valid) const {
        return enabled() && valid && gamma > 1e-9f && raw_abs >= threshold_rad;
    }
};

// Resolve the per-frame threshold from the budget: walk the |raw gradient|
// histogram (over valid, gamma>0 edges) from the steepest bin down until
// spending one more would exceed `budget * total_edges`.
static ResolvedSlopeGuard resolve_slope_guard(
    const std::vector<const std::vector<float> *> &raws,
    const std::vector<const std::vector<float> *> &gammas,
    const std::vector<const std::vector<uint8_t> *> &valids)
{
    std::vector<uint64_t> hist(GUARD_HIST_BINS, 0);
    uint64_t total = 0;
    for (size_t s = 0; s < raws.size(); ++s) {
        const auto &raw = *raws[s]; const auto &gam = *gammas[s]; const auto &val = *valids[s];
        for (size_t i = 0; i < raw.size(); ++i) {
            if (gam[i] > 1e-9f && val[i]) {
                ++total;
                float frac = std::fabs(raw[i]) / PI_F;
                int b = std::min((int)(frac * GUARD_HIST_BINS), GUARD_HIST_BINS - 1);
                ++hist[(size_t)b];
            }
        }
    }
    uint64_t allowed = (uint64_t)((double)SLOPE_GUARD_BUDGET * (double)total);
    float chosen = PI_F;
    uint64_t acc = 0;
    for (int b = GUARD_HIST_BINS - 1; b >= 0; --b) {
        if (acc + hist[(size_t)b] > allowed) {
            chosen = (float)(b + 1) / (float)GUARD_HIST_BINS * PI_F;
            break;
        }
        acc += hist[(size_t)b];
    }
    return ResolvedSlopeGuard{std::max(SLOPE_GUARD_RAD, chosen)};
}

// =============================================================================
// cost/mod.rs: box-filtered (smoothed) + raw phase gradients, and the
// parity Carballo cost assembly (compute_carballo_costs_parity_impl).
// =============================================================================

// Separable box filter, krow taps down rows x kcol taps across cols, nearest-
// edge replication (matches cost/mod.rs::box_filter_2d).
static std::vector<float> box_filter_2d(const std::vector<float> &a, int64_t m, int64_t n,
                                         int krow, int kcol) {
    int64_t lo_c = (kcol - 1) / 2, hi_c = kcol / 2;
    int64_t lo_r = (krow - 1) / 2, hi_r = krow / 2;
    float inv_kc = 1.0f / (float)kcol, inv_kr = 1.0f / (float)krow;

    std::vector<float> tmp((size_t)(m * n));
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int64_t dj = -lo_c; dj <= hi_c; ++dj) {
                int64_t jj = std::min(std::max(j + dj, (int64_t)0), n - 1);
                s += a[(size_t)(i * n + jj)];
            }
            tmp[(size_t)(i * n + j)] = s * inv_kc;
        }
    }
    std::vector<float> out((size_t)(m * n));
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int64_t di = -lo_r; di <= hi_r; ++di) {
                int64_t ii = std::min(std::max(i + di, (int64_t)0), m - 1);
                s += tmp[(size_t)(ii * n + j)];
            }
            out[(size_t)(i * n + j)] = s * inv_kr;
        }
    }
    return out;
}

// Compute per-forward-arc integer Carballo cost, matching
// cost/mod.rs::compute_carballo_costs_parity_impl exactly: biased (non-mask-
// aware) smoothed gradients for the priced alpha, raw (unsmoothed) gradients
// for the slope guard, "both-endpoints-invalid" masking (an edge with only
// ONE invalid endpoint still gets priced -- narrower than "either invalid"),
// the embedded spline-LUT cost, scale baked into CarballoSplineLut::cost().
// `mask` (nonzero=valid) may be null (all valid, guard still applies).
static std::vector<int32_t> compute_carballo_costs(
    const Grid &g, const std::complex<float> *igram, const float *corr,
    const unsigned char *mask, int64_t m_phase, int64_t n_phase, float nlooks,
    int window_par, int window_perp, int gpu_id)
{
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto tock = [&](const char *stage) {
        if (!dbg) return;
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[compute_carballo_costs] %-14s %8.3fs\n", stage,
                     std::chrono::duration<double>(t1 - t0).count());
        t0 = t1;
    };
    std::vector<float> phase_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j) {
            std::complex<float> z = igram[(i + 1) * n_phase + j] * std::conj(igram[i * n_phase + j]);
            phase_dy[(size_t)(i * n_phase + j)] = std::arg(z);
        }
    std::vector<float> phase_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            std::complex<float> z = igram[i * n_phase + (j + 1)] * std::conj(igram[i * n_phase + j]);
            phase_dx[(size_t)(i * (n_phase - 1) + j)] = std::arg(z);
        }
    tock("gradients");
    // Raw gradients are identical to phase_dy/phase_dx above (both are the
    // unsmoothed per-edge arg(z)) -- reuse directly as the guard's signal.
    const std::vector<float> &raw_dy = phase_dy;
    const std::vector<float> &raw_dx = phase_dx;

    // Orientation: parallel extent runs along the gradient's own difference
    // direction (matches cost/mod.rs::smooth_phase_gradients_with_mask).
    // Biased smoothing (no mask arg to box_filter_2d): matches parity mode,
    // which deliberately does NOT use the mask-aware smoother.
    auto phase_dy_s = box_filter_2d(phase_dy, m_phase - 1, n_phase, window_par, window_perp);
    auto phase_dx_s = box_filter_2d(phase_dx, m_phase, n_phase - 1, window_perp, window_par);
    tock("box_filter");

    std::vector<float> cor_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            cor_dy[(size_t)(i * n_phase + j)] =
                std::min(corr[i * n_phase + j], corr[(i + 1) * n_phase + j]);
    std::vector<float> cor_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j)
            cor_dx[(size_t)(i * (n_phase - 1) + j)] =
                std::min(corr[i * n_phase + j], corr[i * n_phase + j + 1]);

    auto is_valid = [&](int64_t i, int64_t j) -> bool {
        return mask == nullptr || mask[i * n_phase + j] != 0;
    };
    // guard_valid: both endpoints valid (raw gradient meaningful there).
    std::vector<uint8_t> guard_valid_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            guard_valid_dy[(size_t)(i * n_phase + j)] = is_valid(i, j) && is_valid(i + 1, j);
    std::vector<uint8_t> guard_valid_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j)
            guard_valid_dx[(size_t)(i * (n_phase - 1) + j)] = is_valid(i, j) && is_valid(i, j + 1);

    tock("coh+valid");
    ResolvedSlopeGuard guard = resolve_slope_guard(
        {&raw_dy, &raw_dx}, {&cor_dy, &cor_dx}, {&guard_valid_dy, &guard_valid_dx});
    tock("guard_resolve");

    std::vector<int32_t> cost((size_t)g.num_forward, 0);

    // zero_mask: both-invalid OR slope-guard-fired -- identical decision for
    // both directions of an edge, so computed once per slab regardless of
    // CPU/GPU path below.
    std::vector<uint8_t> zero_dy(phase_dy_s.size());
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j) {
            size_t idx = (size_t)(i * n_phase + j);
            bool both_invalid = !is_valid(i, j) && !is_valid(i + 1, j);
            zero_dy[idx] = both_invalid ||
                guard.fires(std::fabs(raw_dy[idx]), cor_dy[idx], guard_valid_dy[idx] != 0);
        }
    std::vector<uint8_t> zero_dx(phase_dx_s.size());
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            size_t idx = (size_t)(i * (n_phase - 1) + j);
            bool both_invalid = !is_valid(i, j) && !is_valid(i, j + 1);
            zero_dx[idx] = both_invalid ||
                guard.fires(std::fabs(raw_dx[idx]), cor_dx[idx], guard_valid_dx[idx] != 0);
        }
    tock("zero_mask");

#ifdef CUPHU_WW_GPU_COST
    if (gpu_id >= 0) {
        std::vector<int32_t> pos_dy(phase_dy_s.size()), neg_dy(phase_dy_s.size());
        std::vector<int32_t> pos_dx(phase_dx_s.size()), neg_dx(phase_dx_s.size());
        cuphu_ww_cost_lut_eval_gpu(phase_dy_s.data(), cor_dy.data(), zero_dy.data(),
                                    (int64_t)phase_dy_s.size(), nlooks, gpu_id,
                                    pos_dy.data(), neg_dy.data());
        cuphu_ww_cost_lut_eval_gpu(phase_dx_s.data(), cor_dx.data(), zero_dx.data(),
                                    (int64_t)phase_dx_s.size(), nlooks, gpu_id,
                                    pos_dx.data(), neg_dx.data());
        tock("cost_lut_gpu");
        // dy slab: c_rt = lut(-alpha) = neg, c_lt = lut(alpha) = pos.
        for (int64_t i = 0; i < m_phase - 1; ++i)
            for (int64_t j = 0; j < n_phase; ++j) {
                size_t idx = (size_t)(i * n_phase + j);
                cost[(size_t)g.right_arc(i + 1, j)] = neg_dy[idx];
                cost[(size_t)g.left_arc(i + 1, j + 1)] = pos_dy[idx];
            }
        // dx slab: c_dn = lut(alpha) = pos, c_up = lut(-alpha) = neg.
        for (int64_t i = 0; i < m_phase; ++i)
            for (int64_t j = 0; j < n_phase - 1; ++j) {
                size_t idx = (size_t)(i * (n_phase - 1) + j);
                cost[(size_t)g.down_arc(i, j + 1)] = pos_dx[idx];
                cost[(size_t)g.up_arc(i + 1, j + 1)] = neg_dx[idx];
            }
        tock("cost_remap");
        return cost;
    }
#else
    (void)gpu_id;
#endif

    const CarballoSplineLut &lut = CarballoSplineLut::get();
    // RIGHT/LEFT slabs from vertical pixel edges (alpha = phase_dy).
    for (int64_t i = 0; i < m_phase - 1; ++i) {
        for (int64_t j = 0; j < n_phase; ++j) {
            size_t idx = (size_t)(i * n_phase + j);
            float alpha = phase_dy_s[idx];
            float gamma = cor_dy[idx];
            int32_t c_rt = 0, c_lt = 0;
            if (!zero_dy[idx]) {
                c_rt = lut.cost(-alpha, gamma, nlooks);
                c_lt = lut.cost(alpha, gamma, nlooks);
            }
            cost[(size_t)g.right_arc(i + 1, j)] = c_rt;
            cost[(size_t)g.left_arc(i + 1, j + 1)] = c_lt;
        }
    }
    // DOWN/UP slabs from horizontal pixel edges (alpha = phase_dx).
    for (int64_t i = 0; i < m_phase; ++i) {
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            size_t idx = (size_t)(i * (n_phase - 1) + j);
            float alpha = phase_dx_s[idx];
            float gamma = cor_dx[idx];
            int32_t c_dn = 0, c_up = 0;
            if (!zero_dx[idx]) {
                c_dn = lut.cost(alpha, gamma, nlooks);
                c_up = lut.cost(-alpha, gamma, nlooks);
            }
            cost[(size_t)g.down_arc(i, j + 1)] = c_dn;
            cost[(size_t)g.up_arc(i + 1, j + 1)] = c_up;
        }
    }
    tock("cost_assembly");
    return cost;
}

// =============================================================================
// residue.rs: residue grid from wrapped phase (interior + boundary-frame
// charges). Output shape (m,n) = (m_phase+1, n_phase+1).
// =============================================================================
static inline int32_t cycle_diff(float a, float b) {
    return (int32_t)std::lround((a - b) / TAU);
}

static std::vector<int32_t> compute_residues(const float *wrapped, const unsigned char *mask,
                                              int64_t m_phase, int64_t n_phase) {
    int64_t m = m_phase + 1, n = n_phase + 1;
    std::vector<int32_t> out((size_t)(m * n), 0);
    auto valid = [&](int64_t i, int64_t j) -> bool {
        return mask == nullptr || mask[i * n_phase + j] != 0;
    };
    auto P = [&](int64_t i, int64_t j) { return wrapped[i * n_phase + j]; };

    for (int64_t i = 0; i < m_phase - 1; ++i) {
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            if (!(valid(i, j) && valid(i, j + 1) && valid(i + 1, j) && valid(i + 1, j + 1)))
                continue;
            float p00 = P(i, j), p01 = P(i, j + 1), p10 = P(i + 1, j), p11 = P(i + 1, j + 1);
            int32_t s = cycle_diff(p10, p00) + cycle_diff(p11, p10) + cycle_diff(p01, p11) +
                        cycle_diff(p00, p01);
            out[(size_t)((i + 1) * n + (j + 1))] = s;
        }
    }
    // Boundary-frame charges (sign convention matches residue.rs::compute_with_mask).
    for (int64_t j = 0; j < n_phase - 1; ++j) {
        if (valid(0, j) && valid(0, j + 1))
            out[(size_t)(0 * n + (j + 1))] += cycle_diff(P(0, j + 1), P(0, j));
    }
    for (int64_t j = 0; j < n_phase - 1; ++j) {
        if (valid(m_phase - 1, j) && valid(m_phase - 1, j + 1))
            out[(size_t)(m * n - n + (j + 1))] -=
                cycle_diff(P(m_phase - 1, j + 1), P(m_phase - 1, j));
    }
    for (int64_t i = 0; i < m_phase - 1; ++i) {
        if (valid(i, 0) && valid(i + 1, 0))
            out[(size_t)((i + 1) * n + 0)] -= cycle_diff(P(i + 1, 0), P(i, 0));
    }
    for (int64_t i = 0; i < m_phase - 1; ++i) {
        if (valid(i, n_phase - 1) && valid(i + 1, n_phase - 1))
            out[(size_t)((i + 1) * n + (n - 1))] +=
                cycle_diff(P(i + 1, n_phase - 1), P(i, n_phase - 1));
    }
    return out;
}

// =============================================================================
// network.rs: capacity-1 linear-cost network, no ground node, soft masking
// (matches Network::new_linear_packed -- what unwrap_linear actually uses).
// =============================================================================
struct Network {
    int64_t num_nodes = 0, num_forward = 0;
    std::vector<int32_t> excess;
    std::vector<int64_t> potential;
    std::vector<uint16_t> cost_fwd;
    std::vector<uint8_t> is_saturated; // length 2*num_forward
    std::vector<uint8_t> multi_unit;   // length num_forward; gutter-ring arcs

    Network(const Grid &g, std::vector<int32_t> excess_in, const std::vector<int32_t> &cost_in) {
        num_nodes = g.num_nodes();
        num_forward = g.num_forward;
        excess = std::move(excess_in);
        potential.assign((size_t)num_nodes, 0);
        cost_fwd.resize((size_t)num_forward);
        for (int64_t a = 0; a < num_forward; ++a) {
            int32_t c = cost_in[(size_t)a];
            cost_fwd[(size_t)a] = (uint16_t)std::min(std::max(c, 0), 65535);
        }
        is_saturated.assign((size_t)(2 * num_forward), 0);
        for (int64_t a = 0; a < num_forward; ++a)
            is_saturated[(size_t)(a + num_forward)] = 1; // reverse starts blocked

        mark_gutter_multi_unit(g);
    }

    void mark_gutter_multi_unit(const Grid &g) {
        multi_unit.assign((size_t)num_forward, 0);
        auto mark = [&](int64_t arc) {
            if (arc >= 0 && cost_fwd[(size_t)arc] == 0) multi_unit[(size_t)arc] = 1;
        };
        for (int64_t i = 0; i < g.m - 1; ++i) {
            mark(g.down_arc(i, 0));
            mark(g.up_arc(i + 1, 0));
            mark(g.down_arc(i, g.n - 1));
            mark(g.up_arc(i + 1, g.n - 1));
        }
        for (int64_t j = 0; j < g.n - 1; ++j) {
            mark(g.right_arc(0, j));
            mark(g.left_arc(0, j + 1));
            mark(g.right_arc(g.m - 1, j));
            mark(g.left_arc(g.m - 1, j + 1));
        }
    }

    inline int64_t transpose(int64_t arc) const {
        return arc < num_forward ? arc + num_forward : arc - num_forward;
    }
    inline int32_t arc_cost(int64_t arc) const {
        return arc < num_forward ? (int32_t)cost_fwd[(size_t)arc] : -(int32_t)cost_fwd[(size_t)(arc - num_forward)];
    }
    inline bool is_arc_saturated(int64_t arc) const { return is_saturated[(size_t)arc] != 0; }
    inline int64_t reduced_cost(const Grid &g, int64_t arc) const {
        int64_t t, h; g.arc_endpoints(arc, t, h);
        return (int64_t)arc_cost(arc) - potential[(size_t)t] + potential[(size_t)h];
    }
    void push_unit(int64_t arc) {
        int64_t fwd = arc < num_forward ? arc : arc - num_forward;
        int64_t t = transpose(arc);
        if (multi_unit[(size_t)fwd]) {
            is_saturated[(size_t)t] = 0; // never saturate the pushed direction
        } else {
            is_saturated[(size_t)arc] = 1;
            is_saturated[(size_t)t] = 0;
        }
    }
    inline int32_t arc_flow(int64_t arc) const {
        int64_t fwd = arc < num_forward ? arc : arc - num_forward;
        int64_t rev = fwd + num_forward;
        return (is_saturated[(size_t)fwd] && !is_saturated[(size_t)rev]) ? 1 : 0;
    }
};

// =============================================================================
// shortest_path/dial.rs: Dial's bucket-queue Dijkstra, full-completion,
// multi-source, FIFO buckets (BFS tie-break on the zero-cost masked sea --
// see ssp.rs's module doc for why this matters).
// =============================================================================
struct ShortestPaths {
    std::vector<int64_t> dist;
    std::vector<int64_t> pred_arc;
    std::vector<uint8_t> popped;

    explicit ShortestPaths(int64_t n_nodes) { reset(n_nodes); }
    void reset(int64_t n_nodes) {
        dist.assign((size_t)n_nodes, std::numeric_limits<int64_t>::max());
        pred_arc.assign((size_t)n_nodes, -1);
        popped.assign((size_t)n_nodes, 0);
    }
};

static int64_t max_reduced_cost(const Grid &g, const Network &net) {
    int64_t mx = 0;
    for (int64_t a = 0; a < net.num_forward * 2; ++a) {
        if (net.is_arc_saturated(a)) continue;
        int64_t rc = net.reduced_cost(g, a);
        if (rc > mx) mx = rc;
    }
    return mx;
}

// Multi-source, full-completion Dial's Dijkstra: every reachable node is
// popped and gets an exact finalized distance (matches
// dial.rs::run_full_scratch_into).
static void dial_run_full(const Grid &g, const Network &net, ShortestPaths &sp,
                           std::vector<std::deque<uint32_t>> &buckets) {
    sp.reset(net.num_nodes);
    int64_t max_rc = max_reduced_cost(g, net);
    int64_t k = std::max<int64_t>(max_rc + 1, 1);
    // Caller-owned scratch (matches dial.rs::run_full_scratch_into): reused
    // across calls instead of allocating up to ~k std::deques fresh every
    // pd_pass -- with costs capped around 6908 (parity spline LUT), k can
    // run into the thousands, and this allocation was a measurable fraction
    // of each pass's ~3.5-8s wall time.
    if ((int64_t)buckets.size() < k) buckets.resize((size_t)k);
    for (int64_t b = 0; b < k; ++b) buckets[(size_t)b].clear();

    int64_t pending = 0;
    for (int64_t v = 0; v < net.num_nodes; ++v) {
        if (net.excess[(size_t)v] > 0) {
            sp.dist[(size_t)v] = 0;
            buckets[0].push_back((uint32_t)v);
            ++pending;
        }
    }
    if (pending == 0) return;

    int64_t cur_bucket = 0, cur_dist = 0, bucket_advances = 0;
    std::vector<std::pair<int64_t, int64_t>> out_buf;
    while (pending > 0) {
        while (buckets[(size_t)cur_bucket].empty()) {
            cur_bucket = (cur_bucket + 1) % k;
            ++cur_dist;
            if (++bucket_advances > k) return;
        }
        bucket_advances = 0;
        int64_t u = buckets[(size_t)cur_bucket].front();
        buckets[(size_t)cur_bucket].pop_front();
        --pending;
        if (sp.dist[(size_t)u] != cur_dist) continue;
        sp.popped[(size_t)u] = 1;

        int64_t pot_u = net.potential[(size_t)u];
        g.outgoing(u, out_buf);
        for (auto &pr : out_buf) {
            int64_t arc = pr.first, v = pr.second;
            if (net.is_arc_saturated(arc)) continue;
            int64_t rc = (int64_t)net.arc_cost(arc) - pot_u + net.potential[(size_t)v];
            if (rc < 0 && std::getenv("CUPHU_WW_DEBUG"))
                std::fprintf(stderr, "[dial_run_full] NEGATIVE rc=%lld arc=%lld u=%lld v=%lld\n",
                             (long long)rc, (long long)arc, (long long)u, (long long)v);
            int64_t nd = cur_dist + rc;
            if (nd < sp.dist[(size_t)v]) {
                sp.dist[(size_t)v] = nd;
                sp.pred_arc[(size_t)v] = arc;
                buckets[(size_t)(nd % k)].push_back((uint32_t)v);
                ++pending;
            }
        }
    }
}

// =============================================================================
// primal_dual.rs (run_impl_full_fused, split-layout logic) + ssp.rs
// (run_single_source_scratch, drain_residual_bfs): successive shortest
// paths with node potentials, Dial's algorithm for the shortest-path
// subproblem, a single-source completion tail, and a cost-ignoring BFS
// final balance guard.
// =============================================================================

// One multi-source PD pass: Dijkstra, augment one unit per available source,
// update potentials. Returns the number of units augmented.
static int64_t pd_pass(const Grid &g, Network &net, ShortestPaths &sp,
                        std::vector<std::deque<uint32_t>> &buckets) {
    dial_run_full(g, net, sp, buckets);

    std::vector<int64_t> deficits;
    for (int64_t v = 0; v < net.num_nodes; ++v)
        if (net.excess[(size_t)v] < 0) deficits.push_back(v);

    // (sink, source, arcs) triples for every reached deficit, via the pred chain.
    // Cycle detection uses one reusable epoch-stamped buffer (matches
    // primal_dual.rs's `visited_epoch`) instead of allocating+zeroing an
    // O(num_nodes) vector per sink -- at real-scene node counts (tens of
    // millions) a per-sink allocation would dominate runtime.
    struct PathInfo { int64_t sink, src; std::vector<int64_t> arcs; };
    std::vector<PathInfo> paths;
    static thread_local std::vector<uint32_t> visited_epoch;
    static thread_local uint32_t epoch = 0;
    if ((int64_t)visited_epoch.size() != net.num_nodes)
        visited_epoch.assign((size_t)net.num_nodes, 0);
    for (int64_t sink : deficits) {
        if (!sp.popped[(size_t)sink]) continue;
        std::vector<int64_t> arcs;
        int64_t cur = sink;
        if (++epoch == 0) { std::fill(visited_epoch.begin(), visited_epoch.end(), 0); epoch = 1; }
        visited_epoch[(size_t)cur] = epoch;
        bool cyc = false;
        for (;;) {
            int64_t pa = sp.pred_arc[(size_t)cur];
            if (pa < 0) break;
            arcs.push_back(pa);
            int64_t t, h; g.arc_endpoints(pa, t, h);
            cur = t;
            if (visited_epoch[(size_t)cur] == epoch) { cyc = true; break; }
            visited_epoch[(size_t)cur] = epoch;
        }
        if (!cyc && !arcs.empty()) paths.push_back({sink, cur, std::move(arcs)});
    }
    // Deterministic order: by (source, dist[sink]).
    std::sort(paths.begin(), paths.end(), [&](const PathInfo &a, const PathInfo &b) {
        if (a.src != b.src) return a.src < b.src;
        return sp.dist[(size_t)a.sink] < sp.dist[(size_t)b.sink];
    });
    std::vector<uint8_t> source_used((size_t)net.num_nodes, 0);
    int64_t augmented = 0;
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    for (auto &p : paths) {
        if (source_used[(size_t)p.src]) continue;
        source_used[(size_t)p.src] = 1;
        for (int64_t arc : p.arcs) {
            if (dbg && net.is_arc_saturated(arc))
                std::fprintf(stderr, "[pd_pass] DOUBLE-PUSH on saturated arc=%lld sink=%lld src=%lld\n",
                             (long long)arc, (long long)p.sink, (long long)p.src);
            net.push_unit(arc);
        }
        net.excess[(size_t)p.sink] += 1;
        net.excess[(size_t)p.src] -= 1;
        ++augmented;
    }

    // Potential update: popped nodes get exact dist; unpopped nodes are
    // capped at d_max (the largest popped distance).
    int64_t d_max = 0;
    for (int64_t v = 0; v < net.num_nodes; ++v)
        if (sp.popped[(size_t)v] && sp.dist[(size_t)v] > d_max) d_max = sp.dist[(size_t)v];
    for (int64_t v = 0; v < net.num_nodes; ++v) {
        int64_t dv = sp.popped[(size_t)v] ? sp.dist[(size_t)v] : d_max;
        net.potential[(size_t)v] -= dv;
    }
    return augmented;
}

static int64_t total_excess_magnitude(const Network &net, bool positive) {
    int64_t s = 0;
    for (int32_t e : net.excess) {
        if (positive && e > 0) s += e;
        if (!positive && e < 0) s -= e;
    }
    return s;
}

// Single-source completion pass: one source at a time, early-exiting Dial's
// search the moment the first deficit pops (matches
// ssp.rs::run_single_source_scratch). `max_rc` is a bound carried ACROSS
// sources/units, not rescanned per unit: an O(num_arcs) scan per augmented
// unit is the difference between finishing and never finishing at
// real-scene node counts. On an overflow (a relaxation needs rc >= k, i.e.
// the bound was stale) that source's partial search is discarded and
// max_rc is rescanned tight, then retried -- rare in practice, since
// potentials only grow by bounded amounts per augmentation.
static void ssp_single_source(const Grid &g, Network &net) {
    std::vector<int64_t> sources;
    for (int64_t v = 0; v < net.num_nodes; ++v)
        if (net.excess[(size_t)v] > 0) sources.push_back(v);

    std::vector<int64_t> dist((size_t)net.num_nodes, std::numeric_limits<int64_t>::max());
    std::vector<int64_t> pred_arc((size_t)net.num_nodes, -1);
    std::vector<uint8_t> popped((size_t)net.num_nodes, 0);
    std::vector<int64_t> touched;
    std::vector<std::pair<int64_t, int64_t>> out_buf;
    std::vector<std::deque<uint32_t>> buckets;

    int64_t max_rc = max_reduced_cost(g, net);
    int64_t deficit_units = total_excess_magnitude(net, false);

    for (int64_t src : sources) {
        if (deficit_units == 0) break;
        while (net.excess[(size_t)src] > 0 && deficit_units > 0) {
            int64_t sink = -1, d_sink = 0;

            // Retry loop: re-entered only on k-overflow (a relaxation needed
            // rc >= k because `max_rc` was stale). `touched` is reset at the
            // top of each attempt so a retry starts clean; on final exit
            // (success or genuine exhaustion) `dist`/`pred_arc`/`popped` for
            // `touched` are still live -- the caller below reads them before
            // this function's own post-augment reset.
            for (;;) {
                int64_t k = std::max<int64_t>(max_rc + 1, 1);
                if ((int64_t)buckets.size() < k) buckets.resize((size_t)k);
                for (auto &b : buckets) b.clear();
                for (int64_t v : touched) {
                    dist[(size_t)v] = std::numeric_limits<int64_t>::max();
                    pred_arc[(size_t)v] = -1;
                    popped[(size_t)v] = 0;
                }
                touched.clear();

                dist[(size_t)src] = 0;
                touched.push_back(src);
                buckets[0].push_back((uint32_t)src);
                int64_t pending = 1, cur_bucket = 0, cur_dist = 0, bucket_advances = 0;
                bool overflow = false, neg_rc = false;
                int64_t max_rc_seen = 0;
                sink = -1;

                while (pending > 0) {
                    while (buckets[(size_t)cur_bucket].empty()) {
                        cur_bucket = (cur_bucket + 1) % k;
                        ++cur_dist;
                        if (++bucket_advances > k) break;
                    }
                    if (buckets[(size_t)cur_bucket].empty()) break; // full k-cycle: exhausted
                    bucket_advances = 0;
                    int64_t u = buckets[(size_t)cur_bucket].front();
                    buckets[(size_t)cur_bucket].pop_front();
                    --pending;
                    if (dist[(size_t)u] != cur_dist) continue;
                    popped[(size_t)u] = 1;
                    if (net.excess[(size_t)u] < 0) { sink = u; d_sink = cur_dist; break; }
                    int64_t pot_u = net.potential[(size_t)u];
                    g.outgoing(u, out_buf);
                    for (auto &pr : out_buf) {
                        int64_t arc = pr.first, v = pr.second;
                        if (net.is_arc_saturated(arc)) continue;
                        int64_t rc = (int64_t)net.arc_cost(arc) - pot_u + net.potential[(size_t)v];
                        if (rc < 0) { neg_rc = true; break; }
                        if (rc >= k) { overflow = true; break; }
                        if (rc > max_rc_seen) max_rc_seen = rc;
                        int64_t nd = cur_dist + rc;
                        if (nd < dist[(size_t)v]) {
                            if (dist[(size_t)v] == std::numeric_limits<int64_t>::max()) touched.push_back(v);
                            dist[(size_t)v] = nd;
                            pred_arc[(size_t)v] = arc;
                            buckets[(size_t)(nd % k)].push_back((uint32_t)v);
                            ++pending;
                        }
                    }
                    if (overflow || neg_rc) break;
                }

                if (neg_rc) { sink = -1; break; } // abandon this source; BFS guard cleans up
                if (overflow) { max_rc = max_reduced_cost(g, net); continue; } // retry, tighter k
                if (max_rc_seen > max_rc) max_rc = max_rc_seen;
                break;
            }

            if (sink >= 0) {
                net.excess[(size_t)sink] += 1;
                net.excess[(size_t)src] -= 1;
                --deficit_units;
                int64_t cur = sink;
                for (;;) {
                    int64_t pa = pred_arc[(size_t)cur];
                    if (pa < 0) break;
                    net.push_unit(pa);
                    int64_t t, h; g.arc_endpoints(pa, t, h);
                    cur = t;
                }
                for (int64_t v : touched)
                    if (popped[(size_t)v]) net.potential[(size_t)v] += d_sink - dist[(size_t)v];
            }
            for (int64_t v : touched) {
                dist[(size_t)v] = std::numeric_limits<int64_t>::max();
                pred_arc[(size_t)v] = -1;
                popped[(size_t)v] = 0;
            }
            touched.clear();
            if (sink < 0) break; // stranded (or negative-rc abandon); BFS guard pairs it
        }
    }
}

// Cost-ignoring BFS pairing of any remaining excess/deficit (final safety
// net: guarantees full balance so integration never sees a tear). Matches
// ssp.rs::drain_residual_bfs.
static void drain_residual_bfs(const Grid &g, Network &net) {
    if (total_excess_magnitude(net, false) == 0) return;
    std::vector<int64_t> pred_arc((size_t)net.num_nodes, -1);
    std::vector<uint8_t> visited((size_t)net.num_nodes, 0);
    std::vector<int64_t> touched;
    std::deque<uint32_t> queue;
    std::vector<std::pair<int64_t, int64_t>> out_buf;

    std::vector<int64_t> sources;
    for (int64_t v = 0; v < net.num_nodes; ++v)
        if (net.excess[(size_t)v] > 0) sources.push_back(v);

    // Maintained across units (not rescanned per unit -- same reasoning as
    // ssp_single_source's `deficit_units`).
    int64_t deficit_units = total_excess_magnitude(net, false);

    for (int64_t src : sources) {
        if (deficit_units == 0) break;
        while (net.excess[(size_t)src] > 0 && deficit_units > 0) {
            visited[(size_t)src] = 1;
            touched.push_back(src);
            queue.push_back((uint32_t)src);
            int64_t sink = -1;
            while (!queue.empty()) {
                int64_t u = queue.front(); queue.pop_front();
                if (net.excess[(size_t)u] < 0) { sink = u; break; }
                g.outgoing(u, out_buf);
                for (auto &pr : out_buf) {
                    int64_t arc = pr.first, v = pr.second;
                    if (net.is_arc_saturated(arc) || visited[(size_t)v]) continue;
                    visited[(size_t)v] = 1;
                    touched.push_back(v);
                    pred_arc[(size_t)v] = arc;
                    queue.push_back((uint32_t)v);
                }
            }
            if (sink >= 0) {
                net.excess[(size_t)sink] += 1;
                net.excess[(size_t)src] -= 1;
                --deficit_units;
                int64_t cur = sink;
                for (;;) {
                    int64_t pa = pred_arc[(size_t)cur];
                    if (pa < 0) break;
                    net.push_unit(pa);
                    int64_t t, h; g.arc_endpoints(pa, t, h);
                    cur = t;
                }
            }
            for (int64_t v : touched) { visited[(size_t)v] = 0; pred_arc[(size_t)v] = -1; }
            touched.clear();
            queue.clear();
            if (sink < 0) break;
        }
    }
}

// Run up to `max_iter` pd_pass() calls, stopping early if a pass makes no
// progress. `sp`/`buckets` are caller-owned scratch, reused across calls.
static int run_pd_passes(const Grid &g, Network &net, ShortestPaths &sp,
                          std::vector<std::deque<uint32_t>> &buckets,
                          int max_iter, int64_t &last_excess, bool dbg, const char *tag) {
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        int64_t excess_total = total_excess_magnitude(net, true);
        int64_t deficit_total = total_excess_magnitude(net, false);
        if (excess_total == 0 || deficit_total == 0) break;
        if (excess_total >= last_excess) break;
        last_excess = excess_total;
        auto tpd0 = std::chrono::steady_clock::now();
        pd_pass(g, net, sp, buckets);
        if (dbg) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tpd0).count();
            std::fprintf(stderr, "[solve] %s iter=%d excess=%lld (%.3fs)\n",
                         tag, iter, (long long)total_excess_magnitude(net, true), dt);
        }
    }
    return iter;
}

// Full solve: up to 8 multi-source PD passes (matches unwrap_linear's
// `primal_dual::run_full_dijkstra(&graph, &mut net, 8)`), then single-source
// SSP completion, then the BFS balance guard.

static void solve(const Grid &g, Network &net) {
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    if (dbg) {
        int64_t ex0 = total_excess_magnitude(net, true);
        std::fprintf(stderr, "[solve] initial excess=%lld nodes=%lld arcs=%lld\n",
                     (long long)ex0, (long long)net.num_nodes, (long long)(2 * net.num_forward));
    }
    ShortestPaths sp(net.num_nodes);
    std::vector<std::deque<uint32_t>> buckets;
    int64_t last_excess = std::numeric_limits<int64_t>::max();
    auto ts0 = std::chrono::steady_clock::now();
    int iter = run_pd_passes(g, net, sp, buckets, 8, last_excess, dbg, "after pd_pass");
    if (dbg) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count();
        std::fprintf(stderr, "[solve] PD loop done after %d iters, excess=%lld (PD total %.3fs)\n",
                     iter, (long long)total_excess_magnitude(net, true), dt);
    }

    // NOTE: an "adaptive resume" (a few more batched pd_pass() calls before
    // falling to the per-unit SSP tail, matching primal_dual.rs's
    // run_full_dijkstra wrapper) was tried and measured here -- and reverted.
    // On the venezuela crop it made things dramatically WORSE (solve() 90s ->
    // 174s): each pd_pass call pays a large FIXED cost (a full O(nodes+arcs)
    // graph traversal, a fresh O(arcs) max_reduced_cost scan, and a fresh
    // per-call bucket-vector allocation) that does NOT shrink as the residual
    // source count drops, because the residual graph stays almost fully
    // connected (only ~1% of arcs are saturated by this point) -- so a
    // full-completion Dijkstra still visits nearly the whole graph regardless
    // of how few sources remain. Extending PD passes for a shrinking tail of
    // residues trades a small, bounded-cost path (SSP: ~15.8ms/unit here) for
    // a large, roughly-constant one (~3.5s/pass for single-digit units
    // drained). Left as CPU-only, single-pass-per-unit SSP below; see the
    // implementation plan for where a real win might actually be (fixing
    // dial_run_full's per-call fixed costs, not adding more calls of it).
    if (dbg)
        std::fprintf(stderr, "[solve] starting ssp, excess=%lld\n",
                     (long long)total_excess_magnitude(net, true));
    auto tssp0 = std::chrono::steady_clock::now();
    ssp_single_source(g, net);
    if (dbg) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tssp0).count();
        std::fprintf(stderr, "[solve] after ssp, excess=%lld, starting bfs drain (ssp %.3fs)\n",
                     (long long)total_excess_magnitude(net, true), dt);
    }
    auto tbfs0 = std::chrono::steady_clock::now();
    drain_residual_bfs(g, net);
    if (dbg) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tbfs0).count();
        std::fprintf(stderr, "[solve] bfs drain took %.3fs\n", dt);
    }
    if (dbg)
        std::fprintf(stderr, "[solve] after bfs drain, excess=%lld (should be 0)\n",
                     (long long)total_excess_magnitude(net, true));
}

// =============================================================================
// integrate.rs: integer-cycle phase integration (unmasked path -- matches
// unwrap_linear's use of integrate::integrate; masked pixels still get a
// value here, left for the caller's downstream mask-aware stages, matching
// how CUPHU_INIT_LAPLACE hands its own unw + mask to cuphu_conncomp_gpu
// rather than NaN-ing masked pixels itself).
// =============================================================================
static inline int32_t wrap_n_cycle(float a, float b) {
    return -(int32_t)std::lround((a - b) / TAU);
}

static void integrate(const Grid &g, const Network &net, const float *wrapped,
                       int64_t m_phase, int64_t n_phase, float *unw_out) {
    std::vector<int32_t> col0_cycles((size_t)m_phase, 0);
    for (int64_t i = 1; i < m_phase; ++i) {
        int32_t n_cyc = wrap_n_cycle(wrapped[i * n_phase + 0], wrapped[(i - 1) * n_phase + 0]);
        int64_t fwd = g.right_arc(i, 0), rev = g.left_arc(i, 1);
        int32_t net_flow = net.arc_flow(rev) - net.arc_flow(fwd);
        col0_cycles[(size_t)i] = col0_cycles[(size_t)(i - 1)] + n_cyc + net_flow;
    }
    for (int64_t i = 0; i < m_phase; ++i) {
        int32_t cycles = col0_cycles[(size_t)i];
        unw_out[i * n_phase + 0] = wrapped[i * n_phase + 0] + TAU * (float)cycles;
        for (int64_t j = 1; j < n_phase; ++j) {
            int32_t n_cyc = wrap_n_cycle(wrapped[i * n_phase + j], wrapped[i * n_phase + j - 1]);
            int64_t fwd = g.down_arc(i, j), rev = g.up_arc(i + 1, j);
            int32_t net_flow = net.arc_flow(fwd) - net.arc_flow(rev);
            cycles += n_cyc + net_flow;
            unw_out[i * n_phase + j] = wrapped[i * n_phase + j] + TAU * (float)cycles;
        }
    }
}

} // namespace cuphu_ww

int cuphu_whirlwind_unwrap(
    const float *igram_r, const float *igram_i, const float *corr,
    const unsigned char *mask, int nrow, int ncol, double nlooks, int gpu_id,
    float *unw_out)
{
    using namespace cuphu_ww;
    if (nrow < 2 || ncol < 2) return 1;
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto tock = [&](const char *stage) {
        if (!dbg) return;
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[cuphu_whirlwind_unwrap] %-12s %8.3fs\n", stage,
                     std::chrono::duration<double>(t1 - t0).count());
        t0 = t1;
    };

    int64_t m_phase = nrow, n_phase = ncol;
    std::vector<std::complex<float>> igram((size_t)(m_phase * n_phase));
    std::vector<float> wrapped((size_t)(m_phase * n_phase));
    for (int64_t i = 0; i < m_phase * n_phase; ++i) {
        igram[(size_t)i] = std::complex<float>(igram_r[i], igram_i[i]);
        wrapped[(size_t)i] = std::arg(igram[(size_t)i]);
    }
    tock("wrap");

    Grid g(m_phase + 1, n_phase + 1);
    auto residues = compute_residues(wrapped.data(), mask, m_phase, n_phase);
    tock("residues");
    auto costs = compute_carballo_costs(g, igram.data(), corr, mask, m_phase, n_phase,
                                         (float)nlooks, /*window_par=*/7, /*window_perp=*/7, gpu_id);
    tock("costs");

    Network net(g, std::move(residues), costs);
    tock("network");
    solve(g, net);
    tock("solve");
    integrate(g, net, wrapped.data(), m_phase, n_phase, unw_out);
    tock("integrate");
    return 0;
}
