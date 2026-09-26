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
    // Deterministic order: by (source, dist[sink]). MUST be a stable sort --
    // matches Rust's `sort_by_key` (stable by spec). In a degenerate region
    // (e.g. a large masked block: uniform cost=0 everywhere), many sinks can
    // share the exact same (source, dist) key; std::sort is NOT required to
    // preserve relative order among ties, so an unstable sort here silently
    // picks a DIFFERENT tied winner per source than Rust does, diverging the
    // whole downstream flow/phase from a seemingly-correct, non-crashing,
    // fully-deterministic (within this binary) but wrong computation.
    std::stable_sort(paths.begin(), paths.end(), [&](const PathInfo &a, const PathInfo &b) {
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
    if (std::getenv("CUPHU_WW_DEBUG")) {
        int64_t excess_total = total_excess_magnitude(net, true);
        int64_t multi_unit_sources = 0;
        for (int64_t v : sources) if (net.excess[(size_t)v] > 1) ++multi_unit_sources;
        std::fprintf(stderr,
            "[ssp_single_source] n_sources=%zu excess_total=%lld deficit_units=%lld "
            "n_sources_with_excess>1=%lld\n",
            sources.size(), (long long)excess_total, (long long)deficit_units,
            (long long)multi_unit_sources);
    }

    bool dbg2 = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    int64_t n_overflow_events = 0, n_searches = 0, max_touched = 0, sum_touched = 0;
    double t_rescan = 0.0, t_search = 0.0;

    for (int64_t src : sources) {
        if (deficit_units == 0) break;
        while (net.excess[(size_t)src] > 0 && deficit_units > 0) {
            int64_t sink = -1, d_sink = 0;
            ++n_searches;
            auto t_src0 = std::chrono::steady_clock::now();

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
                if (overflow) {
                    ++n_overflow_events;
                    auto tr0 = std::chrono::steady_clock::now();
                    max_rc = max_reduced_cost(g, net);
                    t_rescan += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tr0).count();
                    continue; // retry, tighter k
                }
                if (max_rc_seen > max_rc) max_rc = max_rc_seen;
                break;
            }
            max_touched = std::max(max_touched, (int64_t)touched.size());
            sum_touched += (int64_t)touched.size();
            t_search += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_src0).count();

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
    if (dbg2) {
        std::fprintf(stderr,
            "[ssp_single_source] n_searches=%lld n_overflow_events=%lld "
            "avg_touched=%.1f max_touched=%lld t_search_total=%.3fs "
            "(of which t_rescan=%.3fs, %.1f%%)\n",
            (long long)n_searches, (long long)n_overflow_events,
            n_searches ? (double)sum_touched / (double)n_searches : 0.0,
            (long long)max_touched, t_search, t_rescan,
            t_search > 0 ? 100.0 * t_rescan / t_search : 0.0);
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
            std::fprintf(stderr, "[solve] %s iter=%d excess=%lld deficit=%lld (%.3fs)\n",
                         tag, iter, (long long)total_excess_magnitude(net, true),
                         (long long)total_excess_magnitude(net, false), dt);
        }
    }
    return iter;
}

// Full solve: up to 8 multi-source PD passes (matches unwrap_linear's
// `primal_dual::run_full_dijkstra(&graph, &mut net, 8)`), then single-source
// SSP completion, then -- if still unbalanced -- a guarded adaptive resume
// (chunked PD + repeated SSP, matching run_full_dijkstra's own fallback),
// then the BFS balance guard.
//
// PD(8)+SSP(1) alone is NOT always sufficient: whirlwind's own Rust source
// documents this directly (primal_dual.rs, run_full_dijkstra's "GUARDED
// ADAPTIVE FALLBACK" comment) -- single-source SSP can strand residues in
// heavily-fragmented residual graphs, which PD(8)+SSP(1) alone leaves
// unresolved. The resume loop below is whirlwind's own fix for that class
// of issue, kept here for fidelity even though it's a no-op whenever excess
// already reaches 0 (which it does whenever the network is correctly
// constructed -- see compute_residues()'s doc comment on the actual bug
// this class of masked-scene divergence traced back to: an incorrectly
// MASKED residue grid, not anything in this solve loop).
//
// An earlier attempt at "a few more pd_pass() calls before the SSP tail" was
// tried and reverted as a regression -- but that was measured ONLY at crop
// scale (~22.5M nodes, ~5000 rows), where PD(8)+SSP(1) already fully
// converges and any extra pd_pass() calls are pure overhead (each pays a
// large FIXED O(nodes+arcs) cost regardless of remaining residue count).
// That conclusion doesn't generalize to scales where PD(8)+SSP(1) leaves a
// real residual -- which is exactly when whirlwind's own resume loop is
// designed to trigger (it's a no-op, by construction, whenever excess
// already reached 0).
static void solve(const Grid &g, Network &net) {
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    if (dbg) {
        int64_t ex0 = total_excess_magnitude(net, true);
        int64_t df0 = total_excess_magnitude(net, false);
        std::fprintf(stderr, "[solve] initial excess=%lld deficit=%lld nodes=%lld arcs=%lld\n",
                     (long long)ex0, (long long)df0, (long long)net.num_nodes,
                     (long long)(2 * net.num_forward));
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

    if (dbg)
        std::fprintf(stderr, "[solve] starting ssp, excess=%lld\n",
                     (long long)total_excess_magnitude(net, true));
    auto tssp0 = std::chrono::steady_clock::now();
    ssp_single_source(g, net);
    if (dbg) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tssp0).count();
        std::fprintf(stderr, "[solve] after ssp, excess=%lld (ssp %.3fs)\n",
                     (long long)total_excess_magnitude(net, true), dt);
    }

    // ── guarded adaptive resume (matches run_full_dijkstra's fallback) ──
    // Only engages when PD(8)+SSP(1) left the network genuinely unbalanced
    // on BOTH sides (a one-sided imbalance has no augmenting path left, so
    // resuming would only burn whole-graph Dijkstra floods discovering
    // that -- report and fall through to the BFS guard instead).
    if (total_excess_magnitude(net, true) > 0 && total_excess_magnitude(net, false) > 0) {
        int cap = 512;
        if (const char *e = std::getenv("CUPHU_WW_PD_CAP")) cap = std::atoi(e);
        int total = 8;
        auto tresume0 = std::chrono::steady_clock::now();
        int rounds = 0;
        while (total_excess_magnitude(net, true) > 0 &&
               total_excess_magnitude(net, false) > 0 && total < cap) {
            int64_t before = total_excess_magnitude(net, true);
            int chunk = std::min(16, cap - total);
            int64_t chunk_last_excess = std::numeric_limits<int64_t>::max();
            run_pd_passes(g, net, sp, buckets, chunk, chunk_last_excess, false, "resume");
            total += chunk;
            ++rounds;
            int64_t after = total_excess_magnitude(net, true);
            if (dbg)
                std::fprintf(stderr, "[solve] adaptive resume round=%d total_pd=%d excess %lld->%lld\n",
                             rounds, total, (long long)before, (long long)after);
            if (after == 0) break;
            ssp_single_source(g, net);
            if (total_excess_magnitude(net, true) == 0) break;
            if (after >= before) {
                if (dbg)
                    std::fprintf(stderr, "[solve] adaptive resume: no progress, stopping\n");
                break;
            }
        }
        if (dbg) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tresume0).count();
            std::fprintf(stderr, "[solve] adaptive resume done, %d rounds, excess=%lld (%.3fs)\n",
                         rounds, (long long)total_excess_magnitude(net, true), dt);
        }
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

// =============================================================================
// conncomp.rs: legacy linear connected components, grown from the ALREADY-
// SOLVED network's own raw arc costs (NOT cuphu's shared phase-diff+
// coherence-threshold cuphu_conncomp_gpu(), which has neither access to nor
// the same semantics as this cost data). Matches conncomp.rs::edge_is_cut /
// grow_components exactly, including the two-forward-arc-per-geometric-edge
// convention (e.g. a horizontal pixel edge is checked via both the DOWN arc
// at its right endpoint and the UP arc one row below -- same duality
// integrate() uses for its column-0/row seeds).
// =============================================================================
// Shared finalizer for both conncomp variants: drop components < min_size,
// keep the top max_ncomps (largest first), renumber survivors 1..=k in
// descending-size order. Matches conncomp.rs::finalize_labels exactly,
// including the stable sort (Rust's `sort_by` is stable).
static void finalize_labels(uint32_t *labels_out, size_t npix,
                             const std::vector<size_t> &sizes, uint32_t next_label,
                             size_t min_size, uint32_t max_ncomps) {
    std::vector<uint32_t> indices;
    for (uint32_t l = 1; l <= next_label; ++l)
        if (sizes[l] >= min_size) indices.push_back(l);
    std::stable_sort(indices.begin(), indices.end(),
                      [&](uint32_t a, uint32_t b) { return sizes[a] > sizes[b]; });
    if (max_ncomps > 0 && indices.size() > max_ncomps) indices.resize(max_ncomps);
    std::vector<uint32_t> renumber((size_t)next_label + 1, 0);
    for (size_t idx = 0; idx < indices.size(); ++idx) renumber[indices[idx]] = (uint32_t)(idx + 1);
    for (size_t p = 0; p < npix; ++p) labels_out[p] = renumber[labels_out[p]];
}

// =============================================================================
// cost/hyp2f1.rs: Gauss hypergeometric 2F1(a,b;c;z) -- needed only for the
// Lee (1994) PDF behind the SNAPHU-style conncomp's variance LUT (the
// unwrap itself uses the embedded parity spline tables and never needs
// this). Ported byte-for-byte: Gauss series for z<0.7, Euler transform
// above, and a positive-term log-space series (ln_hyp2f1_pos) for the PDF
// itself since all its terms are positive in this parameter range.
// =============================================================================
static double gauss_series(double a, double b, double c, double z, int max_terms) {
    double term = 1.0, sum = 1.0;
    for (int k = 0; k < max_terms; ++k) {
        double kf = (double)k;
        double num = (a + kf) * (b + kf) * z;
        double den = (c + kf) * (kf + 1.0);
        term *= num / den;
        sum += term;
        if (std::fabs(term) <= 1e-15 * std::fabs(sum)) break;
    }
    return sum;
}

static double ln_hyp2f1_pos(double a, double b, double c, double z) {
    if (z <= 0.0) return 0.0;
    const double RESCALE = 1e250;
    const double ln_rescale = std::log(RESCALE);
    double term = 1.0, sum = 1.0, ln_scale = 0.0;
    for (int k = 0; k < 200000; ++k) {
        double kf = (double)k;
        term *= (a + kf) * (b + kf) * z / ((c + kf) * (kf + 1.0));
        sum += term;
        if (sum > RESCALE) {
            sum /= RESCALE;
            term /= RESCALE;
            ln_scale += ln_rescale;
        }
        if (term <= 1e-16 * sum) break;
    }
    return std::log(sum) + ln_scale;
}

// =============================================================================
// cost/lee_pdf.rs: Lee (1994) multilook interferometric phase PDF, needed
// only for the SNAPHU-style conncomp's gamma->variance LUT.
// =============================================================================
static double lanczos_lgamma(double x) {
    static const double G = 7.0;
    static const double COEF[9] = {
        0.9999999999998099, 676.5203681218851, -1259.1392167224028,
        771.3234287776531, -176.6150291621406, 12.507343278686905,
        -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7};
    if (x < 0.5) {
        double pi = M_PI;
        return std::log(pi / std::sin(pi * x)) - lanczos_lgamma(1.0 - x);
    }
    x -= 1.0;
    double a = COEF[0];
    for (int i = 1; i < 9; ++i) a += COEF[i] / (x + (double)i);
    double t = x + G + 0.5;
    return 0.5 * std::log(2.0 * M_PI) + (x + 0.5) * std::log(t) - t + std::log(a);
}

static float lee_pdf(float alpha, float gamma, float nlooks) {
    double g = (double)gamma, a = (double)alpha, n = (double)nlooks;
    double g2 = g * g;
    double beta = g * std::cos(a);
    double b2 = beta * beta;
    double one_minus_b2 = std::max(1.0 - b2, 1e-300);

    double ln_t1 = n * std::log(1.0 - g2) - std::log(2.0 * M_PI) + ln_hyp2f1_pos(n, 1.0, 0.5, b2);

    double ln_t2 = lanczos_lgamma(n + 0.5) - lanczos_lgamma(n) + n * std::log(1.0 - g2) -
                   (n + 0.5) * std::log(one_minus_b2) + std::log(std::fabs(beta)) -
                   std::log(2.0 * std::sqrt(M_PI));

    if (beta == 0.0) return (float)std::exp(ln_t1);

    double sum;
    if (beta > 0.0) {
        sum = std::exp(ln_t1) + std::exp(ln_t2);
    } else {
        sum = -std::exp(ln_t1) * std::expm1(ln_t2 - ln_t1);
    }
    return (float)std::max(sum, 0.0);
}

// gamma -> wrapped-phase-variance LUT (numerically integrates the Lee PDF's
// second moment), matches cost/lut.rs::VarianceLut exactly.
struct VarianceLut {
    static constexpr int N_GAMMA_VAR = 1024;
    static constexpr float GAMMA_VAR_LO = 0.0f, GAMMA_VAR_HI = 0.999f;
    std::vector<float> sigma_sq;

    static VarianceLut build(float nlooks) {
        VarianceLut lut;
        lut.sigma_sq.resize(N_GAMMA_VAR);
        const int N_ALPHA = 1024;
        float dalpha = 2.0f * PI_F / (float)N_ALPHA;
        for (int i = 0; i < N_GAMMA_VAR; ++i) {
            float gamma = GAMMA_VAR_LO + (GAMMA_VAR_HI - GAMMA_VAR_LO) * (float)i / (float)(N_GAMMA_VAR - 1);
            float s2 = 0.0f, norm = 0.0f;
            for (int k = 0; k < N_ALPHA; ++k) {
                float alpha = -PI_F + ((float)k + 0.5f) * dalpha;
                float p = lee_pdf(alpha, gamma, nlooks);
                s2 += alpha * alpha * p;
                norm += p;
            }
            lut.sigma_sq[(size_t)i] = norm > 1e-12f ? s2 / norm : (PI_F * PI_F) / 3.0f;
        }
        return lut;
    }

    float eval(float gamma) const {
        float g = std::min(std::max(gamma, GAMMA_VAR_LO), GAMMA_VAR_HI);
        float gi = (g - GAMMA_VAR_LO) / (GAMMA_VAR_HI - GAMMA_VAR_LO) * (float)(N_GAMMA_VAR - 1);
        int i0 = (int)std::floor(gi);
        int i1 = std::min(i0 + 1, N_GAMMA_VAR - 1);
        float f = gi - (float)i0;
        return sigma_sq[(size_t)i0] * (1.0f - f) + sigma_sq[(size_t)i1] * f;
    }
};

// Cached per (nlooks rounded to 0.1), matches lut.rs::get_or_build_variance's
// caching (avoids rebuilding the O(1024*1024) LUT on every call).
static const VarianceLut &get_or_build_variance(float nlooks) {
    static std::mutex mtx;
    static std::unordered_map<int, VarianceLut> cache;
    int key = (int)std::lround(nlooks * 10.0f);
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto ins = cache.emplace(key, VarianceLut::build(nlooks));
    return ins.first->second;
}

// =============================================================================
// cost/lut.rs::CarballoLut + cost/mod.rs::compute_carballo_costs (the SIMPLE,
// analytical-LUT Carballo cost -- NOT the parity spline-table cost used by
// unwrap_linear's actual solve). This is what whirlwind's OWN conncomp_
// algorithm="linear" (components_only/grow_components) actually uses: a
// FRESH cost array, independent of any solve, with a different scale
// (CARBALLO_COST_SCALE=6, not the parity path's implicit 100x) and a wider
// "either-endpoint-invalid" masking rule (not "both-invalid"). Reusing the
// solved parity network's costs here (as an earlier version of this port
// did) is a real bug, confirmed by direct comparison against whirlwind's
// own conncomp_algorithm="linear": the parity cost is far more permissive,
// so reusing it produces drastically more (and wrong) coverage than
// whirlwind's own fresh-cost conncomp actually does.
// =============================================================================
struct CarballoLut {
    static constexpr int N_ALPHA_CARB = 501;
    static constexpr int N_GAMMA_CARB = 101;
    static constexpr float MAX_CARBALLO_COST = 50.0f;
    std::vector<float> values; // [N_GAMMA_CARB x N_ALPHA_CARB], row=gamma, col=alpha

    static CarballoLut build(float nlooks) {
        CarballoLut lut;
        lut.values.assign((size_t)(N_GAMMA_CARB * N_ALPHA_CARB), MAX_CARBALLO_COST);
        const int N_CDF = 2001;
        double dt = 2.0 * M_PI / (double)(N_CDF - 1);
        std::vector<double> cdf((size_t)N_CDF);
        for (int ig = 0; ig < N_GAMMA_CARB; ++ig) {
            float gamma = (float)ig / (float)(N_GAMMA_CARB - 1) * 0.999f;
            cdf[0] = 0.0;
            for (int k = 1; k < N_CDF; ++k) {
                double t0 = -M_PI + (double)(k - 1) * dt;
                double t1 = -M_PI + (double)k * dt;
                double p0 = (double)lee_pdf((float)t0, gamma, nlooks);
                double p1 = (double)lee_pdf((float)t1, gamma, nlooks);
                cdf[(size_t)k] = cdf[(size_t)(k - 1)] + 0.5 * (p0 + p1) * dt;
            }
            double total = std::max(cdf[(size_t)(N_CDF - 1)], 1e-12);
            for (auto &c : cdf) c /= total;
            auto cdf_at = [&](double t) -> double {
                double tc = std::min(std::max(t, -M_PI), M_PI);
                double ki = (tc - (-M_PI)) / dt;
                int k0 = std::min((int)std::floor(ki), N_CDF - 2);
                double f = ki - (double)k0;
                return cdf[(size_t)k0] * (1.0 - f) + cdf[(size_t)(k0 + 1)] * f;
            };
            for (int ia = 0; ia < N_ALPHA_CARB; ++ia) {
                float alpha = -PI_F + (float)ia / (float)(N_ALPHA_CARB - 1) * 2.0f * PI_F;
                float cost;
                if (alpha <= 0.0f) {
                    cost = MAX_CARBALLO_COST;
                } else {
                    double p1 = cdf_at((double)alpha - M_PI);
                    double p0 = 1.0 - p1;
                    if (p1 < 1e-30) {
                        cost = MAX_CARBALLO_COST;
                    } else {
                        cost = std::min(std::max((float)(-std::log(p1 / p0)), 0.0f), MAX_CARBALLO_COST);
                    }
                }
                lut.values[(size_t)(ig * N_ALPHA_CARB + ia)] = cost;
            }
        }
        return lut;
    }

    float eval(float alpha, float gamma) const {
        float a = std::min(std::max(alpha, -PI_F), PI_F);
        float g = std::min(std::max(gamma, 0.0f), 0.999f);
        float ai = (a + PI_F) / (2.0f * PI_F) * (float)(N_ALPHA_CARB - 1);
        float gi = g / 0.999f * (float)(N_GAMMA_CARB - 1);
        int a0 = std::min((int)std::floor(ai), N_ALPHA_CARB - 2);
        int g0 = std::min((int)std::floor(gi), N_GAMMA_CARB - 2);
        float fa = ai - (float)a0, fg = gi - (float)g0;
        float v00 = values[(size_t)(g0 * N_ALPHA_CARB + a0)];
        float v01 = values[(size_t)(g0 * N_ALPHA_CARB + a0 + 1)];
        float v10 = values[(size_t)((g0 + 1) * N_ALPHA_CARB + a0)];
        float v11 = values[(size_t)((g0 + 1) * N_ALPHA_CARB + a0 + 1)];
        float v0 = v00 * (1.0f - fa) + v01 * fa;
        float v1 = v10 * (1.0f - fa) + v11 * fa;
        return v0 * (1.0f - fg) + v1 * fg;
    }
};

static const CarballoLut &get_or_build_carballo(float nlooks) {
    static std::mutex mtx;
    static std::unordered_map<int, CarballoLut> cache;
    int key = (int)std::lround(nlooks * 10.0f);
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto ins = cache.emplace(key, CarballoLut::build(nlooks));
    return ins.first->second;
}

// Fresh, solve-independent Carballo cost array (matches
// cost::compute_carballo_costs exactly): biased (non-mask-aware) smoothing,
// "either-endpoint-invalid" masking -> cost 0, CARBALLO_COST_SCALE=6, no
// SlopeGuard.
static std::vector<int32_t> compute_carballo_costs_analytical(
    const Grid &g, const std::complex<float> *igram, const float *corr,
    const unsigned char *mask, int64_t m_phase, int64_t n_phase, float nlooks,
    int window_par, int window_perp)
{
    const float CARBALLO_COST_SCALE = 6.0f;
    auto is_valid = [&](int64_t i, int64_t j) -> bool {
        return mask == nullptr || mask[i * n_phase + j] != 0;
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
    auto phase_dy_s = box_filter_2d(phase_dy, m_phase - 1, n_phase, window_par, window_perp);
    auto phase_dx_s = box_filter_2d(phase_dx, m_phase, n_phase - 1, window_perp, window_par);

    std::vector<float> cor_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            cor_dy[(size_t)(i * n_phase + j)] = std::min(corr[i * n_phase + j], corr[(i + 1) * n_phase + j]);
    std::vector<float> cor_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j)
            cor_dx[(size_t)(i * (n_phase - 1) + j)] = std::min(corr[i * n_phase + j], corr[i * n_phase + j + 1]);

    const CarballoLut &lut = get_or_build_carballo(nlooks);
    std::vector<int32_t> cost((size_t)g.num_forward, 0);

    // RIGHT/LEFT from vertical (dy) edges: right_arc(i+1,j) / left_arc(i+1,j+1).
    for (int64_t i = 0; i < m_phase - 1; ++i) {
        for (int64_t j = 0; j < n_phase; ++j) {
            size_t idx = (size_t)(i * n_phase + j);
            // "either invalid" (wider than the parity cost's "both invalid").
            bool masked = mask != nullptr && !(is_valid(i, j) && is_valid(i + 1, j));
            float c_rt = 0.0f, c_lt = 0.0f;
            if (!masked) {
                float alpha = phase_dy_s[idx], gamma = cor_dy[idx];
                c_rt = lut.eval(-alpha, gamma);
                c_lt = lut.eval(alpha, gamma);
            }
            int64_t r_arc = g.right_arc(i + 1, j), l_arc = g.left_arc(i + 1, j + 1);
            if (r_arc >= 0) cost[(size_t)r_arc] = (int32_t)std::lround(c_rt * CARBALLO_COST_SCALE);
            if (l_arc >= 0) cost[(size_t)l_arc] = (int32_t)std::lround(c_lt * CARBALLO_COST_SCALE);
        }
    }
    // DOWN/UP from horizontal (dx) edges: down_arc(i,j+1) / up_arc(i+1,j+1).
    for (int64_t i = 0; i < m_phase; ++i) {
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            size_t idx = (size_t)(i * (n_phase - 1) + j);
            bool masked = mask != nullptr && !(is_valid(i, j) && is_valid(i, j + 1));
            float c_dn = 0.0f, c_up = 0.0f;
            if (!masked) {
                float alpha = phase_dx_s[idx], gamma = cor_dx[idx];
                c_dn = lut.eval(alpha, gamma);
                c_up = lut.eval(-alpha, gamma);
            }
            int64_t d_arc = g.down_arc(i, j + 1), u_arc = g.up_arc(i + 1, j + 1);
            if (d_arc >= 0) cost[(size_t)d_arc] = (int32_t)std::lround(c_dn * CARBALLO_COST_SCALE);
            if (u_arc >= 0) cost[(size_t)u_arc] = (int32_t)std::lround(c_up * CARBALLO_COST_SCALE);
        }
    }
    return cost;
}

// conncomp.rs::edge_is_cut / grow_components, now reading a raw, solve-
// independent cost array (see compute_carballo_costs_analytical above)
// instead of a solved Network -- forbidding is redundant with cost=0 for
// any non-negative threshold, since masked arcs already get cost 0 above.
static inline bool edge_is_cut_cost(const std::vector<int32_t> &cost_fwd,
                                     int64_t fwd1, int64_t fwd2, int32_t thresh) {
    int32_t c1 = cost_fwd[(size_t)fwd1], c2 = cost_fwd[(size_t)fwd2];
    return std::min(c1, c2) <= thresh;
}

static void grow_components(const Grid &g, const std::vector<int32_t> &cost_fwd,
                             const unsigned char *mask,
                             int64_t m_phase, int64_t n_phase,
                             int32_t cost_threshold, size_t min_size_px,
                             double min_size_frac, uint32_t max_ncomps,
                             uint32_t *labels_out) {
    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    int64_t n_edge_checks = 0, n_edge_cuts = 0;
    auto valid = [&](int64_t i, int64_t j) -> bool {
        return mask == nullptr || mask[(size_t)(i * n_phase + j)] != 0;
    };
    size_t npix = (size_t)(m_phase * n_phase);
    std::fill(labels_out, labels_out + npix, 0u);

    size_t n_valid = 0;
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            if (valid(i, j)) ++n_valid;
    size_t frac_floor = (size_t)std::ceil(min_size_frac * (double)n_valid);
    size_t min_size = std::max<size_t>({min_size_px, frac_floor, 1});

    std::vector<size_t> sizes;
    sizes.push_back(0); // sizes[0] unused (label 0 = background)
    uint32_t next_label = 0;
    std::deque<std::pair<int64_t, int64_t>> q;

    for (int64_t si = 0; si < m_phase; ++si) {
        for (int64_t sj = 0; sj < n_phase; ++sj) {
            size_t s_idx = (size_t)(si * n_phase + sj);
            if (labels_out[s_idx] != 0 || !valid(si, sj)) continue;
            ++next_label;
            uint32_t label = next_label;
            q.clear();
            q.push_back({si, sj});
            labels_out[s_idx] = label;
            size_t size = 0;
            while (!q.empty()) {
                auto [i, j] = q.front();
                q.pop_front();
                ++size;
                // Right: pixel edge (i,j)-(i,j+1) <-> down(i,j+1) + up(i+1,j+1)
                if (j + 1 < n_phase && labels_out[(size_t)(i * n_phase + j + 1)] == 0 &&
                    valid(i, j + 1)) {
                    int64_t fwd1 = g.down_arc(i, j + 1), fwd2 = g.up_arc(i + 1, j + 1);
                    ++n_edge_checks;
                    bool cut = edge_is_cut_cost(cost_fwd, fwd1, fwd2, cost_threshold);
                    n_edge_cuts += cut;
                    if (!cut) {
                        labels_out[(size_t)(i * n_phase + j + 1)] = label;
                        q.push_back({i, j + 1});
                    }
                }
                // Left: pixel edge (i,j-1)-(i,j) <-> down(i,j) + up(i+1,j)
                if (j >= 1 && labels_out[(size_t)(i * n_phase + j - 1)] == 0 &&
                    valid(i, j - 1)) {
                    int64_t fwd1 = g.down_arc(i, j), fwd2 = g.up_arc(i + 1, j);
                    ++n_edge_checks;
                    bool cut = edge_is_cut_cost(cost_fwd, fwd1, fwd2, cost_threshold);
                    n_edge_cuts += cut;
                    if (!cut) {
                        labels_out[(size_t)(i * n_phase + j - 1)] = label;
                        q.push_back({i, j - 1});
                    }
                }
                // Down: pixel edge (i,j)-(i+1,j) <-> right(i+1,j) + left(i+1,j+1)
                if (i + 1 < m_phase && labels_out[(size_t)((i + 1) * n_phase + j)] == 0 &&
                    valid(i + 1, j)) {
                    int64_t fwd1 = g.right_arc(i + 1, j), fwd2 = g.left_arc(i + 1, j + 1);
                    ++n_edge_checks;
                    bool cut = edge_is_cut_cost(cost_fwd, fwd1, fwd2, cost_threshold);
                    n_edge_cuts += cut;
                    if (!cut) {
                        labels_out[(size_t)((i + 1) * n_phase + j)] = label;
                        q.push_back({i + 1, j});
                    }
                }
                // Up: pixel edge (i-1,j)-(i,j) <-> right(i,j) + left(i,j+1)
                if (i >= 1 && labels_out[(size_t)((i - 1) * n_phase + j)] == 0 &&
                    valid(i - 1, j)) {
                    int64_t fwd1 = g.right_arc(i, j), fwd2 = g.left_arc(i, j + 1);
                    ++n_edge_checks;
                    bool cut = edge_is_cut_cost(cost_fwd, fwd1, fwd2, cost_threshold);
                    n_edge_cuts += cut;
                    if (!cut) {
                        labels_out[(size_t)((i - 1) * n_phase + j)] = label;
                        q.push_back({i - 1, j});
                    }
                }
            }
            sizes.push_back(size);
        }
    }
    if (dbg) {
        std::fprintf(stderr,
            "[grow_components] n_seeds=%u n_edge_checks=%lld n_edge_cuts=%lld (%.4f%%) "
            "threshold=%d\n",
            next_label, (long long)n_edge_checks, (long long)n_edge_cuts,
            n_edge_checks ? 100.0 * n_edge_cuts / n_edge_checks : 0.0, cost_threshold);
    }

    finalize_labels(labels_out, npix, sizes, next_label, min_size, max_ncomps);
}

// =============================================================================
// cost/mod.rs::smooth_phase_gradients_with_mask (mask-aware box filter, sum/
// count-over-valid identity) + compute_snaphu_smooth_costs (SNAPHU convex
// per-arc offset/weight) + conncomp.rs::grow_components_snaphu (ambiguity-
// wiggle reliability test). thicken_cuts is NOT ported (off by default in
// whirlwind's own SnaphuConnCompParams::default()).
// =============================================================================
static std::vector<float> box_filter_2d_masked(const std::vector<float> &a,
                                                 const std::vector<float> &validf,
                                                 int64_t m, int64_t n, int krow, int kcol) {
    // mean_valid = box_filter(a*valid) / box_filter(valid), matches the
    // "two separable box-filter passes then per-pixel divide" identity.
    std::vector<float> av((size_t)(m * n));
    for (int64_t i = 0; i < m * n; ++i) av[(size_t)i] = a[(size_t)i] * validf[(size_t)i];
    auto num = box_filter_2d(av, m, n, krow, kcol);
    auto den = box_filter_2d(validf, m, n, krow, kcol);
    std::vector<float> out((size_t)(m * n));
    for (int64_t i = 0; i < m * n; ++i)
        out[(size_t)i] = den[(size_t)i] > 1e-6f ? num[(size_t)i] / den[(size_t)i] : 0.0f;
    return out;
}

static void compute_snaphu_smooth_costs(
    const Grid &g, const std::complex<float> *igram, const float *corr,
    const unsigned char *mask, int64_t m_phase, int64_t n_phase, float nlooks,
    int window_par, int window_perp,
    std::vector<int32_t> &offsets, std::vector<int32_t> &weights)
{
    auto is_valid = [&](int64_t i, int64_t j) -> bool {
        return mask == nullptr || mask[i * n_phase + j] != 0;
    };
    // raw (unsmoothed) gradients, and the deviation-from-smoothed-mean
    // offset per whirlwind's own doc: offset = raw - mask_aware_smoothed.
    std::vector<float> raw_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j) {
            std::complex<float> z = igram[(i + 1) * n_phase + j] * std::conj(igram[i * n_phase + j]);
            raw_dy[(size_t)(i * n_phase + j)] = std::arg(z);
        }
    std::vector<float> raw_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            std::complex<float> z = igram[i * n_phase + (j + 1)] * std::conj(igram[i * n_phase + j]);
            raw_dx[(size_t)(i * (n_phase - 1) + j)] = std::arg(z);
        }

    std::vector<float> smooth_dy_s, smooth_dx_s;
    if (mask == nullptr) {
        smooth_dy_s = box_filter_2d(raw_dy, m_phase - 1, n_phase, window_par, window_perp);
        smooth_dx_s = box_filter_2d(raw_dx, m_phase, n_phase - 1, window_perp, window_par);
    } else {
        std::vector<float> valid_dy((size_t)((m_phase - 1) * n_phase));
        for (int64_t i = 0; i < m_phase - 1; ++i)
            for (int64_t j = 0; j < n_phase; ++j)
                valid_dy[(size_t)(i * n_phase + j)] = (is_valid(i, j) && is_valid(i + 1, j)) ? 1.0f : 0.0f;
        std::vector<float> valid_dx((size_t)(m_phase * (n_phase - 1)));
        for (int64_t i = 0; i < m_phase; ++i)
            for (int64_t j = 0; j < n_phase - 1; ++j)
                valid_dx[(size_t)(i * (n_phase - 1) + j)] = (is_valid(i, j) && is_valid(i, j + 1)) ? 1.0f : 0.0f;
        smooth_dy_s = box_filter_2d_masked(raw_dy, valid_dy, m_phase - 1, n_phase, window_par, window_perp);
        smooth_dx_s = box_filter_2d_masked(raw_dx, valid_dx, m_phase, n_phase - 1, window_perp, window_par);
    }

    std::vector<float> phase_dy_s(raw_dy.size()), phase_dx_s(raw_dx.size());
    for (size_t i = 0; i < raw_dy.size(); ++i) phase_dy_s[i] = raw_dy[i] - smooth_dy_s[i];
    for (size_t i = 0; i < raw_dx.size(); ++i) phase_dx_s[i] = raw_dx[i] - smooth_dx_s[i];

    std::vector<float> cor_dy((size_t)((m_phase - 1) * n_phase));
    for (int64_t i = 0; i < m_phase - 1; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            cor_dy[(size_t)(i * n_phase + j)] = std::min(corr[i * n_phase + j], corr[(i + 1) * n_phase + j]);
    std::vector<float> cor_dx((size_t)(m_phase * (n_phase - 1)));
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase - 1; ++j)
            cor_dx[(size_t)(i * (n_phase - 1) + j)] = std::min(corr[i * n_phase + j], corr[i * n_phase + j + 1]);

    const int32_t NSHORTCYCLE = 100;
    const float COST_SCALE = 100.0f;
    const VarianceLut &var_lut = get_or_build_variance(nlooks);
    auto alpha_to_offset = [&](float alpha) -> int32_t {
        return (int32_t)std::lround((alpha / (2.0f * PI_F)) * (float)NSHORTCYCLE);
    };
    auto gamma_to_weight = [&](float gamma) -> int32_t {
        float var = std::max(var_lut.eval(gamma), 1e-4f);
        float w = (1.0f / var) * COST_SCALE;
        return (int32_t)std::lround(std::min(w, 1e7f));
    };

    offsets.assign((size_t)g.num_forward, 0);
    weights.assign((size_t)g.num_forward, 0);
    // RIGHT/LEFT from vertical (dy) edges. Same row/col offset convention as
    // the proven compute_carballo_costs: right_arc(i+1,j) / left_arc(i+1,j+1)
    // (the grid-node duality shift, NOT right_arc(i,j)/left_arc(i,j)).
    for (int64_t i = 0; i < m_phase - 1; ++i) {
        for (int64_t j = 0; j < n_phase; ++j) {
            bool masked = mask != nullptr && !(is_valid(i, j) && is_valid(i + 1, j));
            int64_t r_arc = g.right_arc(i + 1, j), l_arc = g.left_arc(i + 1, j + 1);
            size_t idx = (size_t)(i * n_phase + j);
            float alpha = phase_dy_s[idx];
            int32_t w = masked ? 0 : gamma_to_weight(cor_dy[idx]);
            if (r_arc >= 0) { offsets[(size_t)r_arc] = masked ? 0 : alpha_to_offset(-alpha); weights[(size_t)r_arc] = w; }
            if (l_arc >= 0) { offsets[(size_t)l_arc] = masked ? 0 : alpha_to_offset(alpha); weights[(size_t)l_arc] = w; }
        }
    }
    // DOWN/UP from horizontal (dx) edges: down_arc(i,j+1) / up_arc(i+1,j+1).
    for (int64_t i = 0; i < m_phase; ++i) {
        for (int64_t j = 0; j < n_phase - 1; ++j) {
            bool masked = mask != nullptr && !(is_valid(i, j) && is_valid(i, j + 1));
            int64_t d_arc = g.down_arc(i, j + 1), u_arc = g.up_arc(i + 1, j + 1);
            size_t idx = (size_t)(i * (n_phase - 1) + j);
            float alpha = phase_dx_s[idx];
            int32_t w = masked ? 0 : gamma_to_weight(cor_dx[idx]);
            if (d_arc >= 0) { offsets[(size_t)d_arc] = masked ? 0 : alpha_to_offset(alpha); weights[(size_t)d_arc] = w; }
            if (u_arc >= 0) { offsets[(size_t)u_arc] = masked ? 0 : alpha_to_offset(-alpha); weights[(size_t)u_arc] = w; }
        }
    }
}

static void grow_components_snaphu(
    const Grid &g, const std::complex<float> *igram, const float *corr, float nlooks,
    const float *unw, const unsigned char *mask, int64_t m_phase, int64_t n_phase,
    int window_par, int window_perp,
    int64_t reliability_threshold, size_t min_size_px, double min_size_frac,
    uint32_t max_ncomps, bool thicken_cuts, uint32_t *labels_out)
{
    std::vector<int32_t> offsets, weights;
    compute_snaphu_smooth_costs(g, igram, corr, mask, m_phase, n_phase, nlooks,
                                 window_par, window_perp, offsets, weights);
    const int64_t NS = 100;

    auto valid = [&](int64_t i, int64_t j) -> bool {
        bool m = mask == nullptr || mask[(size_t)(i * n_phase + j)] != 0;
        return m && std::isfinite(unw[(size_t)(i * n_phase + j)]);
    };
    auto reliability = [&](int64_t arc, int64_t k) -> int64_t {
        int64_t o = offsets[(size_t)arc], w = weights[(size_t)arc];
        int64_t u = k * NS - o;
        int64_t poscost = w * (NS * NS + 2 * NS * u);
        int64_t negcost = w * (NS * NS - 2 * NS * u);
        return std::min(poscost, negcost);
    };
    auto ambiguity = [&](int64_t ai, int64_t aj, int64_t bi, int64_t bj) -> int64_t {
        float psi_a = std::arg(igram[(size_t)(ai * n_phase + aj)]);
        float psi_b = std::arg(igram[(size_t)(bi * n_phase + bj)]);
        float wrapped_grad = psi_b - psi_a;
        wrapped_grad -= TAU * std::round(wrapped_grad / TAU);
        float unw_grad = unw[(size_t)(bi * n_phase + bj)] - unw[(size_t)(ai * n_phase + aj)];
        return (int64_t)std::lround((unw_grad - wrapped_grad) / TAU);
    };

    bool dbg = std::getenv("CUPHU_WW_DEBUG") != nullptr;
    int64_t n_checked = 0, n_cut = 0;
    double sum_abs_u = 0.0;
    int64_t max_abs_u = 0;

    size_t npix = (size_t)(m_phase * n_phase);
    std::vector<uint8_t> cut_right((size_t)(m_phase * (n_phase - 1)), 1);
    std::vector<uint8_t> cut_down((size_t)((m_phase - 1) * n_phase), 1);

    if (thicken_cuts) {
        // SNAPHU ThickenCosts semantics: per-edge cut STRENGTH
        // max(0, thresh-reliability), laterally smoothed (kernel
        // (2*self+neighbors)/n) so a thin reliable bridge through a wide
        // unreliable band is still cut. Invalid edges carry strength=thresh
        // (matches Rust exactly) and are unconditionally cut besides.
        int64_t invalid_strength = std::max(reliability_threshold, (int64_t)0);
        std::vector<int64_t> s_right((size_t)(m_phase * (n_phase - 1)), invalid_strength);
        std::vector<int64_t> s_down((size_t)((m_phase - 1) * n_phase), invalid_strength);
        for (int64_t i = 0; i < m_phase; ++i)
            for (int64_t j = 0; j < n_phase - 1; ++j)
                if (valid(i, j) && valid(i, j + 1)) {
                    int64_t arc = g.down_arc(i, j + 1);
                    int64_t k = ambiguity(i, j, i, j + 1);
                    s_right[(size_t)(i * (n_phase - 1) + j)] =
                        std::max<int64_t>(reliability_threshold - reliability(arc, k), 0);
                    if (dbg) {
                        int64_t u = k * NS - offsets[(size_t)arc];
                        ++n_checked; sum_abs_u += std::abs((double)u);
                        max_abs_u = std::max(max_abs_u, std::abs(u));
                    }
                }
        for (int64_t i = 0; i < m_phase - 1; ++i)
            for (int64_t j = 0; j < n_phase; ++j)
                if (valid(i, j) && valid(i + 1, j)) {
                    int64_t arc = g.left_arc(i + 1, j + 1);
                    int64_t k = ambiguity(i, j, i + 1, j);
                    s_down[(size_t)(i * n_phase + j)] =
                        std::max<int64_t>(reliability_threshold - reliability(arc, k), 0);
                    if (dbg) {
                        int64_t u = k * NS - offsets[(size_t)arc];
                        ++n_checked; sum_abs_u += std::abs((double)u);
                        max_abs_u = std::max(max_abs_u, std::abs(u));
                    }
                }
        // Smooth RIGHT edges across adjacent rows, DOWN edges across
        // adjacent cols; kernel (2*self + neighbors) / count.
        for (int64_t i = 0; i < m_phase; ++i)
            for (int64_t j = 0; j < n_phase - 1; ++j) {
                int64_t acc = 2 * s_right[(size_t)(i * (n_phase - 1) + j)];
                double cnt = 2.0;
                if (i >= 1) { acc += s_right[(size_t)((i - 1) * (n_phase - 1) + j)]; cnt += 1.0; }
                if (i + 1 < m_phase) { acc += s_right[(size_t)((i + 1) * (n_phase - 1) + j)]; cnt += 1.0; }
                int64_t smoothed = (int64_t)std::llround((double)acc / cnt);
                bool cut = !(valid(i, j) && valid(i, j + 1)) || smoothed > 0;
                cut_right[(size_t)(i * (n_phase - 1) + j)] = cut ? 1 : 0;
                n_cut += cut;
            }
        for (int64_t i = 0; i < m_phase - 1; ++i)
            for (int64_t j = 0; j < n_phase; ++j) {
                int64_t acc = 2 * s_down[(size_t)(i * n_phase + j)];
                double cnt = 2.0;
                if (j >= 1) { acc += s_down[(size_t)(i * n_phase + j - 1)]; cnt += 1.0; }
                if (j + 1 < n_phase) { acc += s_down[(size_t)(i * n_phase + j + 1)]; cnt += 1.0; }
                int64_t smoothed = (int64_t)std::llround((double)acc / cnt);
                bool cut = !(valid(i, j) && valid(i + 1, j)) || smoothed > 0;
                cut_down[(size_t)(i * n_phase + j)] = cut ? 1 : 0;
                n_cut += cut;
            }
    } else {
        for (int64_t i = 0; i < m_phase; ++i)
            for (int64_t j = 0; j < n_phase - 1; ++j)
                if (valid(i, j) && valid(i, j + 1)) {
                    int64_t arc = g.down_arc(i, j + 1);
                    int64_t k = ambiguity(i, j, i, j + 1);
                    bool cut = reliability(arc, k) <= reliability_threshold;
                    cut_right[(size_t)(i * (n_phase - 1) + j)] = cut ? 1 : 0;
                    if (dbg) {
                        int64_t u = k * NS - offsets[(size_t)arc];
                        ++n_checked; n_cut += cut; sum_abs_u += std::abs((double)u);
                        max_abs_u = std::max(max_abs_u, std::abs(u));
                    }
                }
        for (int64_t i = 0; i < m_phase - 1; ++i)
            for (int64_t j = 0; j < n_phase; ++j)
                if (valid(i, j) && valid(i + 1, j)) {
                    int64_t arc = g.left_arc(i + 1, j + 1);
                    int64_t k = ambiguity(i, j, i + 1, j);
                    bool cut = reliability(arc, k) <= reliability_threshold;
                    cut_down[(size_t)(i * n_phase + j)] = cut ? 1 : 0;
                    if (dbg) {
                        int64_t u = k * NS - offsets[(size_t)arc];
                        ++n_checked; n_cut += cut; sum_abs_u += std::abs((double)u);
                        max_abs_u = std::max(max_abs_u, std::abs(u));
                    }
                }
    }
    if (dbg) {
        std::fprintf(stderr,
            "[grow_components_snaphu] n_checked=%lld n_cut=%lld (%.4f%%) "
            "mean|u|=%.3f max|u|=%lld thresh=%lld\n",
            (long long)n_checked, (long long)n_cut,
            n_checked ? 100.0 * n_cut / n_checked : 0.0, sum_abs_u / std::max<int64_t>(n_checked, 1),
            (long long)max_abs_u, (long long)reliability_threshold);
    }

    std::fill(labels_out, labels_out + npix, 0u);
    size_t n_valid = 0;
    for (int64_t i = 0; i < m_phase; ++i)
        for (int64_t j = 0; j < n_phase; ++j)
            if (valid(i, j)) ++n_valid;
    size_t frac_floor = (size_t)std::ceil(min_size_frac * (double)n_valid);
    size_t min_size = std::max<size_t>({min_size_px, frac_floor, 1});

    std::vector<size_t> sizes;
    sizes.push_back(0);
    uint32_t next_label = 0;
    std::deque<std::pair<int64_t, int64_t>> q;
    for (int64_t si = 0; si < m_phase; ++si) {
        for (int64_t sj = 0; sj < n_phase; ++sj) {
            size_t s_idx = (size_t)(si * n_phase + sj);
            if (labels_out[s_idx] != 0 || !valid(si, sj)) continue;
            ++next_label;
            uint32_t label = next_label;
            q.clear();
            q.push_back({si, sj});
            labels_out[s_idx] = label;
            size_t size = 0;
            while (!q.empty()) {
                auto [i, j] = q.front();
                q.pop_front();
                ++size;
                if (j + 1 < n_phase && labels_out[(size_t)(i * n_phase + j + 1)] == 0 &&
                    !cut_right[(size_t)(i * (n_phase - 1) + j)]) {
                    labels_out[(size_t)(i * n_phase + j + 1)] = label;
                    q.push_back({i, j + 1});
                }
                if (j >= 1 && labels_out[(size_t)(i * n_phase + j - 1)] == 0 &&
                    !cut_right[(size_t)(i * (n_phase - 1) + j - 1)]) {
                    labels_out[(size_t)(i * n_phase + j - 1)] = label;
                    q.push_back({i, j - 1});
                }
                if (i + 1 < m_phase && labels_out[(size_t)((i + 1) * n_phase + j)] == 0 &&
                    !cut_down[(size_t)(i * n_phase + j)]) {
                    labels_out[(size_t)((i + 1) * n_phase + j)] = label;
                    q.push_back({i + 1, j});
                }
                if (i >= 1 && labels_out[(size_t)((i - 1) * n_phase + j)] == 0 &&
                    !cut_down[(size_t)((i - 1) * n_phase + j)]) {
                    labels_out[(size_t)((i - 1) * n_phase + j)] = label;
                    q.push_back({i - 1, j});
                }
            }
            sizes.push_back(size);
        }
    }
    finalize_labels(labels_out, npix, sizes, next_label, min_size, max_ncomps);
}

} // namespace cuphu_ww

int cuphu_whirlwind_unwrap(
    const float *igram_r, const float *igram_i, const float *corr,
    const unsigned char *mask, int nrow, int ncol, double nlooks, int gpu_id,
    float *unw_out, int conncomp_algorithm, long reliability_threshold_in,
    int thicken_cuts_in, unsigned int *conncomp_out)
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
    // Residues are computed UNMASKED (matches unwrap_linear_impl's
    // `residue::compute(wrapped_phase.view())` -- no mask argument at all,
    // NOT `compute_with_mask`). Masking only affects costs (arcs get cost=0
    // through masked regions) and the final NaN-out step; residues ignore
    // mask entirely. Passing mask here (as an earlier version of this port
    // did) zeros out residues/boundary charges at every mask-touching cell,
    // producing a substantially different (and wrong) residue grid --
    // confirmed by direct comparison against whirlwind's own
    // WHIRLWIND_DEBUG trace, which reports a different total residue
    // count than the masked version of this code did.
    auto residues = compute_residues(wrapped.data(), nullptr, m_phase, n_phase);
    tock("residues");
    auto costs = compute_carballo_costs(g, igram.data(), corr, mask, m_phase, n_phase,
                                         (float)nlooks, /*window_par=*/7, /*window_perp=*/7, gpu_id);
    tock("costs");
    if (dbg) {
        std::vector<int32_t> sorted_costs(costs);
        std::sort(sorted_costs.begin(), sorted_costs.end());
        size_t nc = sorted_costs.size();
        auto pct = [&](double p) { return sorted_costs[(size_t)(p * (nc - 1))]; };
        std::fprintf(stderr,
            "[cost stats] n=%zu min=%d p1=%d p5=%d p25=%d p50=%d p75=%d p95=%d max=%d "
            "frac<=50: %.4f%%\n",
            nc, sorted_costs[0], pct(0.01), pct(0.05), pct(0.25), pct(0.50), pct(0.75),
            pct(0.95), sorted_costs[nc - 1],
            100.0 * std::count_if(costs.begin(), costs.end(),
                                   [](int32_t c) { return c <= 50; }) / (double)nc);
    }

    Network net(g, std::move(residues), costs);
    tock("network");
    solve(g, net);
    tock("solve");
    integrate(g, net, wrapped.data(), m_phase, n_phase, unw_out);
    tock("integrate");

    // Conncomp: whirlwind offers two algorithms, chosen via
    // conncomp_algorithm (0="linear", 1="snaphu" -- whirlwind-insar's own
    // default; see grow_components_snaphu()). Env vars remain as a debug-
    // only override on top of whatever the caller passed in.
    size_t min_size_px = 100;
    double min_size_frac = 0.0001;
    uint32_t max_ncomps = 1024;
    if (const char *e = std::getenv("CUPHU_WW_CC_MIN_SIZE_PX")) min_size_px = (size_t)std::atoll(e);
    if (const char *e = std::getenv("CUPHU_WW_CC_MAX_NCOMPS")) max_ncomps = (uint32_t)std::atoi(e);

    const char *cc_algo_env = std::getenv("CUPHU_WW_CONNCOMP");
    bool use_snaphu = cc_algo_env ? (std::strcmp(cc_algo_env, "snaphu") == 0)
                                   : (conncomp_algorithm != 0);
    if (use_snaphu) {
        // Defaults (500000, true) match ww.unwrap()'s own Python-wrapper
        // defaults (NOT the bare SnaphuConnCompParams::default() in Rust,
        // which the wrapper overrides): conncomp_reliability=0.5 ->
        // reliability_raw = round(0.5 * CONNCOMP_RELIABILITY_UNIT) with
        // CONNCOMP_RELIABILITY_UNIT = COST_SCALE * nshortcycle^2 = 1e6;
        // conncomp_thicken=True.
        int64_t reliability_threshold = reliability_threshold_in;
        bool thicken_cuts = thicken_cuts_in != 0;
        if (const char *e = std::getenv("CUPHU_WW_CC_RELIABILITY_THRESH"))
            reliability_threshold = std::atoll(e);
        if (const char *e = std::getenv("CUPHU_WW_CC_THICKEN"))
            thicken_cuts = std::atoi(e) != 0;
        grow_components_snaphu(g, igram.data(), corr, (float)nlooks, unw_out, mask,
                                m_phase, n_phase, /*window_par=*/7, /*window_perp=*/7,
                                reliability_threshold, min_size_px, min_size_frac,
                                max_ncomps, thicken_cuts, conncomp_out);
    } else {
        int32_t cost_threshold = 50;
        if (const char *e = std::getenv("CUPHU_WW_CC_COST_THRESH")) cost_threshold = std::atoi(e);
        // Fresh, solve-independent cost array (matches components_only's own
        // network construction) -- NOT net.cost_fwd from the parity-cost
        // solve, which is a different cost model at a different scale.
        auto analytical_cost = compute_carballo_costs_analytical(
            g, igram.data(), corr, mask, m_phase, n_phase, (float)nlooks,
            /*window_par=*/7, /*window_perp=*/7);
        grow_components(g, analytical_cost, mask, m_phase, n_phase, cost_threshold, min_size_px,
                         min_size_frac, max_ncomps, conncomp_out);
    }
    tock("conncomp");
    return 0;
}
