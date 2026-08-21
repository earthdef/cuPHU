/**
 * cuphu_py.cu
 *
 * pybind11 Python extension module: _cuphu_ext
 *
 * Exposes:
 *   _cuphu_ext.unwrap_arrays(igram, corr, nlooks, ...)
 *       → (unw, conncomp) as numpy arrays
 *
 *   _cuphu_ext.build_costs(igram, corr, nlooks, cost_mode, ...)
 *       → costs as numpy structured array (smoothcostT or costT)
 *
 *   _cuphu_ext.gpu_count()
 *       → int  number of CUDA devices available
 *
 *   _cuphu_ext.gpu_name(device_id)
 *       → str  GPU device name
 */

#include "cuphu.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>

namespace py = pybind11;
using namespace pybind11::literals;

/* bring in SNAPHU structures for dtype exposure */
extern "C" {
#include "snaphu.h"
}

/* forward declaration */
extern "C" void cuphu_default_params(CuPhuParams *);
extern "C" void cuphu_default_tile_params(CuPhuTileParams *);
extern "C" void cuphu_default_bridge_params(CuPhuBridgeParams *);

/* ── helpers ─────────────────────────────────────────────────────────────── */

static CuPhuCostMode parse_cost_mode(const std::string &s) {
    if (s == "smooth") return CUPHU_COST_SMOOTH;
    if (s == "defo")   return CUPHU_COST_DEFO;
    if (s == "topo")   return CUPHU_COST_TOPO;
    throw std::invalid_argument("cost must be 'smooth', 'defo', or 'topo'");
}

static CuPhuInitMethod parse_init_method(const std::string &s) {
    if (s == "mst")    return CUPHU_INIT_MST;
    if (s == "mcf")    return CUPHU_INIT_MCF;
    if (s == "laplace") return CUPHU_INIT_LAPLACE;
    throw std::invalid_argument("init must be 'mst', 'mcf', or 'laplace'");
}

static CuPhuRampType parse_ramp_type(const py::object &obj) {
    if (obj.is_none()) return CUPHU_RAMP_NONE;
    std::string s = obj.cast<std::string>();
    if (s == "linear")             return CUPHU_RAMP_LINEAR;
    if (s == "quadratic")          return CUPHU_RAMP_QUADRATIC;
    if (s == "linear_range")       return CUPHU_RAMP_LINEAR_RANGE;
    if (s == "linear_azimuth")     return CUPHU_RAMP_LINEAR_AZIMUTH;
    if (s == "quadratic_range")    return CUPHU_RAMP_QUADRATIC_RANGE;
    if (s == "quadratic_azimuth")  return CUPHU_RAMP_QUADRATIC_AZIMUTH;
    throw std::invalid_argument("bridge_ramp_type must be one of: linear, "
        "quadratic, linear_range, linear_azimuth, quadratic_range, "
        "quadratic_azimuth, or None");
}

/** Validate that arr is a 2-D C-contiguous array of the given dtype. */
template<typename Scalar>
static void check_array_2d(const py::array_t<Scalar, py::array::c_style> &arr,
                            const char *name) {
    if (arr.ndim() != 2)
        throw std::runtime_error(std::string(name) + " must be 2-D");
    if (!arr.flags() & py::array::c_style)
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
}

/** Fill CuPhuParams from kwargs, starting from defaults. */
static CuPhuParams make_params(
    double nlooks,
    double costscale,
    int    nshortcycle,
    int    kperpdpsi,
    int    kpardpsi,
    double defomax,
    double min_conncomp_frac,
    long   max_ncomps,
    long   conncompthresh
) {
    CuPhuParams p;
    cuphu_default_params(&p);
    p.nlooks         = nlooks;
    p.ncorrlooks     = nlooks;
    p.costscale      = costscale;
    p.nshortcycle    = nshortcycle;
    p.kperpdpsi      = kperpdpsi;
    p.kpardpsi       = kpardpsi;
    p.defomax        = defomax;
    p.minconncompfrac= min_conncomp_frac;
    p.maxncomps      = max_ncomps;
    p.conncompthresh = conncompthresh;
    return p;
}

/* ── unwrap_arrays ───────────────────────────────────────────────────────── */

py::tuple py_unwrap_arrays(
    py::array_t<std::complex<float>, py::array::c_style> igram,
    py::array_t<float,               py::array::c_style> corr,
    double  nlooks,
    std::string cost        = "smooth",
    std::string init        = "mcf",
    py::object  mask_obj    = py::none(),
    py::object  mag_obj     = py::none(),
    py::object  orig_mask_obj = py::none(),
    double  costscale       = DEF_COSTSCALE,
    int     nshortcycle     = DEF_NSHORTCYCLE,
    int     kperpdpsi       = DEF_KPERPDPSI,
    int     kpardpsi        = DEF_KPARDPSI,
    double  defomax         = DEF_DEFOMAX,
    double  min_conncomp_frac = DEF_MINCONNCOMPFRAC,
    long    max_ncomps      = DEF_MAXNCOMPS,
    long    conncompthresh  = DEF_CONNCOMPTHRESH,
    int     ntilerow        = 1,
    int     ntilecol        = 1,
    int     tile_rowovrlp   = 0,
    int     tile_colovrlp   = 0,
    int     tilecostthresh  = DEF_TILECOSTTHRESH,
    int     minregionsize   = DEF_MINREGIONSIZE,
    int     nproc           = 1,
    bool    single_tile_reoptimize = false,
    bool    laplace_neighbor_feedback = false,
    int     laplace_neighbor_feedback_feather = 200,
    bool    fix_cycle_spikes = false,
    bool    bridge                  = false,
    int     bridge_radius           = 500,
    int     bridge_min_num_pixel    = 14,
    int     bridge_erosion_size     = 2,
    int     bridge_max_boundary_samples = 4096,
    py::object bridge_ramp_type_obj = py::none(),
    int     bridge_ramp_max_num_sample = 1000000,
    int     gpu_id          = 0
) {
    if (igram.ndim() != 2)
        throw std::runtime_error("igram must be 2-D");
    check_array_2d(corr, "corr");

    int nrow = (int)igram.shape(0);
    int ncol = (int)igram.shape(1);

    if (corr.shape(0) != nrow || corr.shape(1) != ncol)
        throw std::runtime_error("corr must have the same shape as igram");

    if (nlooks < 1.0)
        throw std::runtime_error("nlooks must be >= 1");

    /* Split igram into real/imag flat arrays */
    auto ig_buf = igram.unchecked<2>();
    std::vector<float> igr(nrow * ncol), igi(nrow * ncol);
    for (int r = 0; r < nrow; ++r)
        for (int c = 0; c < ncol; ++c) {
            auto v = ig_buf(r, c);
            igr[(size_t)r * ncol + c] = v.real();
            igi[(size_t)r * ncol + c] = v.imag();
        }

    /* Optional mag */
    const float *mag_ptr = nullptr;
    py::array_t<float, py::array::c_style> mag_arr;
    if (!mag_obj.is_none()) {
        mag_arr = mag_obj.cast<py::array_t<float, py::array::c_style>>();
        check_array_2d(mag_arr, "mag");
        mag_ptr = mag_arr.data();
    }

    /* Optional mask */
    const unsigned char *mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        check_array_2d(mask_arr, "mask");
        mask_ptr = mask_arr.data();
    }

    /* Optional orig_mask -- see cuphu_unwrap()'s docstring. Falls back to
     * mask_ptr (== nullptr if orig_mask is None too) when not given. */
    const unsigned char *orig_mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style> orig_mask_arr;
    if (!orig_mask_obj.is_none()) {
        orig_mask_arr = orig_mask_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        check_array_2d(orig_mask_arr, "orig_mask");
        orig_mask_ptr = orig_mask_arr.data();
    }

    CuPhuParams params = make_params(
        nlooks, costscale, nshortcycle, kperpdpsi, kpardpsi,
        defomax, min_conncomp_frac, max_ncomps, conncompthresh);

    CuPhuTileParams tile;
    cuphu_default_tile_params(&tile);
    tile.ntilerow       = ntilerow;
    tile.ntilecol       = ntilecol;
    tile.rowovrlp       = tile_rowovrlp;
    tile.colovrlp       = tile_colovrlp;
    tile.tilecostthresh = tilecostthresh;
    tile.minregionsize  = minregionsize;
    tile.nproc          = nproc;
    tile.single_tile_reoptimize = single_tile_reoptimize ? 1 : 0;
    tile.laplace_neighbor_feedback = laplace_neighbor_feedback ? 1 : 0;
    tile.laplace_neighbor_feedback_feather = laplace_neighbor_feedback_feather;
    tile.fix_cycle_spikes = fix_cycle_spikes ? 1 : 0;

    CuPhuBridgeParams bp;
    cuphu_default_bridge_params(&bp);
    bp.enabled              = bridge ? 1 : 0;
    bp.radius               = bridge_radius;
    bp.min_num_pixel        = bridge_min_num_pixel;
    bp.erosion_size         = bridge_erosion_size;
    bp.max_boundary_samples = bridge_max_boundary_samples;
    bp.ramp_type            = parse_ramp_type(bridge_ramp_type_obj);
    bp.ramp_max_num_sample  = bridge_ramp_max_num_sample;

    CuPhuResult result = {};
    int rc = cuphu_unwrap(
        igr.data(), igi.data(),
        corr.data(), mag_ptr, mask_ptr,
        nrow, ncol,
        parse_cost_mode(cost),
        parse_init_method(init),
        &params, &tile, &bp, orig_mask_ptr, gpu_id, &result);

    if (rc != 0)
        throw std::runtime_error("cuphu_unwrap failed with code " + std::to_string(rc));

    /* wrap outputs as numpy arrays (take ownership of the malloc'd buffers) */
    py::capsule free_unw(result.unw, [](void *p){ std::free(p); });
    py::capsule free_cc (result.conncomp, [](void *p){ std::free(p); });

    auto unw_arr = py::array_t<float>(
        {(py::ssize_t)nrow, (py::ssize_t)ncol},
        {(py::ssize_t)(ncol * sizeof(float)), (py::ssize_t)sizeof(float)},
        result.unw, free_unw);

    auto cc_arr = py::array_t<uint32_t>(
        {(py::ssize_t)nrow, (py::ssize_t)ncol},
        {(py::ssize_t)(ncol * sizeof(uint32_t)), (py::ssize_t)sizeof(uint32_t)},
        result.conncomp, free_cc);

    return py::make_tuple(unw_arr, cc_arr);
}

/* ── build_costs ─────────────────────────────────────────────────────────── */

py::array py_build_costs(
    py::array_t<std::complex<float>, py::array::c_style> igram,
    py::array_t<float,               py::array::c_style> corr,
    double      nlooks,
    std::string cost        = "smooth",
    double      costscale   = DEF_COSTSCALE,
    int         nshortcycle = DEF_NSHORTCYCLE,
    int         kperpdpsi   = DEF_KPERPDPSI,
    int         kpardpsi    = DEF_KPARDPSI,
    double      defomax     = DEF_DEFOMAX,
    int         gpu_id      = 0
) {
    if (igram.ndim() != 2)
        throw std::runtime_error("igram must be 2-D");

    int nrow = (int)igram.shape(0);
    int ncol = (int)igram.shape(1);

    auto ig_buf = igram.unchecked<2>();
    std::vector<float> igr(nrow * ncol), igi(nrow * ncol);
    for (int r = 0; r < nrow; ++r)
        for (int c = 0; c < ncol; ++c) {
            auto v = ig_buf(r, c);
            igr[(size_t)r * ncol + c] = v.real();
            igi[(size_t)r * ncol + c] = v.imag();
        }

    CuPhuParams params = make_params(
        nlooks, costscale, nshortcycle, kperpdpsi, kpardpsi,
        defomax, DEF_MINCONNCOMPFRAC, DEF_MAXNCOMPS, DEF_CONNCOMPTHRESH);

    void   *costs_ptr  = nullptr;
    size_t  elem_sz    = 0;

    cuphu_build_costs_gpu(
        igr.data(), igi.data(), corr.data(), nullptr, nullptr,
        nrow, ncol,
        parse_cost_mode(cost),
        &params, gpu_id,
        &costs_ptr, &elem_sz);

    if (!costs_ptr)
        throw std::runtime_error("cuphu_build_costs_gpu returned null");

    size_t nrowcost = (size_t)(nrow - 1) * ncol;
    size_t ncolcost = (size_t)nrow * (ncol - 1);
    size_t nbytes   = (nrowcost + ncolcost) * elem_sz;

    /* Return as a flat uint8 byte array; Python side can reinterpret */
    py::capsule free_costs(costs_ptr, [](void *p){ std::free(p); });
    return py::array_t<uint8_t>(
        {(py::ssize_t)nbytes},
        {(py::ssize_t)sizeof(uint8_t)},
        (uint8_t *)costs_ptr, free_costs);
}

/* ── bridge test hook (internal) ────────────────────────────────────────── */

/* Test-only binding: runs the exact phase-bridging orchestration
 * cuphu_unwrap() calls internally on a caller-supplied unw array directly
 * (skipping the igram solve), so pytest can exercise the full wired GPU
 * pipeline against synthetic ground-truth fixtures without needing a real
 * interferogram to induce a genuine MCF ambiguity. */
py::tuple py_bridge_apply_test(
    py::array_t<float, py::array::c_style> unw,
    py::object mask_obj = py::none(),
    int    bridge_radius        = 500,
    int    bridge_min_num_pixel = 14,
    int    bridge_erosion_size  = 2,
    int    bridge_max_boundary_samples = 4096,
    py::object bridge_ramp_type_obj = py::none(),
    int    bridge_ramp_max_num_sample = 1000000,
    int    gpu_id = 0
) {
    check_array_2d(unw, "unw");
    int nrow = (int)unw.shape(0);
    int ncol = (int)unw.shape(1);

    const unsigned char *mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        check_array_2d(mask_arr, "mask");
        mask_ptr = mask_arr.data();
    }

    std::vector<float> unw_buf(unw.data(), unw.data() + (size_t)nrow * ncol);
    std::vector<uint32_t> cc_buf((size_t)nrow * ncol, 0u);

    CuPhuBridgeParams bp;
    cuphu_default_bridge_params(&bp);
    bp.enabled              = 1;
    bp.radius               = bridge_radius;
    bp.min_num_pixel        = bridge_min_num_pixel;
    bp.erosion_size         = bridge_erosion_size;
    bp.max_boundary_samples = bridge_max_boundary_samples;
    bp.ramp_type            = parse_ramp_type(bridge_ramp_type_obj);
    bp.ramp_max_num_sample  = bridge_ramp_max_num_sample;

    cuphu_bridge_apply_test(unw_buf.data(), cc_buf.data(), nrow, ncol, &bp, mask_ptr, gpu_id);

    auto unw_out = py::array_t<float>({(py::ssize_t)nrow, (py::ssize_t)ncol});
    std::memcpy(unw_out.mutable_data(), unw_buf.data(), unw_buf.size() * sizeof(float));
    auto cc_out = py::array_t<uint32_t>({(py::ssize_t)nrow, (py::ssize_t)ncol});
    std::memcpy(cc_out.mutable_data(), cc_buf.data(), cc_buf.size() * sizeof(uint32_t));

    return py::make_tuple(unw_out, cc_out);
}

/* Test-only binding: runs apply_laplace_neighbor_feedback() directly on a
 * caller-supplied, already-stitched unw array. Lets pytest exercise the
 * exact boundary-refinement correction cuphu_unwrap() applies internally,
 * against real (or synthetic) already-tiled results without a full solve. */
py::array_t<float> py_laplace_neighbor_feedback_test(
    py::array_t<float, py::array::c_style> unw,
    py::object mask_obj,
    int ntilerow,
    int ntilecol,
    int row_ovrlp,
    int col_ovrlp,
    int feather_px
) {
    check_array_2d(unw, "unw");
    int nrow = (int)unw.shape(0);
    int ncol = (int)unw.shape(1);

    const unsigned char *mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        check_array_2d(mask_arr, "mask");
        mask_ptr = mask_arr.data();
    }

    std::vector<float> unw_buf(unw.data(), unw.data() + (size_t)nrow * ncol);

    cuphu_laplace_neighbor_feedback_test(
        unw_buf.data(), mask_ptr, nrow, ncol,
        ntilerow, ntilecol, row_ovrlp, col_ovrlp, feather_px);

    auto unw_out = py::array_t<float>({(py::ssize_t)nrow, (py::ssize_t)ncol});
    std::memcpy(unw_out.mutable_data(), unw_buf.data(), unw_buf.size() * sizeof(float));
    return unw_out;
}

/* Test-only binding: runs cuphu_fix_cycle_spikes_test() directly on a
 * caller-supplied unw array. Lets pytest exercise the exact isolated
 * row/column spike correction cuphu_unwrap() applies internally. */
py::array_t<float> py_fix_cycle_spikes_test(
    py::array_t<float, py::array::c_style> unw,
    py::object mask_obj
) {
    check_array_2d(unw, "unw");
    int nrow = (int)unw.shape(0);
    int ncol = (int)unw.shape(1);

    const unsigned char *mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        check_array_2d(mask_arr, "mask");
        mask_ptr = mask_arr.data();
    }

    std::vector<float> unw_buf(unw.data(), unw.data() + (size_t)nrow * ncol);

    cuphu_fix_cycle_spikes_test(unw_buf.data(), mask_ptr, nrow, ncol);

    auto unw_out = py::array_t<float>({(py::ssize_t)nrow, (py::ssize_t)ncol});
    std::memcpy(unw_out.mutable_data(), unw_buf.data(), unw_buf.size() * sizeof(float));
    return unw_out;
}

/* ── device query helpers ─────────────────────────────────────────────────── */

static int py_gpu_count() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

static std::string py_gpu_name(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return std::string(prop.name);
}

static py::dict py_gpu_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    py::dict d;
    d["name"]          = std::string(prop.name);
    d["total_memory"]  = (long long)prop.totalGlobalMem;
    d["sm_count"]      = prop.multiProcessorCount;
    d["compute_cap"]   = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    return d;
}

/* ── module definition ───────────────────────────────────────────────────── */

PYBIND11_MODULE(_cuphu_ext, m) {
    m.doc() = "GPU-accelerated SNAPHU phase unwrapping (CUDA extension)";

    m.def("unwrap_arrays", &py_unwrap_arrays,
        "igram"_a, "corr"_a, "nlooks"_a,
        "cost"_a         = "smooth",
        "init"_a         = "mcf",
        "mask"_a         = py::none(),
        "mag"_a          = py::none(),
        "orig_mask"_a    = py::none(),
        "costscale"_a    = DEF_COSTSCALE,
        "nshortcycle"_a  = DEF_NSHORTCYCLE,
        "kperpdpsi"_a    = DEF_KPERPDPSI,
        "kpardpsi"_a     = DEF_KPARDPSI,
        "defomax"_a      = DEF_DEFOMAX,
        "min_conncomp_frac"_a = DEF_MINCONNCOMPFRAC,
        "max_ncomps"_a   = (long)DEF_MAXNCOMPS,
        "conncompthresh"_a    = (long)DEF_CONNCOMPTHRESH,
        "ntilerow"_a     = 1,
        "ntilecol"_a     = 1,
        "tile_rowovrlp"_a= 0,
        "tile_colovrlp"_a= 0,
        "tilecostthresh"_a    = DEF_TILECOSTTHRESH,
        "minregionsize"_a     = DEF_MINREGIONSIZE,
        "nproc"_a        = 1,
        "single_tile_reoptimize"_a = false,
        "laplace_neighbor_feedback"_a = false,
        "laplace_neighbor_feedback_feather"_a = 200,
        "fix_cycle_spikes"_a = false,
        "bridge"_a                  = false,
        "bridge_radius"_a           = 500,
        "bridge_min_num_pixel"_a    = 14,
        "bridge_erosion_size"_a     = 2,
        "bridge_max_boundary_samples"_a = 4096,
        "bridge_ramp_type"_a = py::none(),
        "bridge_ramp_max_num_sample"_a = 1000000,
        "gpu_id"_a       = 0,
        R"(
Unwrap an interferogram using GPU-accelerated SNAPHU.

Parameters
----------
igram : ndarray, complex64, 2-D
    Complex interferogram.
corr : ndarray, float32, 2-D
    Coherence magnitude in [0, 1].
nlooks : float
    Equivalent number of independent looks.
cost : {'smooth', 'defo', 'topo'}, optional
    Statistical cost mode.
init : {'mcf', 'mst', 'laplace'}, optional
    Initialization algorithm. 'laplace' uses a GPU Conjugate Gradient
    solver on the weighted Laplacian (smooth mode only); skips MCF/MST
    and TreeSolve entirely, typically 50-200x faster.
mask : ndarray, uint8, 2-D, optional
    Valid-pixel mask (0 = invalid).
mag : ndarray, float32, 2-D, optional
    Interferogram magnitude. Derived from igram if None.
orig_mask : ndarray, uint8, 2-D, optional
    The *real*, un-padded validity mask, distinct from `mask` (which the
    caller may have grown, e.g. to give the PCG solve connectivity through
    isolated features). Tile-stitching's overlap-median only trusts pixels
    valid under orig_mask, so a tile's PCG solve getting corrupted through
    a thin padded-through connection can't silently propagate into that
    tile's whole-tile stitching offset. Falls back to `mask` when None --
    no behavior change for callers not padding the mask.
single_tile_reoptimize : bool, optional
    After tiled unwrapping and stitching, rerun a full CPU network-flow
    solve over the whole assembled scene as one tile to clean up tile-
    boundary artifacts (matches snaphu-py's parameter of the same name).
    No effect when ntilerow*ntilecol == 1. Off by default: this is a
    full-scene CPU solve with the same cost profile as snaphu-py's
    single_tile_reoptimize, which can dominate wall time on large scenes.
laplace_neighbor_feedback : bool, optional
    Laplace only (init='laplace'). Refines each internal tile boundary with
    a smoothly-varying (per-row for column boundaries, per-column for row
    boundaries) residual correction on top of the whole-tile bulk offset,
    feathered into the tile interior over laplace_neighbor_feedback_feather
    px. Targets a failure mode the whole-tile constant offset can't reach:
    independently-solved Laplace tiles can show a position-varying (not
    just tile-constant) mismatch along their shared boundary in
    marginal-coherence areas. Off by default. No effect for MCF/MST (their
    whole-tile offset is already exact) or when ntilerow*ntilecol == 1.
laplace_neighbor_feedback_feather : int, optional
    Pixels over which the boundary correction above decays to zero moving
    away from the boundary. Only used when laplace_neighbor_feedback=True.
fix_cycle_spikes : bool, optional
    Detect and correct isolated single row/column whole-2*pi-cycle spikes:
    a row (or column) whose median offset from both immediate neighbors is
    the same nonzero integer multiple of 2*pi, while those neighbors agree
    with each other -- the signature of a degenerate network-flow
    (MCF/MST/reoptimize) solution with a spurious, self-cancelling flow
    loop through one row/column. Off by default.
bridge : bool, optional
    Reconcile whole-2*pi-cycle offsets between disconnected regions of
    unwrapped phase (e.g. regions split apart by a mask) -- a native
    GPU/C++ port of isce3's bridge_unwrapped_phase(). Off by default. v1
    has no ramp/deramp support (isce3's ramp_type); NISAR's own production
    default is already ramp_type=None, so this covers the mode actually
    used in production. Parameter names/defaults mirror NISAR's production
    bridge_* runconfig keys.
bridge_radius : int, optional
    AOI half-size (px) for the per-bridge median phase comparison.
bridge_min_num_pixel : int, optional
    Regions smaller than this are dropped before bridging.
bridge_erosion_size : int, optional
    Structuring-element size (px) for erosion-based thin/small-region
    pruning (two-stage: square SE, then circular SE).
bridge_max_boundary_samples : int, optional
    Cap on boundary points sampled per region for the nearest-neighbor
    region-pair search, bounding cost regardless of any single region's
    true perimeter.
bridge_ramp_type : str or None, optional
    Ramp to fit over the reference region and remove before bridging, add
    back after: one of 'linear', 'quadratic', 'linear_range',
    'linear_azimuth', 'quadratic_range', 'quadratic_azimuth', or None (no
    ramp, default). Matches isce3's bridge_phase.py deramp().
bridge_ramp_max_num_sample : int, optional
    Uniform grid-stride subsample cap for the ramp fit. Default 1e6.
gpu_id : int, optional
    CUDA device index (default 0).

Returns
-------
unw : ndarray, float32, 2-D
    Unwrapped phase in radians.
conncomp : ndarray, uint32, 2-D
    Connected-component labels (0 = no component).
)");

    m.def("build_costs", &py_build_costs,
        "igram"_a, "corr"_a, "nlooks"_a,
        "cost"_a        = "smooth",
        "costscale"_a   = DEF_COSTSCALE,
        "nshortcycle"_a = DEF_NSHORTCYCLE,
        "kperpdpsi"_a   = DEF_KPERPDPSI,
        "kpardpsi"_a    = DEF_KPARDPSI,
        "defomax"_a     = DEF_DEFOMAX,
        "gpu_id"_a      = 0,
        "Build GPU cost arrays and return as a raw byte buffer.");

    m.def("gpu_count", &py_gpu_count,
        "Return the number of CUDA-capable GPUs available.");

    m.def("gpu_name", &py_gpu_name,
        "device_id"_a = 0,
        "Return the name of the CUDA device.");

    m.def("_bridge_apply_test", &py_bridge_apply_test,
        "unw"_a, "mask"_a = py::none(),
        "bridge_radius"_a = 500,
        "bridge_min_num_pixel"_a = 14,
        "bridge_erosion_size"_a = 2,
        "bridge_max_boundary_samples"_a = 4096,
        "bridge_ramp_type"_a = py::none(),
        "bridge_ramp_max_num_sample"_a = 1000000,
        "gpu_id"_a = 0,
        "Test-only: run phase bridging directly on a caller-supplied unw "
        "array (skips the igram solve). Not part of the public API.");

    m.def("_laplace_neighbor_feedback_test", &py_laplace_neighbor_feedback_test,
        "unw"_a, "mask"_a = py::none(),
        "ntilerow"_a, "ntilecol"_a,
        "row_ovrlp"_a, "col_ovrlp"_a,
        "feather_px"_a = 200,
        "Test-only: run the Laplace neighbor-feedback boundary refinement "
        "directly on a caller-supplied, already-stitched unw array. Not "
        "part of the public API.");

    m.def("_fix_cycle_spikes_test", &py_fix_cycle_spikes_test,
        "unw"_a, "mask"_a = py::none(),
        "Test-only: run the isolated row/column whole-cycle spike "
        "correction directly on a caller-supplied unw array. Not part of "
        "the public API.");

    m.def("gpu_info", &py_gpu_info,
        "device_id"_a = 0,
        "Return a dict of device properties.");
}
