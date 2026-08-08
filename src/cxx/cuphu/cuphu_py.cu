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

    CuPhuResult result = {};
    int rc = cuphu_unwrap(
        igr.data(), igi.data(),
        corr.data(), mag_ptr, mask_ptr,
        nrow, ncol,
        parse_cost_mode(cost),
        parse_init_method(init),
        &params, &tile, gpu_id, &result);

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

    m.def("gpu_info", &py_gpu_info,
        "device_id"_a = 0,
        "Return a dict of device properties.");
}
