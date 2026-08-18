/**
 * cuPHU: GPU-accelerated SNAPHU phase unwrapping
 *
 * Public C/C++ API.  GPU data transfers, cost computation, phase integration,
 * and connected-component labeling all run on the device.  The minimum-cost
 * network-flow solver (SNAPHU's TreeSolve) runs on the CPU because it is
 * an inherently sequential algorithm.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── cost mode ────────────────────────────────────────────────────────────── */
typedef enum {
    CUPHU_COST_SMOOTH = 0,
    CUPHU_COST_DEFO   = 1,
    CUPHU_COST_TOPO   = 2,
} CuPhuCostMode;

/* ── init method ──────────────────────────────────────────────────────────── */
typedef enum {
    CUPHU_INIT_MST    = 0,
    CUPHU_INIT_MCF    = 1,
    CUPHU_INIT_LAPLACE = 2,   /* GPU Laplacian PCG — smooth mode only     */
} CuPhuInitMethod;

/* ── run-time parameters passed to the GPU pipeline ─────────────────────── */
typedef struct CuPhuParams {
    /* statistical model */
    double nlooks;           /* equivalent number of independent looks        */
    double ncorrlooks;       /* alias, same as nlooks (for SNAPHU compat)     */
    double defothreshfactor; /* threshold factor for defo decorrelation       */
    double rhosconst1;       /* rho0 = rhosconst1/ncorrlooks + rhosconst2     */
    double rhosconst2;
    double cstd1, cstd2, cstd3; /* rhopow = cstd1*2 + cstd2*log(n) + cstd3*n */
    double sigsqcorr;        /* variance in measured correlation              */
    double costscale;        /* scale for discretizing costs to short int     */
    double nshortcycle;      /* number of integer steps per 2π cycle          */
    long   sigsqshortmin;    /* minimum short value for cost variance         */

    /* topo mode – SAR geometry */
    double orbitradius;      /* orbital radius (m)                            */
    double altitude;         /* SAR altitude (m)                              */
    double earthradius;      /* Earth radius (m)                              */
    double baseline;         /* baseline length (m)                           */
    double baselineangle;    /* baseline angle above horizontal (rad)         */
    double bperp;            /* perpendicular baseline (m)                    */
    int    transmitmode;     /* 2=ping-pong, 1=single-antenna                 */
    long   nlooksrange;      /* range looks                                   */
    long   nlooksaz;         /* azimuth looks                                 */
    long   nlooksother;
    long   ncorrlooksrange;
    long   ncorrlooksaz;
    double nearrange;        /* near-range slant distance (m)                 */
    double dr, da;           /* range/azimuth bin spacing (m)                 */
    double rangeres, azres;  /* resolution (m)                                */
    double lambda;           /* wavelength (m)                                */

    /* scattering model */
    double kds, specularexp, dzrcritfactor;
    int    shadow;
    double dzeimin;
    long   laywidth;
    double layminei, sloperatiofactor, sigsqei;

    /* pdf parameters */
    double dzlaypeak, azdzfactor, dzeifactor, dzeiweight;
    double dzlayfactor, layconst, layfalloffconst, sigsqlayfactor;

    /* defo mode */
    double defoazdzfactor, defomax, defolayconst;

    /* algorithm */
    int    kperpdpsi, kpardpsi; /* boxcar window for phase gradient averaging */
    double p;                    /* Lp exponent (<0 → MAP/statistical)         */
    double maxcost;
    double costscaleambight;
    double dnomincangle;
    double initdzr, initdzstep, threshold;
    long   initmaxflow, arcmaxflowconst, maxflow, nshortcycleL;
    long   cs2scalefactor;

    /* connected components */
    double minconncompfrac;
    long   conncompthresh;
    long   maxncomps;
} CuPhuParams;

/* ── tile geometry ────────────────────────────────────────────────────────── */
typedef struct CuPhuTileParams {
    int ntilerow, ntilecol;
    int rowovrlp, colovrlp;
    int tilecostthresh;
    int minregionsize;
    int nproc;               /* max CPU threads for tile network flow         */
    int ngpustreams;         /* CUDA streams for parallel tile cost compute   */
    int single_tile_reoptimize; /* after tiled stitching, rerun CPU TreeSolve
                                  * over the whole assembled scene as one tile
                                  * to clean up tile-boundary artifacts (see
                                  * snaphu-py's identically-named parameter).
                                  * Off by default -- see cuphu_unwrap()'s
                                  * multi-tile path.                          */
} CuPhuTileParams;

/* ── top-level result handle ─────────────────────────────────────────────── */
typedef struct CuPhuResult {
    float    *unw;           /* nrow*ncol unwrapped phase (radians), row-major */
    uint32_t *conncomp;      /* nrow*ncol connected-component labels           */
    int       nrow, ncol;
} CuPhuResult;

/* ── public API ──────────────────────────────────────────────────────────── */

/**
 * Initialize default parameters (sensible values matching SNAPHU defaults).
 * Call this before customizing and passing to cuphu_unwrap().
 */
void cuphu_default_params(CuPhuParams *p);
void cuphu_default_tile_params(CuPhuTileParams *tp);

/**
 * Full GPU-accelerated unwrapping pipeline.
 *
 * @param igram_r   Real part of complex interferogram, row-major float32
 * @param igram_i   Imaginary part, row-major float32
 * @param corr      Coherence magnitude, row-major float32 in [0,1]
 * @param mag       Amplitude (may be NULL → derived from igram)
 * @param mask      Byte mask (0 = invalid), may be NULL
 * @param nrow      Number of rows
 * @param ncol      Number of columns (line length)
 * @param cost_mode Statistical cost mode
 * @param init_meth Initialization algorithm
 * @param params    Algorithm parameters
 * @param tile      Tiling parameters
 * @param gpu_id    CUDA device ID (0 = first GPU)
 * @param result    Output – caller must free result->unw and result->conncomp
 * @return          0 on success, nonzero on failure
 */
int cuphu_unwrap(
    const float           *igram_r,
    const float           *igram_i,
    const float           *corr,
    const float           *mag,
    const unsigned char   *mask,
    int                    nrow,
    int                    ncol,
    CuPhuCostMode       cost_mode,
    CuPhuInitMethod     init_meth,
    const CuPhuParams  *params,
    const CuPhuTileParams *tile,
    int                    gpu_id,
    CuPhuResult        *result
);

/**
 * GPU-only cost computation.
 * Fills flat arrays rowcosts[nrow-1][ncol] and colcosts[nrow][ncol-1]
 * in SNAPHU's row-major layout (rowcosts first, then colcosts contiguous).
 * The cost element type is either smoothcostT or costT depending on mode.
 * Returns size in bytes of one cost element (sizeof(smoothcostT) or sizeof(costT)).
 */
int cuphu_build_costs_gpu(
    const float           *igram_r,
    const float           *igram_i,
    const float           *corr,
    const float           *mag,
    const short           *weights,       /* row-major, may be NULL */
    int                    nrow,
    int                    ncol,
    CuPhuCostMode       cost_mode,
    const CuPhuParams  *params,
    int                    gpu_id,
    void                  **costs_out,    /* allocated by this function */
    size_t                 *cost_elem_sz
);

/**
 * GPU phase integration: integrate horizontal and vertical flows into phase.
 */
void cuphu_integrate_phase_gpu(
    const float *wrapped_phase,   /* nrow*ncol                    */
    const short *hflows,          /* row arcs: nrow*(ncol-1)... wait,
                                     SNAPHU layout: (nrow-1)*ncol  */
    const short *vflows,          /* col arcs: nrow*(ncol-1)       */
    int          nrow,
    int          ncol,
    int          gpu_id,
    float       *unw_out          /* nrow*ncol                    */
);

/**
 * GPU connected component labeling using parallel union-find.
 */
void cuphu_conncomp_gpu(
    const float    *unw,          /* nrow*ncol unwrapped phase    */
    const float    *corr,         /* nrow*ncol coherence          */
    const unsigned char *mask,    /* may be NULL                  */
    int             nrow,
    int             ncol,
    const short    *poscost,      /* NULL if no MCF solve (e.g. Laplace); */
    const short    *negcost,      /* else flat row-arc-then-col-arc, from */
                                   /* cuphu_incrcost_early_exit() at the   */
                                   /* converged (post-solve) flow          */
    const struct CuPhuParams *params, /* statistical cost model +
                                          conncompthresh/minconncompfrac/maxncomps */
    int             gpu_id,
    uint32_t       *labels_out    /* nrow*ncol                    */
);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ── CUDA error helper (C++ only) ────────────────────────────────────────── */
#ifdef __cplusplus
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(cudaGetErrorString(err)) +
            " (" + file + ":" + std::to_string(line) + ")"
        );
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

/* Device array RAII wrapper */
template<typename T>
class DevArray {
public:
    DevArray() = default;

    explicit DevArray(size_t n) : n_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }

    DevArray(const T *host, size_t n) : n_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    ~DevArray() { if (ptr_) cudaFree(ptr_); }

    /* no copy */
    DevArray(const DevArray&) = delete;
    DevArray& operator=(const DevArray&) = delete;

    /* move */
    DevArray(DevArray&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; }
    DevArray& operator=(DevArray&& o) noexcept {
        if (this != &o) { if (ptr_) cudaFree(ptr_); ptr_ = o.ptr_; n_ = o.n_; o.ptr_ = nullptr; }
        return *this;
    }

    T*     get()  const { return ptr_; }
    size_t size() const { return n_; }

    void to_host(T *dst) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void fill_zero() { CUDA_CHECK(cudaMemset(ptr_, 0, n_ * sizeof(T))); }

private:
    T*     ptr_ = nullptr;
    size_t n_   = 0;
};
#endif /* __cplusplus */
