/**
 * cuphu_solver.cpp
 *
 * C++ bridge between the GPU cost arrays (computed in cuphu_cost.cu) and
 * SNAPHU's CPU network-flow solver.  It:
 *
 *  1. Sets up all SNAPHU internal data structures programmatically from
 *     arrays (no file I/O), matching what SNAPHU normally does after reading
 *     its config file and input binary files.
 *  2. Downloads the GPU-computed cost array to host memory (but keeps the GPU
 *     copy alive for the early-exit incrcost check).
 *  3. Calls SNAPHU's InitNetwork() → optional TreeSolve() solver loop.
 *  4. GPU incrcost early-exit: after MCF/MST init, checks whether all arc
 *     incremental costs are >= 0 on the GPU.  If so, TreeSolve is skipped
 *     entirely (the flow is already at a local optimum for nflow=1).
 *  5. Calls cuphu_integrate_phase_gpu() for phase integration on GPU.
 *  6. Calls cuphu_conncomp_gpu() for connected-component labeling on GPU.
 *  7. Returns unwrapped phase and conncomp labels as flat host arrays.
 */

#include "cuphu.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <queue>

/* ── optional profiling ──────────────────────────────────────────────────────── */
#ifdef CUPHU_PROFILE
#include <chrono>
static long ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
#define TICK() auto _T = std::chrono::steady_clock::now()
#define TOCK(label) fprintf(stderr,"[prof] %-28s %4ld ms\n",(label),ms_since(_T)); \
                    fflush(stderr); _T=std::chrono::steady_clock::now()
#else
#define TICK()       do {} while(0)
#define TOCK(label)  do {} while(0)
#endif

/* SNAPHU C headers */
extern "C" {
#include "snaphu.h"
}

/* CS2 (patched copy in src/cxx/snaphu_tls) keeps its solver state in
 * _Thread_local variables, so concurrent MCFInitFlows calls are safe
 * without serialization. */

/* Work-stealing parallel loop: dispatches ntasks work items across nworkers
 * threads.  Falls back to serial when nworkers <= 1 or ntasks <= 1. */
static void run_parallel(int ntasks, int nworkers,
                         std::function<void(int)> fn)
{
    if (nworkers <= 1 || ntasks <= 1) {
        for (int i = 0; i < ntasks; ++i) fn(i);
        return;
    }
    int w = std::min(nworkers, ntasks);
    std::atomic<int> next(0);
    std::vector<std::thread> workers;
    workers.reserve(w);
    for (int i = 0; i < w; ++i)
        workers.emplace_back([&]() {
            for (int t = next.fetch_add(1); t < ntasks; t = next.fetch_add(1))
                fn(t);
        });
    for (auto& t : workers) t.join();
}

/* Globals defined in snaphu.c (excluded because it has main()).
 * Provide them here so the linker is satisfied.  Informational and
 * progress output (sp1-sp3) is discarded: it is interleaved garbage when
 * tiles run on concurrent threads, and the terminal I/O itself becomes a
 * serialization point.  Errors (sp0) still go to stderr. */
static FILE *null_stream() {
    static FILE *f = fopen("/dev/null", "w");
    return f ? f : stderr;
}
FILE *sp0 = stderr;
FILE *sp1 = null_stream();
FILE *sp2 = null_stream();
FILE *sp3 = null_stream();
char  dumpresults_global  = 0;
char  requestedstop_global = 0;
nodeT NONTREEARC[1]       = {};   /* sentinel: arc not on tree */
void (*CalcCost)(void **, long, long, long, long, long,
                 paramT *, long *, long *) = nullptr;
long (*EvalCost)(void **, short **, long, long, long, paramT *) = nullptr;

/* forward declarations from CUDA translation units */
extern "C" void cuphu_build_smooth_costs_gpu(
    const float*, const float*, const short*, const short*,
    int, int, const CuPhuParams*, int, int,
    smoothcostT**, cudaStream_t);

extern "C" void cuphu_build_defo_costs_gpu(
    const float*, const float*, const short*, const short*,
    int, int, const CuPhuParams*, int, int,
    costT**, cudaStream_t);

extern "C" void cuphu_integrate_phase_gpu(
    const float*, const short*, const short*, int, int, int, float*);

extern "C" void cuphu_conncomp_gpu(
    const float*, const float*, const unsigned char*,
    int, int, double, double, long, int, uint32_t*);

extern "C" void cuphu_wrap_phase(float*, int, cudaStream_t);

extern "C" void cuphu_laplace_unwrap_gpu(
    const smoothcostT *d_smooth_costs,
    const float       *d_phase,
    int                nrow,
    int                ncol,
    double             nshortcycle,
    int                max_iter,
    float              tol,
    int                verbose,
    float             *d_unw,
    cudaStream_t       stream);

extern "C" bool cuphu_incrcost_early_exit(
    const smoothcostT*, const short*, int, long, int, short*, short*, int*);

/* ── default parameter tables ─────────────────────────────────────────────── */
extern "C"
void cuphu_default_params(CuPhuParams *p) {
    std::memset(p, 0, sizeof(*p));
    p->rhosconst1      = DEF_RHOSCONST1;
    p->rhosconst2      = DEF_RHOSCONST2;
    p->cstd1           = DEF_CSTD1;
    p->cstd2           = DEF_CSTD2;
    p->cstd3           = DEF_CSTD3;
    p->defothreshfactor= DEF_DEFOTHRESHFACTOR;
    p->sigsqcorr       = DEF_SIGSQCORR;
    p->nlooks          = DEF_NCORRLOOKS;
    p->ncorrlooks      = DEF_NCORRLOOKS;
    p->costscale       = DEF_COSTSCALE;
    p->nshortcycle     = DEF_NSHORTCYCLE;
    p->sigsqshortmin   = DEF_SIGSQSHORTMIN;
    p->kperpdpsi       = DEF_KPERPDPSI;
    p->kpardpsi        = DEF_KPARDPSI;
    p->p               = DEF_P;
    p->maxcost         = DEF_MAXCOST;
    p->initmaxflow     = DEF_INITMAXFLOW;
    p->arcmaxflowconst = DEF_ARCMAXFLOWCONST;
    p->maxflow         = DEF_MAXFLOW;
    p->cs2scalefactor  = DEF_CS2SCALEFACTOR;
    p->costscaleambight= DEF_COSTSCALEAMBIGHT;
    p->initdzr         = DEF_INITDZR;
    p->initdzstep      = DEF_INITDZSTEP;
    p->threshold       = DEF_THRESHOLD;
    p->dnomincangle    = DEF_DNOMINCANGLE;
    p->orbitradius     = DEF_ORBITRADIUS;
    p->altitude        = DEF_ALTITUDE;
    p->earthradius     = DEF_EARTHRADIUS;
    p->baseline        = DEF_BASELINE;
    p->baselineangle   = DEF_BASELINEANGLE;
    p->bperp           = DEF_BPERP;
    p->transmitmode    = DEF_TRANSMITMODE;
    p->nlooksrange     = DEF_NLOOKSRANGE;
    p->nlooksaz        = DEF_NLOOKSAZ;
    p->nlooksother     = DEF_NLOOKSOTHER;
    p->ncorrlooksrange = DEF_NCORRLOOKSRANGE;
    p->ncorrlooksaz    = DEF_NCORRLOOKSAZ;
    p->nearrange       = DEF_NEARRANGE;
    p->dr              = DEF_DR;
    p->da              = DEF_DA;
    p->rangeres        = DEF_RANGERES;
    p->azres           = DEF_AZRES;
    p->lambda          = DEF_LAMBDA;
    p->kds             = DEF_KDS;
    p->specularexp     = DEF_SPECULAREXP;
    p->dzrcritfactor   = DEF_DZRCRITFACTOR;
    p->shadow          = DEF_SHADOW;
    p->dzeimin         = DEF_DZEIMIN;
    p->laywidth        = DEF_LAYWIDTH;
    p->layminei        = DEF_LAYMINEI;
    p->sloperatiofactor= DEF_SLOPERATIOFACTOR;
    p->sigsqei         = DEF_SIGSQEI;
    p->dzlaypeak       = DEF_DZLAYPEAK;
    p->azdzfactor      = DEF_AZDZFACTOR;
    p->dzeifactor      = DEF_DZEIFACTOR;
    p->dzeiweight      = DEF_DZEIWEIGHT;
    p->dzlayfactor     = DEF_DZLAYFACTOR;
    p->layconst        = DEF_LAYCONST;
    p->layfalloffconst = DEF_LAYFALLOFFCONST;
    p->sigsqlayfactor  = DEF_SIGSQLAYFACTOR;
    p->defoazdzfactor  = DEF_DEFOAZDZFACTOR;
    p->defomax         = DEF_DEFOMAX;
    p->defolayconst    = DEF_DEFOLAYCONST;
    p->minconncompfrac = DEF_MINCONNCOMPFRAC;
    p->conncompthresh  = DEF_CONNCOMPTHRESH;
    p->maxncomps       = DEF_MAXNCOMPS;
}

extern "C"
void cuphu_default_tile_params(CuPhuTileParams *tp) {
    tp->ntilerow       = DEF_NTILEROW;
    tp->ntilecol       = DEF_NTILECOL;
    tp->rowovrlp       = DEF_ROWOVRLP;
    tp->colovrlp       = DEF_COLOVRLP;
    tp->tilecostthresh = DEF_TILECOSTTHRESH;
    tp->minregionsize  = DEF_MINREGIONSIZE;
    tp->nproc          = 1;
    tp->ngpustreams    = 2;
}

/* ── translate CuPhuParams → SNAPHU's paramT ─────────────────────────── */
static void fill_snaphu_params(const CuPhuParams *cp,
                                CuPhuCostMode cost_mode,
                                CuPhuInitMethod init_meth,
                                int nrow, int ncol,
                                paramT *sp) {
    infileT  infiles;  std::memset(&infiles,  0, sizeof(infiles));
    outfileT outfiles; std::memset(&outfiles, 0, sizeof(outfiles));
    SetDefaults(&infiles, &outfiles, sp);

    sp->costmode       = (cost_mode == CUPHU_COST_SMOOTH) ? SMOOTH
                       : (cost_mode == CUPHU_COST_DEFO)   ? DEFO   : TOPO;
    sp->initmethod     = (init_meth  == CUPHU_INIT_MST)   ? MSTINIT : MCFINIT;
    sp->ncorrlooks     = cp->ncorrlooks;
    sp->rhosconst1     = cp->rhosconst1;
    sp->rhosconst2     = cp->rhosconst2;
    sp->cstd1          = cp->cstd1;
    sp->cstd2          = cp->cstd2;
    sp->cstd3          = cp->cstd3;
    sp->sigsqcorr      = cp->sigsqcorr;
    sp->costscale      = cp->costscale;
    sp->nshortcycle    = (long)cp->nshortcycle;
    sp->sigsqshortmin  = cp->sigsqshortmin;
    sp->kperpdpsi      = cp->kperpdpsi;
    sp->kpardpsi       = cp->kpardpsi;
    sp->p              = cp->p;
    sp->maxcost        = cp->maxcost;
    sp->initmaxflow    = cp->initmaxflow;
    sp->arcmaxflowconst= cp->arcmaxflowconst;
    sp->maxflow        = cp->maxflow;
    sp->cs2scalefactor = cp->cs2scalefactor;
    sp->defothreshfactor = cp->defothreshfactor;
    sp->defomax        = cp->defomax;
    sp->minconncompfrac= cp->minconncompfrac;
    sp->conncompthresh = cp->conncompthresh;
    sp->maxncomps      = cp->maxncomps;
    sp->orbitradius    = cp->orbitradius;
    sp->altitude       = cp->altitude;
    sp->earthradius    = cp->earthradius;
    sp->baseline       = cp->baseline;
    sp->baselineangle  = cp->baselineangle;
    sp->bperp          = cp->bperp;
    sp->transmitmode   = cp->transmitmode;
    sp->nlooksrange    = cp->nlooksrange;
    sp->nlooksaz       = cp->nlooksaz;
    sp->nlooksother    = cp->nlooksother;
    sp->ncorrlooksrange= cp->ncorrlooksrange;
    sp->ncorrlooksaz   = cp->ncorrlooksaz;
    sp->nearrange      = cp->nearrange;
    sp->dr             = cp->dr;
    sp->da             = cp->da;
    sp->rangeres       = cp->rangeres;
    sp->azres          = cp->azres;
    sp->lambda         = cp->lambda;
    sp->verbose        = FALSE;
    sp->amplitude      = TRUE;

    if (sp->maxnflowcycles == USEMAXCYCLEFRACTION)
        sp->maxnflowcycles = LRound(sp->maxcyclefraction * nrow * ncol);
}

/* ── build SNAPHU-style 2-D pointer array from flat allocation ──────────── */
template<typename T>
static std::vector<T *> make_2d_ptrs(T *flat, int nrow, int ncol) {
    std::vector<T *> ptrs((size_t)nrow);
    for (int r = 0; r < nrow; ++r) ptrs[r] = flat + (size_t)r * ncol;
    return ptrs;
}

/* ── flatten SNAPHU flows to a contiguous short array ───────────────────── */
/*
 * Output layout (same as GPU cost array):
 *   [0 .. (nrow-1)*ncol - 1]  : row arcs flows[0..nrow-2][0..ncol-1]
 *   [(nrow-1)*ncol .. end]    : col arcs flows[nrow-1+r][0..ncol-2]
 */
static void flatten_flows(
    short **flows, int nrow, int ncol,
    std::vector<short> &out)
{
    size_t nrowcost = (size_t)(nrow - 1) * ncol;
    size_t ncolcost = (size_t)nrow * (ncol - 1);
    out.resize(nrowcost + ncolcost);
    for (int r = 0; r < nrow - 1; ++r)
        for (int c = 0; c < ncol; ++c)
            out[(size_t)r * ncol + c] = flows[r][c];
    for (int r = 0; r < nrow; ++r)
        for (int c = 0; c < ncol - 1; ++c)
            out[nrowcost + (size_t)r * (ncol - 1) + c] = flows[nrow - 1 + r][c];
}

/* ── single-tile solve ──────────────────────────────────────────────────── */
static int solve_tile(
    const float          *h_phase_tile,
    const float          *h_corr_tile,
    const float          *h_mag_tile,
    const unsigned char  *h_mask_tile,
    int                   tile_nrow,
    int                   tile_ncol,
    CuPhuCostMode      cost_mode,
    CuPhuInitMethod    init_meth,
    const CuPhuParams *params,
    int                   gpu_id,
    float                *h_unw_tile,
    uint32_t             *h_conncomp_tile
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    size_t npix  = (size_t)tile_nrow * tile_ncol;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    TICK();

    /* ── upload wrapped phase and corr to GPU ─────────────────────────── */
    DevArray<float> d_phase(h_phase_tile, npix);
    DevArray<float> d_corr (h_corr_tile,  npix);

    cuphu_wrap_phase(d_phase.get(), (int)npix, stream);

    /* ── compute costs on GPU ─────────────────────────────────────────── */
    size_t nrowcost = (size_t)(tile_nrow - 1) * tile_ncol;
    size_t ncolcost = (size_t)tile_nrow * (tile_ncol - 1);
    size_t ncost_total = nrowcost + ncolcost;

    /* Keep GPU smooth-cost array alive for the incrcost early-exit check. */
    smoothcostT *d_smooth_costs = nullptr;  /* only set for SMOOTH mode */
    void        *d_costs_generic = nullptr;
    size_t       cost_elem_sz    = 0;

    if (cost_mode == CUPHU_COST_SMOOTH) {
        cuphu_build_smooth_costs_gpu(
            d_phase.get(), d_corr.get(), nullptr, nullptr,
            tile_nrow, tile_ncol, params,
            params->kperpdpsi, params->kpardpsi,
            &d_smooth_costs, stream);
        d_costs_generic = d_smooth_costs;
        cost_elem_sz    = sizeof(smoothcostT);
    } else {
        costT *d_cost = nullptr;
        cuphu_build_defo_costs_gpu(
            d_phase.get(), d_corr.get(), nullptr, nullptr,
            tile_nrow, tile_ncol, params,
            params->kperpdpsi, params->kpardpsi,
            &d_cost, stream);
        d_costs_generic = d_cost;
        cost_elem_sz    = sizeof(costT);
    }

    /* ── Laplace PCG fast path (smooth mode only) ───────────────────────
     *
     * Bypasses the entire cost download → MCF/MST → InitNetwork → TreeSolve
     * pipeline.  Solves L·u = b on GPU via Jacobi-preconditioned CG.
     * Expected speedup over TreeSolve: ~50-200× for typical InSAR scenes.
     */
    if (init_meth == CUPHU_INIT_LAPLACE) {
        if (cost_mode != CUPHU_COST_SMOOTH)
            throw std::runtime_error("CUPHU_INIT_LAPLACE requires smooth cost mode");

        CUDA_CHECK(cudaStreamSynchronize(stream));
        TOCK("GPU cost compute (no D2H)");

        DevArray<float> d_unw_lap(npix);
        cuphu_laplace_unwrap_gpu(
            d_smooth_costs, d_phase.get(),
            tile_nrow, tile_ncol,
            params->nshortcycle,
            /*max_iter=*/1000, /*tol=*/1e-3f, /*verbose=*/0,
            d_unw_lap.get(), stream);
        TOCK("GPU Laplace PCG");

        CUDA_CHECK(cudaFree(d_smooth_costs));
        d_smooth_costs  = nullptr;
        d_costs_generic = nullptr;

        DevArray<uint32_t> d_conncomp(npix);
        DevArray<float>    d_corr2(h_corr_tile, npix);
        DevArray<uint8_t>  d_mask_lap;
        const uint8_t *mask_lap_ptr = nullptr;
        if (h_mask_tile) {
            d_mask_lap   = DevArray<uint8_t>(h_mask_tile, npix);
            mask_lap_ptr = d_mask_lap.get();
        }
        cuphu_conncomp_gpu(
            d_unw_lap.get(), d_corr2.get(), mask_lap_ptr,
            tile_nrow, tile_ncol,
            /*cost_thresh=*/0.1,
            params->minconncompfrac,
            params->maxncomps,
            gpu_id,
            d_conncomp.get());
        TOCK("GPU conncomp");

        d_unw_lap.to_host(h_unw_tile);
        d_conncomp.to_host(h_conncomp_tile);
        TOCK("D2H download");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }

    /* ── download costs to host ──────────────────────────────────────── */
    std::vector<char> h_costs_buf(ncost_total * cost_elem_sz);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_costs_buf.data(), d_costs_generic,
                          ncost_total * cost_elem_sz,
                          cudaMemcpyDeviceToHost));
    /* NOTE: d_smooth_costs is NOT freed here; freed after early-exit check. */
    if (cost_mode != CUPHU_COST_SMOOTH) {
        CUDA_CHECK(cudaFree(d_costs_generic));
        d_costs_generic = nullptr;
    }
    TOCK("GPU cost compute + D2H");

    /* ── set up SNAPHU parameters ────────────────────────────────────── */
    paramT  sp;
    std::memset(&sp, 0, sizeof(sp));
    fill_snaphu_params(params, cost_mode, init_meth, tile_nrow, tile_ncol, &sp);

    /* ── build SNAPHU 2-D pointer arrays ─────────────────────────────── */
    std::vector<float> h_phase_copy(h_phase_tile, h_phase_tile + npix);
    auto phase2d = make_2d_ptrs(h_phase_copy.data(), tile_nrow, tile_ncol);
    float **phase_pp = phase2d.data();

    std::vector<float> h_mag_copy(h_mag_tile ? npix : 0);
    if (h_mag_tile)
        h_mag_copy.assign(h_mag_tile, h_mag_tile + npix);
    else
        h_mag_copy.assign(npix, 1.0f);
    auto mag2d = make_2d_ptrs(h_mag_copy.data(), tile_nrow, tile_ncol);
    float **mag_pp = mag2d.data();

    if (h_mask_tile)
        for (size_t i = 0; i < npix; ++i)
            if (!h_mask_tile[i]) h_mag_copy[i] = 0.0f;

    /* cost 2-D pointer array: Get2DRowColMem layout */
    size_t ncostrows_full = (size_t)(2 * tile_nrow - 1);
    std::vector<void *> costptrs(ncostrows_full, nullptr);
    char *row_base = h_costs_buf.data();
    char *col_base = row_base + nrowcost * cost_elem_sz;
    for (int r = 0; r < tile_nrow - 1; ++r)
        costptrs[r] = row_base + (size_t)r * tile_ncol * cost_elem_sz;
    for (int r = 0; r < tile_nrow; ++r)
        costptrs[(size_t)(tile_nrow - 1) + r] = col_base + (size_t)r * (tile_ncol - 1) * cost_elem_sz;
    void **costs_pp = costptrs.data();

    /* ── initialize flows ───────────────────────────────────────────── */
    short **flows = nullptr;
    nodeT **nodes = nullptr;
    nodeT   ground;

    SetGridNetworkFunctionPointers();
    if (cost_mode == CUPHU_COST_SMOOTH) {
        CalcCost = CalcCostSmooth;
        EvalCost = EvalCostSmooth;
    } else {
        CalcCost = CalcCostDefo;
        EvalCost = EvalCostDefo;
    }

    /* Both MCFInitFlows (via SolveCS2) and MSTInitFlows call
     * Free2DArray(mstcosts, 2*nrow-1) internally, so mstcosts MUST be
     * allocated through SNAPHU's Get2DRowColMem and NOT wrapped in a
     * std::vector — the callee owns and frees it. */
    {
        short **mst_costs = (short **)Get2DRowColMem(
            tile_nrow, tile_ncol, (int)sizeof(short *), sizeof(short));
        for (int r = 0; r < tile_nrow - 1; ++r)
            for (int c = 0; c < tile_ncol; ++c)
                mst_costs[r][c] = 1;
        for (int r = 0; r < tile_nrow; ++r)
            for (int c = 0; c < tile_ncol - 1; ++c)
                mst_costs[tile_nrow - 1 + r][c] = 1;

        if (sp.initmethod == MSTINIT) {
            MSTInitFlows(phase_pp, &flows, mst_costs,
                         tile_nrow, tile_ncol, &nodes, &ground, sp.initmaxflow);
        } else {
            MCFInitFlows(phase_pp, &flows, mst_costs,
                         tile_nrow, tile_ncol, sp.cs2scalefactor);
        }
        /* mst_costs is freed inside MCFInitFlows/MSTInitFlows; do not free */
    }
    TOCK("InitFlows (MCF or MST)");

    /* ── solver data structures ──────────────────────────────────────── */
    long ngroundarcs, ncycle, nflowdone, mostflow, nflow;
    long candidatebagsize, candidatelistsize;
    candidateT *candidatebag  = nullptr;
    candidateT *candidatelist = nullptr;
    signed char **iscandidate  = nullptr;
    nodeT ***apexes            = nullptr;
    bucketT *bkts              = nullptr;
    long iincrcostfile;
    incrcostT **incrcosts      = nullptr;
    long nnoderow;
    int *nnodesperrow          = nullptr;
    long narcrow;
    int *narcsperrow           = nullptr;
    signed char notfirstloop   = FALSE;
    totalcostT totalcost;

    outfileT outfiles;
    std::memset(&outfiles, 0, sizeof(outfiles));

    int rc = InitNetwork(
        flows, &ngroundarcs, &ncycle, &nflowdone, &mostflow, &nflow,
        &candidatebagsize, &candidatebag,
        &candidatelistsize, &candidatelist,
        &iscandidate, &apexes, &bkts, &iincrcostfile,
        &incrcosts, &nodes, &ground,
        &nnoderow, &nnodesperrow,
        &narcrow, &narcsperrow,
        tile_nrow, tile_ncol,
        &notfirstloop, &totalcost, &sp);

    if (rc != 0)
        throw std::runtime_error("SNAPHU InitNetwork failed");

    MaskNodes(tile_nrow, tile_ncol, nodes, &ground, mag_pp);
    TOCK("InitNetwork");

    /* ── GPU early-exit check (smooth mode only) ─────────────────────── */
    /*
     * Check AFTER InitNetwork, which calls AdjustFlow to modify corner flows.
     * We test the final adjusted flows against the GPU costs: if all arc
     * incremental costs are >= 0 (no single-arc improvement is beneficial)
     * the solution is at a local MCF optimum and TreeSolve can be skipped.
     *
     * For high-coherence InSAR scenes with MCF init, this eliminates the
     * 200–400 ms TreeSolve loop while producing exactly the same result.
     */
    bool gpu_early_exit = false;
    DevArray<short> d_flows_flat_arr;
    DevArray<short> d_poscost_arr;
    DevArray<short> d_negcost_arr;
    DevArray<int>   d_scratch_arr;

    if (cost_mode == CUPHU_COST_SMOOTH && d_smooth_costs != nullptr) {
        std::vector<short> h_flows_flat;
        flatten_flows(flows, tile_nrow, tile_ncol, h_flows_flat);

        int narcs = (int)ncost_total;
        d_flows_flat_arr = DevArray<short>(h_flows_flat.data(), (size_t)narcs);
        d_poscost_arr    = DevArray<short>((size_t)narcs);
        d_negcost_arr    = DevArray<short>((size_t)narcs);
        d_scratch_arr    = DevArray<int>(2u);

        gpu_early_exit = cuphu_incrcost_early_exit(
            d_smooth_costs, d_flows_flat_arr.get(),
            narcs, sp.nshortcycle, /*nflow=*/1,
            d_poscost_arr.get(), d_negcost_arr.get(), d_scratch_arr.get());
        TOCK("GPU incrcost early-exit check");

        if (std::getenv("CUPHU_DISABLE_EARLY_EXIT") != nullptr)
            gpu_early_exit = false;
        if (std::getenv("CUPHU_DEBUG") != nullptr)
            fprintf(stderr, "[cuphu] early-exit check: %s\n",
                    gpu_early_exit ? "PASS (TreeSolve skipped)"
                                   : "fail (TreeSolve runs)");
    }

    /* GPU cost array no longer needed */
    if (d_smooth_costs) {
        CUDA_CHECK(cudaFree(d_smooth_costs));
        d_smooth_costs = nullptr;
    }

    /* ── main solver loop (skipped when GPU early-exit passes) ───────── */
    if (!gpu_early_exit) {
        totalcostT oldtotalcost = totalcost;
        totalcostT mintotalcost = totalcost;
        long nnondecreasedcostiter = 0;

        while (true) {
            SetupIncrFlowCosts(costs_pp, incrcosts, flows, nflow,
                               tile_nrow, narcrow, narcsperrow, &sp);

            nodeT **sourcelist   = nullptr;
            long  *nconnectedarr = nullptr;
            long nsource = SelectSources(
                nodes, mag_pp, &ground, nflow, flows, ngroundarcs,
                tile_nrow, tile_ncol, &sp, &sourcelist, &nconnectedarr);

            SetupTreeSolveNetwork(nodes, &ground, apexes, iscandidate,
                                  nnoderow, nnodesperrow, narcrow, narcsperrow,
                                  tile_nrow, tile_ncol);

            long n = 0;
            for (long isrc = 0; isrc < nsource; ++isrc) {
                nodeT *source = sourcelist[isrc];
                n += TreeSolve(nodes, nullptr, &ground, source,
                               &candidatelist, &candidatebag,
                               &candidatelistsize, &candidatebagsize,
                               bkts, flows, costs_pp, incrcosts, apexes,
                               iscandidate, ngroundarcs, nflow,
                               mag_pp, phase_pp, (char *)"",
                               nnoderow, nnodesperrow, narcrow, narcsperrow,
                               tile_nrow, tile_ncol, &outfiles,
                               nconnectedarr[isrc], &sp);
            }
            std::free(sourcelist);
            std::free(nconnectedarr);

            ncycle    += n;
            nflowdone  = (n <= sp.maxnflowcycles) ? nflowdone + 1 : 1;
            mostflow   = MaxNonMaskFlow(flows, mag_pp, tile_nrow, tile_ncol);
            TOCK("TreeSolve iter");

            if (nnondecreasedcostiter >= 2 * mostflow) break;
            if (nflowdone >= sp.maxflow || nflowdone >= mostflow || sp.p >= 1.0) break;

            nflow++;
            if (nflow > sp.maxflow || nflow > mostflow) {
                nflow = 1;
                notfirstloop = TRUE;
            }
            (void)oldtotalcost; (void)mintotalcost; (void)nnondecreasedcostiter;
        }
    } /* end !gpu_early_exit */

    /* ── flatten flows for GPU phase integration ─────────────────────── */
    size_t nhflows = nrowcost;
    size_t nvflows = ncolcost;
    std::vector<short> h_hflows(nhflows), h_vflows(nvflows);

    for (int r = 0; r < tile_nrow - 1; ++r)
        for (int c = 0; c < tile_ncol; ++c)
            h_hflows[(size_t)r * tile_ncol + c] = flows[r][c];
    for (int r = 0; r < tile_nrow; ++r)
        for (int c = 0; c < tile_ncol - 1; ++c)
            h_vflows[(size_t)r * (tile_ncol - 1) + c] = flows[(tile_nrow - 1) + r][c];

    DevArray<float> d_unw(npix);
    DevArray<short> d_hflows(h_hflows.data(), nhflows);
    DevArray<short> d_vflows(h_vflows.data(), nvflows);
    DevArray<float> d_phase2(h_phase_copy.data(), npix);

    TOCK("flow flatten");
    cuphu_integrate_phase_gpu(
        d_phase2.get(), d_hflows.get(), d_vflows.get(),
        tile_nrow, tile_ncol, gpu_id, d_unw.get());
    TOCK("GPU integrate_phase");

    /* ── connected components on GPU ─────────────────────────────────── */
    DevArray<uint32_t> d_conncomp(npix);
    DevArray<float>    d_corr2(h_corr_tile, npix);
    DevArray<uint8_t>  d_mask_dev;
    const uint8_t *mask_dev_ptr = nullptr;
    if (h_mask_tile) {
        d_mask_dev   = DevArray<uint8_t>(h_mask_tile, npix);
        mask_dev_ptr = d_mask_dev.get();
    }

    cuphu_conncomp_gpu(
        d_unw.get(), d_corr2.get(), mask_dev_ptr,
        tile_nrow, tile_ncol,
        /*cost_thresh=*/0.1,
        params->minconncompfrac,
        params->maxncomps,
        gpu_id,
        d_conncomp.get());
    TOCK("GPU conncomp");

    /* ── download results ─────────────────────────────────────────────── */
    d_unw.to_host(h_unw_tile);
    d_conncomp.to_host(h_conncomp_tile);
    TOCK("D2H download");

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

/* ── main public entry point ─────────────────────────────────────────────── */
extern "C"
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
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    size_t npix = (size_t)nrow * ncol;

    result->unw      = (float *)    std::malloc(npix * sizeof(float));
    result->conncomp = (uint32_t *) std::malloc(npix * sizeof(uint32_t));
    result->nrow     = nrow;
    result->ncol     = ncol;
    if (!result->unw || !result->conncomp) return -1;

    /* derive wrapped phase from complex igram */
    std::vector<float> h_phase(npix);
    for (size_t k = 0; k < npix; ++k) {
        float phi = std::atan2(igram_i[k], igram_r[k]);
        if (phi < 0.0f) phi += 6.28318530717958648f;
        h_phase[k] = phi;
    }

    /* ── single-tile fast path ─────────────────────────────────────────── */
    if (tile->ntilerow == 1 && tile->ntilecol == 1) {
        solve_tile(
            h_phase.data(), corr, mag, mask,
            nrow, ncol,
            cost_mode, init_meth, params, gpu_id,
            result->unw, result->conncomp);
        return 0;
    }

    /* ── multi-tile path ─────────────────────────────────────────────── */
    int ntr      = tile->ntilerow;
    int ntc      = tile->ntilecol;
    int row_ovrlp = tile->rowovrlp;
    int col_ovrlp = tile->colovrlp;
    int ntiles   = ntr * ntc;

    std::memset(result->unw,      0, npix * sizeof(float));
    std::memset(result->conncomp, 0, npix * sizeof(uint32_t));

    /* Pre-build one work descriptor per tile.  Input data is copied here
     * so each worker thread has its own non-overlapping buffer. */
    struct TileWork {
        /* geometry */
        int first_row, first_col;
        int tnrow, tncol;
        int tr, tc;           /* tile indices (needed for overlap trimming) */
        /* input copies */
        std::vector<float>   phase, corr_t;
        std::vector<float>   mag_t;
        std::vector<uint8_t> mask_t;
        /* outputs */
        std::vector<float>    unw;
        std::vector<uint32_t> cc;
    };

    std::vector<TileWork> tiles(ntiles);

    for (int tr = 0; tr < ntr; ++tr) {
        for (int tc = 0; tc < ntc; ++tc) {
            TileWork& tw = tiles[tr * ntc + tc];
            tw.tr = tr;  tw.tc = tc;
            tw.first_row = (int)((long)tr * (nrow - row_ovrlp) / ntr);
            tw.first_col = (int)((long)tc * (ncol - col_ovrlp) / ntc);
            int last_row = (tr == ntr - 1) ? nrow
                         : (int)((long)(tr + 1) * (nrow - row_ovrlp) / ntr) + row_ovrlp;
            int last_col = (tc == ntc - 1) ? ncol
                         : (int)((long)(tc + 1) * (ncol - col_ovrlp) / ntc) + col_ovrlp;
            tw.tnrow = last_row - tw.first_row;
            tw.tncol = last_col - tw.first_col;
            size_t tnpix = (size_t)tw.tnrow * tw.tncol;

            tw.phase.resize(tnpix);
            tw.corr_t.resize(tnpix);
            if (mag)  tw.mag_t.resize(tnpix);
            if (mask) tw.mask_t.resize(tnpix);
            tw.unw.resize(tnpix);
            tw.cc.resize(tnpix);

            for (int r = 0; r < tw.tnrow; ++r) {
                int gr = tw.first_row + r;
                for (int c = 0; c < tw.tncol; ++c) {
                    int gc = tw.first_col + c;
                    size_t ti = (size_t)r * tw.tncol + c;
                    size_t gi = (size_t)gr * ncol + gc;
                    tw.phase[ti]  = h_phase[gi];
                    tw.corr_t[ti] = corr[gi];
                    if (mag)  tw.mag_t[ti]  = mag[gi];
                    if (mask) tw.mask_t[ti] = mask[gi];
                }
            }
        }
    }

    /* Run solve_tile on all tiles in parallel (up to nproc concurrent).
     * MCFInitFlows is serialized via g_cs2_mutex; everything else—
     * TreeSolve, GPU cost/phase/conncomp—runs fully concurrently. */
    run_parallel(ntiles, tile->nproc, [&](int idx) {
        TileWork& tw = tiles[idx];
        solve_tile(
            tw.phase.data(), tw.corr_t.data(),
            tw.mag_t.empty()  ? nullptr : tw.mag_t.data(),
            tw.mask_t.empty() ? nullptr : tw.mask_t.data(),
            tw.tnrow, tw.tncol,
            cost_mode, init_meth, params, gpu_id,
            tw.unw.data(), tw.cc.data());
    });

    /* ── tile stitching: bulk 2π offset per tile ─────────────────────────
     *
     * Each tile is independently unwrapped, so tiles may be offset from
     * their neighbors by an integer multiple of 2π.  We reconcile them with
     * a two-step procedure:
     *
     *  1. For every adjacent tile pair, compute the median phase difference
     *     in the shared overlap pixels (only coherent pixels, corr > 0.05).
     *     Round to the nearest 2π integer → inter-tile offset k_ij.
     *
     *  2. BFS from tile (0,0): propagate cumulative integer offsets across
     *     the tile grid so that every tile's phase is consistent with its
     *     BFS predecessor.
     *
     * The stitching runs entirely on CPU (the overlap is ~ovlp × tile_dim
     * pixels, negligible vs. the GPU tile solve).  nth_element gives O(n)
     * median without sorting.
     */
    std::vector<int> tile_k(ntiles, 0);   /* integer 2π offset per tile */

    if (ntiles > 1 && (row_ovrlp > 0 || col_ovrlp > 0)) {

        auto dump_stats = [](const char *tag, const std::vector<float> &d) {
            if (!std::getenv("CUPHU_DEBUG") || d.empty()) return;
            double sum = 0.0, sumsq = 0.0;
            float lo = d[0], hi = d[0];
            for (float v : d) { sum += v; sumsq += (double)v*v; lo = std::min(lo,v); hi = std::max(hi,v); }
            double mean = sum / d.size();
            double var  = sumsq/d.size() - mean*mean;
            fprintf(stderr, "[cuphu]   %s: n=%zu mean=%.4f std=%.4f min=%.4f max=%.4f\n",
                    tag, d.size(), mean, std::sqrt(std::max(0.0,var)), lo, hi);
        };

        /* Median diff in the horizontal overlap: left tile's rightmost
         * col_ovrlp columns vs right tile's leftmost col_ovrlp columns. */
        auto horiz_k = [&](int li, int ri) -> int {
            const TileWork& L = tiles[li];
            const TileWork& R = tiles[ri];
            if (col_ovrlp <= 0 || col_ovrlp > L.tncol || col_ovrlp > R.tncol)
                return 0;
            std::vector<float> d;
            d.reserve((size_t)std::min(L.tnrow, R.tnrow) * col_ovrlp);
            int nr = std::min(L.tnrow, R.tnrow);
            for (int r = 0; r < nr; ++r) {
                for (int c = 0; c < col_ovrlp; ++c) {
                    size_t lp = (size_t)r * L.tncol + (L.tncol - col_ovrlp + c);
                    size_t rp = (size_t)r * R.tncol + c;
                    if (L.corr_t[lp] < 0.05f || R.corr_t[rp] < 0.05f) continue;
                    float dv = L.unw[lp] - R.unw[rp];
                    /* dv == 0 means the tiles already agree exactly here — the
                     * strongest possible signal that no offset is needed.
                     * Do NOT skip it: an earlier version excluded exact zeros,
                     * which silently threw away the (typically overwhelming)
                     * majority of agreeing pixels and let a handful of
                     * outliers near residues/artifacts dominate the median. */
                    d.push_back(dv);
                }
            }
            dump_stats("horiz_k", d);
            if (d.empty()) return 0;
            auto mid = d.begin() + d.size() / 2;
            std::nth_element(d.begin(), mid, d.end());
            return (int)std::round((double)*mid / (2.0 * M_PI));
        };

        /* Median diff in the vertical overlap: top tile's bottom row_ovrlp
         * rows vs bottom tile's top row_ovrlp rows. */
        auto vert_k = [&](int ti, int bi) -> int {
            const TileWork& T = tiles[ti];
            const TileWork& B = tiles[bi];
            if (row_ovrlp <= 0 || row_ovrlp > T.tnrow || row_ovrlp > B.tnrow)
                return 0;
            std::vector<float> d;
            d.reserve((size_t)std::min(T.tncol, B.tncol) * row_ovrlp);
            int nc = std::min(T.tncol, B.tncol);
            for (int r = 0; r < row_ovrlp; ++r) {
                for (int c = 0; c < nc; ++c) {
                    size_t tp = (size_t)(T.tnrow - row_ovrlp + r) * T.tncol + c;
                    size_t bp = (size_t)r * B.tncol + c;
                    if (T.corr_t[tp] < 0.05f || B.corr_t[bp] < 0.05f) continue;
                    float dv = T.unw[tp] - B.unw[bp];
                    /* see horiz_k: dv == 0 must not be excluded */
                    d.push_back(dv);
                }
            }
            dump_stats("vert_k", d);
            if (d.empty()) return 0;
            auto mid = d.begin() + d.size() / 2;
            std::nth_element(d.begin(), mid, d.end());
            return (int)std::round((double)*mid / (2.0 * M_PI));
        };

        /* BFS from tile (0,0) — propagate cumulative offsets.
         *
         * horiz_k(L, R) = round((L.unw[overlap] - R.unw[overlap]) / 2π) = k
         * means L is k cycles AHEAD of R, so R needs +k*2π to match L.
         * tile_k[R] = tile_k[L] + k so phase_add = tile_k[R] * 2π = +k*2π. */
        std::vector<bool> visited(ntiles, false);
        std::queue<int> bfs;
        bfs.push(0); visited[0] = true;
        while (!bfs.empty()) {
            int idx = bfs.front(); bfs.pop();
            int br = idx / ntc, bc = idx % ntc;
            /* right neighbor */
            if (bc + 1 < ntc) {
                int nb = br * ntc + (bc + 1);
                if (!visited[nb]) {
                    int k = horiz_k(idx, nb);
                    tile_k[nb] = tile_k[idx] + k;
                    if (std::getenv("CUPHU_DEBUG"))
                        fprintf(stderr, "[cuphu] horiz_k(tile%d,tile%d)=%d  tile_k[%d]=%d\n",
                                idx, nb, k, nb, tile_k[nb]);
                    visited[nb] = true; bfs.push(nb);
                }
            }
            /* bottom neighbor */
            if (br + 1 < ntr) {
                int nb = (br + 1) * ntc + bc;
                if (!visited[nb]) {
                    int k = vert_k(idx, nb);
                    tile_k[nb] = tile_k[idx] + k;
                    if (std::getenv("CUPHU_DEBUG"))
                        fprintf(stderr, "[cuphu] vert_k(tile%d,tile%d)=%d  tile_k[%d]=%d\n",
                                idx, nb, k, nb, tile_k[nb]);
                    visited[nb] = true; bfs.push(nb);
                }
            }
        }
    }

    /* Merge results sequentially (comp_offset must accumulate in tile order). */
    uint32_t comp_offset = 0;
    for (int tr = 0; tr < ntr; ++tr) {
        for (int tc = 0; tc < ntc; ++tc) {
            TileWork& tw = tiles[tr * ntc + tc];

            uint32_t local_max = 0;
            for (uint32_t v : tw.cc) local_max = std::max(local_max, v);

            float phase_add = (float)(tile_k[tr * ntc + tc] * (2.0 * M_PI));

            int pr0 = (tr == 0)       ? 0        : row_ovrlp / 2;
            int pc0 = (tc == 0)       ? 0        : col_ovrlp / 2;
            int pr1 = (tr == ntr - 1) ? tw.tnrow : tw.tnrow - row_ovrlp / 2;
            int pc1 = (tc == ntc - 1) ? tw.tncol : tw.tncol - col_ovrlp / 2;

            for (int r = pr0; r < pr1; ++r) {
                int gr = tw.first_row + r;
                for (int c = pc0; c < pc1; ++c) {
                    int gc = tw.first_col + c;
                    size_t ti = (size_t)r * tw.tncol + c;
                    size_t gi = (size_t)gr * ncol + gc;
                    result->unw[gi]      = tw.unw[ti] + phase_add;
                    result->conncomp[gi] = (tw.cc[ti] == 0) ? 0
                                         : tw.cc[ti] + comp_offset;
                }
            }
            comp_offset += local_max;
        }
    }
    return 0;
}

extern "C"
int cuphu_build_costs_gpu(
    const float           *igram_r,
    const float           *igram_i,
    const float           *corr,
    const float           *mag,
    const short           *weights,
    int                    nrow,
    int                    ncol,
    CuPhuCostMode       cost_mode,
    const CuPhuParams  *params,
    int                    gpu_id,
    void                 **costs_out,
    size_t                *cost_elem_sz
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    size_t npix = (size_t)nrow * ncol;

    std::vector<float> h_phase(npix);
    for (size_t k = 0; k < npix; ++k) {
        float phi = std::atan2(igram_i[k], igram_r[k]);
        if (phi < 0.0f) phi += 6.28318530717958648f;
        h_phase[k] = phi;
    }

    DevArray<float> d_phase(h_phase.data(), npix);
    DevArray<float> d_corr(corr, npix);
    cuphu_wrap_phase(d_phase.get(), (int)npix, 0);

    size_t nrowcost = (size_t)(nrow - 1) * ncol;
    size_t ncolcost = (size_t)nrow * (ncol - 1);

    if (cost_mode == CUPHU_COST_SMOOTH) {
        smoothcostT *d_c = nullptr;
        cuphu_build_smooth_costs_gpu(
            d_phase.get(), d_corr.get(), nullptr, nullptr,
            nrow, ncol, params, params->kperpdpsi, params->kpardpsi,
            &d_c, 0);
        size_t nb = (nrowcost + ncolcost) * sizeof(smoothcostT);
        *costs_out    = std::malloc(nb);
        *cost_elem_sz = sizeof(smoothcostT);
        CUDA_CHECK(cudaMemcpy(*costs_out, d_c, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_c));
    } else {
        costT *d_c = nullptr;
        cuphu_build_defo_costs_gpu(
            d_phase.get(), d_corr.get(), nullptr, nullptr,
            nrow, ncol, params, params->kperpdpsi, params->kpardpsi,
            &d_c, 0);
        size_t nb = (nrowcost + ncolcost) * sizeof(costT);
        *costs_out    = std::malloc(nb);
        *cost_elem_sz = sizeof(costT);
        CUDA_CHECK(cudaMemcpy(*costs_out, d_c, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_c));
    }
    return (int)*cost_elem_sz;
}
