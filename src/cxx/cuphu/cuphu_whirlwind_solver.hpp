/**
 * cuphu_whirlwind_solver.hpp
 *
 * Standalone CPU port of whirlwind's validated `unwrap_linear` pipeline:
 * linear (Costantini/Carballo) per-arc cost from the Lee (1994) multilook
 * phase PDF, a capacity-1 residue network (no ground node, soft masking --
 * masked arcs cost 0 and are freely traversed, matching whirlwind exactly),
 * solved by successive shortest paths (multi-source primal-dual + single-
 * source completion) using Dial's bucket-queue Dijkstra, then integrated to
 * unwrapped phase.
 *
 * This is the CUPHU_INIT_WHIRLWIND init path: fully self-contained, plugged
 * into cuphu_solver.cpp the same way CUPHU_INIT_LAPLACE is. Unlike laplace,
 * conncomp is also computed here (whirlwind's own arc-cost-threshold
 * `grow_components`, on the already-solved network) rather than deferred to
 * the caller's shared cuphu_conncomp_gpu() -- that kernel's phase-diff+
 * coherence-threshold rule doesn't have access to (or match) whirlwind's
 * own cost/capacity data.
 *
 * Reference: whirlwind-insar crates/whirlwind-core/src/{lib,network,ssp,
 * primal_dual,grid,residue,integrate,cost/{mod,lut,lee_pdf,hyp2f1}}.rs.
 * Single-threaded C++ port -- correctness first (see the implementation
 * plan's Milestone 1); no rayon-style parallelism or the fused-state cache
 * optimization ported yet.
 */
#pragma once

/**
 * Unwrap one tile with the whirlwind linear/Carballo solver.
 *
 * @param igram_r  Real part of complex interferogram, row-major, nrow*ncol
 * @param igram_i  Imaginary part, row-major, nrow*ncol
 * @param corr     Coherence magnitude in [0,1], row-major, nrow*ncol
 * @param mask     Optional byte mask (nonzero = valid), row-major, nrow*ncol;
 *                 may be nullptr (all valid)
 * @param nrow     Number of pixel rows
 * @param ncol     Number of pixel columns
 * @param nlooks   Effective number of independent looks (>= 1)
 * @param gpu_id   CUDA device to use for the cost-LUT-lookup fast path
 *                 (only active when built with CUPHU_WW_GPU_COST -- the
 *                 standalone/CPU-only build always uses the CPU path
 *                 regardless of this value). -1 forces the CPU path even
 *                 in a GPU-enabled build.
 * @param unw_out  Output unwrapped phase (radians), row-major, nrow*ncol
 * @param conncomp_algorithm  0 = "linear" (default): cut an edge when a
 *                 FRESH, solve-independent analytical Carballo cost is <=
 *                 cost_threshold=50 (matches whirlwind-insar's
 *                 conncomp_algorithm="linear" / ConnCompParams::default()).
 *                 1 = "snaphu": ambiguity-wiggle reliability test on the
 *                 solved phase (matches whirlwind-insar's
 *                 conncomp_algorithm="snaphu", its own default choice).
 *                 Both are validated bit-exact against whirlwind-insar's own
 *                 implementation; NOT cuphu's shared phase-diff+coherence-
 *                 threshold conncomp used by mcf/mst/laplace.
 * @param reliability_threshold  "snaphu" mode only: cut when
 *                 min(poscost,negcost) <= this (raw units,
 *                 COST_SCALE*nshortcycle^2 scale). Matches whirlwind-insar's
 *                 ww.unwrap() default of round(0.5 * 1e6) = 500000. Ignored
 *                 in "linear" mode.
 * @param thicken_cuts  "snaphu" mode only: SNAPHU ThickenCosts lateral cut-
 *                 strength smoothing, so a thin reliable bridge through a
 *                 wide unreliable band still gets cut. Matches whirlwind-
 *                 insar's ww.unwrap() default of True. Ignored in "linear"
 *                 mode.
 * @param conncomp_out  Output connected-component labels (0 = background),
 *                 row-major, nrow*ncol.
 * @return 0 on success, nonzero on failure (e.g. nrow/ncol < 2)
 */
int cuphu_whirlwind_unwrap(
    const float         *igram_r,
    const float         *igram_i,
    const float         *corr,
    const unsigned char *mask,
    int                  nrow,
    int                  ncol,
    double               nlooks,
    int                  gpu_id,
    float               *unw_out,
    int                  conncomp_algorithm,
    long                 reliability_threshold,
    int                  thicken_cuts,
    unsigned int        *conncomp_out
);
