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
 * into cuphu_solver.cpp the same way CUPHU_INIT_LAPLACE is -- it computes
 * `unw` here and the caller runs the shared cuphu_conncomp_gpu() afterward.
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
 * @param unw_out  Output unwrapped phase (radians), row-major, nrow*ncol
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
    float               *unw_out
);
