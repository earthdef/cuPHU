"""Main unwrap() function."""

from __future__ import annotations

import os
from typing import overload

import numpy as np

from cuphu._check import (
    check_bool_or_byte_dtype,
    check_complex_dtype,
    check_cost_mode,
    check_float_dtype,
    check_init_method,
    check_integer_dtype,
    check_shape_match,
)
from cuphu._ext import _cuphu_ext
from cuphu.io import InputDataset, OutputDataset

__all__ = ["unwrap"]

# Default overlap (px) for auto-tiled Laplace runs -- large enough that a
# tile boundary's overlap strip gives a robust median registration estimate
# even if part of it crosses a decorrelated feature, small enough to keep
# redundant (solved-twice) compute a small fraction of tile area.
_DEFAULT_LAPLACE_TILE_OVERLAP = 64


def _auto_laplace_ntiles(nrow, ncol, target_tile_size, row_ovrlp, col_ovrlp):
    """Choose (ntilerow, ntilecol) so every tile has close to the *same*
    edge length in both directions, near target_tile_size (including
    overlap) -- see the *ntiles*/*target_tile_size* docstrings in unwrap()
    for why this matters for ``init='laplace'``.

    Derives one reference edge length from whichever scene dimension is
    larger (rounded to the nearest whole tile count for that axis), then
    applies that same edge length to the other axis too, rather than
    rounding each axis independently against target_tile_size -- two axes
    rounded independently can end up with visibly different actual tile
    sizes depending on how evenly each dimension happens to divide,
    which is really just an accident of the scene's aspect ratio, not a
    deliberate choice. Uses round (not ceil) throughout: ceil-per-axis
    biases toward an extra, disproportionately small "remainder" tile on
    whichever axis doesn't divide evenly (e.g. a 1847px axis at an 874px
    reference edge length ceils to 3 tiles of ~616px rather than rounding
    to 2 tiles of ~924px) -- rounding to nearest avoids that.
    """
    ovrlp = max(row_ovrlp, col_ovrlp)
    step = max(1, target_tile_size - ovrlp)
    max_dim = max(nrow, ncol)
    n_max = max(1, round(max_dim / step))
    edge = max_dim / n_max   # reference tile edge length, shared by both axes

    ntilerow = max(1, round(nrow / edge))
    ntilecol = max(1, round(ncol / edge))
    return ntilerow, ntilecol


def _dilate_mask(valid: np.ndarray, pixels: int) -> np.ndarray:
    """Grow the True (valid) region of a boolean mask by `pixels`, via
    `pixels` rounds of 4-connectivity dilation (a diamond/Manhattan-
    distance-shaped growth, not a true circular Euclidean one -- an
    intentional approximation to avoid a scipy dependency for a feature
    that's a heuristic buffer to begin with, not a precision measurement).
    Pure numpy, no third-party spatial-index dependency, matching cuPHU's
    existing hand-rolled-primitives convention (see e.g. the native bridge
    port's own grid nearest-neighbor search). Cost is O(pixels * n), cheap
    relative to the Laplace solve itself even at `pixels` ~ a few hundred
    on a full-scene array.
    """
    m = valid
    for _ in range(pixels):
        grown = m.copy()
        grown[1:, :] |= m[:-1, :]
        grown[:-1, :] |= m[1:, :]
        grown[:, 1:] |= m[:, :-1]
        grown[:, :-1] |= m[:, 1:]
        m = grown
    return m


# ---------------------------------------------------------------------------
# overloads for static type checking
# ---------------------------------------------------------------------------

@overload
def unwrap(
    igram: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    init: str = "mcf",
    *,
    mask: InputDataset | None = None,
    mask_pad_distance: int = 0,
    mag: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    phase_grad_window: tuple[int, int] = (7, 7),
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | tuple[int, int] | None = None,
    target_tile_size: int = 1024,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    single_tile_reoptimize: bool = False,
    laplace_neighbor_feedback: bool = False,
    laplace_neighbor_feedback_feather: int = 200,
    bridge: bool = False,
    bridge_radius: int = 500,
    bridge_min_num_pixel: int = 14,
    bridge_erosion_size: int = 2,
    bridge_max_boundary_samples: int = 4096,
    reference_pixel: tuple[int, int] | None = (0, 0),
    gpu_id: int = 0,
    unw: OutputDataset,
    conncomp: OutputDataset,
) -> tuple[OutputDataset, OutputDataset]: ...


@overload
def unwrap(
    igram: InputDataset,
    corr: InputDataset,
    nlooks: float,
    cost: str = "smooth",
    init: str = "mcf",
    *,
    mask: InputDataset | None = None,
    mask_pad_distance: int = 0,
    mag: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    phase_grad_window: tuple[int, int] = (7, 7),
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | tuple[int, int] | None = None,
    target_tile_size: int = 1024,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    single_tile_reoptimize: bool = False,
    laplace_neighbor_feedback: bool = False,
    laplace_neighbor_feedback_feather: int = 200,
    bridge: bool = False,
    bridge_radius: int = 500,
    bridge_min_num_pixel: int = 14,
    bridge_erosion_size: int = 2,
    bridge_max_boundary_samples: int = 4096,
    reference_pixel: tuple[int, int] | None = (0, 0),
    gpu_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------------
# implementation
# ---------------------------------------------------------------------------

def unwrap(  # type: ignore[no-untyped-def]
    igram,
    corr,
    nlooks,
    cost="smooth",
    init="mcf",
    *,
    mask=None,
    mask_pad_distance=0,
    mag=None,
    min_conncomp_frac=0.01,
    phase_grad_window=(7, 7),
    ntiles=None,
    tile_overlap=None,
    target_tile_size=1024,
    nproc=1,
    tile_cost_thresh=500,
    min_region_size=100,
    single_tile_reoptimize=False,
    laplace_neighbor_feedback=False,
    laplace_neighbor_feedback_feather=200,
    bridge=False,
    bridge_radius=500,
    bridge_min_num_pixel=14,
    bridge_erosion_size=2,
    bridge_max_boundary_samples=4096,
    reference_pixel=(0, 0),
    gpu_id=0,
    unw=None,
    conncomp=None,
):
    r"""
    Unwrap an interferogram using GPU-accelerated SNAPHU.

    Performs 2-D phase unwrapping using the Statistical-Cost, Network-Flow
    Algorithm for Phase Unwrapping (SNAPHU) [1]_.  Cost computation, phase
    integration, and connected-component labeling run on the GPU; the
    minimum-cost network-flow solver runs on the CPU (it is inherently
    sequential).

    Parameters
    ----------
    igram : array_like, complex64, 2-D
        Complex interferogram. NaN values are replaced with zeros.
    corr : array_like, float32, 2-D
        Sample coherence magnitude in [0, 1]. Same shape as *igram*.
    nlooks : float
        Equivalent number of independent looks (>= 1).
    cost : {'smooth', 'defo', 'topo'}, optional
        Statistical cost mode. Defaults to ``'smooth'``.
    init : {'mcf', 'mst', 'laplace'}, optional
        Initialization algorithm for the unwrapped phase gradients.
        ``'mcf'``/``'mst'`` run SNAPHU's network-flow solver (CPU,
        exact). ``'laplace'`` instead solves a weighted-least-squares
        relaxation via Jacobi-preconditioned CG (GPU, approximate but
        much faster) -- see *ntiles* for why tiling matters for this mode
        on large scenes. Defaults to ``'mcf'``.
    mask : array_like, bool/uint8, 2-D, optional
        Binary valid-pixel mask. Zero means invalid. Defaults to None.
    mask_pad_distance : int, optional
        Grow the valid region of *mask* by this many pixels before solving
        (via ``pixels`` rounds of 4-connectivity dilation), then restore
        the original *mask* for the reported ``conncomp`` (padded pixels
        are excluded from the final output; their raw ``unw`` values are
        left as solved, same as any other masked pixel in this API).

        For ``init='laplace'``: fixes a real failure mode, not just a
        cosmetic one. Narrow or isolated valid features right at a mask
        boundary (e.g. a thin coastal spit next to open water) can become
        tiny, weakly-connected components once masked, where the PCG solve
        and/or bridging has almost nothing reliable to anchor them to --
        confirmed on real data drifting over 100 cycles from the true
        value. Padding the mask before solving gives these features a real
        connection to the rest of the valid region to solve through;
        validated on a real coastal scene reducing the near-boundary
        mismatch (vs. an independent MCF reference) from std=27 rad to
        std=0.7 rad. A starting value of ~64px is reasonable; scale up for
        scenes with wider decorrelated/invalid buffers at valid-region
        edges. Defaults to 0 (no padding, matches prior behavior).

        Also passed internally as ``orig_mask`` (see the C extension's
        ``cuphu_unwrap()`` docstring) so tile-stitching only trusts
        genuinely real boundary data, not the padded-through fiction --
        without this, a tile whose own solve got corrupted through a thin
        padded connection could silently propagate that corruption into
        its entire whole-tile stitching offset, and downstream to every
        tile chained to it. Confirmed on real data: fixed a whole
        tile-column of a real coastal scene that was off from an
        independent MCF reference by several rad on average (isolated
        pixels within it off by 10+ rad), down to matching MCF closely.
    mag : array_like, float32, 2-D, optional
        Interferogram magnitude. Derived from *igram* if None.
    min_conncomp_frac : float, optional
        Minimum connected component size as a fraction of total pixels.
    phase_grad_window : (int, int), optional
        Size of the sliding window for averaging wrapped phase gradients
        in the (perpendicular, parallel) directions.
    ntiles : (int, int) or None, optional
        Number of tiles in (row, column) directions. If None (default):
        for ``init='mcf'``/``'mst'``, defaults to a single tile ``(1, 1)``,
        matching historical behavior. For ``init='laplace'``, defaults to
        an automatically computed tiling that keeps each tile's edge length
        near *target_tile_size* (see below) -- a single huge tile leaves
        the Jacobi-preconditioned CG solve unable to converge on large
        scenes (its iteration count scales with tile edge length), which
        can silently produce whole-cycle errors over large, otherwise
        well-correlated regions. Pass an explicit value to override either
        default, including ``(1, 1)`` to force single-tile Laplace (fine,
        even faster, for scenes already smaller than *target_tile_size*).
    tile_overlap : int or (int, int) or None, optional
        Pixel overlap between adjacent tiles, used to register tiles
        against each other (median offset over the shared region). If
        None (default): 0 for ``init='mcf'``/``'mst'`` (historical
        behavior), 64 for ``init='laplace'`` (needed for the auto-tiling
        above to register tiles correctly -- explicitly pass 0 only if
        also passing ``ntiles=(1, 1)``).
    target_tile_size : int, optional
        Target tile edge length in pixels, including overlap, used only to
        auto-compute *ntiles* when ``init='laplace'`` and *ntiles* is None.
        Empirically, edge lengths of roughly 1000-2000px converge reliably
        within the solver's internal iteration cap without either
        stalling (too large: >~4000px measurably degrades convergence,
        ~8800px can leave whole regions a full cycle wrong) or losing
        registration accuracy (too small: <~700px starts raising
        cycle-disagreement again, from a single tile more often being
        dominated by one bad local feature and from longer inter-tile
        stitching chains). 1024 is a reasonable default across that range;
        tune down for scenes with large decorrelated features, or up for
        speed if a scene is known to be uniformly well-correlated.
    nproc : int, optional
        Maximum number of CPU threads for parallel tile network-flow solves.
    tile_cost_thresh : int, optional
        Cost threshold for determining reliable tile regions.
    min_region_size : int, optional
        Minimum number of pixels in a reliable tile region.
    single_tile_reoptimize : bool, optional
        After tiled unwrapping and stitching, rerun a full CPU network-flow
        solve (SNAPHU's exact TreeSolve, not the GPU-accelerated path) over
        the entire assembled scene as a single tile, seeded from the
        stitched result, to clean up tile-boundary artifacts. Has no effect
        when the effective tiling is ``(1, 1)``.

        cuPHU's own tile stitching (median 2π offset per tile pair,
        propagated via a spanning tree over tile adjacency) is
        coarser than SNAPHU's own per-region secondary-network stitching:
        it applies one constant offset per whole tile (can't correct a
        real discontinuity crossing through the middle of a tile), and has
        no loop-closure correction when the tile grid has cycles (any
        ``ntilerow >= 2 and ntilecol >= 2`` layout). This option is the
        cleanup pass for those cases -- directly analogous to SNAPHU's own
        ``SINGLETILEREOPTIMIZE`` / ``snaphu-py``'s identically-named
        ``single_tile_reoptimize`` parameter. It also merges connected-
        component labels across tile boundaries as a side effect (today,
        without it, the same physical region spanning multiple tiles gets
        distinct label ranges per tile).

        Defaults to ``False``, deliberately diverging from ``snaphu-py``'s
        default of ``True``: this is a full-scene CPU network-flow solve
        that can dominate total wall time on large scenes (the same
        mechanism that made ``snaphu-py``'s reopt pass run for hours on a
        large NISAR scene when left at its default). Enable it for final/
        delivery-quality runs, or when tile-boundary artifacts are visibly
        present; leave it off for speed-sensitive runs.
    laplace_neighbor_feedback : bool, optional
        ``init='laplace'`` only. Refines each internal tile boundary with a
        smoothly-varying (per-row for column boundaries, per-column for row
        boundaries) residual correction on top of the whole-tile bulk
        offset, feathered into the tile interior over
        *laplace_neighbor_feedback_feather* px.

        Targets a failure mode ``single_tile_reoptimize``'s whole-tile
        constant offset can't reach: independently-solved Laplace PCG
        tiles can show a genuinely position-varying mismatch along their
        shared boundary (confirmed on real data: a clean, low-noise ~9 rad
        step localized to specific rows, not present at neighboring rows --
        not explainable by a whole-tile rounding/registration error).
        Cheaper than ``single_tile_reoptimize`` (no full-scene CPU re-solve),
        but a partial fix: on the real boundary this was validated against,
        it reduced the typical row-level mismatch by 5-20%, with the
        remaining residual dominated by genuine per-pixel noise that no
        offset-based correction can remove. Off by default. No effect for
        ``init='mcf'``/``'mst'`` (their whole-tile offset is already exact)
        or when the effective tiling is ``(1, 1)``.
    laplace_neighbor_feedback_feather : int, optional
        Pixels over which the boundary correction above decays to zero
        moving away from the boundary. Only used when
        *laplace_neighbor_feedback* is True.
    bridge : bool, optional
        Reconcile whole-2\ :math:`\pi`-cycle offsets between disconnected
        regions of unwrapped phase (e.g. regions split apart by *mask*) --
        a native GPU/C++ port of isce3's ``bridge_unwrapped_phase()``
        (``isce3.unwrap.bridge_phase``). Labels disconnected regions of
        ``unw != 0`` (8-connectivity), drops small/thin ones
        (*bridge_min_num_pixel*, erosion-based pruning at
        *bridge_erosion_size*), builds a minimum-spanning tree over
        nearest-boundary-point region distances, and applies a per-region
        whole-cycle correction (via AOI-median comparison at
        *bridge_radius*) walking the tree from the largest region outward.
        Defaults to ``False``. v1 has no ramp/deramp support (isce3's
        ``ramp_type``) -- NISAR's own production default is already
        ``ramp_type=None``, so this covers the mode actually used in
        production today. Parameter names/defaults mirror NISAR's
        production ``bridge_*`` runconfig keys for easy migration.
    bridge_radius : int, optional
        AOI half-size (px) for the per-bridge median phase comparison.
        Only used when *bridge* is True.
    bridge_min_num_pixel : int, optional
        Regions smaller than this are dropped before bridging. Only used
        when *bridge* is True.
    bridge_erosion_size : int, optional
        Structuring-element size (px) for erosion-based thin/small-region
        pruning (two-stage: square, then circular). Only used when
        *bridge* is True.
    bridge_max_boundary_samples : int, optional
        Cap on boundary points sampled per region for the nearest-neighbor
        region-pair search, bounding cost regardless of any single
        region's true perimeter. Only used when *bridge* is True.
    reference_pixel : (int, int) or None, optional
        (row, col) to shift the whole output by an integer number of 2*pi
        cycles so that pixel matches its own raw wrapped phase exactly.
        Unwrapped phase is only ever defined up to an arbitrary additive
        constant; SNAPHU's MCF/MST always anchors to (0, 0) (see
        ``phi[0][0] = psi[0][0]`` in ``ext/snaphu/src/snaphu_util.c``),
        while ``'laplace'``'s reference emerges from a global relaxation
        and generally lands elsewhere. Defaults to ``(0, 0)``, matching
        MCF/MST's own convention (a no-op for those methods, since they
        already satisfy it by construction); pass None to disable.
    gpu_id : int, optional
        CUDA device index. Defaults to 0.
    unw : array_like or None, optional
        Pre-allocated output array for the unwrapped phase (float32).
    conncomp : array_like or None, optional
        Pre-allocated output array for connected-component labels (uint32).

    Returns
    -------
    unw : ndarray, float32
        Unwrapped phase in radians.
    conncomp : ndarray, uint32
        Connected-component labels (0 = unassigned).

    References
    ----------
    .. [1] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping
       with use of statistical models for cost functions in nonlinear
       optimization," JOSA A, 18, 338-351 (2001).
    """
    igram   = np.asarray(igram)
    corr    = np.asarray(corr)
    if igram.ndim != 2:
        raise ValueError(f"igram must be 2-D, got ndim={igram.ndim}")

    nrow, ncol = igram.shape
    check_shape_match((nrow, ncol), corr=corr)
    if mask is not None:
        mask = np.asarray(mask)
        check_shape_match((nrow, ncol), mask=mask)
    if mag is not None:
        mag = np.asarray(mag)
        check_shape_match((nrow, ncol), mag=mag)

    check_complex_dtype(igram=igram)
    check_float_dtype(corr=corr)
    if mask is not None:
        check_bool_or_byte_dtype(mask=mask)
    if mag is not None:
        check_float_dtype(mag=mag)

    check_cost_mode(cost)
    check_init_method(init)

    if nlooks < 1.0:
        raise ValueError(f"nlooks must be >= 1, got {nlooks}")

    # normalize tile_overlap -- default depends on init: laplace tiling
    # needs overlap to register tiles (see ntiles below), mcf/mst historically
    # defaulted to none.
    if tile_overlap is None:
        tile_overlap = _DEFAULT_LAPLACE_TILE_OVERLAP if init == "laplace" else 0
    if np.ndim(tile_overlap) == 0:
        tile_overlap = (int(tile_overlap), int(tile_overlap))
    row_ovrlp, col_ovrlp = tile_overlap

    # normalize ntiles -- default depends on init: a single huge tile leaves
    # laplace's PCG solve unable to converge on large scenes (iteration count
    # scales with tile edge length), so auto-tile toward target_tile_size
    # unless the caller passed an explicit ntiles. mcf/mst default unchanged.
    if ntiles is None:
        if init == "laplace":
            ntilerow, ntilecol = _auto_laplace_ntiles(
                nrow, ncol, target_tile_size, row_ovrlp, col_ovrlp)
        else:
            ntilerow, ntilecol = 1, 1
    else:
        ntilerow, ntilecol = int(ntiles[0]), int(ntiles[1])

    # normalize nproc
    if nproc < 1:
        nproc = os.cpu_count() or 1

    # ensure C-contiguous complex64 and float32
    igram_c64 = np.ascontiguousarray(igram, dtype=np.complex64)
    # replace NaN
    nan_mask = ~np.isfinite(igram_c64)
    if nan_mask.any():
        igram_c64 = igram_c64.copy()
        igram_c64[nan_mask] = 0.0

    corr_f32 = np.ascontiguousarray(np.where(np.isfinite(corr), corr, 0.0),
                                    dtype=np.float32)

    mask_u8  = (np.ascontiguousarray(mask, dtype=np.uint8)
                if mask is not None else None)
    mag_f32  = (np.ascontiguousarray(mag, dtype=np.float32)
                if mag is not None else None)

    # mask_pad_distance: solve through a grown valid region (real
    # convergence/connectivity benefit -- see docstring), but only ever
    # report the ORIGINAL mask's validity in the output.
    solve_mask_u8 = mask_u8
    if mask_u8 is not None and mask_pad_distance > 0:
        solve_mask_u8 = _dilate_mask(mask_u8 != 0, int(mask_pad_distance)).astype(np.uint8)

    # Padding + neighbor_feedback interact badly if feedback runs against
    # the *padded* mask: the padded-through pixels are a PCG-continued
    # fiction with no real interferometric signal (that's the whole point
    # of padding -- give the solver connectivity, not trustworthy values),
    # yet feedback's per-row/per-column boundary sampling would treat them
    # as real data. Confirmed on a real scene: this let one small island's
    # single-pixel noise, laundered through ~1000 rows of padded-through
    # water, get reported as directly-measured boundary data and injected
    # as a ~2.5 rad spurious drift into the neighboring tile. Solve with
    # feedback off internally in that case, then reapply feedback
    # separately (same underlying routine) using the ORIGINAL, unpadded
    # mask, so the correction only ever trusts genuinely real boundary
    # pixels -- exactly where mask_pad_distance's own gap-filling logic
    # (bounded fade, not unbounded extrapolation) is designed to help.
    feedback_needs_orig_mask = (
        mask_u8 is not None and mask_pad_distance > 0 and laplace_neighbor_feedback
    )
    internal_feedback = laplace_neighbor_feedback and not feedback_needs_orig_mask

    kperpdpsi, kpardpsi = int(phase_grad_window[0]), int(phase_grad_window[1])

    # call GPU extension
    unw_out, cc_out = _cuphu_ext.unwrap_arrays(
        igram_c64, corr_f32, float(nlooks),
        cost=cost,
        init=init,
        mask=solve_mask_u8,
        orig_mask=mask_u8,
        mag=mag_f32,
        kperpdpsi=kperpdpsi,
        kpardpsi=kpardpsi,
        min_conncomp_frac=float(min_conncomp_frac),
        ntilerow=ntilerow,
        ntilecol=ntilecol,
        tile_rowovrlp=row_ovrlp,
        tile_colovrlp=col_ovrlp,
        tilecostthresh=tile_cost_thresh,
        minregionsize=min_region_size,
        nproc=nproc,
        single_tile_reoptimize=bool(single_tile_reoptimize),
        laplace_neighbor_feedback=bool(internal_feedback),
        laplace_neighbor_feedback_feather=int(laplace_neighbor_feedback_feather),
        bridge=bool(bridge),
        bridge_radius=int(bridge_radius),
        bridge_min_num_pixel=int(bridge_min_num_pixel),
        bridge_erosion_size=int(bridge_erosion_size),
        bridge_max_boundary_samples=int(bridge_max_boundary_samples),
        gpu_id=gpu_id,
    )

    if feedback_needs_orig_mask:
        unw_out = _cuphu_ext._laplace_neighbor_feedback_test(
            np.ascontiguousarray(unw_out, dtype=np.float32),
            mask_u8, ntilerow, ntilecol, row_ovrlp, col_ovrlp,
            int(laplace_neighbor_feedback_feather))

    if mask_u8 is not None and mask_pad_distance > 0:
        # restore the ORIGINAL mask's validity for reporting -- padded
        # pixels solved-through for convergence are not reported as valid.
        cc_out = np.where(mask_u8 != 0, cc_out, 0).astype(cc_out.dtype)

    if reference_pixel is not None:
        # match SNAPHU's own MCF/MST reference convention (see docstring)
        # by shifting the whole output to an integer number of 2*pi
        # cycles from the reference pixel's raw wrapped phase. Uses the
        # same atan2-shifted-to-[0,2*pi) convention as the C extension's
        # own internal h_phase, so this matches exactly what MCF/MST see.
        rp, rc = int(reference_pixel[0]), int(reference_pixel[1])
        ref_phase = float(np.angle(igram_c64[rp, rc]))
        if ref_phase < 0.0:
            ref_phase += 2.0 * np.pi
        n = round((float(unw_out[rp, rc]) - ref_phase) / (2.0 * np.pi))
        if n != 0:
            unw_out = unw_out - np.float32(n * 2.0 * np.pi)

    # write to pre-allocated outputs if provided
    # Use [...] indexing so h5py datasets are written to disk (np.asarray()
    # returns a copy for h5py, making np.copyto a no-op on the file).
    if unw is not None:
        unw[...] = unw_out
        unw_out = unw
    if conncomp is not None:
        conncomp[...] = cc_out.astype(conncomp.dtype)
        cc_out = conncomp

    return unw_out, cc_out
