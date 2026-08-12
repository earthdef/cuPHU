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
    mag: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    phase_grad_window: tuple[int, int] = (7, 7),
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | tuple[int, int] | None = None,
    target_tile_size: int = 1024,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
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
    mag: InputDataset | None = None,
    min_conncomp_frac: float = 0.01,
    phase_grad_window: tuple[int, int] = (7, 7),
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | tuple[int, int] | None = None,
    target_tile_size: int = 1024,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
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
    mag=None,
    min_conncomp_frac=0.01,
    phase_grad_window=(7, 7),
    ntiles=None,
    tile_overlap=None,
    target_tile_size=1024,
    nproc=1,
    tile_cost_thresh=500,
    min_region_size=100,
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

    kperpdpsi, kpardpsi = int(phase_grad_window[0]), int(phase_grad_window[1])

    # call GPU extension
    unw_out, cc_out = _cuphu_ext.unwrap_arrays(
        igram_c64, corr_f32, float(nlooks),
        cost=cost,
        init=init,
        mask=mask_u8,
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
        gpu_id=gpu_id,
    )

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
