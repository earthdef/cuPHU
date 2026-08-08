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
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: int | tuple[int, int] = 0,
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
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: int | tuple[int, int] = 0,
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
    ntiles=(1, 1),
    tile_overlap=0,
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
    init : {'mcf', 'mst'}, optional
        Initialization algorithm for the unwrapped phase gradients.
        Defaults to ``'mcf'``.
    mask : array_like, bool/uint8, 2-D, optional
        Binary valid-pixel mask. Zero means invalid. Defaults to None.
    mag : array_like, float32, 2-D, optional
        Interferogram magnitude. Derived from *igram* if None.
    min_conncomp_frac : float, optional
        Minimum connected component size as a fraction of total pixels.
    phase_grad_window : (int, int), optional
        Size of the sliding window for averaging wrapped phase gradients
        in the (perpendicular, parallel) directions.
    ntiles : (int, int), optional
        Number of tiles in (row, column) directions.
    tile_overlap : int or (int, int), optional
        Pixel overlap between adjacent tiles.
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

    # normalize tile_overlap
    if np.ndim(tile_overlap) == 0:
        tile_overlap = (int(tile_overlap), int(tile_overlap))
    row_ovrlp, col_ovrlp = tile_overlap

    # normalize ntiles
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
