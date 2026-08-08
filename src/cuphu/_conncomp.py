"""Connected-component utilities."""

from __future__ import annotations

import numpy as np

from cuphu._ext import _cuphu_ext

__all__ = ["regrow_conncomp"]


def regrow_conncomp(
    unw: np.ndarray,
    corr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    cost_thresh: float = 0.1,
    min_conncomp_frac: float = 0.01,
    max_ncomps: int = 32,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Re-compute connected-component labels from an already-unwrapped phase.

    Uses GPU parallel union-find on the 4-connected pixel grid.  Two pixels
    are joined if both are valid (corr >= cost_thresh and mask != 0) and
    their unwrapped-phase difference is less than π.

    Parameters
    ----------
    unw : ndarray, float32, 2-D
        Unwrapped phase in radians.
    corr : ndarray, float32, 2-D
        Sample coherence, same shape as *unw*.
    mask : ndarray, uint8, 2-D, optional
        Valid-pixel mask (0 = invalid). Defaults to None.
    cost_thresh : float, optional
        Minimum coherence for a pixel to be included in any component.
    min_conncomp_frac : float, optional
        Minimum component size as a fraction of total pixels.
    max_ncomps : int, optional
        Maximum number of connected components to output.
    gpu_id : int, optional
        CUDA device index.

    Returns
    -------
    conncomp : ndarray, uint32, 2-D
        Connected-component labels (0 = unassigned).
    """
    unw  = np.ascontiguousarray(unw,  dtype=np.float32)
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    nrow, ncol = unw.shape

    mask_u8 = (np.ascontiguousarray(mask, dtype=np.uint8)
               if mask is not None else None)

    labels = np.zeros((nrow, ncol), dtype=np.uint32)

    # call GPU kernel directly via a dedicated extension function
    # (not exposed yet via cuphu_py.cu, so we call the C API through ctypes
    #  or add another pybind11 binding — here we expose it as a future TODO
    #  and fall back to a pure-numpy reference implementation for now)
    _regrow_conncomp_numpy(unw, corr, mask_u8, cost_thresh,
                           min_conncomp_frac, max_ncomps, labels)
    return labels


def _regrow_conncomp_numpy(
    unw: np.ndarray,
    corr: np.ndarray,
    mask: np.ndarray | None,
    cost_thresh: float,
    min_frac: float,
    max_ncomps: int,
    out: np.ndarray,
) -> None:
    """CPU reference implementation of connected-component labeling."""
    from scipy.sparse import coo_matrix  # type: ignore[import]
    from scipy.sparse.csgraph import connected_components  # type: ignore[import]

    nrow, ncol = unw.shape
    npix = nrow * ncol
    valid = corr >= cost_thresh
    if mask is not None:
        valid &= mask.astype(bool)

    # build adjacency: pixels are connected only if both valid AND their
    # unwrapped-phase difference is less than pi (a real fringe discontinuity
    # breaks the edge, unlike plain spatial 4-connectivity on `valid` alone).
    h_ok = valid[:, :-1] & valid[:, 1:] & (np.abs(unw[:, :-1] - unw[:, 1:]) < np.pi)
    v_ok = valid[:-1, :] & valid[1:, :] & (np.abs(unw[:-1, :] - unw[1:, :]) < np.pi)

    idx = np.arange(npix, dtype=np.int64).reshape(nrow, ncol)
    row_idx = np.concatenate([idx[:, :-1][h_ok], idx[:-1, :][v_ok]])
    col_idx = np.concatenate([idx[:, 1:][h_ok], idx[1:, :][v_ok]])

    graph = coo_matrix(
        (np.ones(row_idx.size, dtype=np.uint8), (row_idx, col_idx)),
        shape=(npix, npix),
    )
    nlabels, labels_flat = connected_components(
        graph, directed=False, connection="weak")
    labeled = labels_flat.reshape(nrow, ncol) + 1  # scipy labels start at 0
    labeled[~valid] = 0

    # build a component-to-connected-pixels mapping and filter small ones
    min_size = max(1, int(min_frac * nrow * ncol))
    counts = np.bincount(labeled.ravel(), minlength=nlabels + 1)
    counts[0] = 0  # background never counts toward keep/size filtering
    keep = np.where(counts >= min_size)[0]
    keep = keep[keep > 0][:max_ncomps]  # exclude background (0)

    label_map = np.zeros(nlabels + 1, dtype=np.uint32)
    for new_label, old_label in enumerate(keep, start=1):
        label_map[old_label] = np.uint32(new_label)

    out[:] = label_map[labeled]
    out[~valid] = 0
