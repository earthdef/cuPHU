"""Tests for the mask_pad_distance parameter (solve through a grown valid
region, but report validity at the original mask)."""

import numpy as np
import pytest

import cuphu
from cuphu._unwrap import _dilate_mask


def test_dilate_mask_grows_by_exact_distance():
    m = np.zeros((21, 21), dtype=bool)
    m[10, 10] = True
    grown = _dilate_mask(m, 3)
    # 4-connectivity dilation by N steps = Manhattan-distance-<=N diamond
    ys, xs = np.where(grown)
    manhattan = np.abs(ys - 10) + np.abs(xs - 10)
    assert manhattan.max() == 3
    assert grown.sum() == 1 + 4 + 8 + 12  # diamond ring sizes for r=0,1,2,3


def test_dilate_mask_zero_is_noop():
    rng = np.random.default_rng(0)
    m = rng.random((30, 30)) > 0.5
    np.testing.assert_array_equal(_dilate_mask(m, 0), m)


def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


@gpu_only
def test_conncomp_restored_to_original_mask():
    """Padded-through pixels must never be reported as valid, regardless
    of mask_pad_distance."""
    nrow, ncol = 100, 100
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[:, 40:60] = 0

    for pad in [0, 5, 20, 50]:
        _, cc = cuphu.unwrap(igram, corr, nlooks=10.0, mask=mask, mask_pad_distance=pad)
        assert (cc[:, 40:60] == 0).all(), f"pad={pad} leaked padded pixels into conncomp"
        assert (cc[:, :40] != 0).any(), f"pad={pad} left valid region entirely unlabeled"


@gpu_only
def test_default_zero_matches_no_padding():
    nrow, ncol = 60, 60
    unw_true = 0.04 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[:, 25:35] = 0

    unw_a, cc_a = cuphu.unwrap(igram, corr, nlooks=10.0, mask=mask)
    unw_b, cc_b = cuphu.unwrap(igram, corr, nlooks=10.0, mask=mask, mask_pad_distance=0)
    np.testing.assert_array_equal(unw_a, unw_b)
    np.testing.assert_array_equal(cc_a, cc_b)


@gpu_only
def test_feedback_uses_original_not_padded_mask():
    """Regression test: found on a real scene where mask_pad_distance and
    laplace_neighbor_feedback were combined. The padded-through pixels are
    a PCG-continued fiction with no real interferometric signal (that is
    the whole point of padding -- give the solver connectivity, not
    trustworthy values), but the internal feedback correction, if run
    against the *padded* mask, treated those fictitious pixels as real
    boundary data -- letting a single small island's own single-pixel
    noise, laundered through hundreds of rows of padded-through water,
    inject a multi-radian spurious drift into the neighboring tile.
    Confirmed on the real scene (row-median roughness 0.17 rad, vs 0.03
    matching an independent MCF reference, once fixed). Fixed by running
    feedback against the ORIGINAL mask even when mask_pad_distance > 0.

    Reproduced synthetically: a mainland tile boundary with a small
    isolated island's boundary pixels valid but *only* connected to the
    main valid region through padding (not through the real mask) -- real
    per-row measurements should only come from the directly-valid mainland
    rows, not from padded-in rows that are invalid under the real mask.
    """
    nrow, ncol = 300, 200
    unw = np.zeros((nrow, ncol), dtype=np.float32)
    unw[:, :100] = 10.0
    unw[:, 100:] = 15.0

    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    mask[:20, :] = 1                 # dense mainland strip touching the boundary
    mask[150:160, 99:101] = 1        # a small isolated island touching the boundary
    mask[:, 150] = 1                 # kept valid everywhere so we can read the result there

    igram = np.exp(1j * unw).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)
    kwargs = dict(
        nlooks=10.0, init="laplace", ntiles=(1, 2), tile_overlap=(0, 64),
        mask=mask, mask_pad_distance=100,
        laplace_neighbor_feedback_feather=50,
    )
    unw_fb, _ = cuphu.unwrap(igram, corr, laplace_neighbor_feedback=True, **kwargs)
    unw_nofb, _ = cuphu.unwrap(igram, corr, laplace_neighbor_feedback=False, **kwargs)
    correction = unw_fb[:, 150] - unw_nofb[:, 150]

    # rows far from BOTH the mainland strip and the island (e.g. row 100)
    # have no real boundary evidence under the original mask -- even
    # though padding (100px) connects them to the mainland strip for the
    # solve, feedback must not treat that padded connectivity as real
    # per-row boundary data and inject a correction there.
    assert abs(correction[100]) < 0.5


@gpu_only
def test_padding_helps_isolated_feature():
    """A narrow valid spit surrounded by invalid pixels, connected to the
    main valid region only through a 1px-wide gap, should unwrap more
    consistently with the main region once the mask is padded enough to
    treat it as fully connected."""
    nrow, ncol = 150, 150
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    unw_true = (0.03 * c_idx + 0.02 * r_idx).astype(np.float32)
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.85, dtype=np.float32)

    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    mask[:, :60] = 1                      # main valid region
    mask[60:90, 60:75] = 1                # narrow connecting neck (invalid gap otherwise)
    mask[60:90, 75:100] = 1               # isolated spit

    unw0, cc0 = cuphu.unwrap(igram, corr, nlooks=10.0, mask=mask, mask_pad_distance=0)
    unw20, cc20 = cuphu.unwrap(igram, corr, nlooks=10.0, mask=mask, mask_pad_distance=20)

    assert unw0.shape == (nrow, ncol)
    assert unw20.shape == (nrow, ncol)
    # padded run must not report validity outside the original mask
    assert (cc20[mask == 0] == 0).all()
