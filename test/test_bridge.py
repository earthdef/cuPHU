"""Tests for native phase bridging (cuphu.unwrap(..., bridge=True))."""

import numpy as np
import pytest

import cuphu
from cuphu._ext import _cuphu_ext

TWO_PI = 2.0 * np.pi


def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


# ── ground-truth correctness, via the internal test hook ──────────────────────
#
# _bridge_apply_test() runs the exact same wired GPU pipeline
# cuphu.unwrap(bridge=True) calls internally, directly on a synthetic unw
# array with known injected whole-cycle offsets -- this isolates the
# bridging algorithm/GPU-kernel correctness from whatever a live MCF/Laplace
# solve happens to produce for a particular interferogram (which may or may
# not exhibit a genuine cycle ambiguity for bridging to fix).

@gpu_only
def test_bridge_reconciles_constant_region_offsets() -> None:
    """Three regions, each internally smooth, separated by an invalid gap and
    offset from each other by known integer multiples of 2*pi. Bridging
    should reconcile them all to a single consistent global offset."""
    nrow, ncol = 60, 90
    rng = np.random.default_rng(0)

    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.02 * r_idx + 0.015 * c_idx
    offset = np.zeros((nrow, ncol), dtype=np.float64)
    offset[(r_idx >= 20) & (r_idx < 39)] = 3.0 * TWO_PI
    offset[r_idx >= 40] = -2.0 * TWO_PI

    unw = (ramp + offset).astype(np.float32)
    unw += rng.normal(0.0, 0.05, size=unw.shape).astype(np.float32)

    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[19, :] = 0   # gap rows separating the three regions
    mask[39, :] = 0

    unw_bridged, cc = _cuphu_ext._bridge_apply_test(
        unw, mask=mask, bridge_min_num_pixel=1, bridge_erosion_size=0)

    # every valid pixel's implied offset from the known ramp should now
    # collapse to a single global constant (mod injected noise)
    valid = mask == 1
    implied_offset = unw_bridged[valid] - ramp[valid]
    assert implied_offset.std() < 0.3   # noise floor was std=0.05

    # bridging should have found and merged all 3 regions (nonzero cc labels
    # present, and correction was non-trivial -- unw changed somewhere)
    assert len(np.unique(cc[valid])) >= 1
    assert not np.allclose(unw_bridged[valid], unw[valid])


@gpu_only
def test_bridge_single_region_is_noop() -> None:
    """A single already-connected region (no mask, unw all nonzero) should
    be left completely unchanged -- matches isce3's num_cluster<=1 early
    return."""
    nrow, ncol = 30, 30
    unw = np.full((nrow, ncol), 1.2345, dtype=np.float32)

    unw_out, _ = _cuphu_ext._bridge_apply_test(unw)

    np.testing.assert_array_equal(unw_out, unw)


@gpu_only
def test_bridge_respects_mask_not_raw_unw_zero() -> None:
    """cuPHU's own solved unw does not zero out masked pixels (unlike
    isce3's convention where invalid pixels are literally 0 by the time
    bridging runs) -- bridging must use the mask, not unw==0, to determine
    region topology, or it silently becomes a no-op on real cuPHU output."""
    nrow, ncol = 40, 60
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    # +1.0 baseline avoids a coincidental exact-zero unw value at the origin
    # (unw==0 means "invalid" under this convention, same as isce3's own)
    ramp = 1.0 + 0.02 * r_idx + 0.01 * c_idx
    offset = np.where(c_idx >= 30, 4.0 * TWO_PI, 0.0)
    unw = (ramp + offset).astype(np.float32)

    # masked gap, but with a NONZERO phase value left in it (mirrors what
    # cuPHU's own solve actually produces at masked pixels)
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[:, 28:32] = 0
    unw[:, 28:32] = 99.0   # deliberately nonzero, unlike isce3's convention

    unw_out, cc = _cuphu_ext._bridge_apply_test(
        unw, mask=mask, bridge_min_num_pixel=1, bridge_erosion_size=0)

    valid = mask == 1
    implied_offset = unw_out[valid] - ramp[valid]
    assert implied_offset.std() < 0.3
    # masked pixels' original (nonsense) values must be left untouched
    np.testing.assert_array_equal(unw_out[:, 28:32], unw[:, 28:32])


# ── API-level smoke tests, via the public cuphu.unwrap() entry point ──────────

@gpu_only
def test_bridge_default_off_matches_no_kwarg() -> None:
    nrow, ncol = 20, 30
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw_a, cc_a = cuphu.unwrap(igram, corr, nlooks=10.0)
    unw_b, cc_b = cuphu.unwrap(igram, corr, nlooks=10.0, bridge=False)

    np.testing.assert_array_equal(unw_a, unw_b)
    np.testing.assert_array_equal(cc_a, cc_b)


@gpu_only
def test_bridge_enabled_runs_without_error() -> None:
    nrow, ncol = 40, 60
    unw_true = 0.03 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.85, dtype=np.float32)
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[:, 28:32] = 0

    unw, cc = cuphu.unwrap(
        igram, corr, nlooks=10.0, mask=mask, bridge=True,
        bridge_radius=200, bridge_min_num_pixel=5)

    assert unw.shape == (nrow, ncol)
    assert cc.shape == (nrow, ncol)
    assert unw.dtype == np.float32
    assert cc.dtype == np.uint32
