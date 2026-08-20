"""Tests for isolated row/column whole-cycle spike correction
(cuphu.unwrap(..., fix_cycle_spikes=True))."""

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

@gpu_only
def test_fix_cycle_spikes_corrects_isolated_row() -> None:
    """A single row shifted by a whole 2*pi cycle from both its immediate
    neighbors, on an otherwise smooth ramp, should be corrected back."""
    nrow, ncol = 60, 40
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.02 * r_idx + 0.01 * c_idx
    unw = ramp.astype(np.float32).copy()
    unw[27, :] -= np.float32(TWO_PI)   # isolated spike row

    unw_out = _cuphu_ext._fix_cycle_spikes_test(unw)

    np.testing.assert_allclose(unw_out, ramp, atol=1e-3)


@gpu_only
def test_fix_cycle_spikes_corrects_isolated_column() -> None:
    nrow, ncol = 40, 60
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.015 * r_idx + 0.02 * c_idx
    unw = ramp.astype(np.float32).copy()
    unw[:, 33] += np.float32(2.0 * TWO_PI)   # isolated spike column

    unw_out = _cuphu_ext._fix_cycle_spikes_test(unw)

    np.testing.assert_allclose(unw_out, ramp, atol=1e-3)


@gpu_only
def test_fix_cycle_spikes_ignores_smooth_ramp() -> None:
    """A row-to-row difference that does not round to a common nonzero
    cycle count on both sides (a genuine smooth gradient) must be left
    untouched -- only a clean isolated spike is a false positive risk."""
    nrow, ncol = 50, 50
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.05 * r_idx + 0.03 * c_idx
    unw = ramp.astype(np.float32)

    unw_out = _cuphu_ext._fix_cycle_spikes_test(unw)

    np.testing.assert_array_equal(unw_out, unw)


@gpu_only
def test_fix_cycle_spikes_ignores_sustained_shift() -> None:
    """A sustained multi-row shifted band (not an isolated single row) must
    not be touched -- neighbors on both sides of the band's edges disagree
    with each other, so no row rounds to a common cycle count with both."""
    nrow, ncol = 60, 40
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.02 * r_idx + 0.01 * c_idx
    unw = ramp.astype(np.float32).copy()
    unw[20:40, :] -= np.float32(TWO_PI)   # a whole shifted band, not isolated

    unw_out = _cuphu_ext._fix_cycle_spikes_test(unw)

    np.testing.assert_array_equal(unw_out, unw)


@gpu_only
def test_fix_cycle_spikes_respects_mask() -> None:
    """Masked pixels must not be sampled into the row/column median, and
    must not be corrected even inside a detected spike row."""
    nrow, ncol = 60, 40
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    ramp = 0.02 * r_idx + 0.01 * c_idx
    unw = ramp.astype(np.float32).copy()
    unw[27, :] -= np.float32(TWO_PI)

    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[27, :10] = 0   # part of the spike row is masked out, >20 still valid

    unw_out = _cuphu_ext._fix_cycle_spikes_test(unw, mask=mask)

    # masked pixels untouched (still show the raw, uncorrected value)
    np.testing.assert_array_equal(unw_out[27, :10], unw[27, :10])
    # valid pixels in the spike row are corrected
    np.testing.assert_allclose(unw_out[27, 10:], ramp[27, 10:], atol=1e-3)


# ── API-level smoke test, via the public cuphu.unwrap() entry point ───────────

@gpu_only
def test_fix_cycle_spikes_default_off_matches_no_kwarg() -> None:
    nrow, ncol = 20, 30
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw_a, cc_a = cuphu.unwrap(igram, corr, nlooks=10.0)
    unw_b, cc_b = cuphu.unwrap(igram, corr, nlooks=10.0, fix_cycle_spikes=False)

    np.testing.assert_array_equal(unw_a, unw_b)
    np.testing.assert_array_equal(cc_a, cc_b)


@gpu_only
def test_fix_cycle_spikes_enabled_runs_without_error() -> None:
    nrow, ncol = 30, 30
    unw_true = 0.04 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.85, dtype=np.float32)

    unw, cc = cuphu.unwrap(igram, corr, nlooks=10.0, fix_cycle_spikes=True)

    assert unw.shape == (nrow, ncol)
    assert cc.shape == (nrow, ncol)
