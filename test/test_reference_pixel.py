"""Tests for reference_pixel: shifting the whole output by an integer
number of 2*pi cycles to match a given pixel's own raw wrapped phase,
reproducing SNAPHU's MCF/MST convention (anchored to pixel (0, 0))."""

import numpy as np
import pytest

import cuphu


def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


def _wrapped_phase_convention(igram: np.ndarray) -> np.ndarray:
    """Matches the C extension's internal atan2-shifted-to-[0,2*pi) phase."""
    phi = np.angle(igram)
    return np.where(phi < 0, phi + 2 * np.pi, phi)


@gpu_only
def test_default_reference_pixel_matches_own_wrapped_phase_for_laplace():
    nrow, ncol = 60, 90
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1)) + 10.0
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace")
    wrapped = _wrapped_phase_convention(igram)
    # exact match at (0, 0), not just congruent -- the whole point of the
    # default reference_pixel=(0, 0) shift
    assert unw[0, 0] == pytest.approx(wrapped[0, 0], abs=1e-4)


@gpu_only
def test_reference_pixel_none_disables_shift():
    nrow, ncol = 60, 90
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1)) + 10.0
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw_shifted, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace")
    unw_raw, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace",
                               reference_pixel=None)
    diff = (unw_shifted - unw_raw)[0, 0]
    n = diff / (2 * np.pi)
    # None should differ from the default only by the (possibly zero)
    # whole-cycle shift the default applies
    assert abs(n - round(n)) < 1e-3


@gpu_only
def test_mcf_reference_pixel_is_a_noop():
    """MCF/MST already anchor to (0, 0) by construction (SNAPHU's
    IntegratePhase()), so the default shift should not change anything."""
    nrow, ncol = 60, 90
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1)) + 10.0
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw_default, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="mcf")
    unw_noshift, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="mcf",
                                   reference_pixel=None)
    np.testing.assert_array_equal(unw_default, unw_noshift)


@gpu_only
def test_custom_reference_pixel():
    nrow, ncol = 60, 90
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1)) + 10.0
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    rp, rc = 30, 45
    unw, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace",
                           reference_pixel=(rp, rc))
    wrapped = _wrapped_phase_convention(igram)
    assert unw[rp, rc] == pytest.approx(wrapped[rp, rc], abs=1e-4)


@gpu_only
def test_reference_pixel_safe_when_masked_invalid():
    """Well-defined even if the reference pixel is masked out -- the
    wrapped phase used is the raw interferogram value, independent of
    mask validity (matches SNAPHU's own C reference, which doesn't check
    mask validity at its (0, 0) anchor either)."""
    nrow, ncol = 60, 90
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1)) + 10.0
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[0, 0] = 0

    unw, cc = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace", mask=mask)
    assert np.isfinite(unw).all()
    assert cc[0, 0] == 0  # still correctly reported invalid
