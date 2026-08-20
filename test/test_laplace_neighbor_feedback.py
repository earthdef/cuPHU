"""Tests for the Laplace neighbor-feedback tile-boundary refinement."""

import numpy as np
import pytest

import cuphu
from cuphu._ext import _cuphu_ext


def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


# ── ground-truth correctness, via the internal test hook ──────────────────────

@gpu_only
def test_reduces_constant_offset_between_tiles() -> None:
    """Two tiles offset by a constant 5 rad at the boundary; the correction
    should pull the boundary values together."""
    nrow, ncol = 50, 200
    unw = np.zeros((nrow, ncol), dtype=np.float32)
    unw[:, :100] = 10.0
    unw[:, 100:] = 15.0
    mask = np.ones((nrow, ncol), dtype=np.uint8)

    fixed = _cuphu_ext._laplace_neighbor_feedback_test(
        unw, mask, ntilerow=1, ntilecol=2, row_ovrlp=0, col_ovrlp=64, feather_px=50)

    # boundary column pair should now match closely (was a clean 5.0 gap)
    assert abs(fixed[0, 100] - fixed[0, 99]) < 0.5
    # far from the boundary (beyond the feather distance), values are untouched
    np.testing.assert_allclose(fixed[:, :90], unw[:, :90])
    np.testing.assert_allclose(fixed[:, 160:], unw[:, 160:])


@gpu_only
def test_does_not_worsen_boundary_with_local_gradient() -> None:
    """Regression test: an earlier implementation used a multi-column window
    median as the boundary-value estimate. That's fine for a flat signal,
    but with a real local *gradient* across the window (e.g. real
    topography/deformation phase, not just noise), the window's median
    represents its middle column, not the pixel actually at the boundary --
    this improved the wider-neighborhood match while making the *tight*
    single-pixel-pair mismatch worse (the metric that actually defines a
    visible seam). Fixed by using only the single pixel immediately
    adjacent to the boundary. A synthetic case without a gradient does NOT
    reproduce this bug (verified) -- the gradient is essential here."""
    nrow, ncol = 500, 200
    r_idx, c_idx = np.mgrid[0:nrow, 0:ncol]
    # a real local gradient (ramp) present on BOTH sides, continuous across
    # the boundary at col=100 -- i.e. already well-matched, no true offset
    # needed, but with real signal structure a window-median would blur.
    unw = (0.15 * c_idx).astype(np.float32)
    mask = np.ones((nrow, ncol), dtype=np.uint8)

    def tight_mismatch(u):
        return np.abs(u[:, 100] - u[:, 99]).mean()

    before = tight_mismatch(unw)
    fixed = _cuphu_ext._laplace_neighbor_feedback_test(
        unw, mask, ntilerow=1, ntilecol=2, row_ovrlp=0, col_ovrlp=64, feather_px=200)
    after = tight_mismatch(fixed)

    assert after <= before * 1.1  # allow tiny fluctuation, not a regression


@gpu_only
def test_sparse_boundary_data_gets_interpolated_not_skipped() -> None:
    """Regression test: found on a real scene where a tile boundary crosses
    mostly-water with only a few scattered rows of land (small islands)
    touching the boundary pixels directly. The original implementation
    computed a correction only for rows with directly-valid boundary
    pixels and left every other row's correction at exactly zero -- since
    those "have data" rows are sparse and interleaved with "no data" rows,
    the *applied* correction alternated between a real value and zero from
    one row to the next, producing visible horizontal banding across the
    whole feather zone (confirmed via row-median roughness on the real
    scene: 0.17 rad with the bug vs 0.03 rad -- matching the reference --
    once fixed). Fixed by interpolating the correction across rows lacking
    direct boundary data instead of leaving them uncorrected."""
    nrow, ncol = 60, 200
    unw = np.zeros((nrow, ncol), dtype=np.float32)
    unw[:, :100] = 10.0
    unw[:, 100:] = 15.0  # clean 5 rad offset needing correction

    # boundary pixels (cols 99, 100) are valid on only 1 row in 10 --
    # mimicking sparse islands touching a mostly-water tile boundary --
    # but the rest of each tile (e.g. col 150) is valid everywhere, so the
    # feather-zone application still reaches every row.
    mask = np.ones((nrow, ncol), dtype=np.uint8)
    mask[:, 99:101] = 0
    mask[::10, 99:101] = 1

    fixed = _cuphu_ext._laplace_neighbor_feedback_test(
        unw, mask, ntilerow=1, ntilecol=2, row_ovrlp=0, col_ovrlp=64, feather_px=50)

    correction = fixed[:, 110] - unw[:, 110]
    have_rows = correction[::10]
    sparse_rows = np.delete(correction, np.arange(0, nrow, 10))

    # rows without direct boundary data must still receive a real,
    # non-trivial correction (the bug left these at exactly 0)...
    assert np.abs(sparse_rows).min() > 1.0
    # ...and it should be close to the directly-measured rows' correction,
    # not an arbitrary interpolation artifact.
    np.testing.assert_allclose(sparse_rows, np.median(have_rows), atol=1.0)
    # no row-to-row jump should approach the full 5 rad offset -- that
    # alternating on/off pattern was exactly the banding bug.
    assert np.abs(np.diff(correction)).max() < 2.0


@gpu_only
def test_far_from_any_measurement_correction_fades_to_zero() -> None:
    """Regression test: found on a real scene where a tile boundary crosses
    mostly-water, with only one small ~16-row cluster of islands touching
    it and no other data for hundreds of rows in either direction. An
    earlier fix (interpolate/hold across every gap, unbounded) turned that
    one cluster's measurement into a fabricated multi-radian drift spread
    across the whole empty span -- confirmed on the real scene as a
    visible band covering ~250 rows of pure water fed by a single 16-row
    island cluster. Correction must instead fade to zero within a bounded
    distance of the nearest real measurement, leaving genuinely
    unsupported rows uncorrected."""
    nrow, ncol = 400, 200
    unw = np.zeros((nrow, ncol), dtype=np.float32)
    unw[:, :100] = 10.0
    unw[:, 100:] = 15.0  # clean 5 rad offset

    # only rows 190-200 have valid boundary-adjacent pixels; everything
    # else near the boundary is masked out (water), but col 150 (used to
    # read back the applied correction) stays valid everywhere.
    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    mask[:, 150] = 1
    mask[190:200, 99:101] = 1

    fixed = _cuphu_ext._laplace_neighbor_feedback_test(
        unw, mask, ntilerow=1, ntilecol=2, row_ovrlp=0, col_ovrlp=64, feather_px=50)
    correction = fixed[:, 150] - unw[:, 150]

    # far from the measured cluster (>2*SMOOTH_WIN=42 rows away), the
    # bug's unbounded interpolation would otherwise still show a
    # substantial fabricated correction; it must be at or near zero.
    assert abs(correction[0]) < 0.5
    assert abs(correction[-1]) < 0.5
    assert abs(correction[50]) < 0.5
    assert abs(correction[350]) < 0.5


@gpu_only
def test_single_tile_is_noop() -> None:
    nrow, ncol = 40, 40
    unw = np.full((nrow, ncol), 3.5, dtype=np.float32)
    fixed = _cuphu_ext._laplace_neighbor_feedback_test(
        unw, None, ntilerow=1, ntilecol=1, row_ovrlp=0, col_ovrlp=0, feather_px=200)
    np.testing.assert_array_equal(fixed, unw)


# ── API-level smoke tests ──────────────────────────────────────────────────────

@gpu_only
def test_default_off_matches_no_kwarg() -> None:
    nrow, ncol = 30, 30
    unw_true = 0.05 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw_a, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace")
    unw_b, _ = cuphu.unwrap(igram, corr, nlooks=10.0, init="laplace",
                            laplace_neighbor_feedback=False)
    np.testing.assert_array_equal(unw_a, unw_b)


@gpu_only
def test_enabled_runs_without_error() -> None:
    nrow, ncol = 60, 90
    unw_true = 0.04 * np.arange(ncol)[None, :] * np.ones((nrow, 1))
    igram = np.exp(1j * unw_true).astype(np.complex64)
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)

    unw, cc = cuphu.unwrap(
        igram, corr, nlooks=10.0, init="laplace", ntiles=(2, 1),
        laplace_neighbor_feedback=True, laplace_neighbor_feedback_feather=50)
    assert unw.shape == (nrow, ncol)
    assert cc.shape == (nrow, ncol)
