"""Tests for cuphu.unwrap().

These tests do NOT require a GPU to run; they exercise the Python-layer
input validation and the CPU fallback paths only.  GPU-specific tests are
marked with `pytest.mark.gpu` and skipped when no GPU is available.
"""

import numpy as np
import pytest

import cuphu
from cuphu._check import check_cost_mode, check_init_method


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def small_igram(rng: np.random.Generator) -> np.ndarray:
    """50×60 complex64 interferogram with known wrapped phase."""
    nrow, ncol = 50, 60
    # quadratic phase ramp — easy to unwrap
    r = np.arange(nrow, dtype=np.float32)
    c = np.arange(ncol, dtype=np.float32)
    true_phase = 0.1 * r[:, None] + 0.2 * c[None, :]  # linear, few cycles
    igram = np.exp(1j * true_phase).astype(np.complex64)
    return igram


@pytest.fixture()
def small_corr(rng: np.random.Generator) -> np.ndarray:
    """50×60 float32 coherence all equal to 0.9."""
    return np.full((50, 60), 0.9, dtype=np.float32)


# ── input-validation tests (no GPU needed) ────────────────────────────────────

def test_invalid_cost_mode() -> None:
    with pytest.raises(ValueError, match="cost must be"):
        check_cost_mode("bilinear")


def test_invalid_init_method() -> None:
    with pytest.raises(ValueError, match="init must be"):
        check_init_method("dijkstra")


def test_igram_not_2d(small_corr: np.ndarray) -> None:
    igram_3d = np.ones((2, 50, 60), dtype=np.complex64)
    with pytest.raises(ValueError, match="2-D"):
        cuphu.unwrap(igram_3d, small_corr, 10.0)  # type: ignore[arg-type]


def test_shape_mismatch(small_igram: np.ndarray) -> None:
    bad_corr = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        cuphu.unwrap(small_igram, bad_corr, 10.0)


def test_nlooks_too_small(small_igram: np.ndarray, small_corr: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nlooks"):
        cuphu.unwrap(small_igram, small_corr, 0.5)


def test_wrong_igram_dtype(small_corr: np.ndarray) -> None:
    igram_real = np.ones((50, 60), dtype=np.float32)
    with pytest.raises(TypeError, match="complex"):
        cuphu.unwrap(igram_real, small_corr, 10.0)  # type: ignore[arg-type]


def test_wrong_corr_dtype(small_igram: np.ndarray) -> None:
    bad_corr = np.ones((50, 60), dtype=np.int16)
    with pytest.raises(TypeError, match="floating"):
        cuphu.unwrap(small_igram, bad_corr, 10.0)  # type: ignore[arg-type]


# ── GPU tests (skipped when no device is available) ────────────────────────────

def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


@gpu_only
def test_unwrap_returns_correct_shape(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    unw, cc = cuphu.unwrap(small_igram, small_corr, nlooks=10.0)
    assert unw.shape   == small_igram.shape
    assert cc.shape    == small_igram.shape
    assert unw.dtype   == np.float32
    assert cc.dtype    == np.uint32


@gpu_only
def test_unwrap_output_finite(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    unw, _ = cuphu.unwrap(small_igram, small_corr, nlooks=10.0)
    assert np.isfinite(unw).all()


@gpu_only
def test_unwrap_phase_is_consistent(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    """Rewrapping the unwrapped phase should match the input phase."""
    unw, _ = cuphu.unwrap(small_igram, small_corr, nlooks=10.0)
    wrapped_back = np.angle(np.exp(1j * unw))
    input_phase  = np.angle(small_igram)
    diff = np.abs(wrapped_back - input_phase)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    # expect < 0.1 rad residual on this noiseless interferogram
    assert float(np.nanpercentile(diff, 99)) < 0.1


@gpu_only
def test_unwrap_with_mask(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    mask = np.ones((50, 60), dtype=np.uint8)
    mask[20:30, 20:40] = 0  # mask out a rectangular region
    unw, cc = cuphu.unwrap(small_igram, small_corr, nlooks=10.0, mask=mask)
    assert unw.shape == (50, 60)
    # masked pixels should have conncomp label 0
    assert (cc[20:30, 20:40] == 0).all()


@gpu_only
def test_unwrap_cost_modes(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    for mode in ("smooth", "defo"):
        unw, cc = cuphu.unwrap(small_igram, small_corr, nlooks=10.0, cost=mode)
        assert np.isfinite(unw).all()


@gpu_only
def test_unwrap_init_methods(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    for method in ("mst", "mcf"):
        unw, cc = cuphu.unwrap(small_igram, small_corr, nlooks=10.0, init=method)
        assert np.isfinite(unw).all()


@gpu_only
def test_unwrap_pre_allocated_output(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    unw_buf     = np.zeros((50, 60), dtype=np.float32)
    conncomp_buf = np.zeros((50, 60), dtype=np.uint32)
    out_unw, out_cc = cuphu.unwrap(
        small_igram, small_corr, nlooks=10.0,
        unw=unw_buf, conncomp=conncomp_buf)
    # the returned arrays should be the same objects
    assert out_unw is unw_buf
    assert out_cc  is conncomp_buf
    assert np.isfinite(unw_buf).all()


@gpu_only
@pytest.mark.parametrize("ntiles", [(1, 1), (2, 2), (2, 3)])
def test_unwrap_tiled_matches_single(
    ntiles: tuple[int, int],
    small_igram: np.ndarray,
    small_corr: np.ndarray,
) -> None:
    """Tiled result should be close to single-tile result."""
    unw_ref, _ = cuphu.unwrap(small_igram, small_corr, nlooks=10.0,
                                  ntiles=(1, 1))
    unw_tiled, _ = cuphu.unwrap(small_igram, small_corr, nlooks=10.0,
                                    ntiles=ntiles, tile_overlap=4)
    # allow a small absolute difference due to tile-boundary effects
    valid = np.isfinite(unw_ref) & np.isfinite(unw_tiled)
    diff = np.abs(unw_ref[valid] - unw_tiled[valid])
    assert float(np.percentile(diff, 95)) < 0.5  # < 0.5 rad at 95th percentile


@gpu_only
def test_gpu_info() -> None:
    info = cuphu.gpu_info(0)
    assert "name" in info
    assert "total_memory" in info
    assert info["total_memory"] > 0


@gpu_only
def test_nan_input_does_not_crash(small_corr: np.ndarray) -> None:
    igram = np.full((50, 60), np.nan + 0j, dtype=np.complex64)
    # all NaN → all zero phase → unwrap should return finite zeros
    unw, _ = cuphu.unwrap(igram, small_corr, nlooks=10.0)
    assert np.isfinite(unw).all()
