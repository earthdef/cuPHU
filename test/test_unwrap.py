"""Tests for cuphu.unwrap().

These tests do NOT require a GPU to run; they exercise the Python-layer
input validation and the CPU fallback paths only.  GPU-specific tests are
marked with `pytest.mark.gpu` and skipped when no GPU is available.
"""

import os

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


# ── single_tile_reoptimize ──────────────────────────────────────────────────
#
# _reopt_igram/_reopt_corr deliberately use more noise and lower coherence
# than small_igram/small_corr: cuphu's own tile stitching (median 2pi offset
# per tile, no per-region reconciliation, no loop-closure correction) tends
# to already get "nice" data right, which would make these tests pass
# trivially without actually exercising TreeSolve's refinement. This fixture
# was confirmed during development (CUPHU_DEBUG=1) to produce a real,
# non-trivial correction: seeded (pre-TreeSolve) cost ~4.04M -> final
# (post-TreeSolve) cost ~3.96M, a genuine ~2% improvement, not a no-op.

@pytest.fixture()
def reopt_igram() -> np.ndarray:
    """200x200 complex64, noisier/lower-coherence than small_igram -- see
    module note above for why that matters for these specific tests."""
    rng = np.random.default_rng(11)
    nrow, ncol = 200, 200
    r = np.arange(nrow, dtype=np.float32)
    c = np.arange(ncol, dtype=np.float32)
    phase = (0.5 * r[:, None] + 0.6 * c[None, :]
             + rng.normal(scale=1.8, size=(nrow, ncol)).astype(np.float32))
    return np.exp(1j * phase).astype(np.complex64)


@pytest.fixture()
def reopt_corr() -> np.ndarray:
    return np.full((200, 200), 0.25, dtype=np.float32)


@pytest.fixture()
def reopt_conncomp_igram() -> np.ndarray:
    """200x200 complex64, distinct from reopt_igram: this specific
    phase/noise/coherence combination was confirmed during development to
    naturally fragment into one conncomp label per tile pre-reopt (unlike
    reopt_igram, which forms a single region even without reopt for this
    fixture's cost-improvement purposes) -- needed for
    test_reopt_merges_conncomp_across_tiles to exercise the actual gap."""
    rng = np.random.default_rng(5)
    nrow, ncol = 200, 200
    r = np.arange(nrow, dtype=np.float32)
    c = np.arange(ncol, dtype=np.float32)
    phase = (0.15 * r[:, None] + 0.2 * c[None, :]
             + rng.normal(scale=0.5, size=(nrow, ncol)).astype(np.float32))
    return np.exp(1j * phase).astype(np.complex64)


@pytest.fixture()
def reopt_conncomp_corr() -> np.ndarray:
    return np.full((200, 200), 0.6, dtype=np.float32)


@gpu_only
def test_reopt_single_tile_is_noop(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    """single_tile_reoptimize has no effect when the effective tiling is
    (1, 1) -- confirms the ntiles>1 gate in cuphu_unwrap()'s C++ driver."""
    unw_off, cc_off = cuphu.unwrap(small_igram, small_corr, nlooks=10.0,
                                    ntiles=(1, 1), single_tile_reoptimize=False)
    unw_on, cc_on = cuphu.unwrap(small_igram, small_corr, nlooks=10.0,
                                  ntiles=(1, 1), single_tile_reoptimize=True)
    np.testing.assert_array_equal(unw_off, unw_on)
    np.testing.assert_array_equal(cc_off, cc_on)


@gpu_only
def test_reopt_supersedes_neighbor_feedback(
    reopt_igram: np.ndarray, reopt_corr: np.ndarray
) -> None:
    """laplace_neighbor_feedback is a no-op when single_tile_reoptimize is
    True."""
    unw_fb, cc_fb = cuphu.unwrap(
        reopt_igram, reopt_corr, nlooks=4.0, init="laplace",
        ntiles=(2, 2), tile_overlap=4,
        single_tile_reoptimize=True, laplace_neighbor_feedback=True)
    unw_nofb, cc_nofb = cuphu.unwrap(
        reopt_igram, reopt_corr, nlooks=4.0, init="laplace",
        ntiles=(2, 2), tile_overlap=4,
        single_tile_reoptimize=True, laplace_neighbor_feedback=False)
    np.testing.assert_array_equal(unw_fb, unw_nofb)
    np.testing.assert_array_equal(cc_fb, cc_nofb)


@gpu_only
def test_reopt_default_off_unchanged(
    small_igram: np.ndarray, small_corr: np.ndarray
) -> None:
    """Omitting single_tile_reoptimize must match passing it explicitly
    False -- confirms zero behavior change to the default path."""
    unw_default, cc_default = cuphu.unwrap(
        small_igram, small_corr, nlooks=10.0, ntiles=(2, 2), tile_overlap=4)
    unw_explicit, cc_explicit = cuphu.unwrap(
        small_igram, small_corr, nlooks=10.0, ntiles=(2, 2), tile_overlap=4,
        single_tile_reoptimize=False)
    np.testing.assert_array_equal(unw_default, unw_explicit)
    np.testing.assert_array_equal(cc_default, cc_explicit)


@gpu_only
def test_reopt_improves_cost_at_tile_boundary(
    reopt_igram: np.ndarray, reopt_corr: np.ndarray, capfd: pytest.CaptureFixture
) -> None:
    """TreeSolve is a monotonically-non-increasing local search from a
    feasible start, so the post-reopt total cost must be <= the seeded
    (pre-TreeSolve) cost -- and, on this fixture, genuinely lower (not just
    equal), confirming reopt does real refinement work rather than a no-op.
    EvaluateTotalCost has no direct Python binding; captured via the
    existing CUPHU_DEBUG stderr print instead (cuphu_solver.cpp)."""
    os.environ["CUPHU_DEBUG"] = "1"
    try:
        unw, cc = cuphu.unwrap(reopt_igram, reopt_corr, nlooks=4.0,
                                cost="smooth", init="mcf",
                                ntiles=(2, 2), tile_overlap=4,
                                single_tile_reoptimize=True)
    finally:
        del os.environ["CUPHU_DEBUG"]

    assert np.isfinite(unw).all()

    lines = capfd.readouterr().err.splitlines()
    seed_idx = next((i for i, l in enumerate(lines) if "seeded flow cost" in l), None)
    assert seed_idx is not None, "expected a 'seeded flow cost' debug line from the reopt pass"
    # only lines after the seed line can belong to the reopt pass itself
    # (per-tile solves print their own "final total cost" earlier, during
    # tiling) -- if the reopt call's own early-exit fires, no such line
    # follows at all, which should fail loudly here, not silently grab a
    # per-tile line.
    final_after_seed = [l for l in lines[seed_idx:] if "final total cost" in l]
    assert final_after_seed, "expected a 'final total cost' line after the reopt pass"

    seed_cost = float(lines[seed_idx].rsplit(":", 1)[1])
    reopt_final_cost = float(final_after_seed[0].rsplit(":", 1)[1])

    assert reopt_final_cost <= seed_cost
    assert reopt_final_cost < seed_cost * 0.99, (
        "expected a measurable (>1%) improvement on this fixture, got "
        f"seed={seed_cost} final={reopt_final_cost} -- if this fixture "
        "stopped exercising real TreeSolve refinement, it needs revisiting"
    )


@gpu_only
@pytest.mark.parametrize("ntiles", [(2, 2), (2, 3)])
def test_reopt_merges_conncomp_across_tiles(
    ntiles: tuple[int, int],
    reopt_conncomp_igram: np.ndarray,
    reopt_conncomp_corr: np.ndarray,
) -> None:
    """Without reopt, cuphu's tile stitching never merges connected-
    component labels across tile boundaries (comp_offset just concatenates
    per-tile label spaces) -- documents that known gap on a fixture
    confirmed (CUPHU_DEBUG, during development) to fragment into multiple
    labels pre-reopt, and confirms reopt fixes it as a side effect
    (conncomp is fully recomputed over the assembled, re-solved scene)."""
    _, cc_off = cuphu.unwrap(reopt_conncomp_igram, reopt_conncomp_corr,
                              nlooks=8.0, ntiles=ntiles, tile_overlap=16,
                              single_tile_reoptimize=False)
    _, cc_on = cuphu.unwrap(reopt_conncomp_igram, reopt_conncomp_corr,
                             nlooks=8.0, ntiles=ntiles, tile_overlap=16,
                             single_tile_reoptimize=True)
    n_off = np.unique(cc_off).size
    n_on = np.unique(cc_on).size
    assert n_off > 1, (
        "expected today's known gap: multiple conncomp labels pre-reopt "
        f"(got {n_off}) -- fixture may no longer exercise the fragmentation "
        "this test is meant to document"
    )
    assert n_on < n_off, "expected reopt to merge labels across tile boundaries"


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
