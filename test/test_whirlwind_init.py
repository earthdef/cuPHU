"""Tests for init='whirlwind': a standalone CPU successive-shortest-paths
solver over a linear (Costantini/Carballo) cost, ported from whirlwind-insar's
validated `unwrap_linear` pipeline. Unlike 'mcf'/'mst'/'laplace' it does not
share cuphu's GPU cost table, ground node, or hard-mask machinery -- see
cuphu_whirlwind_solver.hpp and the implementation plan for why.

Where the `whirlwind` Python package is importable, these tests compare
directly against it: since this port matches whirlwind's own algorithm and
cost model (not cuphu's), that -- not cuphu's other init methods -- is the
correctness oracle. Cross-checks against 'mcf' are included too, but looser
(different cost model; agreement is expected to be good, not exact).
"""
import numpy as np
import pytest

import cuphu

try:
    import whirlwind as _ww
    HAVE_WHIRLWIND = True
except ImportError:
    HAVE_WHIRLWIND = False

TWO_PI = 2 * np.pi


def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")
needs_whirlwind = pytest.mark.skipif(not HAVE_WHIRLWIND, reason="whirlwind package not installed")


def _cycle_agreement(a: np.ndarray, b: np.ndarray, valid: np.ndarray):
    """% of `valid` pixels where a and b agree exactly after removing the
    arbitrary whole-scene 2pi offset between two independent MCF solves."""
    diff = (a - b)[valid]
    n_best = round(float(np.median(diff)) / TWO_PI)
    resid = diff - n_best * TWO_PI
    k = np.round(diff / TWO_PI - n_best).astype(np.int64)
    disagree_pct = float((k != 0).mean() * 100.0)
    return n_best, float(resid.std()), disagree_pct


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(101)


def _synth_igram(nrow: int, ncol: int, gamma: float, nlooks: int, rng: np.random.Generator,
                  ramp_scale: float = 0.15):
    """Multilook-averaged synthetic interferogram with a smooth ramp truth
    and Lee-PDF-consistent coherence noise (mixes true and random phasors,
    matching the statistical model the Carballo cost itself assumes)."""
    yy, xx = np.mgrid[0:nrow, 0:ncol]
    truth = (ramp_scale * (xx + 0.7 * yy)
             + 3.0 * np.sin(xx / 12.0) * np.cos(yy / 9.0)).astype(np.float32)
    acc = np.zeros((nrow, ncol), dtype=np.complex128)
    for _ in range(nlooks):
        noise_phase = rng.uniform(-np.pi, np.pi, size=(nrow, ncol))
        look = gamma * np.exp(1j * truth) + (1 - gamma) * np.exp(1j * noise_phase)
        acc += look / np.abs(look)
    igram = (acc / nlooks).astype(np.complex64)
    corr = np.abs(acc / nlooks).astype(np.float32)
    return truth, igram, corr


# ── basic smoke tests ────────────────────────────────────────────────────────

@gpu_only
def test_whirlwind_init_listed_as_valid():
    from cuphu._check import check_init_method
    check_init_method("whirlwind")  # must not raise


@gpu_only
def test_whirlwind_init_output_finite(rng: np.random.Generator):
    _, igram, corr = _synth_igram(50, 60, gamma=0.85, nlooks=8, rng=rng)
    unw, cc = cuphu.unwrap(igram, corr, nlooks=8.0, init="whirlwind")
    assert unw.shape == (50, 60)
    assert np.isfinite(unw).all()
    assert cc.max() >= 1


@gpu_only
def test_whirlwind_init_rejects_non_smooth_cost(rng: np.random.Generator):
    _, igram, corr = _synth_igram(50, 60, gamma=0.85, nlooks=8, rng=rng)
    with pytest.raises(Exception):
        cuphu.unwrap(igram, corr, nlooks=8.0, init="whirlwind", cost="defo")


# ── agreement with Python whirlwind (the actual correctness oracle here) ────

@gpu_only
@needs_whirlwind
@pytest.mark.parametrize("nrow,ncol,nlooks,gamma,seed", [
    (64, 64, 8, 0.85, 1),
    (80, 100, 12, 0.8, 2),
    (64, 64, 4, 0.6, 3),  # noisier / fewer looks
])
def test_matches_python_whirlwind(nrow, ncol, nlooks, gamma, seed):
    rng = np.random.default_rng(seed)
    _, igram, corr = _synth_igram(nrow, ncol, gamma=gamma, nlooks=nlooks, rng=rng)

    unw_cuphu, _ = cuphu.unwrap(igram, corr, nlooks=float(nlooks), init="whirlwind")
    unw_py, _ = _ww.unwrap(igram, corr, nlooks=float(nlooks))

    valid = np.isfinite(unw_cuphu) & np.isfinite(unw_py)
    _, resid_std, disagree_pct = _cycle_agreement(unw_cuphu, unw_py, valid)
    # Same algorithm, same cost model, independent C++ vs Rust implementation:
    # expect near-exact agreement, not just "close".
    assert disagree_pct < 0.5, f"cycle-disagreement {disagree_pct:.3f}% vs python whirlwind"
    assert resid_std < 0.05, f"residual std {resid_std:.4f} rad vs python whirlwind"


@gpu_only
@needs_whirlwind
def test_matches_python_whirlwind_with_mask():
    rng = np.random.default_rng(4)
    nrow, ncol = 48, 48
    _, igram, corr = _synth_igram(nrow, ncol, gamma=0.85, nlooks=6, rng=rng)
    mask = np.ones((nrow, ncol), dtype=bool)
    mask[:, ncol // 2] = False  # disconnect left/right halves

    unw_cuphu, _ = cuphu.unwrap(igram, corr, nlooks=6.0, init="whirlwind", mask=mask)
    unw_py, _ = _ww.unwrap(igram, corr, nlooks=6.0, mask=mask)

    valid = mask & np.isfinite(unw_cuphu) & np.isfinite(unw_py)
    _, resid_std, disagree_pct = _cycle_agreement(unw_cuphu, unw_py, valid)
    assert disagree_pct < 0.5
    assert resid_std < 0.05


# ── cross-check against cuphu's own mcf (looser: different cost model) ──────

@gpu_only
def test_broadly_agrees_with_mcf(rng: np.random.Generator):
    _, igram, corr = _synth_igram(64, 64, gamma=0.85, nlooks=8, rng=rng)
    unw_ww, cc_ww = cuphu.unwrap(igram, corr, nlooks=8.0, init="whirlwind")
    unw_mcf, cc_mcf = cuphu.unwrap(igram, corr, nlooks=8.0, init="mcf")

    both = (cc_ww > 0) & (cc_mcf > 0)
    assert both.mean() > 0.5, "expected most of this clean scene to be trusted by both"
    _, resid_std, disagree_pct = _cycle_agreement(unw_ww, unw_mcf, both)
    # Different cost models (linear/Carballo vs SNAPHU quadratic-smooth): not
    # expected to be bit-exact, but should mostly agree on a clean scene.
    assert disagree_pct < 5.0, f"cycle-disagreement {disagree_pct:.2f}% vs mcf"


# ── multi-tile: no solver-side integration work expected (see the
# implementation plan's Milestone 3), so this pins that it actually holds. ──

@gpu_only
@pytest.mark.parametrize("ntiles", [(1, 1), (2, 2)])
def test_tiled_matches_single_tile(ntiles, rng: np.random.Generator):
    _, igram, corr = _synth_igram(120, 100, gamma=0.85, nlooks=8, rng=rng)
    unw_ref, _ = cuphu.unwrap(igram, corr, nlooks=8.0, init="whirlwind", ntiles=(1, 1))
    unw_tiled, _ = cuphu.unwrap(igram, corr, nlooks=8.0, init="whirlwind",
                                 ntiles=ntiles, tile_overlap=16, nproc=1)
    valid = np.isfinite(unw_ref) & np.isfinite(unw_tiled)
    _, resid_std, disagree_pct = _cycle_agreement(unw_ref, unw_tiled, valid)
    assert disagree_pct < 1.0
    assert resid_std < 0.1


# ── conncomp_min_coherence ──────────────────────────────────────────────────

@needs_whirlwind
@pytest.mark.parametrize("nlooks", [1.0, 4.0, 7.43, 16.0, 100.0, 1000.0])
def test_min_coherence_helpers_match_python_whirlwind(nlooks):
    from cuphu._unwrap import _conncomp_min_coherence_auto, _reliability_from_coherence

    assert _conncomp_min_coherence_auto(nlooks) == pytest.approx(
        _ww.conncomp_min_coherence_auto(nlooks))
    for gamma in [1e-4, 0.05, 0.1, 0.3, 0.9, 0.9999]:
        assert _reliability_from_coherence(gamma, nlooks) == pytest.approx(
            _ww.conncomp_reliability_from_coherence(gamma, nlooks))


@gpu_only
def test_min_coherence_overrides_reliability(rng: np.random.Generator):
    from cuphu._unwrap import _reliability_from_coherence

    _, igram, corr = _synth_igram(64, 64, gamma=0.6, nlooks=4, rng=rng)
    kw = dict(nlooks=4.0, init="whirlwind", conncomp_algorithm="snaphu")
    _, cc_gamma = cuphu.unwrap(igram, corr, conncomp_min_coherence=0.3,
                               conncomp_reliability=0.5, **kw)
    _, cc_rel = cuphu.unwrap(igram, corr,
                             conncomp_reliability=_reliability_from_coherence(0.3, 4.0), **kw)
    np.testing.assert_array_equal(cc_gamma, cc_rel)


@gpu_only
@needs_whirlwind
def test_min_coherence_auto_matches_python_whirlwind():
    rng = np.random.default_rng(5)
    _, igram, corr = _synth_igram(64, 64, gamma=0.6, nlooks=4, rng=rng)

    _, cc_cuphu = cuphu.unwrap(igram, corr, nlooks=4.0, init="whirlwind",
                               conncomp_algorithm="snaphu", conncomp_min_coherence="auto")
    _, cc_py = _ww.unwrap(igram, corr, nlooks=4.0, conncomp_min_coherence="auto")
    np.testing.assert_array_equal(cc_cuphu > 0, cc_py > 0)


@gpu_only
@pytest.mark.parametrize("value", ["bogus", 0.0, 1.0])
def test_min_coherence_rejects_invalid(value, rng: np.random.Generator):
    _, igram, corr = _synth_igram(16, 16, gamma=0.9, nlooks=4, rng=rng)
    with pytest.raises(ValueError, match="conncomp_min_coherence"):
        cuphu.unwrap(igram, corr, nlooks=4.0, init="whirlwind",
                     conncomp_algorithm="snaphu", conncomp_min_coherence=value)
