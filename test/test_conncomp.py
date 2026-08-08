"""Tests for connected-component labeling."""

import numpy as np
import pytest

import cuphu
from cuphu._conncomp import _regrow_conncomp_numpy


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def simple_unw() -> np.ndarray:
    """A 20×30 unwrapped phase with two smooth plateaus separated by a jump."""
    unw = np.zeros((20, 30), dtype=np.float32)
    unw[:, 15:] += 10.0     # large jump at column 15 → two components
    return unw


@pytest.fixture()
def simple_corr() -> np.ndarray:
    return np.full((20, 30), 0.8, dtype=np.float32)


# ── CPU fallback tests ────────────────────────────────────────────────────────

def test_two_components(simple_unw: np.ndarray, simple_corr: np.ndarray) -> None:
    out = np.zeros((20, 30), dtype=np.uint32)
    _regrow_conncomp_numpy(
        simple_unw, simple_corr, None, 0.1, 0.01, 32, out)
    # should find exactly 2 components
    unique = np.unique(out)
    unique = unique[unique > 0]
    assert len(unique) == 2


def test_all_masked_returns_zeros(simple_corr: np.ndarray) -> None:
    unw  = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    out  = np.zeros((10, 10), dtype=np.uint32)
    _regrow_conncomp_numpy(unw, simple_corr[:10, :10], mask, 0.1, 0.0, 32, out)
    assert (out == 0).all()


def test_min_frac_filters_small_components() -> None:
    """A single-pixel component is smaller than min_frac → excluded."""
    unw  = np.zeros((20, 20), dtype=np.float32)
    corr = np.full((20, 20), 0.9, dtype=np.float32)
    # create a 1-px isolated island with a huge phase jump
    unw[10, 10] = 100.0
    out = np.zeros((20, 20), dtype=np.uint32)
    # min_frac = 0.05 → min_size = 20 px → isolated pixel filtered out
    _regrow_conncomp_numpy(unw, corr, None, 0.1, 0.05, 32, out)
    assert out[10, 10] == 0   # the outlier pixel has label 0


def test_max_ncomps_limits_output() -> None:
    """max_ncomps=1 means only the largest component gets a label."""
    nrow, ncol = 20, 30
    unw  = np.zeros((nrow, ncol), dtype=np.float32)
    corr = np.full((nrow, ncol), 0.9, dtype=np.float32)
    # create 3 disconnected regions via large phase jumps
    unw[:, 10:20] += 20.0
    unw[:, 20:]   += 40.0
    out = np.zeros((nrow, ncol), dtype=np.uint32)
    _regrow_conncomp_numpy(unw, corr, None, 0.1, 0.0, 1, out)
    unique = np.unique(out)
    assert 1 in unique   # at least one labeled component
    assert 2 not in unique   # max_ncomps=1 prevents label 2


def test_regrow_conncomp_api(simple_unw: np.ndarray, simple_corr: np.ndarray) -> None:
    """High-level regrow_conncomp() function runs without error."""
    cc = cuphu.regrow_conncomp(simple_unw, simple_corr, cost_thresh=0.1)
    assert cc.shape == simple_unw.shape
    assert cc.dtype == np.uint32


# ── GPU tests ─────────────────────────────────────────────────────────────────

def _has_gpu() -> bool:
    try:
        return cuphu.gpu_count() > 0
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


@gpu_only
def test_gpu_conncomp_matches_cpu() -> None:
    """GPU connected-component output should agree with CPU reference.

    Uses a low-coherence gap (not a raw phase jump) to separate the two
    regions: an arbitrary phase-value jump is not guaranteed to survive a
    real wrap/unwrap round trip (a solver has no reason to preserve a jump
    that wraps to something under pi — it will just recover a smooth
    surface, correctly, from its point of view). A coherence gap breaks
    connectivity regardless of what any unwrapper's cycle assignment does.
    """
    nrow, ncol = 20, 30
    corr = np.full((nrow, ncol), 0.8, dtype=np.float32)
    corr[:, 14:16] = 0.0  # decorrelated strip splits the scene in two

    # smooth, unambiguous phase ramp on each side (no wrap/unwrap residue)
    unw = np.zeros((nrow, ncol), dtype=np.float32)
    unw += 0.05 * np.arange(ncol)[None, :]
    igram = np.exp(1j * unw).astype(np.complex64)

    # CPU reference, using the same known-good unw directly
    cpu_out = np.zeros((nrow, ncol), dtype=np.uint32)
    _regrow_conncomp_numpy(unw, corr, None, 0.1, 0.01, 32, cpu_out)

    # GPU result via the full unwrap pipeline
    _, gpu_cc = cuphu.unwrap(igram, corr, nlooks=10.0, cost="smooth")

    # both should find 2 components (labels may differ, counts should match)
    cpu_counts = np.bincount(cpu_out.ravel())[1:]  # exclude 0
    gpu_counts = np.bincount(gpu_cc.ravel())[1:]

    assert len(cpu_counts) == 2
    assert len(cpu_counts) == len(gpu_counts)
    assert sorted(cpu_counts.tolist()) == sorted(gpu_counts.tolist())
