"""Tests for cuphu.get_effective_looks()."""

import pytest

import cuphu


def test_matches_known_value():
    n_e = cuphu.get_effective_looks(
        range_looks=13, azimuth_looks=16,
        range_spacing=2.33, azimuth_spacing=2.75,
        range_resolution=7.5, azimuth_resolution=6.0,
    )
    assert n_e == pytest.approx(29.616888888888894)


def test_below_naive_product_when_oversampled():
    # spacing < resolution (oversampled data): effective looks < k_r * k_a
    n_e = cuphu.get_effective_looks(
        range_looks=13, azimuth_looks=16,
        range_spacing=2.33, azimuth_spacing=2.75,
        range_resolution=7.5, azimuth_resolution=6.0,
    )
    assert n_e < 13 * 16


def test_equals_naive_product_at_nyquist():
    # spacing == resolution: no oversampling correction, reduces to k_r * k_a
    n_e = cuphu.get_effective_looks(
        range_looks=13, azimuth_looks=16,
        range_spacing=3.0, azimuth_spacing=4.0,
        range_resolution=3.0, azimuth_resolution=4.0,
    )
    assert n_e == pytest.approx(13 * 16)


def test_scales_linearly_with_looks():
    kwargs = dict(range_spacing=2.33, azimuth_spacing=2.75,
                 range_resolution=7.5, azimuth_resolution=6.0)
    n1 = cuphu.get_effective_looks(range_looks=1, azimuth_looks=1, **kwargs)
    n2 = cuphu.get_effective_looks(range_looks=2, azimuth_looks=3, **kwargs)
    assert n2 == pytest.approx(6 * n1)
