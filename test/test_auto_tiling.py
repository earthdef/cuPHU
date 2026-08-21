"""Tests for auto-tiling heuristics (pure Python, no GPU needed)."""

from cuphu._unwrap import _auto_ntiles_by_target_size


def test_auto_ntiles_targets_edge_length():
    ntilerow, ntilecol = _auto_ntiles_by_target_size(
        18240, 13162, target_tile_size=1024, row_ovrlp=64, col_ovrlp=64)
    # edge lengths should land reasonably close to target_tile_size
    assert 700 < 18240 / ntilerow < 2000
    assert 700 < 13162 / ntilecol < 2000


def test_auto_ntiles_stays_single_tile_on_small_scene():
    # a scene already smaller than target_tile_size shouldn't tile at all
    ntilerow, ntilecol = _auto_ntiles_by_target_size(
        200, 200, target_tile_size=1024, row_ovrlp=64, col_ovrlp=64)
    assert (ntilerow, ntilecol) == (1, 1)


def test_auto_ntiles_respects_aspect_ratio():
    # taller-than-wide scene should get more row tiles than column tiles
    ntilerow, ntilecol = _auto_ntiles_by_target_size(
        18240, 13162, target_tile_size=1024, row_ovrlp=64, col_ovrlp=64)
    assert ntilerow >= ntilecol


def test_auto_ntiles_square_scene_is_square_tiled():
    ntilerow, ntilecol = _auto_ntiles_by_target_size(
        4096, 4096, target_tile_size=1024, row_ovrlp=64, col_ovrlp=64)
    assert ntilerow == ntilecol
