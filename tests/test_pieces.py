"""Parça tanımları için birim testleri."""

from env.pieces import (
    PIECES,
    get_piece_cells,
    get_piece_size,
    get_random_pieces,
    get_all_piece_names,
)


def test_all_pieces_valid() -> None:
    """Tüm parça isimleri için get_piece_cells() çalışmalı, liste dönmeli."""
    for name in get_all_piece_names():
        cells = get_piece_cells(name)
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(row, list) for row in cells)


def test_piece_sizes_correct() -> None:
    """Bilinen parçalar için get_piece_size() doğru (rows, cols) döndürmeli."""
    assert get_piece_size("tek") == (1, 1)
    assert get_piece_size("yatay_3") == (1, 3)
    assert get_piece_size("dikey_3") == (3, 1)
    assert get_piece_size("kare_2x2") == (2, 2)
    assert get_piece_size("kare_3x3") == (3, 3)
    assert get_piece_size("L_sag") == (3, 2)


def test_random_pieces_count() -> None:
    """get_random_pieces(3) her zaman uzunluğu 3 olan liste döndürmeli."""
    pieces = get_random_pieces(3)
    assert len(pieces) == 3
    for p in pieces:
        assert p in PIECES


def test_random_pieces_reproducible() -> None:
    """Aynı seed ile get_random_pieces() aynı sonucu vermeli."""
    a = get_random_pieces(5, seed=123)
    b = get_random_pieces(5, seed=123)
    assert a == b
