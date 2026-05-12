"""Board sınıfı için birim testleri."""

import numpy as np
import pytest

from env.board import Board


class TestBoardPlace:
    """Yerleştirme işlemi testleri."""

    def test_place_single_cell(self) -> None:
        """Tek hücrelik bloğu yerleştir, grid'de 1 adet 1 olsun."""
        board = Board()
        board.place("tek", 0, 0)
        assert board.grid[0, 0] == 1
        assert board.count_filled() == 1

    def test_place_horizontal_3(self) -> None:
        """Yatay 3'lü bloğu bir satıra yerleştir, doğru hücreler dolsun."""
        board = Board()
        board.place("yatay_3", 2, 3)
        assert board.grid[2, 3] == 1
        assert board.grid[2, 4] == 1
        assert board.grid[2, 5] == 1
        assert board.count_filled() == 3

    def test_place_out_of_bounds(self) -> None:
        """Sınır dışı koordinata yerleştirme ValueError fırlatmalı."""
        board = Board()
        with pytest.raises(ValueError):
            board.place("yatay_3", 0, 6)

    def test_place_overlap(self) -> None:
        """Dolu hücrenin üzerine yerleştirme ValueError fırlatmalı."""
        board = Board()
        board.place("tek", 0, 0)
        with pytest.raises(ValueError):
            board.place("tek", 0, 0)


class TestBoardClear:
    """Satır ve sütun silme testleri."""

    def test_row_clear(self) -> None:
        """Bir satırı tamamen doldur, place() sonrasında silinsin ve 1 döndürsün."""
        board = Board()
        board.grid[7, :] = 1
        board.grid[7, 0] = 0
        cleared = board.place("tek", 7, 0)
        assert cleared >= 1
        assert np.sum(board.grid[7, :]) == 0

    def test_col_clear(self) -> None:
        """Bir sütunu tamamen doldur, place() sonrasında silinsin ve 1 döndürsün."""
        board = Board()
        board.grid[:, 0] = 1
        board.grid[0, 0] = 0
        cleared = board.place("tek", 0, 0)
        assert cleared >= 1
        assert np.sum(board.grid[:, 0]) == 0

    def test_simultaneous_row_col_clear(self) -> None:
        """Aynı hamleyle hem satır hem sütun silinebilmeli."""
        board = Board()
        board.grid[7, :] = 1
        board.grid[:, 7] = 1
        board.grid[7, 7] = 0
        cleared = board.place("tek", 7, 7)
        assert cleared >= 2


class TestBoardMetrics:
    """Heuristik metrik testleri."""

    def test_holes_count(self) -> None:
        """Manuel tahta kur, count_holes() beklenen değeri döndürsün."""
        board = Board()
        board.grid[2, 0] = 1
        # Satır 2'de dolu, satır 3-7 boş → 5 hole sütun 0'da
        assert board.count_holes() == 5

    def test_aggregate_height(self) -> None:
        """Manuel tahta kur, aggregate_height() beklenen değeri döndürsün."""
        board = Board()
        board.grid[6, 0] = 1  # yükseklik = 8-6 = 2
        board.grid[4, 3] = 1  # yükseklik = 8-4 = 4
        assert board.aggregate_height() == 2 + 4

    def test_bumpiness(self) -> None:
        """Manuel tahta kur, bumpiness() beklenen değeri döndürsün."""
        board = Board()
        board.grid[6, 0] = 1  # sütun 0 yükseklik = 2
        board.grid[4, 1] = 1  # sütun 1 yükseklik = 4
        # |2-4| + |4-0| + |0-0|*5 = 2 + 4 = 6
        assert board.bumpiness() == 2 + 4


class TestBoardCopy:
    """Kopya bağımsızlık testi."""

    def test_board_copy_independence(self) -> None:
        """copy() sonrası orijinali değiştirmek kopyayı etkilememeli."""
        board = Board()
        board.place("kare_2x2", 0, 0)
        board_copy = board.copy()

        board.place("tek", 4, 4)
        assert board.grid[4, 4] == 1
        assert board_copy.grid[4, 4] == 0
