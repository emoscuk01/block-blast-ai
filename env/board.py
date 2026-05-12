"""8×8 oyun tahtasının tüm temel operasyonlarını içeren modül."""

from __future__ import annotations

import numpy as np

from env.pieces import get_piece_cells, get_piece_size


class Board:
    """8×8 Block Blast oyun tahtası."""

    ROWS: int = 8
    COLS: int = 8

    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

    # ------------------------------------------------------------------
    # Temel işlemler
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Tahtayı sıfırla, boş grid döndür."""
        self.grid = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        return self.grid.copy()

    def copy(self) -> "Board":
        """Derin kopya döndür. Simülasyon için kritik."""
        new_board = Board()
        new_board.grid = np.copy(self.grid)
        return new_board

    # ------------------------------------------------------------------
    # Yerleştirme
    # ------------------------------------------------------------------

    def can_place(self, piece_name: str, row: int, col: int) -> bool:
        """
        Bloğu (row, col) sol üst köşesine yerleştirmek mümkün mü?
        Tahta sınırları ve çakışma kontrolü yapar.
        """
        cells = get_piece_cells(piece_name)
        p_rows, p_cols = len(cells), len(cells[0])

        if row < 0 or col < 0 or row + p_rows > self.ROWS or col + p_cols > self.COLS:
            return False

        for r in range(p_rows):
            for c in range(p_cols):
                if cells[r][c] == 1 and self.grid[row + r, col + c] != 0:
                    return False
        return True

    def place(self, piece_name: str, row: int, col: int) -> int:
        """
        Bloğu yerleştir, doldurulan satır/sütunları sil.
        Silinen satır+sütun toplamını döndürür.
        can_place() False ise ValueError fırlatır.
        """
        if not self.can_place(piece_name, row, col):
            raise ValueError(
                f"'{piece_name}' parçası ({row}, {col}) konumuna yerleştirilemez."
            )

        cells = get_piece_cells(piece_name)
        p_rows, p_cols = len(cells), len(cells[0])

        for r in range(p_rows):
            for c in range(p_cols):
                if cells[r][c] == 1:
                    self.grid[row + r, col + c] = 1

        cleared = self._clear_lines()
        return cleared

    # ------------------------------------------------------------------
    # Satır / sütun silme
    # ------------------------------------------------------------------

    def _clear_lines(self) -> int:
        """Dolu satır ve sütunları tespit edip siler, toplam silinen sayıyı döndürür."""
        rows_to_clear: list[int] = []
        cols_to_clear: list[int] = []

        for r in range(self.ROWS):
            if self.is_full_row(r):
                rows_to_clear.append(r)

        for c in range(self.COLS):
            if self.is_full_col(c):
                cols_to_clear.append(c)

        for r in rows_to_clear:
            self.grid[r, :] = 0

        for c in cols_to_clear:
            self.grid[:, c] = 0

        return len(rows_to_clear) + len(cols_to_clear)

    # ------------------------------------------------------------------
    # Sorgulama
    # ------------------------------------------------------------------

    def get_valid_placements(self, piece_name: str) -> list[tuple[int, int]]:
        """Bu blok için geçerli tüm (row, col) konumlarını döndür."""
        p_rows, p_cols = get_piece_size(piece_name)
        placements: list[tuple[int, int]] = []
        for r in range(self.ROWS - p_rows + 1):
            for c in range(self.COLS - p_cols + 1):
                if self.can_place(piece_name, r, c):
                    placements.append((r, c))
        return placements

    def get_grid(self) -> np.ndarray:
        """Mevcut grid'in kopyasını döndür (dışarıdan değiştirilemez)."""
        return self.grid.copy()

    def is_full_row(self, row: int) -> bool:
        """Satır tamamen dolu mu?"""
        return bool(np.all(self.grid[row, :] == 1))

    def is_full_col(self, col: int) -> bool:
        """Sütun tamamen dolu mu?"""
        return bool(np.all(self.grid[:, col] == 1))

    # ------------------------------------------------------------------
    # Heuristik metrikleri
    # ------------------------------------------------------------------

    def count_holes(self) -> int:
        """
        Hole: bir sütunda, en üstteki dolu hücreden aşağıda kalan boş hücre.
        """
        holes = 0
        for c in range(self.COLS):
            found_filled = False
            for r in range(self.ROWS):
                if self.grid[r, c] == 1:
                    found_filled = True
                elif found_filled:
                    holes += 1
        return holes

    def aggregate_height(self) -> int:
        """
        Her sütundaki en yüksek dolu hücrenin tahtanın altından
        uzaklığını toplar. Yüksek değer = tahta dolmaya yakın.
        """
        total = 0
        for c in range(self.COLS):
            for r in range(self.ROWS):
                if self.grid[r, c] == 1:
                    total += self.ROWS - r
                    break
        return total

    def bumpiness(self) -> int:
        """
        Komşu sütun yükseklikleri arasındaki mutlak fark toplamı.
        """
        heights: list[int] = []
        for c in range(self.COLS):
            h = 0
            for r in range(self.ROWS):
                if self.grid[r, c] == 1:
                    h = self.ROWS - r
                    break
            heights.append(h)

        bump = 0
        for i in range(len(heights) - 1):
            bump += abs(heights[i] - heights[i + 1])
        return bump

    def count_filled(self) -> int:
        """Tahtadaki toplam dolu hücre sayısı."""
        return int(np.sum(self.grid))

    def count_empty(self) -> int:
        """Tahtadaki toplam boş hücre sayısı."""
        return int(self.ROWS * self.COLS - np.sum(self.grid))

    # ------------------------------------------------------------------
    # Gösterim
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Board(dolu={self.count_filled()}, boş={self.count_empty()})"

    def __str__(self) -> str:
        header = "  " + " ".join(str(c) for c in range(self.COLS))
        lines = [header]
        for r in range(self.ROWS):
            row_str = " ".join("#" if self.grid[r, c] else "." for c in range(self.COLS))
            lines.append(f"{r} {row_str}")
        return "\n".join(lines)
