"""Tahta koordinatı → piksel koordinatı dönüşümü.

'Parça 0'ı tahta konumu (3, 5)'e yerleştir' bilgisini
piksel koordinatlarına çevirir.
"""

from __future__ import annotations

import logging
from typing import Optional

from env.pieces import get_piece_size

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """Oyun koordinatlarını ekran piksel koordinatlarına çevirir."""

    def __init__(self, config: dict) -> None:
        self.config = config

        tl = config["board"]["top_left"]
        br = config["board"]["bottom_right"]
        self.board_left: int = tl[0]
        self.board_top: int = tl[1]
        self.board_w: int = br[0] - tl[0]
        self.board_h: int = br[1] - tl[1]
        self.cell_w: float = self.board_w / 8
        self.cell_h: float = self.board_h / 8

        ptl = config["pieces_area"]["top_left"]
        pbr = config["pieces_area"]["bottom_right"]
        self.pieces_left: int = ptl[0]
        self.pieces_top: int = ptl[1]
        self.pieces_w: int = pbr[0] - ptl[0]
        self.pieces_h: int = pbr[1] - ptl[1]

        self.screen_w: int = config["screen_size"][0]
        self.screen_h: int = config["screen_size"][1]

        self.drag_offset_x: int = config.get("drag_offset_x", 0)
        self.drag_offset_y: int = config.get("drag_offset_y", 0)

    def board_cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """Tahta hücresinin merkez piksel koordinatını döndürür."""
        cell_x = int(self.board_left + col * self.cell_w + self.cell_w / 2)
        cell_y = int(self.board_top + row * self.cell_h + self.cell_h / 2)
        return (cell_x, cell_y)

    def piece_slot_to_pixel(self, slot_index: int) -> tuple[int, int]:
        """Parça slotunun (0, 1, 2) merkez piksel koordinatını döndürür."""
        slot_w = self.pieces_w / 3
        x = int(self.pieces_left + slot_index * slot_w + slot_w / 2)
        y = int(self.pieces_top + self.pieces_h / 2)
        return (x, y)

    def calculate_drag_target(
        self,
        piece_name: str,
        target_row: int,
        target_col: int,
    ) -> tuple[int, int]:
        """
        Parçanın sürükleneceği hedef koordinatı hesaplar.

        Block Blast'ta sürükleme hedefi parçanın merkezinin
        hücre grubu merkeziyle hizalanacağı noktadır.
        """
        p_rows, p_cols = get_piece_size(piece_name)

        center_row = target_row + p_rows / 2
        center_col = target_col + p_cols / 2

        target_x = int(self.board_left + center_col * self.cell_w)
        target_y = int(self.board_top + center_row * self.cell_h)

        target_x += self.drag_offset_x
        target_y += self.drag_offset_y

        return (target_x, target_y)

    def validate_coordinates(self, x: int, y: int) -> bool:
        """Koordinatların ekran sınırları içinde olup olmadığını kontrol eder."""
        if x < 0 or y < 0:
            logger.warning("Negatif koordinat: (%d, %d)", x, y)
            return False
        if x >= self.screen_w or y >= self.screen_h:
            logger.warning(
                "Koordinat ekran dışı: (%d, %d), ekran: %dx%d",
                x, y, self.screen_w, self.screen_h,
            )
            return False
        return True
