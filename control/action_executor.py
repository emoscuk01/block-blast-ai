"""Model kararını ADB hareketine çeviren modül."""

from __future__ import annotations

import logging
import time
from typing import Optional

from control.adb_controller import ADBController
from control.coordinate_mapper import CoordinateMapper

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Model kararını (piece_index, row, col) üçlüsünden gerçek dokunma hareketine çevirir."""

    MOVE_DELAY: float = 1.5
    TURN_DELAY: float = 2.0
    RETRY_COUNT: int = 2

    def __init__(self, adb: ADBController, mapper: CoordinateMapper) -> None:
        self.adb = adb
        self.mapper = mapper

    def execute_action(
        self,
        piece_index: int,
        row: int,
        col: int,
        piece_name: str,
    ) -> bool:
        """
        Tek bir hamleyi gerçekleştirir.
        1. Parça slotunun piksel koordinatını hesapla
        2. Sürükleme hedefini hesapla
        3. ADB drag_piece() ile hareketi yap
        4. Başarısızsa yeniden dene
        """
        piece_x, piece_y = self.mapper.piece_slot_to_pixel(piece_index)
        target_x, target_y = self.mapper.calculate_drag_target(piece_name, row, col)

        if not self.mapper.validate_coordinates(piece_x, piece_y):
            logger.error("Parça koordinatı ekran dışı: (%d, %d)", piece_x, piece_y)
            return False
        if not self.mapper.validate_coordinates(target_x, target_y):
            logger.error("Hedef koordinat ekran dışı: (%d, %d)", target_x, target_y)
            return False

        for attempt in range(1, self.RETRY_COUNT + 1):
            logger.info(
                "Hamle: %s (slot %d) → (%d,%d) | piksel (%d,%d)→(%d,%d) [deneme %d]",
                piece_name, piece_index, row, col,
                piece_x, piece_y, target_x, target_y, attempt,
            )

            success = self.adb.drag_piece(piece_x, piece_y, target_x, target_y)
            if success:
                time.sleep(self.MOVE_DELAY)
                return True

            logger.warning("Hamle başarısız, tekrar deneniyor... (%d/%d)", attempt, self.RETRY_COUNT)
            time.sleep(0.5)

        logger.error(
            "Hamle %d denemede başarısız oldu: %s → (%d,%d)",
            self.RETRY_COUNT, piece_name, row, col,
        )
        return False

    def execute_turn(
        self,
        actions: list[tuple[int, int, int]],
        piece_names: list[str],
    ) -> bool:
        """
        Bir turdaki hamlelerin tamamını sırayla gerçekleştirir.
        Herhangi biri başarısız olursa False döndürür.
        """
        for i, (piece_index, row, col) in enumerate(actions):
            if i >= len(piece_names):
                logger.error("Parça ismi eksik: aksiyon %d", i)
                return False

            piece_name = piece_names[i]
            if not self.execute_action(piece_index, row, col, piece_name):
                return False

            if i < len(actions) - 1:
                time.sleep(0.5)

        time.sleep(self.TURN_DELAY)
        return True
