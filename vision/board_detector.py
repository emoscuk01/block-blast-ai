"""Ekran görüntüsünden 8×8 board matrisini çıkaran modül."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 60


class BoardDetector:
    """Ekran görüntüsünden 8×8 binary tahta matrisi çıkarır."""

    def __init__(self, config: dict) -> None:
        self.config = config
        tl = config["board"]["top_left"]
        br = config["board"]["bottom_right"]
        self.board_x: int = tl[0]
        self.board_y: int = tl[1]
        self.board_w: int = br[0] - tl[0]
        self.board_h: int = br[1] - tl[1]
        self.cell_w: float = self.board_w / 8
        self.cell_h: float = self.board_h / 8
        self.threshold: int = config.get("brightness_threshold", DEFAULT_THRESHOLD)

    def detect(self, screenshot: Image.Image) -> np.ndarray:
        """
        8×8 binary matris döndürür (0=boş, 1=dolu). dtype=np.int8

        Hücrenin %40-%60 bölgesinin medyan parlaklığına bakarak karar verir.
        """
        img_array = np.array(screenshot)
        board = np.zeros((8, 8), dtype=np.int8)

        for row in range(8):
            for col in range(8):
                brightness = self._cell_brightness(img_array, row, col)
                board[row, col] = 1 if brightness > self.threshold else 0

        return board

    def detect_with_confidence(
        self, screenshot: Image.Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (board_matrix, confidence_matrix) döndürür.
        confidence: her hücre için 0.0-1.0 arası güven skoru.
        """
        img_array = np.array(screenshot)
        board = np.zeros((8, 8), dtype=np.int8)
        confidence = np.zeros((8, 8), dtype=np.float32)

        for row in range(8):
            for col in range(8):
                brightness = self._cell_brightness(img_array, row, col)

                if brightness > self.threshold:
                    board[row, col] = 1
                    conf = min(1.0, (brightness - self.threshold) / 100.0)
                else:
                    board[row, col] = 0
                    conf = min(1.0, (self.threshold - brightness) / 60.0)

                confidence[row, col] = max(0.1, conf)

        return board, confidence

    def set_threshold(self, threshold: int) -> None:
        """Parlaklık eşiğini runtime'da değiştirir."""
        self.threshold = threshold
        logger.info("Parlaklık eşiği güncellendi: %d", threshold)

    def _cell_brightness(self, img_array: np.ndarray, row: int, col: int) -> float:
        """Hücrenin merkez bölgesinin medyan parlaklığını hesaplar."""
        cx = int(self.board_x + col * self.cell_w + self.cell_w * 0.5)
        cy = int(self.board_y + row * self.cell_h + self.cell_h * 0.5)

        half_w = int(self.cell_w * 0.1)
        half_h = int(self.cell_h * 0.1)
        half_w = max(half_w, 2)
        half_h = max(half_h, 2)

        y1 = max(0, cy - half_h)
        y2 = min(img_array.shape[0], cy + half_h)
        x1 = max(0, cx - half_w)
        x2 = min(img_array.shape[1], cx + half_w)

        region = img_array[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Parlaklık: 0.299*R + 0.587*G + 0.114*B
        if len(region.shape) == 3 and region.shape[2] >= 3:
            brightness = (
                0.299 * region[:, :, 0].astype(np.float32)
                + 0.587 * region[:, :, 1].astype(np.float32)
                + 0.114 * region[:, :, 2].astype(np.float32)
            )
            return float(np.median(brightness))

        return float(np.median(region))
