"""Vision modülü — tüm bilgisayarlı görü bileşenlerini bir araya getirir."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from vision.screen_capture import ScreenCapture
from vision.calibration import Calibrator
from vision.board_detector import BoardDetector
from vision.piece_detector import PieceDetector
from vision.debug_overlay import draw_board_overlay, draw_pieces_overlay, save_debug_frame

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Tüm vision modüllerini birleştiren ana sınıf. Aşama 4'te kullanılır."""

    def __init__(self, debug: bool = False, device_id: Optional[str] = None) -> None:
        self.debug = debug
        self.capture = ScreenCapture(device_id=device_id)
        self._calibrator = Calibrator(self.capture)
        self._config = self._calibrator.load_config()
        self.board_detector = BoardDetector(self._config)
        self.piece_detector = PieceDetector(self._config)
        self._frame_counter = 0

    def get_game_state(self) -> Optional[dict]:
        """
        Tek çağrıyla tam oyun durumunu döndürür.

        Döndürür:
        {
            "board": np.ndarray (8×8),
            "pieces": list[str | None],  # 3 parça
            "confidence": float,          # Ortalama tespit güveni
            "screenshot": PIL.Image,      # Ham görüntü
        }

        Tespit başarısızsa None döndürür.
        """
        try:
            screenshot = self.capture.capture()
        except Exception as e:
            logger.error("Ekran görüntüsü alınamadı: %s", e)
            return None

        try:
            board, confidence = self.board_detector.detect_with_confidence(screenshot)
        except Exception as e:
            logger.error("Tahta tespiti başarısız: %s", e)
            return None

        try:
            pieces = self.piece_detector.detect_pieces(screenshot)
        except Exception as e:
            logger.error("Parça tespiti başarısız: %s", e)
            pieces = [None, None, None]

        avg_confidence = float(np.mean(confidence))

        if self.debug:
            self._frame_counter += 1
            try:
                draw_board_overlay(screenshot, board, self._config, confidence)
                save_debug_frame(
                    screenshot.copy(), board, pieces, self._frame_counter
                )
            except Exception as e:
                logger.warning("Debug overlay oluşturulamadı: %s", e)

        return {
            "board": board,
            "pieces": pieces,
            "confidence": avg_confidence,
            "screenshot": screenshot,
        }

    def is_game_over(self, screenshot: Optional[Image.Image] = None) -> bool:
        """
        Oyun bitti ekranını tespit eder.
        Renk histogramı ile koyu ekran oranını kontrol eder.
        """
        if screenshot is None:
            try:
                screenshot = self.capture.capture()
            except Exception:
                return False

        img_array = np.array(screenshot)

        brightness = (
            0.299 * img_array[:, :, 0].astype(np.float32)
            + 0.587 * img_array[:, :, 1].astype(np.float32)
            + 0.114 * img_array[:, :, 2].astype(np.float32)
        )

        # Oyun bitti ekranında genellikle karartılmış bir overlay var
        dark_ratio = np.mean(brightness < 30)
        if dark_ratio > 0.6:
            logger.info("Oyun bitti ekranı tespit edildi (koyu oran: %.2f)", dark_ratio)
            return True

        return False

    @property
    def config(self) -> dict:
        """Kalibrasyon ayarlarını döndürür."""
        return self._config
