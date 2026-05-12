"""Vision modülü için birim testleri.

ADB gerektirmeyen testler: sentetik görüntülerle board_detector ve
piece_detector'ın temel mantığını doğrular.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vision.board_detector import BoardDetector
from vision.piece_detector import PieceDetector
from vision.calibration import Calibrator, CONFIG_PATH


def _make_test_config() -> dict:
    """Test için sahte kalibrasyon config'i oluşturur."""
    return {
        "screen_size": [400, 800],
        "board": {
            "top_left": [0, 0],
            "bottom_right": [400, 400],
            "cell_size": [50, 50],
        },
        "pieces_area": {
            "top_left": [0, 420],
            "bottom_right": [400, 600],
        },
        "brightness_threshold": 60,
    }


def _make_empty_board_image() -> Image.Image:
    """Tüm hücreleri koyu (boş) olan 400×800 test görüntüsü oluşturur."""
    img = np.zeros((800, 400, 3), dtype=np.uint8)
    img[:, :] = [10, 10, 10]
    return Image.fromarray(img, "RGB")


def _make_full_row_image(row_idx: int) -> Image.Image:
    """Belirtilen satırın hücrelerini parlak yapan test görüntüsü oluşturur."""
    img = np.zeros((800, 400, 3), dtype=np.uint8)
    img[:, :] = [10, 10, 10]
    cell_h = 400 // 8
    y1 = row_idx * cell_h
    y2 = y1 + cell_h
    img[y1:y2, :] = [200, 200, 200]
    return Image.fromarray(img, "RGB")


def test_board_detector_empty_board() -> None:
    """Tüm hücreler boş ekranda board_detector 8×8 sıfır matrisi döndürmeli."""
    config = _make_test_config()
    detector = BoardDetector(config)
    img = _make_empty_board_image()
    board = detector.detect(img)
    assert board.shape == (8, 8)
    assert np.sum(board) == 0, f"Boş tahta beklendi, {np.sum(board)} dolu hücre var"


def test_board_detector_full_row() -> None:
    """Test görüntüsüyle bilinen bir satırı doğru tespit etmeli."""
    config = _make_test_config()
    detector = BoardDetector(config)
    img = _make_full_row_image(3)
    board = detector.detect(img)
    assert board.shape == (8, 8)
    assert np.sum(board[3, :]) == 8, f"3. satır tamamen dolu olmalı, {np.sum(board[3, :])} dolu"
    other_rows = np.delete(board, 3, axis=0)
    assert np.sum(other_rows) == 0, "Diğer satırlar boş olmalı"


def test_piece_detector_known_piece() -> None:
    """PieceDetector'ın piece-shape eşleme mantığını test eder."""
    config = _make_test_config()
    detector = PieceDetector(config)

    shape = np.array([[1, 1, 1]], dtype=np.int8)
    result = detector._shape_to_piece_name(shape)
    assert result == "yatay_3", f"Beklenen 'yatay_3', alınan '{result}'"


def test_calibration_loads() -> None:
    """Config dosyası varsa Calibrator hatasız yüklemeli."""
    test_config = _make_test_config()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    backup = None
    if CONFIG_PATH.exists():
        backup = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(test_config, f)

        from vision.screen_capture import ScreenCapture
        cap = ScreenCapture.__new__(ScreenCapture)
        cap.device_id = None
        calibrator = Calibrator(cap)
        loaded = calibrator.load_config()
        assert "board" in loaded
        assert "pieces_area" in loaded
    finally:
        if backup is not None:
            CONFIG_PATH.write_text(backup, encoding="utf-8")
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()


def test_board_detector_confidence() -> None:
    """detect_with_confidence() güven matrisi döndürmeli."""
    config = _make_test_config()
    detector = BoardDetector(config)
    img = _make_empty_board_image()
    board, confidence = detector.detect_with_confidence(img)
    assert board.shape == (8, 8)
    assert confidence.shape == (8, 8)
    assert np.all(confidence >= 0.0)
    assert np.all(confidence <= 1.0)
