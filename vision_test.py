"""
Manuel vision test scripti.
Çalıştır: python vision_test.py
Telefon bağlı ve Block Blast açık olmalı.

Yapacakları:
1. Ekran görüntüsü al
2. Tahta ve parçaları tespit et
3. debug_overlay.png kaydet
4. Terminale tespit sonuçlarını yazdır
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vision_test")


def main() -> None:
    """Vision pipeline'ı test eder."""
    from vision.screen_capture import ScreenCapture
    from vision.calibration import Calibrator, CONFIG_PATH
    from vision.board_detector import BoardDetector
    from vision.piece_detector import PieceDetector
    from vision.debug_overlay import draw_board_overlay, draw_pieces_overlay

    # 1. Bağlantı kontrolü
    print("=" * 50)
    print("Block Blast AI — Vision Test")
    print("=" * 50)

    capture = ScreenCapture()
    if not capture.test_connection():
        print("\nHATA: ADB bağlantısı kurulamadı!")
        print("Kontrol listesi:")
        print("  1. Telefon USB ile bağlı mı?")
        print("  2. USB hata ayıklama açık mı?")
        print("  3. 'adb devices' komutunu terminalde dene")
        sys.exit(1)

    print("ADB bağlantısı başarılı!")
    screen_size = capture.get_screen_size()
    print(f"Ekran boyutu: {screen_size[0]}x{screen_size[1]}")

    # 2. Kalibrasyon
    calibrator = Calibrator(capture)
    if not CONFIG_PATH.exists():
        print("\nKalibrasyon dosyası bulunamadı. İnteraktif kalibrasyon başlatılıyor...")
        config = calibrator.run_interactive()
    else:
        config = calibrator.load_config()
        print(f"Kalibrasyon yüklendi: {CONFIG_PATH}")

    # 3. Ekran görüntüsü al
    print("\nEkran görüntüsü alınıyor...")
    screenshot = capture.capture()
    screenshot.save("calibration_data/test_screenshot.png")
    print("Kaydedildi: calibration_data/test_screenshot.png")

    # 4. Tahta tespiti
    print("\nTahta tespit ediliyor...")
    board_detector = BoardDetector(config)
    board, confidence = board_detector.detect_with_confidence(screenshot)

    print("\nTespit edilen tahta (# = dolu, . = boş):")
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = " ".join("#" if board[r, c] else "." for c in range(8))
        print(f"{r} {row_str}")

    print(f"\nDolu hücre: {int(board.sum())}/64")
    print(f"Ortalama güven: {float(confidence.mean()):.2f}")
    print(f"Min güven: {float(confidence.min()):.2f}")

    # 5. Parça tespiti
    print("\nParçalar tespit ediliyor...")
    piece_detector = PieceDetector(config)
    pieces = piece_detector.detect_pieces(screenshot)
    print(f"Tespit edilen parçalar: {pieces}")

    # 6. Debug overlay
    overlay = draw_board_overlay(screenshot, board, config, confidence)
    overlay = draw_pieces_overlay(overlay, pieces, config)
    overlay.save("calibration_data/debug_overlay.png")
    print("\nDebug overlay kaydedildi: calibration_data/debug_overlay.png")

    print("\n" + "=" * 50)
    print("Vision test tamamlandı!")
    print("=" * 50)


if __name__ == "__main__":
    main()
