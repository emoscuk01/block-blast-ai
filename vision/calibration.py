"""İlk kurulumda koordinatları ayarlayan kalibrasyon modülü."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from vision.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("calibration_data/config.json")


class Calibrator:
    """Ekran koordinatlarını interaktif olarak kalibre eden sınıf."""

    def __init__(self, screen_capture: ScreenCapture) -> None:
        self.screen_capture = screen_capture

    def run_interactive(self) -> dict:
        """
        Adım adım kalibrasyon sihirbazı:
        1. Ekran görüntüsü al ve kaydet
        2. Kullanıcıdan koordinatları iste
        3. Doğrula ve config.json'a kaydet
        """
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 50)
        print("Block Blast AI — Kalibrasyon Sihirbazı")
        print("=" * 50)

        try:
            screenshot = self.screen_capture.capture()
            screenshot_path = CONFIG_PATH.parent / "screenshot.png"
            screenshot.save(str(screenshot_path))
            screen_w, screen_h = screenshot.size
            print(f"\nEkran görüntüsü kaydedildi: {screenshot_path}")
            print(f"Ekran boyutu: {screen_w}x{screen_h}")
        except Exception as e:
            print(f"\nEkran görüntüsü alınamadı: {e}")
            print("Manuel koordinat girişine devam ediliyor...")
            screen_w = int(input("Ekran genişliği (piksel): "))
            screen_h = int(input("Ekran yüksekliği (piksel): "))

        print(f"\nLütfen screenshot.png dosyasını açın ve aşağıdaki koordinatları girin.")
        print("(Paint, GIMP veya herhangi bir resim düzenleyicide pikselleri görebilirsiniz)\n")

        print("--- TAHTA KOORDİNATLARI ---")
        board_tl_x = int(input("Tahtanın sol üst köşesi X: "))
        board_tl_y = int(input("Tahtanın sol üst köşesi Y: "))
        board_br_x = int(input("Tahtanın sağ alt köşesi X: "))
        board_br_y = int(input("Tahtanın sağ alt köşesi Y: "))

        print("\n--- PARÇA ALANI KOORDİNATLARI ---")
        pieces_tl_x = int(input("Parça alanının sol üst köşesi X: "))
        pieces_tl_y = int(input("Parça alanının sol üst köşesi Y: "))
        pieces_br_x = int(input("Parça alanının sağ alt köşesi X: "))
        pieces_br_y = int(input("Parça alanının sağ alt köşesi Y: "))

        board_w = board_br_x - board_tl_x
        board_h = board_br_y - board_tl_y
        if board_w <= 0 or board_h <= 0:
            raise ValueError("Tahta koordinatları geçersiz (negatif boyut)")

        cell_w = board_w / 8
        cell_h = board_h / 8

        config = {
            "screen_size": [screen_w, screen_h],
            "board": {
                "top_left": [board_tl_x, board_tl_y],
                "bottom_right": [board_br_x, board_br_y],
                "cell_size": [round(cell_w), round(cell_h)],
            },
            "pieces_area": {
                "top_left": [pieces_tl_x, pieces_tl_y],
                "bottom_right": [pieces_br_x, pieces_br_y],
            },
            "drag_offset_x": 0,
            "drag_offset_y": 0,
            "brightness_threshold": 60,
            "calibrated_at": datetime.now().isoformat(),
        }

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\nKalibrasyon kaydedildi: {CONFIG_PATH}")
        print(f"Hücre boyutu: {cell_w:.1f} x {cell_h:.1f} piksel")
        return config

    def load_config(self) -> dict:
        """config.json varsa yükler, yoksa run_interactive() çağırır."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Kalibrasyon yüklendi: %s", CONFIG_PATH)
            return config
        logger.info("Kalibrasyon dosyası bulunamadı, interaktif kalibrasyon başlatılıyor.")
        return self.run_interactive()

    def get_cell_coordinates(self, config: dict) -> list[list[tuple[int, int, int, int]]]:
        """
        Tahta koordinatlarından 8×8 = 64 hücrenin (x, y, w, h) değerlerini hesaplar.
        Her hücre eşit boyutta varsayılır.
        """
        tl = config["board"]["top_left"]
        br = config["board"]["bottom_right"]
        board_x, board_y = tl[0], tl[1]
        board_w = br[0] - tl[0]
        board_h = br[1] - tl[1]
        cell_w = board_w / 8
        cell_h = board_h / 8

        cells: list[list[tuple[int, int, int, int]]] = []
        for row in range(8):
            row_cells: list[tuple[int, int, int, int]] = []
            for col in range(8):
                x = int(board_x + col * cell_w)
                y = int(board_y + row * cell_h)
                w = int(cell_w)
                h = int(cell_h)
                row_cells.append((x, y, w, h))
            cells.append(row_cells)
        return cells
