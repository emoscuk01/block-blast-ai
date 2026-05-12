"""Tespiti görselleştiren debug modülü. Üretim kodunda kullanılmaz."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

DEBUG_FRAMES_DIR = Path("debug_frames")

# Renkler (RGB)
COLOR_GRID = (45, 45, 68)
COLOR_EMPTY = (0, 200, 0)
COLOR_FILLED = (255, 0, 0)
COLOR_LOW_CONF = (255, 255, 0)


def draw_board_overlay(
    screenshot: Image.Image,
    board_matrix: np.ndarray,
    config: dict,
    confidence: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Ekran görüntüsü üzerine tahta grid'ini ve tespit sonuçlarını çizer.
    - Yeşil kare: boş hücre sınırı
    - Kırmızı: dolu hücre
    - Sarı: düşük güvenli hücre
    Sonucu debug_overlay.png olarak kaydeder ve döndürür.
    """
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)

    tl = config["board"]["top_left"]
    br = config["board"]["bottom_right"]
    board_x, board_y = tl[0], tl[1]
    board_w = br[0] - tl[0]
    board_h = br[1] - tl[1]
    cell_w = board_w / 8
    cell_h = board_h / 8

    for row in range(8):
        for col in range(8):
            x1 = int(board_x + col * cell_w)
            y1 = int(board_y + row * cell_h)
            x2 = int(x1 + cell_w)
            y2 = int(y1 + cell_h)

            if board_matrix[row, col] == 1:
                color = COLOR_FILLED
            else:
                color = COLOR_EMPTY

            if confidence is not None and confidence[row, col] < 0.5:
                color = COLOR_LOW_CONF

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            if board_matrix[row, col] == 1:
                draw.rectangle(
                    [x1 + 3, y1 + 3, x2 - 3, y2 - 3],
                    fill=(*COLOR_FILLED, 80),
                )

    output_path = Path("calibration_data/debug_overlay.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info("Debug overlay kaydedildi: %s", output_path)
    return img


def draw_pieces_overlay(
    screenshot: Image.Image,
    detected_pieces: list[Optional[str]],
    config: dict,
) -> Image.Image:
    """Parça alanı üzerine tespit edilen parça isimlerini yazar."""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)

    tl = config["pieces_area"]["top_left"]
    br = config["pieces_area"]["bottom_right"]
    area_w = br[0] - tl[0]
    slot_w = area_w // 3

    for i, piece_name in enumerate(detected_pieces):
        x = tl[0] + i * slot_w + slot_w // 2
        y = tl[1] - 20
        text = piece_name if piece_name else "None"
        draw.text((x - 30, y), text, fill=(255, 255, 255))

    return img


def save_debug_frame(
    screenshot: Image.Image,
    board: np.ndarray,
    pieces: list,
    frame_id: int,
) -> None:
    """debug_frames/ klasörüne tam debug görüntüsünü kaydeder."""
    DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    draw = ImageDraw.Draw(screenshot)
    info_text = f"Frame #{frame_id} | Parçalar: {pieces}"
    draw.text((10, 10), info_text, fill=(255, 255, 0))

    board_text = "Tahta:\n"
    for row in range(8):
        row_str = " ".join("#" if board[row, c] else "." for c in range(8))
        board_text += row_str + "\n"

    draw.text((10, 30), board_text, fill=(255, 255, 255))

    path = DEBUG_FRAMES_DIR / f"frame_{frame_id:04d}.png"
    screenshot.save(str(path))
    logger.debug("Debug frame kaydedildi: %s", path)
