"""Ekrandaki 3 bloğu tespit edip pieces.py isimleriyle eşleyen modül.

İki yaklaşım implement edilmiştir:
  A — Şablon eşleme (cv2.matchTemplate)
  B — Şekil tespiti (kontur analizi + IoU karşılaştırma, tercih edilen)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from env.pieces import PIECES, get_piece_cells, get_piece_size

logger = logging.getLogger(__name__)

MIN_IOU_THRESHOLD = 0.5


class PieceDetector:
    """Ekrandaki 3 blok slotunu tespit eder ve isimlendirmeye çalışır."""

    def __init__(self, config: dict) -> None:
        self.config = config
        tl = config["pieces_area"]["top_left"]
        br = config["pieces_area"]["bottom_right"]
        self.area_x: int = tl[0]
        self.area_y: int = tl[1]
        self.area_w: int = br[0] - tl[0]
        self.area_h: int = br[1] - tl[1]
        self.threshold: int = config.get("brightness_threshold", 60)

        self._piece_shapes: dict[str, np.ndarray] = {}
        for name, cells in PIECES.items():
            self._piece_shapes[name] = np.array(cells, dtype=np.int8)

    def detect_pieces(self, screenshot: Image.Image) -> list[Optional[str]]:
        """
        3 elemanlı liste döndürür.
        Her eleman: pieces.py'daki parça ismi veya None (boş slot).
        Önce Yaklaşım B'yi dener, güven düşükse Yaklaşım A'ya geçer.
        """
        results: list[Optional[str]] = []

        for slot_idx in range(3):
            region = self._extract_piece_region(screenshot, slot_idx)
            name = self._detect_by_shape(region)
            if name is None:
                name = self._match_template(region)
            results.append(name)

        return results

    def _extract_piece_region(
        self, screenshot: Image.Image, slot_index: int
    ) -> np.ndarray:
        """Config'den parça alanını alır, 3 slot'a böler, ilgili bölgeyi döndürür."""
        slot_w = self.area_w // 3
        x = self.area_x + slot_index * slot_w
        y = self.area_y
        w = slot_w
        h = self.area_h

        cropped = screenshot.crop((x, y, x + w, y + h))
        return np.array(cropped)

    def _detect_by_shape(self, piece_region: np.ndarray) -> Optional[str]:
        """
        Yaklaşım B: Renk filtresiyle parça piksellerini bulur,
        binary grid'e çevirir, pieces.py şekilleriyle IoU karşılaştırır.
        """
        if piece_region.size == 0:
            return None

        # Parlaklık haritası
        if len(piece_region.shape) == 3 and piece_region.shape[2] >= 3:
            brightness = (
                0.299 * piece_region[:, :, 0].astype(np.float32)
                + 0.587 * piece_region[:, :, 1].astype(np.float32)
                + 0.114 * piece_region[:, :, 2].astype(np.float32)
            )
        else:
            brightness = piece_region.astype(np.float32)

        binary = (brightness > self.threshold).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        total_area = piece_region.shape[0] * piece_region.shape[1]
        if area < total_area * 0.02:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        roi = binary[y : y + h, x : x + w]

        grid = self._binary_to_grid(roi)
        if grid is None:
            return None

        return self._shape_to_piece_name(grid)

    def _match_template(self, piece_region: np.ndarray) -> Optional[str]:
        """Yaklaşım A: Parlaklık tabanlı basit grid karşılaştırma."""
        if piece_region.size == 0:
            return None

        if len(piece_region.shape) == 3 and piece_region.shape[2] >= 3:
            brightness = (
                0.299 * piece_region[:, :, 0].astype(np.float32)
                + 0.587 * piece_region[:, :, 1].astype(np.float32)
                + 0.114 * piece_region[:, :, 2].astype(np.float32)
            )
        else:
            brightness = piece_region.astype(np.float32)

        binary = (brightness > self.threshold).astype(np.uint8)
        if np.sum(binary) < 5:
            return None

        ys, xs = np.where(binary == 1)
        if len(ys) == 0:
            return None

        roi = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        grid = self._binary_to_grid(roi)
        if grid is None:
            return None

        return self._shape_to_piece_name(grid)

    def _binary_to_grid(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Binary ROI'yi küçük bir grid'e dönüştürür (1-5 × 1-5 boyutunda)."""
        if roi.size == 0:
            return None

        h, w = roi.shape
        filled_pixels = np.sum(roi)
        if filled_pixels < 3:
            return None

        best_grid: Optional[np.ndarray] = None
        best_score = -1.0

        for rows in range(1, 6):
            for cols in range(1, 6):
                if rows * cols > 9:
                    continue
                cell_h = h / rows
                cell_w = w / cols
                if cell_h < 2 or cell_w < 2:
                    continue

                grid = np.zeros((rows, cols), dtype=np.int8)
                for r in range(rows):
                    for c in range(cols):
                        cy = int(r * cell_h + cell_h * 0.5)
                        cx = int(c * cell_w + cell_w * 0.5)
                        cy = min(cy, h - 1)
                        cx = min(cx, w - 1)
                        y1 = max(0, int(r * cell_h + cell_h * 0.3))
                        y2 = min(h, int(r * cell_h + cell_h * 0.7))
                        x1 = max(0, int(c * cell_w + cell_w * 0.3))
                        x2 = min(w, int(c * cell_w + cell_w * 0.7))
                        region = roi[y1:y2, x1:x2]
                        if region.size > 0 and np.mean(region) > 0.5:
                            grid[r, c] = 1

                if np.sum(grid) == 0:
                    continue

                matched = self._shape_to_piece_name_with_score(grid)
                if matched is not None:
                    name, iou = matched
                    if iou > best_score:
                        best_score = iou
                        best_grid = grid

        return best_grid

    def _shape_to_piece_name(self, shape_matrix: np.ndarray) -> Optional[str]:
        """
        Tespit edilen binary matrisi pieces.py şekilleriyle karşılaştırır.
        En yüksek IoU skorunu veren ismi döndürür.
        """
        result = self._shape_to_piece_name_with_score(shape_matrix)
        if result is None:
            return None
        name, iou = result
        return name

    def _shape_to_piece_name_with_score(
        self, shape_matrix: np.ndarray
    ) -> Optional[tuple[str, float]]:
        """İsim ve IoU skoru birlikte döndürür."""
        best_name: Optional[str] = None
        best_iou = 0.0
        sr, sc = shape_matrix.shape

        for name, ref_shape in self._piece_shapes.items():
            rr, rc = ref_shape.shape
            if rr != sr or rc != sc:
                continue
            intersection = np.sum((shape_matrix == 1) & (ref_shape == 1))
            union = np.sum((shape_matrix == 1) | (ref_shape == 1))
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_name = name

        if best_iou < MIN_IOU_THRESHOLD:
            logger.debug(
                "Parça eşleşmesi düşük güvenli: en iyi IoU=%.2f (%s)",
                best_iou,
                best_name,
            )
            return None

        return (best_name, best_iou) if best_name else None
