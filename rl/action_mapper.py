"""Aksiyon indeksi ↔ (piece_index, row, col) dönüşümü.

DQN tek bir tamsayı üretir; bu modül o tamsayıyı oyun aksiyonuna çevirir.
Toplam: 3 parça × 8 satır × 8 sütun = 192 aksiyon.
"""

from __future__ import annotations

import numpy as np

TOTAL_ACTIONS: int = 192  # 3 * 8 * 8


def action_to_tuple(action: int) -> tuple[int, int, int]:
    """
    0–191 arası tamsayıyı (piece_index, row, col) üçlüsüne çevirir.

    piece_index = action // 64
    row         = (action % 64) // 8
    col         = action % 8
    """
    piece_index = action // 64
    row = (action % 64) // 8
    col = action % 8
    return (piece_index, row, col)


def tuple_to_action(piece_index: int, row: int, col: int) -> int:
    """(piece_index, row, col) üçlüsünü 0–191 arası tamsayıya çevirir."""
    return piece_index * 64 + row * 8 + col


def get_valid_action_mask(env) -> np.ndarray:
    """
    Shape: (192,), dtype: bool
    env.get_valid_actions() listesini alır,
    geçerli indeksleri True, geçersizleri False yapar.
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    for piece_idx, row, col in env.get_valid_actions():
        idx = tuple_to_action(piece_idx, row, col)
        mask[idx] = True
    return mask
