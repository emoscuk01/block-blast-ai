"""Oyunda kullanılan tüm blok şekillerinin tanımı ve yardımcı fonksiyonlar."""

from __future__ import annotations

import random
from typing import Optional

PIECES: dict[str, list[list[int]]] = {
    "tek":          [[1]],
    "yatay_2":      [[1, 1]],
    "dikey_2":      [[1], [1]],
    "yatay_3":      [[1, 1, 1]],
    "dikey_3":      [[1], [1], [1]],
    "yatay_4":      [[1, 1, 1, 1]],
    "dikey_4":      [[1], [1], [1], [1]],
    "yatay_5":      [[1, 1, 1, 1, 1]],
    "dikey_5":      [[1], [1], [1], [1], [1]],
    "kare_2x2":     [[1, 1], [1, 1]],
    "kare_3x3":     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "L_sag":        [[1, 0], [1, 0], [1, 1]],
    "L_sol":        [[0, 1], [0, 1], [1, 1]],
    "L_ust":        [[1, 1, 1], [1, 0, 0]],
    "L_alt":        [[1, 0, 0], [1, 1, 1]],
    "J_sag":        [[1, 1], [1, 0], [1, 0]],
    "J_sol":        [[1, 1], [0, 1], [0, 1]],
    "T_sag":        [[1, 1, 1], [0, 1, 0]],
    "T_sol":        [[0, 1, 0], [1, 1, 1]],
    "T_dikey":      [[1, 0], [1, 1], [1, 0]],
    "S_yatay":      [[0, 1, 1], [1, 1, 0]],
    "S_dikey":      [[1, 0], [1, 1], [0, 1]],
    "Z_yatay":      [[1, 1, 0], [0, 1, 1]],
    "Z_dikey":      [[0, 1], [1, 1], [1, 0]],
    "kose_sol_ust": [[1, 1], [1, 0]],
    "kose_sag_ust": [[1, 1], [0, 1]],
    "kose_sol_alt": [[1, 0], [1, 1]],
    "kose_sag_alt": [[0, 1], [1, 1]],
}

_PIECE_NAMES: list[str] = list(PIECES.keys())


def get_piece_cells(piece_name: str) -> list[list[int]]:
    """Verilen isimli bloğun 2D matrisini döndür."""
    if piece_name not in PIECES:
        raise ValueError(f"Bilinmeyen parça ismi: {piece_name}")
    return PIECES[piece_name]


def get_piece_size(piece_name: str) -> tuple[int, int]:
    """(satır_sayısı, sütun_sayısı) döndür."""
    cells = get_piece_cells(piece_name)
    return (len(cells), len(cells[0]))


def get_random_pieces(n: int = 3, seed: Optional[int] = None) -> list[str]:
    """n adet rastgele blok ismi döndür. Seed verilirse tekrarlanabilir."""
    rng = random.Random(seed)
    return [rng.choice(_PIECE_NAMES) for _ in range(n)]


def get_all_piece_names() -> list[str]:
    """Tüm blok isimlerini liste olarak döndür."""
    return list(_PIECE_NAMES)
