"""State'i sinir ağına uygun düz NumPy vektörüne çeviren fonksiyonlar.

Observation vektörü:
[ tahta (64) | parça_0 (25) | parça_1 (25) | parça_2 (25) | blok_kaldı (3) ]
Toplam: 142 boyutlu float32 vektör
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.pieces import get_piece_cells

OBS_SIZE: int = 142  # 64 + 25*3 + 3


def encode_piece(piece_name: Optional[str]) -> np.ndarray:
    """
    Tek bir parçayı 5×5 padding'li (25,) float32 vektöre çevirir.
    piece_name None ise sıfır vektörü döndürür.
    """
    padded = np.zeros((5, 5), dtype=np.float32)
    if piece_name is not None:
        cells = get_piece_cells(piece_name)
        for r, row in enumerate(cells):
            for c, val in enumerate(row):
                padded[r, c] = float(val)
    return padded.flatten()


def encode_observation(obs_dict: dict) -> np.ndarray:
    """
    GameEnv.get_observation() çıktısını (142,) float32 vektöre çevirir.
    """
    board_flat = obs_dict["board"].flatten().astype(np.float32)

    piece_vecs: list[np.ndarray] = []
    for piece_arr in obs_dict["pieces"]:
        piece_vecs.append(piece_arr.flatten().astype(np.float32))

    blocks_remaining = obs_dict["blocks_remaining"]
    remaining_onehot = np.zeros(3, dtype=np.float32)
    if blocks_remaining == 3:
        remaining_onehot[0] = 1.0
    elif blocks_remaining == 2:
        remaining_onehot[1] = 1.0
    elif blocks_remaining == 1:
        remaining_onehot[2] = 1.0

    return np.concatenate([board_flat] + piece_vecs + [remaining_onehot])
