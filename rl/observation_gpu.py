"""GPU Batched Observation Encoder.

Observation vektörü (CPU versiyonuyla aynı format):
    [ tahta (64) | parça_0 (25) | parça_1 (25) | parça_2 (25) | blok_kaldı (3) ]
    Toplam: 142 boyutlu float32 vektör

Fark: Tüm N ortamı tek seferde, GPU tensörleriyle encode eder.
"""

from __future__ import annotations

import torch

OBS_SIZE: int = 142  # 64 + 25*3 + 3


def encode_observation_batch(
    boards: torch.Tensor,
    pieces_5x5: torch.Tensor,
    pieces_remaining: torch.BoolTensor,
) -> torch.Tensor:
    """Batched observation encoding — tamamen GPU'da.

    Args:
        boards:           [N, 8, 8]    — tahta durumu
        pieces_5x5:       [N, 3, 5, 5] — 3 parçanın 5×5 padded tensörleri
        pieces_remaining: [N, 3]       — parça slot'u hâlâ mevcut mu

    Returns:
        obs: [N, 142] float32
    """
    N = boards.shape[0]

    # Board flatten: [N, 64]
    board_flat = boards.reshape(N, 64)

    # Pieces flatten: [N, 3, 25] -> [N, 75]
    pieces_flat = pieces_5x5.reshape(N, 3, 25).reshape(N, 75)

    # Blocks remaining one-hot: [N, 3]
    # blocks_remaining sayısı = pieces_remaining.sum(dim=1)
    blocks_count = pieces_remaining.long().sum(dim=1)  # [N]  0, 1, 2, or 3
    remaining_onehot = torch.zeros(N, 3, dtype=torch.float32, device=boards.device)
    # 3 -> [1,0,0], 2 -> [0,1,0], 1 -> [0,0,1], 0 -> [0,0,0]
    mask_3 = blocks_count == 3
    mask_2 = blocks_count == 2
    mask_1 = blocks_count == 1
    remaining_onehot[mask_3, 0] = 1.0
    remaining_onehot[mask_2, 1] = 1.0
    remaining_onehot[mask_1, 2] = 1.0

    # Concat: [N, 64 + 75 + 3] = [N, 142]
    return torch.cat([board_flat, pieces_flat, remaining_onehot], dim=1)
