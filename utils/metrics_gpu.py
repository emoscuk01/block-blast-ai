"""GPU Batched Reward Hesaplama.

Tüm reward metriklerini (aggregate_height, bumpiness, holes, line_clear)
batched GPU tensörleriyle hesaplar. Python loop YOK.
"""

from __future__ import annotations

import torch

BOARD_SIZE: int = 8

# Reward katsayıları — scripts/config.py'den import edilir, yoksa fallback
try:
    from scripts.config import (
        LINE_CLEAR_COEF,
        AGG_HEIGHT_COEF,
        BUMPINESS_COEF,
        HOLE_COEF,
        SURVIVAL_BONUS_PER_STEP,
        GAME_OVER_PENALTY,
    )
except ImportError:
    LINE_CLEAR_COEF = 20.0
    AGG_HEIGHT_COEF = 0.35
    BUMPINESS_COEF = 0.25
    HOLE_COEF = 5.0
    SURVIVAL_BONUS_PER_STEP = 4.0
    GAME_OVER_PENALTY = 35.0


def _aggregate_height_batch(boards: torch.Tensor) -> torch.Tensor:
    """[N, 8, 8] -> [N] aggregate height."""
    filled = (boards > 0).float()
    has_any = filled.any(dim=1)  # [N, 8]
    first_filled_row = filled.argmax(dim=1)  # [N, 8]
    heights = (BOARD_SIZE - first_filled_row) * has_any.long()
    return heights.sum(dim=1).float()


def _bumpiness_batch(boards: torch.Tensor) -> torch.Tensor:
    """[N, 8, 8] -> [N] bumpiness."""
    filled = (boards > 0).float()
    has_any = filled.any(dim=1)
    first_filled_row = filled.argmax(dim=1)
    heights = (BOARD_SIZE - first_filled_row) * has_any.long()
    diffs = (heights[:, 1:] - heights[:, :-1]).abs()
    return diffs.sum(dim=1).float()


def _holes_batch(boards: torch.Tensor) -> torch.Tensor:
    """[N, 8, 8] -> [N] hole count."""
    filled = (boards > 0)
    cum_filled, _ = filled.cummax(dim=1)
    holes = cum_filled & (~filled)
    return holes.sum(dim=(1, 2)).float()


def compute_reward_batch(
    boards_after: torch.Tensor,
    lines_cleared: torch.Tensor,
    dones: torch.BoolTensor,
) -> torch.Tensor:
    """Batched reward hesaplama.

    Args:
        boards_after:  [N, 8, 8] — yerleştirme sonrası tahta
        lines_cleared: [N] — silinen satır+sütun sayısı
        dones:         [N] — oyun bitti mi

    Returns:
        rewards: [N] float32
    """
    s = lines_cleared.float()
    a = _aggregate_height_batch(boards_after)
    b = _bumpiness_batch(boards_after)
    h = _holes_batch(boards_after)

    reward = (
        LINE_CLEAR_COEF * s
        - AGG_HEIGHT_COEF * a
        - BUMPINESS_COEF * b
        - HOLE_COEF * h
    )

    # Hayatta kalma bonusu / game over cezası
    reward = torch.where(
        dones,
        reward - GAME_OVER_PENALTY,
        reward + SURVIVAL_BONUS_PER_STEP,
    )

    return reward
