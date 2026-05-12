"""Reward ve heuristik hesaplamalarında kullanılan saf (side-effect'siz) fonksiyonlar."""

from __future__ import annotations

import numpy as np

from env.pieces import get_piece_cells, get_piece_size

BOARD_ROWS = 8
BOARD_COLS = 8

# ---------------------------------------------------------------------------
# Reward sabitleri — scripts/config.py varsa oradan okunur; yoksa eski
# hardcoded değerler (geriye uyumluluk) kullanılır.
# ---------------------------------------------------------------------------
try:
    from scripts.config import (
        LINE_CLEAR_COEF,
        AGG_HEIGHT_COEF,
        BUMPINESS_COEF,
        HOLE_COEF,
        SURVIVAL_BONUS_PER_STEP,
        GAME_OVER_PENALTY,
        REGRET_PENALTY_PER_BLOCKED_PIECE,
    )
except ImportError:
    # Fallback: scripts paketi yoksa veya bağımsız çalıştırılıyorsa
    REGRET_PENALTY_PER_BLOCKED_PIECE = 15.0
    SURVIVAL_BONUS_PER_STEP = 4.0
    GAME_OVER_PENALTY = 35.0
    LINE_CLEAR_COEF = 20.0
    AGG_HEIGHT_COEF = 0.35
    BUMPINESS_COEF = 0.25
    HOLE_COEF = 5.0


def board_aggregate_height(board: np.ndarray) -> int:
    """Board numpy array'i alır, aggregate height hesaplar."""
    total = 0
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS):
            if board[r, c] == 1:
                total += BOARD_ROWS - r
                break
    return total


def board_bumpiness(board: np.ndarray) -> int:
    """Board numpy array'i alır, bumpiness hesaplar."""
    heights: list[int] = []
    for c in range(BOARD_COLS):
        h = 0
        for r in range(BOARD_ROWS):
            if board[r, c] == 1:
                h = BOARD_ROWS - r
                break
        heights.append(h)

    bump = 0
    for i in range(len(heights) - 1):
        bump += abs(heights[i] - heights[i + 1])
    return bump


def board_holes(board: np.ndarray) -> int:
    """
    Hole tanımı: dolu bir hücrenin altında kalan boş hücre.
    Sütun bazlı tarar: üstten ilk dolu hücreyi bulur,
    onun altındaki tüm boşları sayar.
    """
    holes = 0
    for c in range(BOARD_COLS):
        found_filled = False
        for r in range(BOARD_ROWS):
            if board[r, c] == 1:
                found_filled = True
            elif found_filled:
                holes += 1
    return holes


def _can_place_on_board(board: np.ndarray, piece_name: str) -> bool:
    """Verilen parçanın tahta üzerinde herhangi bir yere yerleştirilip yerleştirilemeyeceğini kontrol eder."""
    cells = get_piece_cells(piece_name)
    p_rows, p_cols = len(cells), len(cells[0])

    for r in range(BOARD_ROWS - p_rows + 1):
        for c in range(BOARD_COLS - p_cols + 1):
            can = True
            for pr in range(p_rows):
                for pc in range(p_cols):
                    if cells[pr][pc] == 1 and board[r + pr, c + pc] != 0:
                        can = False
                        break
                if not can:
                    break
            if can:
                return True
    return False


def compute_reward(
    board_before: np.ndarray,
    board_after: np.ndarray,
    lines_cleared: int,
    game_over: bool,
) -> float:
    """
    Ana reward fonksiyonu.

    R = LINE_CLEAR_COEF*S - AGG_HEIGHT_COEF*A - BUMPINESS_COEF*B - HOLE_COEF*H
        + (oyun devam → SURVIVAL_BONUS_PER_STEP) - (game_over → GAME_OVER_PENALTY)

    S = silinen satır + sütun toplamı
    A,B,H = board_after metrikleri
    """
    s = lines_cleared
    a = board_aggregate_height(board_after)
    b = board_bumpiness(board_after)
    h = board_holes(board_after)

    reward = (
        (LINE_CLEAR_COEF * s)
        - (AGG_HEIGHT_COEF * a)
        - (BUMPINESS_COEF * b)
        - (HOLE_COEF * h)
    )
    # LEARNING FIX: Geçerli yerleştirme sonrası hayatta kalındı → pozitif taban (yanmama sinyali).
    if not game_over:
        reward += SURVIVAL_BONUS_PER_STEP
    else:
        reward -= GAME_OVER_PENALTY
    return reward


def compute_regret(board: np.ndarray, upcoming_pieces: list[str]) -> float:
    """
    Pişmanlık skoru: mevcut tahtada upcoming_pieces içinden
    kaçı hiçbir yere yerleştirilemiyor?

    Her sığmayan parça için REGRET_PENALTY_PER_BLOCKED_PIECE cezası uygular (eskiden 50).
    """
    penalty = 0.0
    for piece_name in upcoming_pieces:
        if piece_name is not None and not _can_place_on_board(board, piece_name):
            penalty -= REGRET_PENALTY_PER_BLOCKED_PIECE
    return penalty


def composite_score(
    board: np.ndarray,
    upcoming_pieces: list[str],
    lines_cleared: int = 0,
) -> float:
    """
    Heuristik agent'ın hamle değerlendirmesinde kullandığı tek sayı.
    compute_reward + compute_regret kombinasyonu.
    Hamle simülasyonu yapıldıktan sonra bu fonksiyon çağrılır.
    """
    a = board_aggregate_height(board)
    b = board_bumpiness(board)
    h = board_holes(board)

    score = (
        (LINE_CLEAR_COEF * lines_cleared)
        - (AGG_HEIGHT_COEF * a)
        - (BUMPINESS_COEF * b)
        - (HOLE_COEF * h)
    )
    score += compute_regret(board, upcoming_pieces)
    return score
