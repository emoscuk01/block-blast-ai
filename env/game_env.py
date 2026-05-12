"""Ana oyun ortamı sınıfı: step(), reset(), render() arayüzü."""

from __future__ import annotations

import copy
import random
from typing import Optional

import numpy as np

from env.board import Board
from env.pieces import get_piece_cells, get_random_pieces
from utils.metrics import compute_reward


class GameEnv:
    """Block Blast oyun simülatörü — RL uyumlu step/reset arayüzü."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self.board: Board = Board()
        self.current_pieces: list[Optional[str]] = [None, None, None]
        self.pieces_placed: int = 0
        self.score: int = 0
        self.turn: int = 0
        self.done: bool = False
        self.total_lines_cleared: int = 0

    # ------------------------------------------------------------------
    # Yaşam döngüsü
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Oyunu sıfırla. Observation dict döndür."""
        if self.seed is not None:
            self._rng = random.Random(self.seed)
        self.board.reset()
        self.score = 0
        self.turn = 1
        self.pieces_placed = 0
        self.done = False
        self.total_lines_cleared = 0
        self._deal_new_pieces()
        return self.get_observation()

    # ------------------------------------------------------------------
    # Aksiyon
    # ------------------------------------------------------------------

    def step(self, piece_index: int, row: int, col: int) -> tuple[dict, float, bool, dict]:
        """
        Tek bir bloğu (piece_index) verilen konuma (row, col) yerleştirir.

        Döndürür: (observation, reward, done, info)
        """
        if self.done:
            raise ValueError("Oyun zaten bitti, step() çağrılamaz.")

        if piece_index < 0 or piece_index > 2:
            raise ValueError(f"piece_index 0-2 arasında olmalı, verilen: {piece_index}")

        piece_name = self.current_pieces[piece_index]
        if piece_name is None:
            raise ValueError(f"Parça {piece_index} zaten yerleştirildi (None).")

        if not self.board.can_place(piece_name, row, col):
            raise ValueError(
                f"'{piece_name}' parçası ({row}, {col}) konumuna yerleştirilemez."
            )

        board_before = self.board.get_grid()
        lines_cleared = self.board.place(piece_name, row, col)
        board_after = self.board.get_grid()

        self.total_lines_cleared += lines_cleared

        line_score = lines_cleared * 10
        if lines_cleared > 1:
            line_score += (lines_cleared - 1) * 5
        self.score += line_score

        self.current_pieces[piece_index] = None
        self.pieces_placed += 1

        if self.pieces_placed >= 3:
            self._start_new_turn()

        game_over = self._check_game_over()
        self.done = game_over

        reward = compute_reward(board_before, board_after, lines_cleared, game_over)

        info = {
            "lines_cleared": lines_cleared,
            "score": self.score,
            "turn": self.turn,
        }

        return self.get_observation(), reward, self.done, info

    # ------------------------------------------------------------------
    # Geçerli aksiyonlar
    # ------------------------------------------------------------------

    def get_valid_actions(self) -> list[tuple[int, int, int]]:
        """Mevcut state'te geçerli tüm (piece_index, row, col) üçlülerini döndür."""
        actions: list[tuple[int, int, int]] = []
        for idx, piece_name in enumerate(self.current_pieces):
            if piece_name is None:
                continue
            for r, c in self.board.get_valid_placements(piece_name):
                actions.append((idx, r, c))
        return actions

    # ------------------------------------------------------------------
    # Gözlem
    # ------------------------------------------------------------------

    def get_observation(self) -> dict:
        """State'in tam temsili."""
        pieces_arrays: list[np.ndarray] = []
        pieces_remaining: list[bool] = []

        for piece_name in self.current_pieces:
            if piece_name is not None:
                cells = get_piece_cells(piece_name)
                padded = np.zeros((5, 5), dtype=np.float32)
                for r, row in enumerate(cells):
                    for c, val in enumerate(row):
                        padded[r, c] = float(val)
                pieces_arrays.append(padded)
                pieces_remaining.append(True)
            else:
                pieces_arrays.append(np.zeros((5, 5), dtype=np.float32))
                pieces_remaining.append(False)

        blocks_remaining = sum(1 for p in self.current_pieces if p is not None)

        return {
            "board": self.board.get_grid().astype(np.float32),
            "pieces": pieces_arrays,
            "pieces_remaining": pieces_remaining,
            "blocks_remaining": blocks_remaining,
            "score": self.score,
            "turn": self.turn,
        }

    # ------------------------------------------------------------------
    # Görselleştirme
    # ------------------------------------------------------------------

    def render(self, mode: str = "ascii") -> str:
        """Terminalde yazdırılabilir string döndür."""
        blocks_remaining = sum(1 for p in self.current_pieces if p is not None)
        lines: list[str] = []
        lines.append(f"Tur: {self.turn}  |  Skor: {self.score}  |  Parçalar kaldı: {blocks_remaining}")
        lines.append("")
        lines.append(str(self.board))
        lines.append("")
        piece_names = [p if p is not None else "None" for p in self.current_pieces]
        lines.append(f"Mevcut parçalar: [{', '.join(piece_names)}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Klonlama
    # ------------------------------------------------------------------

    def clone(self) -> "GameEnv":
        """Mevcut state'in derin kopyasını döndür."""
        new_env = GameEnv.__new__(GameEnv)
        new_env.seed = self.seed
        new_env._rng = random.Random()
        new_env._rng.setstate(self._rng.getstate())
        new_env.board = self.board.copy()
        new_env.current_pieces = list(self.current_pieces)
        new_env.pieces_placed = self.pieces_placed
        new_env.score = self.score
        new_env.turn = self.turn
        new_env.done = self.done
        new_env.total_lines_cleared = self.total_lines_cleared
        return new_env

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _deal_new_pieces(self) -> None:
        """Yeni 3 parça dağıt."""
        names = [self._rng.choice(list(_get_all_names())) for _ in range(3)]
        self.current_pieces = names

    def _start_new_turn(self) -> None:
        """Yeni tura geç: 3 yeni parça dağıt, sayaçları sıfırla."""
        self.turn += 1
        self.pieces_placed = 0
        self._deal_new_pieces()

    def _check_game_over(self) -> bool:
        """Kalan parçalardan hiçbiri yerleştirilemiyorsa oyun biter."""
        for piece_name in self.current_pieces:
            if piece_name is None:
                continue
            if self.board.get_valid_placements(piece_name):
                return False
        return True

    # ------------------------------------------------------------------
    # Gösterim
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GameEnv(tur={self.turn}, skor={self.score}, "
            f"done={self.done}, dolu={self.board.count_filled()})"
        )

    def __str__(self) -> str:
        return self.render()


def _get_all_names() -> list[str]:
    """Döngüsel import'u önlemek için pieces modülünden isimleri getirir."""
    from env.pieces import get_all_piece_names
    return get_all_piece_names()
