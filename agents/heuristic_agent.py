"""Matematiksel skor fonksiyonuyla oynayan heuristik baseline bot."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from agents.base_agent import BaseAgent
from utils.metrics import composite_score

if TYPE_CHECKING:
    from env.game_env import GameEnv


class HeuristicAgent(BaseAgent):
    """Tüm geçerli hamleleri simüle ederek en yüksek composite_score'u seçen deterministik bot."""

    def __init__(self, weights: Optional[dict] = None) -> None:
        self.weights = weights or {
            "lines": 10,
            "height": 0.5,
            "bump": 0.3,
            "holes": 8,
            "regret": 50,
        }

    def select_action(self, env: "GameEnv") -> Optional[tuple[int, int, int]]:
        """
        En yüksek composite_score'u veren (piece_index, row, col) hamlesini döndür.
        Geçerli hamle yoksa None döndür.
        """
        actions = env.get_valid_actions()
        if not actions:
            return None

        best_action: Optional[tuple[int, int, int]] = None
        best_score = float("-inf")

        for action in actions:
            score = self.evaluate_move(env, action[0], action[1], action[2])
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def evaluate_move(self, env: "GameEnv", piece_index: int, row: int, col: int) -> float:
        """Tek bir hamlenin composite_score'unu döndür."""
        clone = env.clone()
        _obs, _reward, _done, info = clone.step(piece_index, row, col)

        remaining = [p for p in clone.current_pieces if p is not None]
        board_array = clone.board.get_grid()
        lines_cleared = info.get("lines_cleared", 0)

        return composite_score(board_array, remaining, lines_cleared)
