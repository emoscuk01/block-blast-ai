"""agent_loop ↔ dashboard arası JSON dosya köprüsü.

agent_loop her turda live_state.json'ı günceller,
dashboard her N saniyede bir okur.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_FILE = Path("dashboard/live_state.json")


class StateBridge:
    """agent_loop tarafında kullanılır — veri yazar."""

    def update(
        self,
        turn: int,
        board: np.ndarray,
        pieces: list[Optional[str]],
        last_action: Optional[tuple[int, int, int]],
        last_reward: float,
        score: int,
        confidence: float,
        q_values: Optional[list[float]] = None,
    ) -> None:
        """
        Mevcut state'i JSON dosyasına yazar.
        Atomic write: önce .tmp yaz, sonra rename.
        """
        action_dict = None
        if last_action is not None:
            pi, r, c = last_action
            piece_name = pieces[pi] if pi < len(pieces) else None
            action_dict = {
                "piece_index": pi,
                "row": r,
                "col": c,
                "piece_name": piece_name,
            }

        state = {
            "turn": turn,
            "timestamp": time.time(),
            "board": board.tolist() if isinstance(board, np.ndarray) else board,
            "pieces": pieces,
            "last_action": action_dict,
            "last_reward": last_reward,
            "score": score,
            "confidence": confidence,
            "q_values": q_values,
            "is_game_over": False,
        }

        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATE_FILE.with_suffix(".tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            os.replace(str(tmp_path), str(STATE_FILE))
        except Exception as e:
            logger.error("State yazılamadı: %s", e)

    def mark_game_over(self, final_score: int, total_turns: int) -> None:
        """Oyun bitince is_game_over=True yazar."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
            else:
                state = {}

            state["is_game_over"] = True
            state["score"] = final_score
            state["turn"] = total_turns
            state["timestamp"] = time.time()

            tmp_path = STATE_FILE.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            os.replace(str(tmp_path), str(STATE_FILE))
        except Exception as e:
            logger.error("Game over yazılamadı: %s", e)


class StateReader:
    """Dashboard tarafında kullanılır — veri okur."""

    def read(self) -> Optional[dict]:
        """JSON dosyasını okur ve dict döndürür. Dosya yoksa/bozuksa None."""
        if not STATE_FILE.exists():
            return None
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("State okunamadı: %s", e)
            return None

    def is_fresh(self, max_age_seconds: float = 10.0) -> bool:
        """Son güncelleme max_age_seconds'dan yeni mi?"""
        state = self.read()
        if state is None:
            return False
        ts = state.get("timestamp", 0)
        return (time.time() - ts) < max_age_seconds
