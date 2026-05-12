"""Tamamen rastgele hamle yapan bot (alt kıyas)."""

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from env.game_env import GameEnv


class RandomAgent(BaseAgent):
    """Geçerli aksiyonlar arasından uniform rastgele seçim yapar."""

    def select_action(self, env: "GameEnv") -> Optional[tuple[int, int, int]]:
        """get_valid_actions() listesinden uniform rastgele bir üçlü seç."""
        actions = env.get_valid_actions()
        if not actions:
            return None
        return random.choice(actions)
