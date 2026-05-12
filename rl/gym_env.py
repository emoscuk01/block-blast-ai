"""GameEnv'i Gymnasium standardına saran wrapper.

Stable-Baselines3'ün anlayacağı step/reset arayüzünü sağlar.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.game_env import GameEnv
from rl.action_mapper import TOTAL_ACTIONS, action_to_tuple, get_valid_action_mask
from rl.observation import encode_observation, OBS_SIZE
from utils.metrics import compute_reward, compute_regret


class BlockBlastGymEnv(gym.Env):
    """Block Blast oyununu Gymnasium arayüzüyle sunan ortam."""

    metadata = {"render_modes": ["human", "ascii"]}

    MAX_INVALID_STREAK: int = 50

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.game_env = GameEnv(seed=seed)
        self.render_mode = render_mode
        self._seed = seed
        self._invalid_streak: int = 0

        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Gymnasium API: (observation, info) döndür."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.game_env = GameEnv(seed=seed)
        self._invalid_streak = 0
        obs_dict = self.game_env.reset()
        obs = encode_observation(obs_dict)
        info: dict[str, Any] = {
            "score": 0,
            "turn": 1,
            "action_mask": self.get_action_mask(),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Gymnasium API: (obs, reward, terminated, truncated, info) döndür.

        Geçersiz aksiyon gelirse geçerli aksiyonlardan rastgele biri seçilip uygulanır.
        """
        mask = get_valid_action_mask(self.game_env)
        piece_idx, row, col = action_to_tuple(action)

        if not mask[action]:
            valid_actions = self.game_env.get_valid_actions()
            if not valid_actions:
                self.game_env.done = True
                obs = encode_observation(self.game_env.get_observation())
                info: dict[str, Any] = {
                    "score": self.game_env.score,
                    "turn": self.game_env.turn,
                    "invalid_action": True,
                    "action_mask": self.get_action_mask(),
                }
                return obs, -100.0, True, False, info
            import random
            piece_idx, row, col = random.choice(valid_actions)

        board_before = self.game_env.board.get_grid()
        obs_dict, base_reward, done, step_info = self.game_env.step(piece_idx, row, col)
        board_after = self.game_env.board.get_grid()

        lines_cleared = step_info.get("lines_cleared", 0)
        reward = compute_reward(board_before, board_after, lines_cleared, done)

        remaining = [p for p in self.game_env.current_pieces if p is not None]
        if remaining:
            reward += compute_regret(board_after, remaining)

        # REWARD FIX: Manuel /100 kaldırıldı — ölçeklemeyi yalnızca VecNormalize yapıyor (train.py).

        obs = encode_observation(obs_dict)
        info = {
            "score": self.game_env.score,
            "turn": self.game_env.turn,
            "lines_cleared": lines_cleared,
            "invalid_action": False,
            "action_mask": self.get_action_mask(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), done, False, info

    def render(self) -> Optional[str]:
        """render_mode='ascii' ise GameEnv.render() sonucunu yazdırır."""
        output = self.game_env.render(mode="ascii")
        if self.render_mode in ("human", "ascii"):
            print(output)
        return output

    def get_action_mask(self) -> np.ndarray:
        """Shape: (192,) bool array. SB3 MaskableDQN için."""
        return get_valid_action_mask(self.game_env)

    def action_masks(self) -> np.ndarray:
        """get_action_mask() ile aynı, SB3 naming convention için alias."""
        return self.get_action_mask()
