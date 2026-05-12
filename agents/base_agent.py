"""Tüm agent'ların türediği soyut temel sınıf."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from env.game_env import GameEnv


class BaseAgent(ABC):
    """Block Blast agent soyut arayüzü."""

    @abstractmethod
    def select_action(self, env: "GameEnv") -> Optional[tuple[int, int, int]]:
        """
        Mevcut ortam durumuna göre en iyi (piece_index, row, col) hamlesini seç.
        Geçerli hamle yoksa None döndür.
        """
        ...
