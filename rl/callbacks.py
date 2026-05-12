"""Eğitim sırasında metrikleri kaydeden ve erken durdurma uygulayan callback'ler."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class TrainingMetricsCallback(BaseCallback):
    """
    Her episode bitişinde şunları loglar:
    - ep_score: o oyundaki toplam oyun skoru
    - ep_turns: kaç tur sürdü
    - ep_reward: toplam RL reward'ı
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_scores: list[int] = []
        self._episode_turns: list[int] = []
        self._episode_rewards: list[float] = []
        self._current_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Her step sonrası çağrılır. Episode bittiyse metrikleri toplar ve loglar."""
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if rewards is not None:
            for r in rewards:
                self._current_rewards.append(float(r))

        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                score = info.get("score", 0)
                turn = info.get("turn", 0)
                ep_reward = ep_info.get("r", 0.0)
                # Monitor: gerçek ortam adımı sayısı (~ parça yerleştirme). ep_turns = oyun turu (≈ adım/3).
                ep_len = ep_info.get("l", 0)

                self._episode_scores.append(score)
                self._episode_turns.append(turn)
                self._episode_rewards.append(float(ep_reward))

                self.logger.record("custom/ep_score", score)
                self.logger.record("custom/ep_turns", turn)
                self.logger.record("custom/ep_len_steps", float(ep_len))
                self.logger.record("custom/ep_reward", float(ep_reward))

                if len(self._episode_scores) >= 10:
                    self.logger.record(
                        "custom/avg_score_10",
                        np.mean(self._episode_scores[-10:]),
                    )
                    self.logger.record(
                        "custom/avg_turns_10",
                        np.mean(self._episode_turns[-10:]),
                    )

        if self.num_timesteps % self.log_freq == 0 and self.verbose:
            n_eps = len(self._episode_scores)
            if n_eps > 0:
                recent = self._episode_scores[-min(10, n_eps) :]
                print(
                    f"[{self.num_timesteps} step] "
                    f"Episode: {n_eps}, "
                    f"Son 10 ort. skor: {np.mean(recent):.1f}"
                )

        return True


class HeuristicComparisonCallback(BaseCallback):
    """
    Her N step'te bir eğitilen modeli heuristik agent ile kıyaslar.
    Sonucu TensorBoard'a "rl_vs_heuristic_ratio" olarak loglar.
    """

    def __init__(
        self,
        eval_env,
        heuristic_agent: "BaseAgent",
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.heuristic_agent = heuristic_agent
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._heuristic_avg: Optional[float] = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        if self._heuristic_avg is None:
            self._heuristic_avg = self._run_heuristic()

        rl_avg = self._run_rl_model()

        ratio = rl_avg / max(self._heuristic_avg, 1.0)

        self.logger.record("eval/rl_avg_score", rl_avg)
        self.logger.record("eval/heuristic_avg_score", self._heuristic_avg)
        self.logger.record("eval/rl_vs_heuristic_ratio", ratio)

        if self.verbose:
            print(
                f"\n[Eval @ {self.num_timesteps} step] "
                f"RL ort: {rl_avg:.1f}, "
                f"Heuristik ort: {self._heuristic_avg:.1f}, "
                f"Oran: {ratio:.2f}x\n"
            )

        return True

    def _run_heuristic(self) -> float:
        """Heuristik agent ile n_eval_episodes oyun oynatıp ortalama skoru döndürür."""
        from env.game_env import GameEnv

        scores: list[int] = []
        for i in range(self.n_eval_episodes):
            env = GameEnv(seed=5000 + i)
            env.reset()
            while not env.done:
                action = self.heuristic_agent.select_action(env)
                if action is None:
                    break
                env.step(*action)
            scores.append(env.score)
        return float(np.mean(scores))

    def _run_rl_model(self) -> float:
        """Eğitilen RL modelini eval_env üzerinde n_eval_episodes oynatıp ortalama skoru döndürür."""
        scores: list[float] = []
        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                done = terminated or truncated
            scores.append(float(info.get("score", 0)))
        return float(np.mean(scores))
