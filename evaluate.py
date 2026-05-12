"""
Block Blast AI — Model Değerlendirme Scripti

Kullanım:
    python evaluate.py --model models/ppo_v1 --games 100
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from rl.gym_env import BlockBlastGymEnv
from env.game_env import GameEnv
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent


def evaluate_dqn(model_path: str, n_games: int = 100) -> dict[str, Any]:
    """Eğitilmiş MaskablePPO modelini değerlendir."""
    model = MaskablePPO.load(model_path)
    scores: list[int] = []
    turns: list[int] = []

    for i in range(n_games):
        env = BlockBlastGymEnv(seed=i)
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        scores.append(info.get("score", 0))
        turns.append(info.get("turn", 0))

    return {
        "avg_score": np.mean(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_turns": np.mean(turns),
    }


def evaluate_heuristic(n_games: int = 100) -> dict[str, Any]:
    """Heuristik agent'ı değerlendir."""
    agent = HeuristicAgent()
    scores: list[int] = []
    turns: list[int] = []

    for i in range(n_games):
        env = GameEnv(seed=i)
        env.reset()
        while not env.done:
            action = agent.select_action(env)
            if action is None:
                break
            env.step(*action)
        scores.append(env.score)
        turns.append(env.turn)

    return {
        "avg_score": np.mean(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_turns": np.mean(turns),
    }


def evaluate_random(n_games: int = 100) -> dict[str, Any]:
    """Random agent'ı değerlendir."""
    agent = RandomAgent()
    scores: list[int] = []
    turns: list[int] = []

    for i in range(n_games):
        env = GameEnv(seed=i)
        env.reset()
        while not env.done:
            action = agent.select_action(env)
            if action is None:
                break
            env.step(*action)
        scores.append(env.score)
        turns.append(env.turn)

    return {
        "avg_score": np.mean(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_turns": np.mean(turns),
    }


def main(args: argparse.Namespace) -> None:
    """Üç agent'ı kıyaslayan tablo yazdır."""
    n = args.games
    print(f"\n=== DEĞERLENDİRME SONUÇLARI (N={n} oyun) ===\n")

    print("DQN modeli değerlendiriliyor...")
    dqn_stats = evaluate_dqn(args.model, n)

    print("HeuristicAgent değerlendiriliyor...")
    heuristic_stats = evaluate_heuristic(n)

    print("RandomAgent değerlendiriliyor...")
    random_stats = evaluate_random(n)
    print()

    h_avg = heuristic_stats["avg_score"]

    results = {
        f"DQN (eğitilmiş)": dqn_stats,
        "HeuristicAgent": heuristic_stats,
        "RandomAgent": random_stats,
    }

    header = (
        f"{'Agent':<20} | {'Ort. Skor':>10} | {'Maks Skor':>10} | "
        f"{'Ort. Tur':>9} | {'Heuristik Oranı':>16}"
    )
    print(header)
    print("-" * len(header))

    for name, stats in results.items():
        ratio = stats["avg_score"] / max(h_avg, 1.0)
        print(
            f"{name:<20} | {stats['avg_score']:>10.1f} | {stats['max_score']:>10} | "
            f"{stats['avg_turns']:>9.1f} | {ratio:>15.2f}x"
        )

    print()
    dqn_ratio = dqn_stats["avg_score"] / max(h_avg, 1.0)
    if dqn_ratio >= 1.0:
        print(f"DQN heuristik baseline'ı geçti! (oran: {dqn_ratio:.2f}x)")
    else:
        print(f"DQN henüz heuristik seviyesine ulaşamadı (oran: {dqn_ratio:.2f}x)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Blast Model Değerlendirme")
    parser.add_argument("--model", type=str, required=True, help="Model dosya yolu")
    parser.add_argument("--games", type=int, default=100, help="Oyun sayısı")
    args = parser.parse_args()
    main(args)
