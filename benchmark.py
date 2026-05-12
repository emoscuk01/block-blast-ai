"""İki agent'ı belirli sayıda oyun üzerinde karşılaştıran benchmark scripti."""

from __future__ import annotations

import time
from typing import Any

from env.game_env import GameEnv
from agents.base_agent import BaseAgent


def run_episode(agent: BaseAgent, env: GameEnv) -> dict[str, Any]:
    """Tek bir oyunu baştan sona oynat."""
    obs = env.reset()
    total_lines = 0

    while not env.done:
        action = agent.select_action(env)
        if action is None:
            break
        piece_index, row, col = action
        obs, reward, done, info = env.step(piece_index, row, col)
        total_lines += info.get("lines_cleared", 0)

    return {
        "score": env.score,
        "turns": env.turn,
        "lines_cleared": total_lines,
    }


def benchmark(agent: BaseAgent, n_games: int = 100, seed_start: int = 0) -> dict[str, Any]:
    """
    n_games kadar oyun oynat, istatistikleri hesapla.
    Her oyun için seed = seed_start + i kullanılır (tekrarlanabilirlik).
    """
    scores: list[int] = []
    turns: list[int] = []
    lines: list[int] = []

    for i in range(n_games):
        env = GameEnv(seed=seed_start + i)
        result = run_episode(agent, env)
        scores.append(result["score"])
        turns.append(result["turns"])
        lines.append(result["lines_cleared"])

    return {
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_turns": sum(turns) / len(turns),
        "scores": scores,
        "turns": turns,
        "lines": lines,
    }


def compare(agents: dict[str, BaseAgent], n_games: int = 100) -> None:
    """Agent sözlüğü al, hepsini benchmark et, tabloyu yazdır."""
    results: dict[str, dict] = {}

    print(f"\n=== BENCHMARK SONUÇLARI (N={n_games} oyun) ===\n")

    for name, agent in agents.items():
        t0 = time.time()
        stats = benchmark(agent, n_games=n_games)
        elapsed = time.time() - t0
        results[name] = stats
        print(f"  {name} tamamlandı ({elapsed:.1f}s)")

    print()
    header = f"{'Agent':<20} | {'Ort. Skor':>10} | {'Maks Skor':>10} | {'Min Skor':>9} | {'Ort. Tur':>9} | {'Kazanma %':>9}"
    print(header)
    print("-" * len(header))

    for name, stats in results.items():
        print(
            f"{name:<20} | {stats['avg_score']:>10.1f} | {stats['max_score']:>10} | "
            f"{stats['min_score']:>9} | {stats['avg_turns']:>9.1f} | {'—':>9}"
        )

    names = list(results.keys())
    if len(names) >= 2:
        first = results[names[0]]["avg_score"]
        second = results[names[1]]["avg_score"]
        if second > 0:
            ratio = first / second
            print(f"\n{names[0]} / {names[1]} oran: {ratio:.1f}x")


if __name__ == "__main__":
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent

    agents = {
        "HeuristicAgent": HeuristicAgent(),
        "RandomAgent": RandomAgent(),
    }
    compare(agents, n_games=100)
