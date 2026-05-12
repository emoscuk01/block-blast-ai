"""Terminalde ASCII görselleştirmeyle tek bir oyun gösterimi."""

from __future__ import annotations

import sys

from env.game_env import GameEnv
from agents.heuristic_agent import HeuristicAgent


def play_demo(seed: int = 42, max_turns: int = 200, auto: bool = False) -> None:
    """HeuristicAgent ile tek bir oyun oynat ve terminale yazdır."""
    env = GameEnv(seed=seed)
    agent = HeuristicAgent()
    obs = env.reset()

    step_count = 0
    while not env.done and env.turn <= max_turns:
        action = agent.select_action(env)
        if action is None:
            break

        piece_index, row, col = action
        piece_name = env.current_pieces[piece_index]

        print("\033[2J\033[H", end="")  # Ekranı temizle
        print(env.render())
        print(f"Seçilen hamle: parça={piece_index} ({piece_name}), konum=({row},{col})")

        if not auto:
            try:
                input("Bekleniyor... (Enter'a bas)")
            except (EOFError, KeyboardInterrupt):
                print("\nOyun sonlandırıldı.")
                return

        obs, reward, done, info = env.step(piece_index, row, col)
        step_count += 1

    print("\033[2J\033[H", end="")
    print(env.render())
    print(f"\n{'='*40}")
    print(f"OYUN BİTTİ!")
    print(f"Final Skor : {env.score}")
    print(f"Toplam Tur : {env.turn}")
    print(f"Toplam Adım: {step_count}")
    print(f"{'='*40}")


if __name__ == "__main__":
    auto = "--auto" in sys.argv
    seed = 42
    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
    play_demo(seed=seed, auto=auto)
