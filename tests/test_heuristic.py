"""Heuristik agent ve metrik testleri."""

import numpy as np

from env.game_env import GameEnv
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from utils.metrics import compute_regret


def test_heuristic_beats_random() -> None:
    """50 oyunda HeuristicAgent ortalama skoru RandomAgent'ı geçmeli."""
    heuristic = HeuristicAgent()
    random_agent = RandomAgent()
    n_games = 50

    h_scores: list[int] = []
    r_scores: list[int] = []

    for i in range(n_games):
        env = GameEnv(seed=i)
        env.reset()
        while not env.done:
            action = heuristic.select_action(env)
            if action is None:
                break
            env.step(*action)
        h_scores.append(env.score)

        env = GameEnv(seed=i)
        env.reset()
        while not env.done:
            action = random_agent.select_action(env)
            if action is None:
                break
            env.step(*action)
        r_scores.append(env.score)

    avg_h = sum(h_scores) / n_games
    avg_r = sum(r_scores) / n_games
    assert avg_h > avg_r, f"Heuristik ({avg_h:.1f}) random'dan ({avg_r:.1f}) iyi olmalı"


def test_heuristic_no_invalid_moves() -> None:
    """HeuristicAgent hiçbir zaman geçersiz hamle seçmemeli."""
    agent = HeuristicAgent()
    env = GameEnv(seed=99)
    env.reset()

    move_count = 0
    while not env.done and move_count < 500:
        action = agent.select_action(env)
        if action is None:
            break
        piece_index, row, col = action
        valid_actions = env.get_valid_actions()
        assert action in valid_actions, f"Geçersiz hamle: {action}"
        env.step(piece_index, row, col)
        move_count += 1


def test_evaluate_move_returns_float() -> None:
    """evaluate_move() her zaman float döndürmeli."""
    agent = HeuristicAgent()
    env = GameEnv(seed=0)
    env.reset()

    actions = env.get_valid_actions()
    assert len(actions) > 0

    score = agent.evaluate_move(env, actions[0][0], actions[0][1], actions[0][2])
    assert isinstance(score, float)


def test_regret_penalizes_blocked_pieces() -> None:
    """Tamamen dolu tahtada regret skoru maksimum olmalı."""
    board = np.ones((8, 8), dtype=np.float32)
    penalty = compute_regret(board, ["kare_3x3", "yatay_5", "L_sag"])
    # 3 parça × REGRET_PENALTY_PER_BLOCKED_PIECE (15) — metrics.py ile uyumlu
    assert penalty == -45.0, f"Beklenen -45.0, alınan {penalty}"
