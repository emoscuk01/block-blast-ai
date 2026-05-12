"""Block Blast AI — Evaluate & Promote.

Yeni nesil (challenger) ile mevcut usta (master) arasında N oyunluk karşılaştırma.
Win-rate yeterli ise challenger yeni master olur.

Kullanım:
    python -m scripts.evaluate_and_promote --gen 1
    python -m scripts.evaluate_and_promote --gen 1 --n-games 20  # Hızlı test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# Proje kökünü path'e ekle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.config import (
    EVAL_N_GAMES,
    EVAL_SEED_BASE,
    PROMOTION_WIN_RATE,
    gen_dir,
    GENERATION_LOG,
    LOGS_DIR,
)
from env.game_env import GameEnv
from agents.heuristic_agent import HeuristicAgent
from rl.observation import encode_observation
from rl.action_mapper import action_to_tuple, get_valid_action_mask


# =========================================================================
# Agent Oynatma Fonksiyonları
# =========================================================================

def play_heuristic_game(seed: int) -> dict:
    """HeuristicAgent ile tek oyun oyna.

    Returns:
        {"score": int, "turns": int, "moves": int}
    """
    agent = HeuristicAgent()
    env = GameEnv(seed=seed)
    env.reset()

    moves = 0
    while not env.done:
        action = agent.select_action(env)
        if action is None:
            break
        env.step(*action)
        moves += 1

    return {"score": env.score, "turns": env.turn, "moves": moves}


def play_model_game(model, seed: int, vecnorm_path: str | None = None) -> dict:
    """MaskablePPO modeli ile tek oyun oyna.

    Args:
        model: MaskablePPO modeli
        seed: Oyun seed'i
        vecnorm_path: VecNormalize istatistik dosyasi (varsa obs normalize edilir)

    Returns:
        {"score": int, "turns": int, "moves": int}
    """
    # VecNormalize yukle (egitimde kullanildiysa)
    obs_normalizer = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        from stable_baselines3.common.vec_env import VecNormalize as VN
        import pickle
        try:
            with open(vecnorm_path, "rb") as f:
                vecnorm_data = pickle.load(f)
            # VecNormalize obs_rms (running mean/std) al
            if hasattr(vecnorm_data, 'obs_rms'):
                obs_normalizer = vecnorm_data.obs_rms
        except Exception:
            pass

    env = GameEnv(seed=seed)
    env.reset()

    moves = 0
    while not env.done:
        obs_dict = env.get_observation()
        obs_vec = encode_observation(obs_dict)
        mask = get_valid_action_mask(env)

        # Obs normalize et (egitimle ayni scale)
        if obs_normalizer is not None:
            obs_vec = (obs_vec - obs_normalizer.mean) / np.sqrt(obs_normalizer.var + 1e-8)
            obs_vec = np.clip(obs_vec, -10.0, 10.0).astype(np.float32)

        # Gecerli hamle yoksa dur
        if not mask.any():
            break

        action_int, _ = model.predict(
            obs_vec,
            deterministic=True,
            action_masks=mask,
        )
        piece_idx, row, col = action_to_tuple(int(action_int))

        # Gecerlilik kontrolu
        if not mask[int(action_int)]:
            valid = env.get_valid_actions()
            if not valid:
                break
            piece_idx, row, col = valid[0]

        try:
            env.step(piece_idx, row, col)
            moves += 1
        except ValueError:
            break

    return {"score": env.score, "turns": env.turn, "moves": moves}


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_generation(
    gen: int,
    n_games: int | None = None,
    verbose: bool = True,
) -> dict:
    """Yeni nesil (gen) vs usta (gen-1) karşılaştırması.

    Gen 0 durumu:
        - Master = HeuristicAgent (baseline)
        - Challenger yok — sadece heuristik performansı ölçülür

    Gen 1+ durumu:
        - Master = Gen (gen-1) modeli veya HeuristicAgent (gen-1 == 0 ise)
        - Challenger = Gen (gen) modeli

    Returns:
        {"gen": int, "status": str, "win_rate": float, ...}
    """
    n_games = n_games or EVAL_N_GAMES
    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_games))

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Gen {gen}] Değerlendirme — {n_games} oyun")
        print(f"{'='*60}")

    # --- Gen 0 = sadece heuristik baseline ---
    if gen == 0:
        if verbose:
            print(f"  Gen 0: HeuristicAgent baseline ölçümü...")

        scores = []
        for i, seed in enumerate(seeds):
            result = play_heuristic_game(seed)
            scores.append(result["score"])
            if verbose and (i + 1) % 20 == 0:
                print(f"    Oyun {i+1}/{n_games}: ort={np.mean(scores):.1f}")

        avg_score = float(np.mean(scores))
        result = {
            "gen": 0,
            "type": "heuristic_baseline",
            "status": "BASELINE",
            "avg_score": round(avg_score, 2),
            "median_score": round(float(np.median(scores)), 2),
            "max_score": int(np.max(scores)),
            "min_score": int(np.min(scores)),
            "n_games": n_games,
        }
        if verbose:
            print(f"\n  Heuristik Baseline:")
            print(f"    Ortalama skor: {avg_score:.1f}")
            print(f"    Medyan skor  : {result['median_score']:.1f}")
            print(f"    Max / Min    : {result['max_score']} / {result['min_score']}")

        return result

    # --- Gen 1+ = challenger vs master ---
    # Challenger modeli yükle
    challenger_path = os.path.join(gen_dir(gen), "rl_finetuned.zip")
    if not os.path.exists(challenger_path):
        # Best model'e bak
        challenger_path = os.path.join(gen_dir(gen), "best", "best_model.zip")
    if not os.path.exists(challenger_path):
        raise FileNotFoundError(
            f"Gen {gen} modeli bulunamadı: {challenger_path}"
        )

    from sb3_contrib import MaskablePPO

    challenger = MaskablePPO.load(challenger_path, device="cpu")
    # VecNormalize istatistikleri
    challenger_vecnorm = os.path.join(gen_dir(gen), "vecnormalize.pkl")
    if not os.path.exists(challenger_vecnorm):
        challenger_vecnorm = None
    if verbose:
        print(f"  Challenger: {challenger_path}")
        if challenger_vecnorm:
            print(f"  VecNorm  : {challenger_vecnorm}")

    # Master — gen-1 == 0 ise HeuristicAgent, aksi halde önceki model
    master_is_heuristic = (gen - 1) == 0
    master_model = None
    if not master_is_heuristic:
        master_path = os.path.join(gen_dir(gen - 1), "rl_finetuned.zip")
        if not os.path.exists(master_path):
            master_path = os.path.join(gen_dir(gen - 1), "best", "best_model.zip")
        if os.path.exists(master_path):
            master_model = MaskablePPO.load(master_path, device="cpu")
            if verbose:
                print(f"  Master   : {master_path}")
        else:
            master_is_heuristic = True
            if verbose:
                print(f"  Master   : HeuristicAgent (model bulunamadı: {master_path})")

    if master_is_heuristic and verbose:
        print(f"  Master   : HeuristicAgent (Gen 0)")

    # --- Oyunları oyna ---
    challenger_scores = []
    master_scores = []
    challenger_wins = 0
    draws = 0

    t_start = time.time()
    for i, seed in enumerate(seeds):
        # Challenger
        c_result = play_model_game(challenger, seed, vecnorm_path=challenger_vecnorm)
        challenger_scores.append(c_result["score"])

        # Master
        if master_is_heuristic:
            m_result = play_heuristic_game(seed)
        else:
            master_vecnorm = os.path.join(gen_dir(gen - 1), "vecnormalize.pkl")
            if not os.path.exists(master_vecnorm):
                master_vecnorm = None
            m_result = play_model_game(master_model, seed, vecnorm_path=master_vecnorm)
        master_scores.append(m_result["score"])

        if c_result["score"] > m_result["score"]:
            challenger_wins += 1
        elif c_result["score"] == m_result["score"]:
            draws += 1

        if verbose and (i + 1) % 20 == 0:
            wr = challenger_wins / (i + 1)
            print(
                f"    Oyun {i+1}/{n_games}: "
                f"challenger={c_result['score']}, master={m_result['score']} | "
                f"Win-rate: {wr:.1%}"
            )

    elapsed = time.time() - t_start

    # --- Sonuçlar ---
    win_rate = challenger_wins / n_games
    promoted = win_rate >= PROMOTION_WIN_RATE

    result = {
        "gen": gen,
        "status": "PROMOTED" if promoted else "REJECTED",
        "win_rate": round(win_rate, 4),
        "challenger_wins": challenger_wins,
        "master_wins": n_games - challenger_wins - draws,
        "draws": draws,
        "challenger_avg_score": round(float(np.mean(challenger_scores)), 2),
        "challenger_median": round(float(np.median(challenger_scores)), 2),
        "master_avg_score": round(float(np.mean(master_scores)), 2),
        "master_median": round(float(np.median(master_scores)), 2),
        "master_type": "heuristic" if master_is_heuristic else f"gen_{gen-1}",
        "n_games": n_games,
        "eval_time_s": round(elapsed, 1),
    }

    if verbose:
        status_icon = "[OK]" if promoted else "[X]"
        print(f"\n  {'='*50}")
        print(f"  {status_icon} Gen {gen} -> {result['status']}")
        print(f"  Win-rate       : {win_rate:.1%} (esik: {PROMOTION_WIN_RATE:.0%})")
        print(f"  Challenger ort : {result['challenger_avg_score']:.1f}")
        print(f"  Master ort     : {result['master_avg_score']:.1f}")
        print(f"  Galibiyet      : {challenger_wins}W / {result['master_wins']}L / {draws}D")
        print(f"  Sure           : {elapsed:.1f}s")
        print(f"  {'='*50}")

    return result


# =========================================================================
# Generation History Logger
# =========================================================================

def log_generation_result(result: dict) -> None:
    """Nesil sonucunu generation_history.json'a ekle."""
    from datetime import datetime

    result["timestamp"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(GENERATION_LOG), exist_ok=True)

    history: list[dict] = []
    if os.path.exists(GENERATION_LOG):
        try:
            with open(GENERATION_LOG, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(result)

    with open(GENERATION_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Nesil logu güncellendi: {GENERATION_LOG}")


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Block Blast Evaluate & Promote")
    parser.add_argument("--gen", type=int, required=True, help="Değerlendirilecek nesil numarası")
    parser.add_argument("--n-games", type=int, default=None, help=f"Maç sayısı (varsayılan: {EVAL_N_GAMES})")
    args = parser.parse_args()

    result = evaluate_generation(gen=args.gen, n_games=args.n_games)
    log_generation_result(result)


if __name__ == "__main__":
    main()
