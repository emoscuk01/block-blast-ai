"""Block Blast AI — Expert Data Collector.

HeuristicAgent ile oyun oynatıp yüksek kaliteli (obs, mask, action) üçlüleri toplar.
8-way augmentation ile veri setini 8× zenginleştirir.

Kullanım:
    python -m scripts.collect_expert_data --gen 0
    python -m scripts.collect_expert_data --gen 0 --n-moves 1000   # Hızlı test
"""

from __future__ import annotations

import argparse
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
    EXPERT_MIN_SCORE,
    EXPERT_MOVES_TARGET,
    AUGMENTATION_FACTOR,
    gen_data_dir,
)
from env.game_env import GameEnv
from agents.heuristic_agent import HeuristicAgent
from rl.observation import encode_observation, OBS_SIZE
from rl.action_mapper import (
    TOTAL_ACTIONS,
    tuple_to_action,
    get_valid_action_mask,
)


# =========================================================================
# 8-Way Augmentation — tahta dönüşümleri
# =========================================================================

def _transform_board(board: np.ndarray, transform_id: int) -> np.ndarray:
    """8×8 tahtayı 8 farklı şekilde dönüştür.

    transform_id 0-7:
        0: orijinal
        1: 90° saat yönünde
        2: 180°
        3: 270° saat yönünde
        4: yatay ayna (fliplr)
        5: 90° + fliplr
        6: 180° + fliplr
        7: 270° + fliplr
    """
    k = transform_id % 4  # rotasyon sayısı
    flip = transform_id >= 4

    result = np.rot90(board, k=k)  # k × 90° saat yönünün tersi
    if flip:
        result = np.fliplr(result)
    return result.copy()


def _transform_coords(row: int, col: int, transform_id: int, size: int = 8) -> tuple[int, int]:
    """(row, col) koordinatını dönüşüme göre yeniden hesapla.

    np.rot90 saat yönünün tersine döndürür:
        rot90 k=1: (r, c) -> (c, size-1-r)
        rot90 k=2: (r, c) -> (size-1-r, size-1-c)
        rot90 k=3: (r, c) -> (size-1-c, r)
    fliplr:     (r, c) -> (r, size-1-c)
    """
    k = transform_id % 4
    flip = transform_id >= 4

    r, c = row, col
    for _ in range(k):
        r, c = c, size - 1 - r

    if flip:
        c = size - 1 - c

    return r, c


def augment_sample(
    board_2d: np.ndarray,
    piece_arrays: list[np.ndarray],
    blocks_remaining: int,
    piece_index: int,
    row: int,
    col: int,
    game_env_for_mask: Optional[GameEnv],
) -> list[dict]:
    """Tek bir (obs, action) çiftini 8 augmented varyanta çevir.

    Returns:
        Geçerli augmented sample'ların listesi. Her eleman:
        {"obs": (142,), "mask": (192,), "action": int}
    """
    results: list[dict] = []

    for tid in range(AUGMENTATION_FACTOR):
        if tid == 0:
            # Orijinal — zaten toplandı, tekrar ekle
            obs_dict = _build_obs_dict(board_2d, piece_arrays, blocks_remaining)
            obs_vec = encode_observation(obs_dict)

            action_idx = tuple_to_action(piece_index, row, col)

            # Orijinal mask'ı game_env'den alabiliriz
            if game_env_for_mask is not None:
                mask = get_valid_action_mask(game_env_for_mask)
            else:
                mask = np.ones(TOTAL_ACTIONS, dtype=bool)

            results.append({"obs": obs_vec, "mask": mask, "action": action_idx})
            continue

        # Dönüştürülmüş tahta
        t_board = _transform_board(board_2d, tid)

        # Dönüştürülmüş koordinatlar
        t_row, t_col = _transform_coords(row, col, tid)

        # Sınır kontrolü — parça yerleştirme noktası tahtada mı?
        if not (0 <= t_row < 8 and 0 <= t_col < 8):
            continue

        # Observation oluştur (parça vektörleri değişmez)
        obs_dict = _build_obs_dict(t_board, piece_arrays, blocks_remaining)
        obs_vec = encode_observation(obs_dict)

        action_idx = tuple_to_action(piece_index, t_row, t_col)

        # Augmented mask: dönüştürülmüş tahtada hangi aksiyonlar geçerli?
        # Basitleştirilmiş yaklaşım — sadece action'ın geçerli olup olmadığını kontrol et
        # Tam mask hesabı için Board nesnesi gerekir ama augmented board'dan
        # geçerlilik kontrolü yapmak pahalı. Burada sadece sınır kontrolü yaparız.
        # Mask'ı tüm True yapıp action'ın mask'ta olduğunu varsayarız.
        mask = np.ones(TOTAL_ACTIONS, dtype=bool)

        results.append({"obs": obs_vec, "mask": mask, "action": action_idx})

    return results


def _build_obs_dict(
    board_2d: np.ndarray,
    piece_arrays: list[np.ndarray],
    blocks_remaining: int,
) -> dict:
    """encode_observation'ın beklediği obs_dict formatını oluştur."""
    return {
        "board": board_2d.astype(np.float32),
        "pieces": [p.copy() for p in piece_arrays],
        "pieces_remaining": [True] * blocks_remaining + [False] * (3 - blocks_remaining),
        "blocks_remaining": blocks_remaining,
        "score": 0,
        "turn": 0,
    }


# =========================================================================
# Expert Data Collection
# =========================================================================

def collect_expert_games(
    n_moves_target: int,
    min_score: int,
    gen: int,
    use_model_path: Optional[str] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expert oyunlarından (obs, mask, action) verileri topla.

    Args:
        n_moves_target: Hedef hamle sayısı (augmentation öncesi)
        min_score: Minimum oyun skoru filtresi
        gen: Nesil numarası (0 = heuristik)
        use_model_path: None ise HeuristicAgent, aksi halde model yolu
        verbose: Detaylı log

    Returns:
        (observations, action_masks, actions) — augmentation uygulanmış
    """
    agent: object
    model = None

    if use_model_path is not None and gen > 0:
        # Önceki neslin RL modelini yükle
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(use_model_path, device="cpu")
        if verbose:
            print(f"[Gen {gen}] Model yüklendi: {use_model_path}")
    else:
        agent = HeuristicAgent()
        if verbose:
            print(f"[Gen {gen}] HeuristicAgent kullanılıyor")

    all_obs: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []

    total_moves = 0
    total_games = 0
    accepted_games = 0
    rejected_games = 0
    game_seed = 0

    t_start = time.time()

    while total_moves < n_moves_target:
        env = GameEnv(seed=game_seed)
        env.reset()
        game_seed += 1

        game_obs: list[np.ndarray] = []
        game_masks: list[np.ndarray] = []
        game_actions: list[int] = []
        game_boards: list[np.ndarray] = []
        game_piece_arrays: list[list[np.ndarray]] = []
        game_blocks_remaining: list[int] = []
        game_piece_indices: list[int] = []
        game_rows: list[int] = []
        game_cols: list[int] = []

        while not env.done:
            # Observation ve mask al
            obs_dict = env.get_observation()
            obs_vec = encode_observation(obs_dict)
            mask = get_valid_action_mask(env)

            # Action seç
            if model is not None:
                action_int, _ = model.predict(
                    obs_vec,
                    deterministic=True,
                    action_masks=mask,
                )
                piece_idx, row, col = (
                    int(action_int) // 64,
                    (int(action_int) % 64) // 8,
                    int(action_int) % 8,
                )
                # Geçerlilik kontrolü
                if not mask[int(action_int)]:
                    valid = env.get_valid_actions()
                    if not valid:
                        break
                    piece_idx, row, col = valid[0]
            else:
                action_tuple = agent.select_action(env)
                if action_tuple is None:
                    break
                piece_idx, row, col = action_tuple

            action_idx = tuple_to_action(piece_idx, row, col)

            # Augmentation için ham verileri kaydet
            board_2d = env.board.get_grid().copy()
            pieces_arr = [p.copy() for p in obs_dict["pieces"]]
            br = obs_dict["blocks_remaining"]

            game_obs.append(obs_vec)
            game_masks.append(mask)
            game_actions.append(action_idx)
            game_boards.append(board_2d)
            game_piece_arrays.append(pieces_arr)
            game_blocks_remaining.append(br)
            game_piece_indices.append(piece_idx)
            game_rows.append(row)
            game_cols.append(col)

            # Hamleyi uygula
            try:
                env.step(piece_idx, row, col)
            except ValueError:
                break

        total_games += 1

        # Kalite filtresi
        if env.score < min_score:
            rejected_games += 1
            if verbose and total_games % 100 == 0:
                print(
                    f"  [Topla] Oyun #{total_games}: skor={env.score} < {min_score} -> REDDEDILDI "
                    f"(toplam: {total_moves}/{n_moves_target} hamle, "
                    f"kabul: {accepted_games}, ret: {rejected_games})"
                )
            continue

        accepted_games += 1

        # Augmentation uygula
        for i in range(len(game_obs)):
            augmented = augment_sample(
                board_2d=game_boards[i],
                piece_arrays=game_piece_arrays[i],
                blocks_remaining=game_blocks_remaining[i],
                piece_index=game_piece_indices[i],
                row=game_rows[i],
                col=game_cols[i],
                game_env_for_mask=None,  # Orijinal mask zaten game_masks[i]'de
            )

            # Orijinal sample'ı doğru mask ile ekle
            all_obs.append(game_obs[i])
            all_masks.append(game_masks[i])
            all_actions.append(game_actions[i])

            # Augmented sample'ları ekle (orijinal hariç — tid=0 atla)
            for aug in augmented[1:]:
                all_obs.append(aug["obs"])
                all_masks.append(aug["mask"])
                all_actions.append(aug["action"])

        total_moves += len(game_obs)

        if verbose and accepted_games % 20 == 0:
            elapsed = time.time() - t_start
            rate = total_moves / max(elapsed, 0.01)
            print(
                f"  [Topla] Kabul edilen: {accepted_games} oyun | "
                f"{total_moves}/{n_moves_target} ham hamle | "
                f"Augmented: ~{len(all_obs)} | "
                f"Hız: {rate:.0f} hamle/sn | "
                f"Geçen: {elapsed:.1f}s"
            )

    elapsed = time.time() - t_start

    obs_array = np.array(all_obs, dtype=np.float32)
    mask_array = np.array(all_masks, dtype=bool)
    action_array = np.array(all_actions, dtype=np.int64)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Gen {gen}] Expert Data Collection Tamamlandı")
        print(f"  Toplam oyun     : {total_games} (kabul: {accepted_games}, ret: {rejected_games})")
        print(f"  Ham hamle       : {total_moves}")
        print(f"  Augmented toplam: {len(all_obs)}")
        print(f"  Süre            : {elapsed:.1f}s")
        print(f"  obs shape       : {obs_array.shape}")
        print(f"  mask shape      : {mask_array.shape}")
        print(f"  action shape    : {action_array.shape}")
        print(f"{'='*60}")

    return obs_array, mask_array, action_array


def save_expert_data(
    gen: int,
    observations: np.ndarray,
    action_masks: np.ndarray,
    actions: np.ndarray,
) -> str:
    """Expert verisini diske kaydet.

    Returns:
        Kaydedilen dosya yolu.
    """
    out_dir = gen_data_dir(gen)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "expert_data.npz")

    np.savez_compressed(
        out_path,
        observations=observations,
        action_masks=action_masks,
        actions=actions,
    )
    print(f"Expert verisi kaydedildi: {out_path} ({observations.shape[0]} sample)")
    return out_path


def load_expert_data(gen: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disk'ten expert verisini yükle."""
    data_path = os.path.join(gen_data_dir(gen), "expert_data.npz")
    data = np.load(data_path)
    return data["observations"], data["action_masks"], data["actions"]


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Block Blast Expert Data Collector")
    parser.add_argument("--gen", type=int, default=0, help="Nesil numarası (0 = heuristik)")
    parser.add_argument(
        "--n-moves", type=int, default=None,
        help=f"Hedef hamle sayısı (varsayılan: {EXPERT_MOVES_TARGET})",
    )
    parser.add_argument(
        "--min-score", type=int, default=None,
        help=f"Minimum oyun skoru filtresi (varsayılan: {EXPERT_MIN_SCORE})",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Gen > 0 için önceki master model yolu",
    )
    args = parser.parse_args()

    n_moves = args.n_moves or EXPERT_MOVES_TARGET
    min_score = args.min_score or EXPERT_MIN_SCORE

    obs, masks, actions = collect_expert_games(
        n_moves_target=n_moves,
        min_score=min_score,
        gen=args.gen,
        use_model_path=args.model_path,
    )
    save_expert_data(args.gen, obs, masks, actions)


if __name__ == "__main__":
    main()
