"""Block Blast AI -- Master Control (Orkestra Sefi).

Tum Self-Play pipeline'ini nesiller halinde otomatik dondurur:
    Gen N verisi toplanir -> Gen N+1 BC on-egitim -> RL fine-tune -> Degerlendirme -> Terfi?

Kullanim:
    python -m scripts.master_control
    python -m scripts.master_control --max-gen 10 --target-score 300
    python -m scripts.master_control --start-gen 3 --resume  # Kaldigi yerden devam

    # Hizli test (kucuk parametreler):
    python -m scripts.master_control --max-gen 2 --expert-moves 1000 --rl-timesteps 100000 --eval-games 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

# Proje kökünü path'e ekle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.config import (
    EXPERT_MOVES_TARGET,
    EXPERT_MIN_SCORE,
    RL_TIMESTEPS,
    RL_N_ENVS,
    RL_BATCH_SIZE,
    EVAL_N_GAMES,
    PROMOTION_WIN_RATE,
    gen_dir,
    gen_data_dir,
    GENERATION_LOG,
    LOGS_DIR,
    MODELS_DIR,
)

from scripts.collect_expert_data import collect_expert_games, save_expert_data
from scripts.pretrain_apprentice import pretrain
from scripts.rl_fine_tune import rl_fine_tune
from scripts.evaluate_and_promote import evaluate_generation, log_generation_result


# =========================================================================
# Pipeline Fonksiyonları
# =========================================================================

def _find_master_model(current_master_gen: int) -> Optional[str]:
    """Mevcut master'in model yolunu bul.

    Gen 0 = HeuristicAgent -> None doner.
    Gen 1+ = rl_finetuned.zip veya best/best_model.zip
    """
    if current_master_gen == 0:
        return None

    paths = [
        os.path.join(gen_dir(current_master_gen), "rl_finetuned.zip"),
        os.path.join(gen_dir(current_master_gen), "best", "best_model.zip"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def run_pipeline(
    max_gen: int = 100,
    start_gen: int = 0,
    target_score: Optional[float] = None,
    expert_moves: int | None = None,
    expert_min_score: int | None = None,
    rl_timesteps: int | None = None,
    rl_n_envs: int | None = None,
    rl_batch_size: int | None = None,
    eval_n_games: int | None = None,
    device: str = "auto",
    vec_env: str = "subproc",
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Self-Play pipeline döngüsünü çalıştır.

    Args:
        max_gen: Maksimum nesil sayısı
        start_gen: Başlangıç nesli
        target_score: Hedef ortalama skor (ulaşılınca dur)
        expert_moves: Her nesilde toplanacak hamle sayısı
        expert_min_score: Minimum oyun skoru filtresi
        rl_timesteps: PPO toplam adım
        rl_n_envs: Paralel ortam sayısı
        rl_batch_size: PPO minibatch
        eval_n_games: Değerlendirme maç sayısı
        device: "auto", "cuda", "cpu"
        vec_env: "subproc" veya "dummy"
        dry_run: True ise sadece akışı göster, çalıştırma
        verbose: Detaylı log
    """
    expert_moves = expert_moves or EXPERT_MOVES_TARGET
    expert_min_score = expert_min_score or EXPERT_MIN_SCORE
    rl_timesteps = rl_timesteps or RL_TIMESTEPS
    rl_n_envs = rl_n_envs or RL_N_ENVS
    rl_batch_size = rl_batch_size or RL_BATCH_SIZE
    eval_n_games = eval_n_games or EVAL_N_GAMES

    current_master_gen = start_gen
    pipeline_start = time.time()

    print(f"\n{'#'*70}")
    print(f"#  BLOCK BLAST AI -- SELF-PLAY PIPELINE")
    print(f"#  Başlangıç nesli : {start_gen}")
    print(f"#  Maksimum nesil  : {max_gen}")
    print(f"#  Hedef skor      : {target_score or 'Yok (sınırsız döngü)'}")
    print(f"#  Expert hamle    : {expert_moves:,}")
    print(f"#  RL adım         : {rl_timesteps:,}")
    print(f"#  Eval maç        : {eval_n_games}")
    print(f"#  Cihaz           : {device}")
    print(f"#  Dry-run         : {dry_run}")
    print(f"{'#'*70}\n")

    if dry_run:
        print("[DRY-RUN] Pipeline akisi simule ediliyor...\n")

    # =====================================================================
    # Adım 0: Baseline ölçümü (eğer start_gen == 0)
    # =====================================================================
    if start_gen == 0:
        print(f"\n{'='*60}")
        print(f"ASAMA 0: Heuristik Baseline Olcumu")
        print(f"{'='*60}")

        if dry_run:
            print(f"  [DRY] HeuristicAgent ile {eval_n_games} oyun oynanacak")
            print(f"  [DRY] Sonuc -> generation_history.json'a loglanacak")
        else:
            baseline_result = evaluate_generation(gen=0, n_games=eval_n_games, verbose=verbose)
            log_generation_result(baseline_result)
            print(f"\n  Heuristik baseline skoru: {baseline_result['avg_score']:.1f}")

        current_master_gen = 0

    # =====================================================================
    # Ana Dongu: Nesil N -> N+1
    # =====================================================================
    for gen in range(max(start_gen, 1), max_gen + 1):
        gen_start = time.time()

        print(f"\n{'*'*70}")
        print(f"*  NESIL {gen} BASLIYOR")
        print(f"*  Master: Gen {current_master_gen} {'(Heuristik)' if current_master_gen == 0 else ''}")
        print(f"{'*'*70}")

        try:
            # -----------------------------------------------------------------
            # Adım 1: Expert Data Collection
            # -----------------------------------------------------------------
            print(f"\n--- Adim 1/4: Expert Veri Toplama (Gen {current_master_gen} -> data) ---")

            if dry_run:
                print(f"  [DRY] {expert_moves:,} hamle toplanacak (min_score={expert_min_score})")
                print(f"  [DRY] 8-way augmentation -> ~{expert_moves * 8:,} sample")
                print(f"  [DRY] Kayit: data/gen_{current_master_gen}/expert_data.npz")
            else:
                # Veri zaten varsa atla
                data_path = os.path.join(gen_data_dir(current_master_gen), "expert_data.npz")
                if os.path.exists(data_path):
                    print(f"  Expert verisi zaten mevcut: {data_path} -> atlaniyor")
                else:
                    master_model_path = _find_master_model(current_master_gen)
                    obs, masks, actions = collect_expert_games(
                        n_moves_target=expert_moves,
                        min_score=expert_min_score,
                        gen=current_master_gen,
                        use_model_path=master_model_path,
                        verbose=verbose,
                    )
                    save_expert_data(current_master_gen, obs, masks, actions)

            # -----------------------------------------------------------------
            # Adım 2: Behavior Cloning Ön-Eğitim
            # -----------------------------------------------------------------
            print(f"\n--- Adim 2/4: Behavior Cloning On-Egitim (Gen {gen}) ---")

            if dry_run:
                print(f"  [DRY] BC egitimi: data/gen_{current_master_gen} -> models/gen_{gen}/")
                print(f"  [DRY] MaskablePPO policy agi supervised egitilecek")
            else:
                pretrain(
                    gen=gen,
                    data_gen=current_master_gen,
                    device_str=device,
                    verbose=verbose,
                )

            # -----------------------------------------------------------------
            # Adım 3: RL Fine-Tuning
            # -----------------------------------------------------------------
            print(f"\n--- Adim 3/4: RL Fine-Tune (Gen {gen}) ---")

            if dry_run:
                print(f"  [DRY] {rl_timesteps:,} adım MaskablePPO eğitimi")
                print(f"  [DRY] {rl_n_envs} paralel ortam, batch_size={rl_batch_size}")
                print(f"  [DRY] KL divergence monitoring + experience replay aktif")
            else:
                rl_fine_tune(
                    gen=gen,
                    timesteps=rl_timesteps,
                    n_envs=rl_n_envs,
                    batch_size=rl_batch_size,
                    device_str=device,
                    vec_env_type=vec_env,
                    verbose=verbose,
                )

            # -----------------------------------------------------------------
            # Adım 4: Değerlendirme & Terfi
            # -----------------------------------------------------------------
            print(f"\n--- Adim 4/4: Degerlendirme (Gen {gen} vs Gen {current_master_gen}) ---")

            if dry_run:
                print(f"  [DRY] {eval_n_games} oyunluk karşılaştırma")
                print(f"  [DRY] Win-rate >= {PROMOTION_WIN_RATE:.0%} -> PROMOTED")
                eval_result = {
                    "gen": gen,
                    "status": "DRY_RUN",
                    "win_rate": 0.0,
                    "challenger_avg_score": 0.0,
                }
            else:
                eval_result = evaluate_generation(
                    gen=gen,
                    n_games=eval_n_games,
                    verbose=verbose,
                )
                log_generation_result(eval_result)

            # -----------------------------------------------------------------
            # Terfi kararı
            # -----------------------------------------------------------------
            gen_elapsed = time.time() - gen_start
            promoted = eval_result.get("status") == "PROMOTED"

            if promoted:
                current_master_gen = gen
                print(f"\n  [OK] Gen {gen} PROMOTED -> Yeni Master!")
            else:
                print(f"\n  [X] Gen {gen} REJECTED -- Master hala Gen {current_master_gen}")

            print(f"  Nesil süresi: {gen_elapsed:.0f}s ({gen_elapsed/3600:.1f}h)")

            # Hedef skor kontrolü
            if target_score is not None and not dry_run:
                challenger_avg = eval_result.get("challenger_avg_score", 0.0)
                if challenger_avg >= target_score:
                    print(f"\n  [HEDEF] SKORA ULASILDI: {challenger_avg:.1f} >= {target_score}")
                    break

        except KeyboardInterrupt:
            print(f"\n\n[!] Pipeline kullanici tarafindan durduruldu (Gen {gen})")
            break
        except Exception as e:
            print(f"\n[X] Gen {gen} HATA: {e}")
            traceback.print_exc()

            # Hatayı logla
            error_result = {
                "gen": gen,
                "status": "ERROR",
                "error": str(e),
            }
            if not dry_run:
                log_generation_result(error_result)

            print(f"  Bir sonraki nesile geciliyor...")
            continue

    # =====================================================================
    # Özet
    # =====================================================================
    total_elapsed = time.time() - pipeline_start

    print(f"\n{'#'*70}")
    print(f"#  PIPELINE TAMAMLANDI")
    print(f"#  Son Master   : Gen {current_master_gen}")
    print(f"#  Toplam süre  : {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"#  Nesil logu   : {GENERATION_LOG}")
    print(f"{'#'*70}")

    # Özet raporu
    if os.path.exists(GENERATION_LOG) and not dry_run:
        try:
            with open(GENERATION_LOG, "r", encoding="utf-8") as f:
                history = json.load(f)
            print(f"\n  Nesil Geçmişi:")
            print(f"  {'Gen':<5} {'Durum':<12} {'Win-Rate':<10} {'Ort. Skor':<12} {'Zaman'}")
            print(f"  {'-'*55}")
            for entry in history:
                gen_n = entry.get("gen", "?")
                status = entry.get("status", entry.get("type", "?"))
                wr = entry.get("win_rate", "-")
                avg = entry.get("challenger_avg_score", entry.get("avg_score", "-"))
                ts = entry.get("timestamp", "")[:19]
                wr_str = f"{wr:.1%}" if isinstance(wr, float) else str(wr)
                avg_str = f"{avg:.1f}" if isinstance(avg, (int, float)) else str(avg)
                print(f"  {gen_n:<5} {status:<12} {wr_str:<10} {avg_str:<12} {ts}")
        except Exception:
            pass


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block Blast Self-Play Pipeline Orkestra Şefi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Tam pipeline (Vast.ai RTX 5090):
  python -m scripts.master_control --max-gen 20

  # Hızlı test (lokal):
  python -m scripts.master_control --max-gen 2 --expert-moves 1000 \\
      --rl-timesteps 100000 --eval-games 10 --vec-env dummy --rl-n-envs 4

  # Dry-run (akışı kontrol et):
  python -m scripts.master_control --dry-run

  # Kaldığı yerden devam:
  python -m scripts.master_control --start-gen 3 --max-gen 20
        """,
    )
    parser.add_argument("--max-gen", type=int, default=100, help="Maksimum nesil sayısı")
    parser.add_argument("--start-gen", type=int, default=0, help="Başlangıç nesli")
    parser.add_argument("--target-score", type=float, default=None, help="Hedef ortalama skor")
    parser.add_argument("--expert-moves", type=int, default=None, help=f"Expert hamle sayısı (varsayılan: {EXPERT_MOVES_TARGET:,})")
    parser.add_argument("--expert-min-score", type=int, default=None, help=f"Min oyun skoru (varsayılan: {EXPERT_MIN_SCORE})")
    parser.add_argument("--rl-timesteps", type=int, default=None, help=f"RL adım sayısı (varsayılan: {RL_TIMESTEPS:,})")
    parser.add_argument("--rl-n-envs", type=int, default=None, help=f"Paralel ortam (varsayılan: {RL_N_ENVS})")
    parser.add_argument("--rl-batch-size", type=int, default=None, help=f"Batch size (varsayılan: {RL_BATCH_SIZE})")
    parser.add_argument("--eval-games", type=int, default=None, help=f"Eval maç sayısı (varsayılan: {EVAL_N_GAMES})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"])
    parser.add_argument("--dry-run", action="store_true", help="Sadece akışı göster, çalıştırma")
    args = parser.parse_args()

    run_pipeline(
        max_gen=args.max_gen,
        start_gen=args.start_gen,
        target_score=args.target_score,
        expert_moves=args.expert_moves,
        expert_min_score=args.expert_min_score,
        rl_timesteps=args.rl_timesteps,
        rl_n_envs=args.rl_n_envs,
        rl_batch_size=args.rl_batch_size,
        eval_n_games=args.eval_games,
        device=args.device,
        vec_env=args.vec_env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
