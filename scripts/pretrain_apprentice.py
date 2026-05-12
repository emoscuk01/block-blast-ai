"""Block Blast AI — Behavior Cloning (BC) Pre-Training.

Expert verisiyle MaskablePPO policy ağını supervised learning ile ön-eğitir.
Action masking + LR warm-up + CrossEntropy loss.

Kullanım:
    python -m scripts.pretrain_apprentice --gen 1
    python -m scripts.pretrain_apprentice --gen 1 --epochs 5 --batch-size 256  # Hızlı test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Proje kökünü path'e ekle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.config import (
    BC_BATCH_SIZE,
    BC_EPOCHS,
    BC_LR_START,
    BC_LR_TARGET,
    BC_WARMUP_STEPS,
    BC_VALIDATION_SPLIT,
    POLICY_ARCH_PI,
    POLICY_ARCH_VF,
    gen_dir,
    gen_data_dir,
    LOGS_DIR,
)
from scripts.collect_expert_data import load_expert_data
from rl.gym_env import BlockBlastGymEnv
from rl.action_mapper import TOTAL_ACTIONS
from rl.observation import OBS_SIZE


# =========================================================================
# LR Warm-up Scheduler
# =========================================================================

class WarmupScheduler:
    """İlk N adımda LR'yi linearly artır, sonra sabit tut."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 start_lr: float, target_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            lr = self.start_lr + (self.target_lr - self.start_lr) * progress
        else:
            lr = self.target_lr

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# =========================================================================
# Policy Forward Pass (MaskablePPO internals)
# =========================================================================

def get_policy_logits(policy: nn.Module, observations: torch.Tensor) -> torch.Tensor:
    """MaskablePPO policy ağından ham logitleri çıkart.

    SB3 MaskableActorCriticPolicy yapısı:
        features = extract_features(obs, self.pi_features_extractor)
        latent_pi = mlp_extractor.forward_actor(features)
        action_logits = action_net(latent_pi)

    Returns:
        (batch_size, TOTAL_ACTIONS) logit tensor
    """
    # features_extractor (MlpPolicy'de genellikle identity / FlattenExtractor)
    features = policy.extract_features(observations, policy.pi_features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    logits = policy.action_net(latent_pi)
    return logits


# =========================================================================
# Behavior Cloning Training
# =========================================================================

def pretrain(
    gen: int,
    data_gen: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device_str: str = "auto",
    verbose: bool = True,
) -> str:
    """Expert verisiyle BC ön-eğitim yap.

    Args:
        gen: Eğitilecek modelin nesil numarası
        data_gen: Veri kaynağı nesil numarası (None ise gen-1)
        epochs: BC epoch sayısı
        batch_size: Minibatch boyutu
        device_str: "auto", "cuda", "cpu"
        verbose: Detaylı log

    Returns:
        Kaydedilen model yolu
    """
    data_gen = data_gen if data_gen is not None else (gen - 1)
    epochs = epochs or BC_EPOCHS
    batch_size = batch_size or BC_BATCH_SIZE

    # Cihaz seçimi
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Gen {gen}] Behavior Cloning Ön-Eğitim")
        print(f"  Veri kaynağı : Gen {data_gen}")
        print(f"  Cihaz        : {device}")
        print(f"  Epochs       : {epochs}")
        print(f"  Batch size   : {batch_size}")
        print(f"  LR warm-up   : {BC_LR_START} -> {BC_LR_TARGET} ({BC_WARMUP_STEPS} adim)")
        print(f"{'='*60}\n")

    # --- Veri yükle ---
    observations, action_masks, actions = load_expert_data(data_gen)

    if verbose:
        print(f"  Yüklenen veri: {observations.shape[0]} sample")
        print(f"  Observation  : {observations.shape}")
        print(f"  Unique action: {len(np.unique(actions))}")

    # --- Train / validation split ---
    n_total = len(observations)
    n_val = max(1, int(n_total * BC_VALIDATION_SPLIT))
    n_train = n_total - n_val

    # Shuffle
    perm = np.random.permutation(n_total)
    observations = observations[perm]
    action_masks = action_masks[perm]
    actions = actions[perm]

    train_obs = torch.tensor(observations[:n_train], dtype=torch.float32)
    train_masks = torch.tensor(action_masks[:n_train], dtype=torch.bool)
    train_actions = torch.tensor(actions[:n_train], dtype=torch.long)

    val_obs = torch.tensor(observations[n_train:], dtype=torch.float32)
    val_masks = torch.tensor(action_masks[n_train:], dtype=torch.bool)
    val_actions = torch.tensor(actions[n_train:], dtype=torch.long)

    train_dataset = TensorDataset(train_obs, train_masks, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if verbose:
        print(f"  Train samples: {n_train}, Val samples: {n_val}")

    # --- MaskablePPO modeli oluştur ---
    # Geçici DummyVecEnv — sadece space tanımları için
    dummy_env = DummyVecEnv([lambda: Monitor(BlockBlastGymEnv())])

    policy_kwargs = dict(net_arch=[dict(pi=list(POLICY_ARCH_PI), vf=list(POLICY_ARCH_VF))])

    model = MaskablePPO(
        "MlpPolicy",
        env=dummy_env,
        learning_rate=BC_LR_TARGET,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )
    dummy_env.close()

    policy = model.policy
    policy.train()

    # --- Optimizer & Scheduler ---
    # Sadece policy (actor) parametrelerini eğit, value head'i dondur
    actor_params = list(policy.mlp_extractor.policy_net.parameters()) + \
                   list(policy.action_net.parameters())

    optimizer = torch.optim.Adam(actor_params, lr=BC_LR_START)
    scheduler = WarmupScheduler(optimizer, BC_WARMUP_STEPS, BC_LR_START, BC_LR_TARGET)

    loss_fn = nn.CrossEntropyLoss()

    # --- Eğitim ---
    history: list[dict] = []
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_epoch = time.time()

        policy.train()
        for batch_obs, batch_masks, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)

            # Forward pass — ham logitler
            logits = get_policy_logits(policy, batch_obs)

            # Action masking — geçersiz aksiyonların logitlerini -inf yap
            logits[~batch_masks] = -1e8

            # Loss
            loss = loss_fn(logits, batch_actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            optimizer.step()
            lr = scheduler.step()
            global_step += 1

            # Metrikler
            epoch_loss += loss.item() * batch_obs.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == batch_actions).sum().item()
            epoch_total += batch_obs.size(0)

        avg_train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        # --- Validation ---
        policy.eval()
        with torch.no_grad():
            v_obs = val_obs.to(device)
            v_masks = val_masks.to(device)
            v_actions = val_actions.to(device)

            v_logits = get_policy_logits(policy, v_obs)
            v_logits[~v_masks] = -1e8
            val_loss = loss_fn(v_logits, v_actions).item()
            v_preds = v_logits.argmax(dim=1)
            val_acc = (v_preds == v_actions).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epoch_time = time.time() - t_epoch

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "lr": round(lr, 8),
            "time_s": round(epoch_time, 2),
        }
        history.append(record)

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss: {avg_train_loss:.4f} | acc: {train_acc:.3f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f} | "
                f"lr: {lr:.2e} | {epoch_time:.1f}s"
            )

    # --- Kaydet ---
    out_dir = gen_dir(gen)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "pretrained_policy")
    model.save(model_path)

    # Eğitim logunu kaydet
    log_path = os.path.join(out_dir, "bc_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  Model kaydedildi  : {model_path}.zip")
        print(f"  Eğitim logu       : {log_path}")
        print(f"  En iyi val_loss   : {best_val_loss:.4f}")
        print(f"  Son val_acc       : {history[-1]['val_acc']:.3f}")

    return f"{model_path}.zip"


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Block Blast BC Pre-Training")
    parser.add_argument("--gen", type=int, required=True, help="Eğitilecek modelin nesil numarası")
    parser.add_argument("--data-gen", type=int, default=None, help="Veri kaynağı nesil numarası")
    parser.add_argument("--epochs", type=int, default=None, help=f"Epoch sayısı (varsayılan: {BC_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Batch size (varsayılan: {BC_BATCH_SIZE})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    pretrain(
        gen=args.gen,
        data_gen=args.data_gen,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
