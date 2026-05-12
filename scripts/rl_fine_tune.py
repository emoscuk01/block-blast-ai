"""Block Blast AI — RL Fine-Tuning (PPO).

BC ile ön-eğitilmiş modeli MaskablePPO ile geliştir.
Catastrophic Forgetting önleyiciler: KL Divergence monitoring + Experience Replay.

Kullanım:
    python -m scripts.rl_fine_tune --gen 1
    python -m scripts.rl_fine_tune --gen 1 --timesteps 1000000  # Hızlı test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Proje kökünü path'e ekle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# PERF: Çok sayıda Subproc worker varken thread şişmesini önle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from scripts.config import (
    RL_TIMESTEPS,
    RL_N_ENVS,
    RL_N_STEPS,
    RL_BATCH_SIZE,
    RL_N_EPOCHS,
    RL_LEARNING_RATE,
    RL_ENT_COEF,
    RL_GAMMA,
    RL_GAE_LAMBDA,
    RL_CLIP_RANGE_START,
    RL_CLIP_RANGE_END,
    RL_VF_COEF,
    RL_MAX_GRAD_NORM,
    RL_TARGET_KL,
    RL_REPLAY_RATIO,
    RL_KL_THRESHOLD,
    POLICY_ARCH_PI,
    POLICY_ARCH_VF,
    GPU_ENV_ENABLED,
    GPU_N_ENVS,
    GPU_N_STEPS,
    GPU_BATCH_SIZE,
    GPU_N_EPOCHS,
    gen_dir,
    gen_data_dir,
    LOGS_DIR,
)
from rl.gym_env import BlockBlastGymEnv
from rl.action_mapper import TOTAL_ACTIONS


# =========================================================================
# Linear Schedule (clip_range için)
# =========================================================================

def linear_schedule(initial_value: float, final_value: float):
    """SB3 callable schedule: progress_remaining 1.0 -> 0.0."""
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


# =========================================================================
# KL Divergence Monitoring Callback
# =========================================================================

class KLDivergenceCallback(BaseCallback):
    """Her PPO update sonrası, mevcut politikayı referans politikayla karşılaştırır.

    KL divergence sınırı aşılırsa LR'yi geçici olarak düşürür.
    """

    def __init__(
        self,
        reference_model_path: str,
        kl_threshold: float = 0.05,
        sample_size: int = 2048,
        check_freq: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.reference_model_path = reference_model_path
        self.kl_threshold = kl_threshold
        self.sample_size = sample_size
        self.check_freq = check_freq
        self._ref_policy: Optional[nn.Module] = None
        self._rollout_count = 0
        self._original_lr: Optional[float] = None
        self._lr_reduced = False
        self._sample_obs: Optional[torch.Tensor] = None

    def _on_training_start(self) -> None:
        """Referans modeli yükle ve sample observation'ları hazırla."""
        try:
            ref_model = MaskablePPO.load(
                self.reference_model_path,
                device=self.model.device,
            )
            self._ref_policy = ref_model.policy
            self._ref_policy.eval()
            # Parametre gradyanlarını kapat
            for param in self._ref_policy.parameters():
                param.requires_grad = False

            if self.verbose:
                print(f"[KL Monitor] Referans model yüklendi: {self.reference_model_path}")
        except Exception as e:
            print(f"[KL Monitor] UYARI: Referans model yüklenemedi: {e}")
            self._ref_policy = None

    def _on_rollout_end(self) -> None:
        """Her N rollout'ta bir KL divergence hesapla."""
        if self._ref_policy is None:
            return

        self._rollout_count += 1
        if self._rollout_count % self.check_freq != 0:
            return

        try:
            kl = self._compute_kl()
            self.logger.record("safety/kl_divergence", kl)

            if kl > self.kl_threshold:
                if not self._lr_reduced:
                    # LR'yi yarıya düşür
                    current_lr = self.model.learning_rate
                    if callable(current_lr):
                        # Schedule fonksiyonu — dokunma
                        pass
                    else:
                        self._original_lr = current_lr
                        self.model.learning_rate = current_lr * 0.5
                        self._lr_reduced = True
                        if self.verbose:
                            print(
                                f"[KL Monitor] KL={kl:.4f} > {self.kl_threshold} -> "
                                f"LR {current_lr:.2e} -> {current_lr * 0.5:.2e}"
                            )
            else:
                if self._lr_reduced and self._original_lr is not None:
                    self.model.learning_rate = self._original_lr
                    self._lr_reduced = False
                    if self.verbose:
                        print(f"[KL Monitor] KL={kl:.4f} normal -> LR {self._original_lr:.2e} geri yuklendi")

        except Exception as e:
            if self.verbose:
                print(f"[KL Monitor] KL hesaplama hatası: {e}")

    def _compute_kl(self) -> float:
        """Sample observation'lar üzerinde KL(current || reference) hesapla."""
        # Rollout buffer'dan sample al
        buf = self.model.rollout_buffer
        if buf.observations is None or len(buf.observations) == 0:
            return 0.0

        # Observations shape is typically (buffer_size, n_envs, obs_dim)
        # We need to flatten the first two dimensions to sample individual observations
        obs_flat = buf.observations.reshape(-1, buf.observations.shape[-1])
        
        n = min(self.sample_size, len(obs_flat))
        indices = np.random.choice(len(obs_flat), size=n, replace=False)
        sample_obs = torch.tensor(
            obs_flat[indices], dtype=torch.float32
        ).to(self.model.device)

        # Squeeze eğer gereksiz boyut varsa
        if sample_obs.dim() == 3 and sample_obs.shape[1] == 1:
            sample_obs = sample_obs.squeeze(1)

        with torch.no_grad():
            # Current policy logits
            cur_features = self.model.policy.extract_features(
                sample_obs, self.model.policy.pi_features_extractor
            )
            cur_latent = self.model.policy.mlp_extractor.forward_actor(cur_features)
            cur_logits = self.model.policy.action_net(cur_latent)
            cur_log_probs = torch.log_softmax(cur_logits, dim=-1)

            # Reference policy logits
            ref_features = self._ref_policy.extract_features(
                sample_obs, self._ref_policy.pi_features_extractor
            )
            ref_latent = self._ref_policy.mlp_extractor.forward_actor(ref_features)
            ref_logits = self._ref_policy.action_net(ref_latent)
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

        # KL(current || reference) = sum(p_cur * (log_p_cur - log_p_ref))
        cur_probs = torch.exp(cur_log_probs)
        kl = (cur_probs * (cur_log_probs - ref_log_probs)).sum(dim=-1).mean().item()
        return max(kl, 0.0)

    def _on_step(self) -> bool:
        return True


# =========================================================================
# Experience Replay Callback (Catastrophic Forgetting Önleyici)
# =========================================================================

class ExperienceReplayCallback(BaseCallback):
    """PPO rollout sonrası, usta verisinden %15 oranında supervised gradient adımı atar.

    Bu, modelin önceki neslin bilgisini tamamen unutmasını önler.
    """

    def __init__(
        self,
        expert_data_path: str,
        replay_ratio: float = 0.15,
        replay_batch_size: int = 512,
        replay_steps_per_rollout: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.expert_data_path = expert_data_path
        self.replay_ratio = replay_ratio
        self.replay_batch_size = replay_batch_size
        self.replay_steps_per_rollout = replay_steps_per_rollout

        self._expert_obs: Optional[torch.Tensor] = None
        self._expert_masks: Optional[torch.Tensor] = None
        self._expert_actions: Optional[torch.Tensor] = None
        self._loss_fn = nn.CrossEntropyLoss()

    def _on_training_start(self) -> None:
        """Expert verisini yükle."""
        try:
            data = np.load(self.expert_data_path)
            obs = data["observations"]
            masks = data["action_masks"]
            actions = data["actions"]

            # replay_ratio kadar rastgele sample al
            n_keep = int(len(obs) * self.replay_ratio)
            if n_keep < self.replay_batch_size:
                n_keep = min(len(obs), self.replay_batch_size * 10)

            indices = np.random.choice(len(obs), size=n_keep, replace=False)

            self._expert_obs = torch.tensor(obs[indices], dtype=torch.float32)
            self._expert_masks = torch.tensor(masks[indices], dtype=torch.bool)
            self._expert_actions = torch.tensor(actions[indices], dtype=torch.long)

            if self.verbose:
                print(
                    f"[Replay] Expert verisi yüklendi: {n_keep}/{len(obs)} sample "
                    f"(%{self.replay_ratio*100:.0f})"
                )
        except Exception as e:
            print(f"[Replay] UYARI: Expert verisi yüklenemedi: {e}")

    def _on_rollout_end(self) -> None:
        """Her rollout sonrası usta verisinden supervised gradient adımları at."""
        if self._expert_obs is None:
            return

        policy = self.model.policy
        device = self.model.device
        policy.train()

        # Actor parametreleri
        actor_params = list(policy.mlp_extractor.policy_net.parameters()) + \
                       list(policy.action_net.parameters())
        optimizer = torch.optim.Adam(actor_params, lr=1e-4)  # Düşük LR — nazik güncelleme

        total_loss = 0.0
        for _ in range(self.replay_steps_per_rollout):
            # Rastgele batch seç
            indices = torch.randint(0, len(self._expert_obs), (self.replay_batch_size,))
            batch_obs = self._expert_obs[indices].to(device)
            batch_masks = self._expert_masks[indices].to(device)
            batch_actions = self._expert_actions[indices].to(device)

            # Forward
            features = policy.extract_features(batch_obs, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            logits[~batch_masks] = -1e8

            loss = self._loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(self.replay_steps_per_rollout, 1)
        self.logger.record("replay/bc_loss", avg_loss)

    def _on_step(self) -> bool:
        return True


# =========================================================================
# VecNormalize Kaydetme Callback
# =========================================================================

class SaveVecNormalizeCallback(BaseCallback):
    """En iyi model bulunduğunda VecNormalize istatistiklerini de kaydet."""

    def __init__(self, save_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            vec_norm.save(self.save_path)
        return True


# =========================================================================
# RL Fine-Tune
# =========================================================================

def rl_fine_tune(
    gen: int,
    timesteps: int | None = None,
    n_envs: int | None = None,
    batch_size: int | None = None,
    device_str: str = "auto",
    vec_env_type: str = "subproc",
    verbose: bool = True,
) -> str:
    """BC ile ön-eğitilmiş modeli PPO ile geliştir.

    Args:
        gen: Nesil numarası
        timesteps: Toplam PPO adımı
        n_envs: Paralel ortam sayısı
        batch_size: PPO minibatch
        device_str: "auto", "cuda", "cpu"
        vec_env_type: "subproc" veya "dummy"
        verbose: Detaylı log

    Returns:
        Kaydedilen model yolu
    """
    timesteps = timesteps or RL_TIMESTEPS
    n_envs = n_envs or RL_N_ENVS
    batch_size = batch_size or RL_BATCH_SIZE

    # --- GPU Setup ---
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        else:
            device = "cpu"
            print("UYARI: CUDA bulunamadı — eğitim CPU'da çalışıyor.")
    else:
        device = device_str

    # --- Dizinler ---
    model_dir = gen_dir(gen)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"gen_{gen}_rl_{timestamp}")

    # --- Ortam ---
    use_gpu_env = (device == "cuda") and GPU_ENV_ENABLED
    if use_gpu_env:
        from rl.gpu_vec_env import GpuVecEnv
        n_envs = GPU_N_ENVS  # GPU'da daha fazla paralel ortam
        print(f"[Gen {gen}] GPU ENV: {n_envs} paralel ortam tek GPU tensöründe")
        train_env = GpuVecEnv(n_envs=n_envs, device=torch.device("cuda"))
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )
    else:
        if vec_env_type == "subproc" and n_envs > 1:
            vec_cls = SubprocVecEnv
        else:
            vec_cls = DummyVecEnv

        train_env = make_vec_env(
            BlockBlastGymEnv,
            n_envs=n_envs,
            seed=0,
            wrapper_class=Monitor,
            vec_env_cls=vec_cls,
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )

    # Eval ortamı
    eval_env = DummyVecEnv([lambda: Monitor(BlockBlastGymEnv(seed=None))])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # --- Rollout boyut kontrolü ---
    n_steps = GPU_N_STEPS if use_gpu_env else RL_N_STEPS
    rollout_size = n_steps * n_envs
    eff_batch = min(batch_size, rollout_size)
    eff_epochs = RL_N_EPOCHS

    # GPU auto-tune
    if device == "cuda":
        if use_gpu_env:
            eff_batch = min(GPU_BATCH_SIZE, rollout_size)
            eff_epochs = GPU_N_EPOCHS
        else:
            floor_batch = min(32768, rollout_size)
            epoch_floor = 8
            eff_batch = max(eff_batch, floor_batch)
            eff_batch = min(eff_batch, rollout_size)
            eff_epochs = max(eff_epochs, epoch_floor)

    # --- Model yükle ---
    pretrained_path = os.path.join(model_dir, "pretrained_policy.zip")
    if os.path.exists(pretrained_path):
        model = MaskablePPO.load(pretrained_path, env=train_env, device=device)
        model.tensorboard_log = log_dir

        # PPO hiperparametrelerini güncelle
        model.n_steps = n_steps
        model.batch_size = eff_batch
        model.n_epochs = eff_epochs
        model.ent_coef = RL_ENT_COEF
        model.learning_rate = RL_LEARNING_RATE
        model.gamma = RL_GAMMA
        model.gae_lambda = RL_GAE_LAMBDA
        model.clip_range = linear_schedule(RL_CLIP_RANGE_START, RL_CLIP_RANGE_END)
        model.vf_coef = RL_VF_COEF
        model.max_grad_norm = RL_MAX_GRAD_NORM
        model.target_kl = RL_TARGET_KL
        # Buffer'ı doğru tipte yeniden oluştur (n_steps değiştiği için)
        from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
        model.rollout_buffer = MaskableRolloutBuffer(
            buffer_size=n_steps,
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=n_envs,
        )

        if verbose:
            print(f"[Gen {gen}] Ön-eğitimli model yüklendi: {pretrained_path}")
    else:
        # Sıfırdan oluştur (BC verisi yoksa)
        policy_kwargs = dict(net_arch=[dict(pi=list(POLICY_ARCH_PI), vf=list(POLICY_ARCH_VF))])
        model = MaskablePPO(
            "MlpPolicy",
            env=train_env,
            learning_rate=RL_LEARNING_RATE,
            n_steps=n_steps,
            batch_size=eff_batch,
            n_epochs=eff_epochs,
            gamma=RL_GAMMA,
            gae_lambda=RL_GAE_LAMBDA,
            clip_range=linear_schedule(RL_CLIP_RANGE_START, RL_CLIP_RANGE_END),
            ent_coef=RL_ENT_COEF,
            vf_coef=RL_VF_COEF,
            max_grad_norm=RL_MAX_GRAD_NORM,
            target_kl=RL_TARGET_KL,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )
        if verbose:
            print(f"[Gen {gen}] Sıfırdan model oluşturuldu (BC verisi bulunamadı)")

    # --- Callbacks ---
    callbacks = []

    # 1) KL Divergence Monitor
    kl_ref_path = pretrained_path if os.path.exists(pretrained_path) else None
    if kl_ref_path:
        kl_cb = KLDivergenceCallback(
            reference_model_path=kl_ref_path,
            kl_threshold=RL_KL_THRESHOLD,
            verbose=1,
        )
        callbacks.append(kl_cb)

    # 2) Experience Replay
    prev_gen = gen - 1
    expert_data_path = os.path.join(gen_data_dir(prev_gen), "expert_data.npz")
    if os.path.exists(expert_data_path):
        replay_cb = ExperienceReplayCallback(
            expert_data_path=expert_data_path,
            replay_ratio=RL_REPLAY_RATIO,
            verbose=1,
        )
        callbacks.append(replay_cb)
    elif verbose:
        print(f"[Gen {gen}] Expert replay verisi bulunamadı: {expert_data_path}")

    # 3) Eval Callback
    best_model_dir = os.path.join(model_dir, "best")
    os.makedirs(best_model_dir, exist_ok=True)
    vecnorm_path = os.path.join(best_model_dir, "vecnormalize.pkl")

    save_vecnorm_cb = SaveVecNormalizeCallback(save_path=vecnorm_path, verbose=1)
    eval_cb = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        use_masking=True,
        callback_on_new_best=save_vecnorm_cb,
        verbose=1,
    )
    callbacks.append(eval_cb)

    # --- Eğitim ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Gen {gen}] RL Fine-Tune Başlıyor")
        print(f"  Toplam adım  : {timesteps:,}")
        print(f"  Paralel ortam: {n_envs} ({vec_env_type})")
        print(f"  Rollout      : n_steps={n_steps} -> buffer={rollout_size:,}")
        print(f"  batch_size   : {eff_batch}  |  n_epochs: {eff_epochs}")
        print(f"  ent_coef     : {RL_ENT_COEF}")
        print(f"  KL threshold : {RL_KL_THRESHOLD}")
        print(f"  Replay ratio : {RL_REPLAY_RATIO}")
        print(f"  Cihaz        : {device}")
        print(f"  Log dizini   : {log_dir}")
        print(f"{'='*60}\n")

    t_start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t_start

    # --- Kaydet ---
    final_model_path = os.path.join(model_dir, "rl_finetuned")
    final_vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    model.save(final_model_path)
    train_env.save(final_vecnorm_path)

    if verbose:
        print(f"\n[Gen {gen}] RL Fine-Tune Tamamlandı ({elapsed:.0f}s)")
        print(f"  Son model       : {final_model_path}.zip")
        print(f"  VecNormalize    : {final_vecnorm_path}")
        print(f"  En iyi model    : {best_model_dir}/best_model.zip")
        print(f"  TensorBoard     : tensorboard --logdir {log_dir}")

    train_env.close()
    eval_env.close()

    return f"{final_model_path}.zip"


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Block Blast RL Fine-Tune")
    parser.add_argument("--gen", type=int, required=True, help="Nesil numarası")
    parser.add_argument("--timesteps", type=int, default=None, help=f"Toplam adım (varsayılan: {RL_TIMESTEPS:,})")
    parser.add_argument("--n-envs", type=int, default=None, help=f"Paralel ortam (varsayılan: {RL_N_ENVS})")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Batch size (varsayılan: {RL_BATCH_SIZE})")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"])
    args = parser.parse_args()

    rl_fine_tune(
        gen=args.gen,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        device_str=args.device,
        vec_env_type=args.vec_env,
    )


if __name__ == "__main__":
    main()
