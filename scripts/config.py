"""Block Blast AI — Self-Play Pipeline Merkezi Konfigürasyonu.

Tüm katsayılar, hiperparametreler ve yol tanımları bu dosyadan okunur.
utils/metrics.py de bu dosyadaki reward sabitlerini import eder.
"""

from __future__ import annotations

import os

# =========================================================================
# 1. Reward Katsayıları (utils/metrics.py bu değerleri kullanır)
# =========================================================================
LINE_CLEAR_COEF: float = 20.0
AGG_HEIGHT_COEF: float = 0.35
BUMPINESS_COEF: float = 0.25
HOLE_COEF: float = 5.0
SURVIVAL_BONUS_PER_STEP: float = 4.0
GAME_OVER_PENALTY: float = 35.0
REGRET_PENALTY_PER_BLOCKED_PIECE: float = 15.0

# =========================================================================
# 2. Expert Data Collection
# =========================================================================
EXPERT_MIN_SCORE: int = 100          # Sadece 100+ skor oyunlarını kaydet
EXPERT_MOVES_TARGET: int = 100_000   # Hedef hamle sayısı (augmentation öncesi)
AUGMENTATION_FACTOR: int = 8         # 8-way augmentation çarpanı

# =========================================================================
# 3. Pretrain — Behavior Cloning
# =========================================================================
BC_BATCH_SIZE: int = 512
BC_EPOCHS: int = 30
BC_LR_START: float = 1e-6           # Warm-up başlangıç LR
BC_LR_TARGET: float = 3e-4          # Warm-up hedef LR
BC_WARMUP_STEPS: int = 5_000        # İlk N adımda LR kademeli artış
BC_VALIDATION_SPLIT: float = 0.05   # Eğitim verisinin %5'i doğrulama için

# =========================================================================
# 4. RL Fine-Tune
# =========================================================================
RL_TIMESTEPS: int = 20_000_000       # PPO toplam adım
RL_N_ENVS: int = 128                 # Paralel ortam sayısı
RL_N_STEPS: int = 2048               # Ortam başına rollout uzunluğu
RL_BATCH_SIZE: int = 8192            # PPO minibatch (VRAM'e göre ayarla)
RL_N_EPOCHS: int = 4                 # PPO epoch
RL_LEARNING_RATE: float = 3e-4       # PPO learning rate
RL_ENT_COEF: float = 0.08           # Entropy katsayısı
RL_GAMMA: float = 0.99
RL_GAE_LAMBDA: float = 0.95
RL_CLIP_RANGE_START: float = 0.2
RL_CLIP_RANGE_END: float = 0.1
RL_VF_COEF: float = 0.5
RL_MAX_GRAD_NORM: float = 0.5
RL_TARGET_KL: float = 0.02
RL_REPLAY_RATIO: float = 0.15       # Usta verisinin %15'i replay buffer'da
RL_KL_THRESHOLD: float = 0.05       # KL divergence sınırı

# =========================================================================
# 5. Evaluation
# =========================================================================
EVAL_N_GAMES: int = 100              # Değerlendirme maç sayısı
EVAL_SEED_BASE: int = 7000           # Deterministik seed başlangıcı
PROMOTION_WIN_RATE: float = 0.55     # %55 -> terfi

# =========================================================================
# 6. GPU-Vectorized Environment
# =========================================================================
GPU_ENV_ENABLED: bool = True          # GPU ortam kullanımı (CUDA varsa)
GPU_N_ENVS: int = 2048               # GPU'da paralel ortam sayısı (CPU'da 128 idi)
GPU_N_STEPS: int = 512               # GPU ortam başına rollout uzunluğu
GPU_BATCH_SIZE: int = 65536           # GPU batch size (5090 32GB VRAM yeter)
GPU_N_EPOCHS: int = 10               # GPU ile daha fazla epoch — GPU hep dolu

# =========================================================================
# 7. Network Architecture
# =========================================================================
POLICY_ARCH_PI: list[int] = [512, 512, 256]
POLICY_ARCH_VF: list[int] = [512, 512, 256]

# =========================================================================
# 7. Paths
# =========================================================================
# Proje kök dizini: block_blast_ai/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
GENERATION_LOG: str = os.path.join(LOGS_DIR, "generation_history.json")


def gen_dir(gen: int) -> str:
    """Nesil bazlı model dizini: models/gen_N/"""
    return os.path.join(MODELS_DIR, f"gen_{gen}")


def gen_data_dir(gen: int) -> str:
    """Nesil bazlı veri dizini: data/gen_N/"""
    return os.path.join(DATA_DIR, f"gen_{gen}")
