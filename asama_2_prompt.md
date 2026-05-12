# Block Blast AI — Aşama 2: RL Modeli
# Gymnasium Wrapper + Stable-Baselines3 DQN

---

## BAĞLAM

Aşama 0+1 tamamlandı. Elimizde şunlar var:
- `env/game_env.py` → `GameEnv` sınıfı (step, reset, clone, get_valid_actions)
- `env/board.py` → `Board` sınıfı (8×8 tahta, satır/sütun silme)
- `env/pieces.py` → 28 blok şekli
- `utils/metrics.py` → reward, regret, composite_score
- `agents/heuristic_agent.py` → baseline bot (12.7x random'ı geçiyor)

Bu aşamada `GameEnv`'i Gymnasium standardına sarıp Stable-Baselines3 ile
DQN eğitiyoruz. Mevcut kodlara **dokunma**, sadece üstüne ekle.

---

## DOSYA YAPISI (SADECE YENİ DOSYALAR)

```
block_blast_ai/
├── rl/
│   ├── __init__.py
│   ├── gym_env.py          # Gymnasium wrapper
│   ├── action_mapper.py    # Aksiyon indeksi ↔ (piece, row, col) dönüşümü
│   ├── observation.py      # State'i sinir ağına uygun vektöre çevir
│   └── callbacks.py        # Eğitim sırasında loglama ve erken durdurma
├── train.py                # Ana eğitim scripti
├── evaluate.py             # Eğitilmiş modeli test et, heuristik ile kıyasla
├── models/                 # Kaydedilen model dosyaları (boş klasör, .gitkeep)
│   └── .gitkeep
└── requirements_rl.txt     # gymnasium, stable-baselines3, tensorboard
```

---

## MODÜL 1: `rl/action_mapper.py`

### Görev
Aksiyon uzayını düzleştir. DQN tek bir tamsayı üretir;
bu tamsayıyı `(piece_index, row, col)` üçlüsüne çevirmemiz gerekir.

### Tasarım
- `piece_index`: 0, 1, 2 → 3 seçenek
- `row`: 0–7 → 8 seçenek
- `col`: 0–7 → 8 seçenek
- Toplam: 3 × 8 × 8 = **192 aksiyon**

```python
TOTAL_ACTIONS = 192  # 3 * 8 * 8

def action_to_tuple(action: int) -> tuple[int, int, int]:
    """
    0–191 arası tamsayıyı (piece_index, row, col) üçlüsüne çevir.
    Formül:
        piece_index = action // 64
        row         = (action % 64) // 8
        col         = action % 8
    """

def tuple_to_action(piece_index: int, row: int, col: int) -> int:
    """
    (piece_index, row, col) üçlüsünü 0–191 arası tamsayıya çevir.
    Formül: piece_index * 64 + row * 8 + col
    """

def get_valid_action_mask(env) -> np.ndarray:
    """
    Shape: (192,), dtype: bool
    env.get_valid_actions() listesini al,
    geçerli indeksleri True, geçersizleri False yap.
    DQN'nin geçersiz hamle seçmesini engellemek için kullanılır.
    """
```

---

## MODÜL 2: `rl/observation.py`

### Görev
`GameEnv.get_observation()` dict'ini sinir ağına verebileceğimiz
düz bir NumPy vektörüne çevir.

### Observation vektörü tasarımı

```
[ tahta (64) | parça_0 (25) | parça_1 (25) | parça_2 (25) | blok_kaldı (3) ]
Toplam: 64 + 75 + 3 = 142 boyutlu vektör, dtype=float32
```

- **Tahta (64):** 8×8 grid'i flatten et, 0.0/1.0 değerleri
- **Parça (25):** Her parçayı 5×5 padding'li matrise yerleştir ve flatten et.
  Parça yerleştirilmişse (None) → 25 adet 0.0
- **Blok kaldı (3):** One-hot encoding. Turda 3 blok kaldıysa [1,0,0],
  2 kaldıysa [0,1,0], 1 kaldıysa [0,0,1]

```python
OBS_SIZE = 142  # 64 + 25*3 + 3

def encode_observation(obs_dict: dict) -> np.ndarray:
    """
    GameEnv.get_observation() çıktısını (142,) float32 vektöre çevir.
    """

def encode_piece(piece_name: str | None) -> np.ndarray:
    """
    Tek bir parçayı 5×5 padding'li (25,) float32 vektöre çevir.
    piece_name None ise sıfır vektörü döndür.
    """
```

---

## MODÜL 3: `rl/gym_env.py`

### Görev
`GameEnv`'i `gymnasium.Env` subclass'ına sar.
Bu wrapper Stable-Baselines3'ün anlayacağı standart arayüzü sağlar.

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.game_env import GameEnv
from rl.action_mapper import TOTAL_ACTIONS, action_to_tuple, get_valid_action_mask
from rl.observation import encode_observation, OBS_SIZE

class BlockBlastGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "ascii"]}

    def __init__(self, seed: int = None, render_mode: str = None):
        super().__init__()
        self.game_env = GameEnv(seed=seed)
        self.render_mode = render_mode

        # Aksiyon uzayı: 192 discrete aksiyon
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Observation uzayı: 142 boyutlu [0,1] vektör
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        Gymnasium API: (observation, info) döndür.
        GameEnv.reset() çağır, encode_observation ile vektöre çevir.
        """

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Gymnasium API: (obs, reward, terminated, truncated, info) döndür.
        truncated her zaman False.

        AKSIYON MASKELEME:
        - action geçerli değilse (get_valid_action_mask False döndürüyorsa)
          küçük bir ceza ver (-5.0) ve oyuna devam et (terminated=False).
          Bu sayede model geçersiz hamleleri öğrenir ama oyun ölmez.
        - Geçerli aksiyonsa GameEnv.step() çağır, reward hesapla.

        reward nasıl hesaplanır:
          1. GameEnv.step() döndürdüğü bilgilerle compute_reward() çağır.
          2. compute_regret() ile pişmanlık cezasını ekle.
          3. Toplam reward'ı döndür.
        """

    def render(self):
        """render_mode="ascii" ise GameEnv.render() sonucunu yazdır."""

    def get_action_mask(self) -> np.ndarray:
        """
        Shape: (192,) bool array.
        SB3'ün MaskableDQN'i için gerekli.
        Şimdilik standart DQN kullanıyoruz ama bu metodu hazır tut.
        """

    def action_masks(self) -> np.ndarray:
        """get_action_mask() ile aynı, SB3 naming convention için alias."""
```

---

## MODÜL 4: `rl/callbacks.py`

### Görev
Eğitim sırasında metrikleri kaydet ve erken durdurma uygula.

```python
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

class TrainingMetricsCallback(BaseCallback):
    """
    Her episode bitişinde şunları logla:
    - ep_score: o oyundaki toplam oyun skoru
    - ep_turns: kaç tur sürdü
    - ep_reward: toplam RL reward'ı
    TensorBoard ve terminale yaz.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        ...

    def _on_step(self) -> bool:
        """
        Her step sonrası çağrılır.
        Episode bittiyse (done=True) metrikleri topla ve logla.
        True döndür (eğitim devam etsin).
        """

class HeuristicComparisonCallback(BaseCallback):
    """
    Her N step'te bir eğitilen modeli heuristik agent ile kıyasla.
    Sonucu TensorBoard'a "rl_vs_heuristic_ratio" olarak logla.
    Bu grafik "model ne zaman baseline'ı geçti" sorusunu yanıtlar.
    """
    def __init__(self, eval_env, heuristic_agent, eval_freq: int = 10000, n_eval_episodes: int = 20):
        ...
```

---

## MODÜL 5: `train.py`

### Görev
Modeli eğit, kaydet, logla. Bu script hem local hem Vast.ai'de çalışacak.

```python
"""
Kullanım:
    python train.py                          # Varsayılan ayarlar
    python train.py --timesteps 1000000      # Daha uzun eğitim
    python train.py --resume models/dqn_v1   # Kaldığı yerden devam
"""
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from rl.gym_env import BlockBlastGymEnv
from rl.callbacks import TrainingMetricsCallback, HeuristicComparisonCallback
from agents.heuristic_agent import HeuristicAgent

def make_env(seed: int = 0):
    """Monitor ile sarılmış tek env döndür."""
    env = BlockBlastGymEnv(seed=seed)
    return Monitor(env)

def train(args):
    # Eğitim ortamı: 4 paralel env (CPU'yu daha iyi kullan)
    train_env = make_vec_env(make_env, n_envs=4)

    # Değerlendirme ortamı: tek env, sabit seed
    eval_env = Monitor(BlockBlastGymEnv(seed=9999))

    # DQN konfigürasyonu
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,       # Replay buffer boyutu
        learning_starts=10_000,    # Bu kadar random adımdan sonra öğrenmeye başla
        batch_size=64,
        tau=1.0,                   # Target network güncelleme oranı
        gamma=0.99,                # İndirim faktörü
        train_freq=4,              # Her 4 step'te bir güncelle
        target_update_interval=1000,
        exploration_fraction=0.2,  # Toplam adımın %20'sinde epsilon düşsün
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/",
        device="auto",             # GPU varsa kullan, yoksa CPU
    )

    # Kaldığı yerden devam et
    if args.resume:
        model = DQN.load(args.resume, env=train_env)
        print(f"Model yüklendi: {args.resume}")

    # Callback'ler
    metrics_cb = TrainingMetricsCallback(log_freq=1000)
    heuristic_cb = HeuristicComparisonCallback(
        eval_env=eval_env,
        heuristic_agent=HeuristicAgent(),
        eval_freq=20_000,
        n_eval_episodes=20
    )

    # Eğitim
    model.learn(
        total_timesteps=args.timesteps,
        callback=[metrics_cb, heuristic_cb],
        reset_num_timesteps=not args.resume,
    )

    # Kaydet
    save_path = f"models/dqn_v{args.version}"
    model.save(save_path)
    print(f"Model kaydedildi: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    train(args)
```

---

## MODÜL 6: `evaluate.py`

### Görev
Eğitilmiş modeli yükle, hem random hem heuristik ile kıyasla.

```python
"""
Kullanım:
    python evaluate.py --model models/dqn_v1 --games 100
"""

# Çıktı formatı:
# === DEĞERLENDİRME SONUÇLARI (N=100 oyun) ===
#
# Agent              | Ort. Skor | Maks Skor | Ort. Tur | Heuristik Oranı
# -------------------|-----------|-----------|----------|-----------------
# DQN (eğitilmiş)   |    XXXX.X |    XXXXX  |    XXX.X |          X.XXx
# HeuristicAgent     |     182.8 |      585  |     16.4 |          1.00x
# RandomAgent        |      14.4 |       90  |      4.8 |          0.08x
```

---

## `requirements_rl.txt`

```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
tensorboard>=2.15.0
torch>=2.0.0
```

---

## VAST.AI DEPLOYMENT NOTU

Vast.ai'de çalıştırmak için `train.py` değişmez. Sadece şunları yap:

```bash
# Vast.ai'de Docker container başlatınca:
pip install -r requirements.txt
pip install -r requirements_rl.txt
python train.py --timesteps 2000000 --version 2

# TensorBoard'u görmek için (Vast.ai port forwarding ile):
tensorboard --logdir ./logs/ --port 6006
```

---

## KOD STİLİ

- Tüm yorum ve docstring'ler Türkçe
- Type hint'ler zorunlu
- `make_vec_env` ile 4 paralel ortam kullan (eğitimi hızlandırır)
- Model kayıt ismi versiyonlu olsun: `dqn_v1`, `dqn_v2` ...
- Her eğitim başlangıcında `logs/` altına timestamp'li klasör aç

---

## KONTROL LİSTESİ

- [ ] `BlockBlastGymEnv` gymnasium checker'dan geçiyor
      (`gymnasium.utils.env_checker.check_env(env)`)
- [ ] `train.py` hatasız çalışıyor, `models/` altına dosya kaydediyor
- [ ] TensorBoard `logs/` klasöründe grafik oluşturuyor
- [ ] `evaluate.py` üç agent'ı kıyaslayan tablo veriyor
- [ ] 500k step sonrası DQN en az heuristik'in %50'sine ulaşıyor
- [ ] 1M step sonrası DQN heuristik baseline'ı geçiyor
