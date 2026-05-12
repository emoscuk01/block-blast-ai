# Block Blast AI — Self-Play / Iterative Learning Mimarisi

Mevcut HeuristicAgent'tan başlayarak nesiller (generations) halinde kendi kendini geliştiren bir MaskablePPO eğitim pipeline'ı oluşturulacak. Sistem 5 ana script + 1 config modülünden oluşacak.

## Mevcut Kodla Entegrasyon

Yeni sistem mevcut modülleri **doğrudan kullanır**, hiçbir mevcut dosya değiştirilmez:

| Mevcut Modül | Kullanım |
|---|---|
| `env.game_env.GameEnv` | Expert veri toplama, evaluation oyunları |
| `env.board.Board` | 8-way augmentation (grid dönüşümleri) |
| `rl.gym_env.BlockBlastGymEnv` | RL fine-tuning ortamı |
| `rl.observation.encode_observation`, `OBS_SIZE` | Observation vektörü üretimi |
| `rl.action_mapper.*` | Action encoding/decoding, mask üretimi |
| `utils.metrics.*` | Reward hesaplama (katsayılar config'den okunacak) |
| `agents.heuristic_agent.HeuristicAgent` | Gen 0 uzman veri kaynağı |

## User Review Required

> [!IMPORTANT]
> **Reward Katsayıları:** `utils/metrics.py` içindeki sabit katsayılar (`LINE_CLEAR_COEF`, `SURVIVAL_BONUS_PER_STEP` vb.) şu an hardcoded. İki yol var:
> 1. **Değiştirmeden bırak** — `config.py` sadece scripts içinde kullanılır, `metrics.py` dokunulmaz. (Self-play scriptleri kendi reward'ını hesaplamaz, `BlockBlastGymEnv.step` zaten `metrics.py` kullanıyor.)
> 2. **`metrics.py`'yi config'den okuyacak şekilde güncelle** — Tüm sistemi etkiler.
>
> **Önerim:** Seçenek 1 daha güvenli. `config.py` referans dokümantasyon olarak katsayıları tutar, ama `metrics.py`'deki değerler ground truth olarak kalır. Böylece mevcut `train.py` ve testler bozulmaz.

> [!WARNING]
> **`utils/metrics.py` Güncellemesi:** Kullanıcı açıkça "config.py'den oku" dediği için, `metrics.py`'deki 6 sabitin `config.py`'den import edilmesi planlanmıştır. Bu mevcut `train.py`, `evaluate.py`, ve testleri etkileyebilir. Tüm mevcut importlar (`from utils.metrics import ...`) çalışmaya devam edecek çünkü sabitler hâlâ `metrics.py` namespace'inde olacak.

## Open Questions

> [!IMPORTANT]
> **8-Way Augmentation & Action Mapping:** 8×8 tahtayı 90°/180°/270° döndürme ve aynalama yaparken, **parça şekilleri simetrik değil** (L_sag ≠ L_sol). Augmentation sırasında:
> - Tahta döndürülür → action koordinatları (row, col) yeniden hesaplanır
> - **Parça indeksi değişmez** (aynı slot'ta aynı parça kalır)
> - Parça şeklinin kendisi döndürülmez — sadece tahtadaki yerleşim pozisyonu dönüştürülür
>
> Bu yaklaşımı onaylıyor musun? Alternatif: parça şeklini de döndürmek (daha karmaşık, PIECES sözlüğünde bazı rotasyonlar zaten mevcut).

---

## Proposed Changes

### Dosya Yapısı

```
block_blast_ai/
├── scripts/                          # [NEW] Tüm self-play scriptleri
│   ├── __init__.py                   # [NEW] Package marker
│   ├── config.py                     # [NEW] Merkezi konfigürasyon
│   ├── collect_expert_data.py        # [NEW] Heuristik uzman veri toplama
│   ├── pretrain_apprentice.py        # [NEW] Behavior Cloning ön-eğitim
│   ├── rl_fine_tune.py               # [NEW] PPO fine-tuning (Vast.ai)
│   ├── evaluate_and_promote.py       # [NEW] Nesil vs nesil değerlendirme
│   └── master_control.py            # [NEW] Orkestra döngüsü
├── utils/
│   └── metrics.py                    # [MODIFY] Sabitleri config'den import et
└── models/
    ├── gen_0/                        # [NEW] Runtime — heuristik baseline verileri
    ├── gen_1/                        # [NEW] Runtime — ilk çırak modeli
    └── ...
```

---

### 1. Config Modülü

#### [NEW] [config.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/config.py)

Merkezi konfigürasyon — tüm katsayılar, hiperparametreler, yollar:

```python
# --- Reward Katsayıları (metrics.py bunları import edecek) ---
LINE_CLEAR_COEF = 20.0
AGG_HEIGHT_COEF = 0.35
BUMPINESS_COEF = 0.25
HOLE_COEF = 5.0
SURVIVAL_BONUS_PER_STEP = 4.0
GAME_OVER_PENALTY = 35.0
REGRET_PENALTY_PER_BLOCKED_PIECE = 15.0

# --- Expert Data Collection ---
EXPERT_MIN_SCORE = 100           # Sadece 100+ skor oyunları kaydet
EXPERT_MOVES_TARGET = 100_000    # Hedef hamle sayısı
AUGMENTATION_FACTOR = 8          # 8-way augmentation

# --- Pretrain (Behavior Cloning) ---
BC_BATCH_SIZE = 512
BC_EPOCHS = 30
BC_LR_START = 1e-6               # Warm-up başlangıç LR
BC_LR_TARGET = 3e-4              # Warm-up hedef LR
BC_WARMUP_STEPS = 5000           # İlk 5000 adım warm-up

# --- RL Fine-Tune ---
RL_TIMESTEPS = 20_000_000
RL_N_ENVS = 128
RL_N_STEPS = 2048
RL_BATCH_SIZE = 8192
RL_ENT_COEF = 0.08
RL_REPLAY_RATIO = 0.15           # Usta verisinin %15'i replay'de
RL_KL_THRESHOLD = 0.05           # KL divergence sınırı

# --- Evaluation ---
EVAL_N_GAMES = 100               # Değerlendirme maç sayısı
PROMOTION_WIN_RATE = 0.55        # %55 win-rate → terfi

# --- Network Architecture ---
POLICY_ARCH_PI = [512, 512, 256]
POLICY_ARCH_VF = [512, 512, 256]

# --- Paths ---
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_DIR = "data"
GENERATION_LOG = "logs/generation_history.json"
```

---

### 2. Expert Data Collector

#### [NEW] [collect_expert_data.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/collect_expert_data.py)

**Sorumluluk:** `HeuristicAgent` ile oyun oynatıp yüksek kaliteli `(observation, action_mask, expert_action)` üçlüleri toplar.

**Akış:**
1. `GameEnv` oluştur → `HeuristicAgent.select_action(env)` ile oyna
2. Her hamlede `encode_observation()` + `get_valid_action_mask()` + `tuple_to_action()` kaydet
3. Oyun bittiğinde: skor < `EXPERT_MIN_SCORE` → o oyunun tüm verisini at
4. Yeterli hamle toplanınca **8-way augmentation** uygula
5. Veriyi `data/gen_N/expert_data.npz` olarak kaydet

**8-Way Augmentation detayı:**
```
Orijinal tahta (8×8) → 4 rotasyon × 2 (flip/no-flip) = 8 varyant
Her varyant için:
  - board_flat: np.rot90(board, k) + optional np.fliplr
  - row, col: dönüşüme göre yeniden hesapla
  - piece slot vektörleri: değişmez (parça şekli aynı kalır)
  - action index: yeni (piece_idx, new_row, new_col) → tuple_to_action
  - action_mask: yeni tahta üzerinden get_valid_action_mask ile yeniden hesapla
```

> [!NOTE]
> Augmentation sırasında dönüştürülmüş action'ın yeni mask'ta geçerli olup olmadığı kontrol edilir. Geçersizse o augmented sample atılır.

**Çıktı formatı:** `expert_data.npz` — keys: `observations (N, 142)`, `action_masks (N, 192)`, `actions (N,)`

---

### 3. Pretrain (Behavior Cloning)

#### [NEW] [pretrain_apprentice.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/pretrain_apprentice.py)

**Sorumluluk:** Uzman verileriyle MaskablePPO'nun policy ağını supervised learning ile ön-eğitir.

**Teknik Detaylar:**

1. **MaskablePPO modeli oluştur** (aynı mimariyle: `MlpPolicy`, `net_arch`, obs/action space)
2. **Policy ağını çıkart**: `model.policy.mlp_extractor` + `model.policy.action_net`
3. **Custom training loop:**
   ```python
   for epoch in range(BC_EPOCHS):
       for batch in dataloader:
           obs, masks, expert_actions = batch
           logits = policy_forward(obs)           # (batch, 192)
           logits[~masks] = -1e8                   # Geçersiz aksiyonları maskele
           loss = CrossEntropyLoss(logits, expert_actions)
           loss.backward()
           optimizer.step()
   ```
4. **LR Warm-up Schedule:**
   - Adım 0–5000: `lr = BC_LR_START + (BC_LR_TARGET - BC_LR_START) * (step / BC_WARMUP_STEPS)`
   - Adım 5000+: `lr = BC_LR_TARGET` (sabit)
5. **Kayıt:** `models/gen_N/pretrained_policy.zip` + training loss grafiği

> [!IMPORTANT]
> Behavior Cloning sadece **policy ağını** eğitir. Value function (`vf`) eğitilmez — RL fine-tuning aşamasında PPO kendi value head'ini öğrenecek.

---

### 4. RL Fine-Tune

#### [NEW] [rl_fine_tune.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/rl_fine_tune.py)

**Sorumluluk:** BC ile ön-eğitilmiş modeli PPO ile geliştir. Catastrophic forgetting önleyiciler aktif.

**Teknik Detaylar:**

1. **Model yükleme:** `MaskablePPO.load("models/gen_N/pretrained_policy.zip", env=vec_env)`
2. **Ortam:** `SubprocVecEnv` × 128 + `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)`
3. **KL Divergence Monitoring (Custom Callback):**
   ```python
   class KLDivergenceCallback(BaseCallback):
       """Her PPO update sonrası, mevcut politikayı referans (usta) politikayla karşılaştırır."""
       def __init__(self, reference_model_path, kl_threshold=0.05):
           self.ref_policy = load_reference_policy(reference_model_path)
           self.kl_threshold = kl_threshold
       
       def _on_rollout_end(self):
           # Sample bir batch observation'la her iki politikanın log-prob'larını karşılaştır
           kl = compute_kl_divergence(self.ref_policy, self.model.policy, sample_obs)
           self.logger.record("safety/kl_divergence", kl)
           if kl > self.kl_threshold:
               # LR'yi geçici olarak düşür veya uyarı ver
               self.model.learning_rate *= 0.5
               logger.warning(f"KL={kl:.4f} > threshold, LR halved")
   ```
4. **Experience Replay Buffer (%15 usta verisi):**
   - Her PPO rollout'tan sonra, batch'in %15'ini önceki neslin expert verisinden al
   - Bu verilerle ek bir supervised gradient adımı at (policy ağı üzerinde)
   - Böylece model önceki neslin bilgisini tamamen unutmaz
5. **Kayıt:** `models/gen_N/rl_finetuned.zip` + `models/gen_N/vecnormalize.pkl`

**GPU Optimizasyonları (RTX 5090):**
- `torch.backends.cudnn.benchmark = True`
- `torch.set_float32_matmul_precision("high")`
- `batch_size=8192` (32GB VRAM'e uygun)
- `n_envs=128` ile `SubprocVecEnv`

---

### 5. Evaluate & Promote

#### [NEW] [evaluate_and_promote.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/evaluate_and_promote.py)

**Sorumluluk:** Yeni nesil (challenger) ile mevcut usta (master) arasında 100 oyunluk karşılaştırma.

**Akış:**
1. Her iki modeli (veya Gen 0 için HeuristicAgent) **aynı seed setleriyle** oynat
2. Karşılaştırma metriği: **oyun skoru** (daha yüksek = daha iyi)
3. Win-rate hesapla: `challenger_wins / total_games`
4. `win_rate >= PROMOTION_WIN_RATE (0.55)` → challenger yeni master olur

**Detay:**
- **Gen 0 master = HeuristicAgent** (model yok, doğrudan `select_action` ile oynar)
- **Gen 1+ master = önceki neslin RL modeli** (`MaskablePPO.load` + `predict`)
- Seeds: `range(7000, 7000 + EVAL_N_GAMES)` — deterministik, tekrarlanabilir
- Çıktı: `{"gen": N, "win_rate": 0.62, "challenger_avg": 185.3, "master_avg": 142.7, "promoted": true}`

---

### 6. Master Control (Orkestra)

#### [NEW] [master_control.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/scripts/master_control.py)

**Sorumluluk:** Tüm pipeline'ı nesiller halinde döngüsel çalıştırır.

**Döngü:**
```
for gen in range(start_gen, max_gen):
    1. collect_expert_data(gen)     # 100k hamle (Gen 0: heuristik, Gen N: önceki master model)
    2. pretrain_apprentice(gen)     # BC ön-eğitim
    3. rl_fine_tune(gen)            # 20M adım PPO
    4. promoted = evaluate_and_promote(gen)
    5. if promoted:
           master = gen model
           log_generation(gen, "PROMOTED")
       else:
           log_generation(gen, "REJECTED")
           # Opsiyonel: daha fazla RL adımıyla tekrar dene
```

**Loglama:** `logs/generation_history.json`
```json
[
  {"gen": 0, "type": "heuristic_baseline", "avg_score": 142.3, "timestamp": "..."},
  {"gen": 1, "status": "PROMOTED", "win_rate": 0.62, "avg_score": 185.3, ...},
  {"gen": 2, "status": "REJECTED", "win_rate": 0.48, "avg_score": 178.1, ...},
  {"gen": 3, "status": "PROMOTED", "win_rate": 0.58, "avg_score": 201.5, ...}
]
```

---

### 7. Metrics Güncellemesi

#### [MODIFY] [metrics.py](file:///c:/Users/Emre%20Polat/Desktop/Python/block%20blast%20ai/block_blast_ai/utils/metrics.py)

Sabitleri `scripts.config`'den import edecek şekilde güncelle:

```diff
-REGRET_PENALTY_PER_BLOCKED_PIECE = 15.0
-SURVIVAL_BONUS_PER_STEP = 4.0
-GAME_OVER_PENALTY = 35.0
-LINE_CLEAR_COEF = 20.0
-AGG_HEIGHT_COEF = 0.35
-BUMPINESS_COEF = 0.25
-HOLE_COEF = 5.0
+try:
+    from scripts.config import (
+        LINE_CLEAR_COEF, AGG_HEIGHT_COEF, BUMPINESS_COEF, HOLE_COEF,
+        SURVIVAL_BONUS_PER_STEP, GAME_OVER_PENALTY,
+        REGRET_PENALTY_PER_BLOCKED_PIECE,
+    )
+except ImportError:
+    # Fallback: scripts paketi yoksa veya bağımsız çalıştırılıyorsa
+    LINE_CLEAR_COEF = 20.0
+    AGG_HEIGHT_COEF = 0.35
+    BUMPINESS_COEF = 0.25
+    HOLE_COEF = 5.0
+    SURVIVAL_BONUS_PER_STEP = 4.0
+    GAME_OVER_PENALTY = 35.0
+    REGRET_PENALTY_PER_BLOCKED_PIECE = 15.0
```

> [!TIP]
> `try/except ImportError` ile geriye uyumluluk korunur. Mevcut testler ve `train.py` değişiklik olmadan çalışmaya devam eder.

---

## Dosya Boyutu Tahminleri

| Dosya | Tahmini Satır |
|---|---|
| `scripts/config.py` | ~60 |
| `scripts/collect_expert_data.py` | ~250 |
| `scripts/pretrain_apprentice.py` | ~280 |
| `scripts/rl_fine_tune.py` | ~320 |
| `scripts/evaluate_and_promote.py` | ~200 |
| `scripts/master_control.py` | ~250 |
| `utils/metrics.py` değişiklik | ~10 satır diff |
| **Toplam yeni kod** | **~1360 satır** |

---

## Verification Plan

### Automated Tests
1. **Config import testi:** `python -c "from scripts.config import LINE_CLEAR_COEF; print(LINE_CLEAR_COEF)"`
2. **Metrics geriye uyumluluk:** Mevcut `pytest tests/` geçmeli
3. **Expert data smoke test:** `python -m scripts.collect_expert_data --n-moves 1000 --gen 0` (küçük set)
4. **Pretrain smoke test:** 1000 hamlelik veriyle 2 epoch BC
5. **Evaluation smoke test:** 5 oyunluk hızlı karşılaştırma
6. **Master control dry-run:** `--dry-run` flag ile pipeline akışını doğrula

### Manual Verification
- Vast.ai RTX 5090'da tam 20M adım RL fine-tune testi (kullanıcı tarafından)
- TensorBoard'da KL divergence, loss, avg_score grafiklerinin kontrolü
