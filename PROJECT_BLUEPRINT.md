# Block Blast AI — PROJECT_BLUEPRINT

Bu doküman, projeye sıfırdan dahil olacak bir uzman LLM veya geliştiricinin doğrudan doğru sınıfları, indeksleri ve katsayılarla kod yazabilmesi için hazırlanmıştır. Kaynak gerçeği (ground truth) olarak Python paketi `block_blast_ai/` altındaki dosyalar esas alınmıştır.

---

## 1. Project Overview

### Amaç

**Block Blast RL**, mobil “block puzzle” tarzı bir oyunun **8×8 tahta** üzerindeki yerleştirme dinamiğini simüle eder; amaç **uzun süre hayatta kalmak**, **satır/sütun temizlemek** ve **ölümcül tahta yapılarını** (yükseklik, girinti çıkıntı, delik) cezalandıran bir ödül sinyali ile **Pekiştirmeli Öğrenme (RL)** eğitimidir.

### Ana kütüphaneler

| Bileşen | Kütüphane | Kullanım |
|--------|-----------|----------|
| Ortam API | **Gymnasium** (`gymnasium`) | `gym.Env`: `reset`, `step`, `observation_space`, `action_space` |
| RL algoritması | **Stable-Baselines3** + **sb3-contrib** | `MaskablePPO`, `MlpPolicy`, `VecNormalize`, `Monitor`, callback’ler |
| Maskeli aksiyon | **sb3-contrib** | Geçersiz yerleştirmelerin politikadan çıkarılması (`use_masking=True`) |
| Derin öğrenme | **PyTorch** (`torch`) | Politika ağı; opsiyonel `torch.compile`, CUDA TF32 ayarları |

Temel RL bağımlılıkları: `block_blast_ai/requirements_rl.txt`.

### Donanım hedefi (RTX 5090 / yüksek paralellik)

`train.py` açıklamalarına göre:

- Ortam adımları **CPU’da** Python ile simüle edilir; **GPU** ağırlıklı olarak **PPO gradyan güncellemelerinde** kullanılır (TensorBoard’da düşük GPU ortalaması beklenen bir durum olabilir).
- CUDA algılandığında: `cudnn.benchmark = True`, isteğe bağlı `torch.set_float32_matmul_precision("high")` (Ampere+ / RTX serisi için TF32 matmul).
- **Önerilen yoğun profil örneği**: `--heavy-gpu`, `--compile-policy`, `--n-envs 128`, çok çekirdekli CPU (ör. EPYC) ile `SubprocVecEnv`.
- Windows’ta `SubprocVecEnv` pickle/spawn sorunları için `--vec-env dummy --n-envs 4` debug yolu belirtilmiştir.

---

## 2. Directory Tree

Kök: `block blast ai/`  
Python paketi ve çalışma dizini genelde: `block_blast_ai/` (scriptler buradan çalıştırılır).

```
block blast ai/
├── PROJECT_BLUEPRINT.md          # Bu dosya
└── block_blast_ai/
    ├── train.py                  # MaskablePPO eğitimi, VecNormalize, hiperparametre CLI
    ├── evaluate.py               # PPO / Heuristic / Random karşılaştırma
    ├── play_demo.py              # İnsan/örnek oyun döngüsü
    ├── benchmark.py              # Basit performans ölçümü
    ├── agent_loop.py             # Görüntü → observation/mask → model/heuristic → ADB kontrol
    ├── dashboard_app.py          # Streamlit canlı panel giriş noktası
    ├── requirements.txt          # Genel
    ├── requirements_rl.txt       # Gymnasium, SB3, sb3-contrib, torch, tensorboard
    ├── requirements_dashboard.txt
    ├── requirements_control.txt
    ├── requirements_vision.txt
    ├── asama_*.prompt.md         # Aşama bazlı tasarım notları (Türkçe)
    │
    ├── env/                      # Oyun simülatörü (RL’den bağımsız çekirdek)
    │   ├── game_env.py           # GameEnv: reset/step/get_observation/get_valid_actions/clone
    │   ├── board.py              # Board: 8×8 grid, yerleştirme, satır/sütun temizleme
    │   └── pieces.py             # PIECES sözlüğü, get_piece_cells, get_random_pieces
    │
    ├── rl/                       # Gymnasium + RL köprüsü
    │   ├── gym_env.py            # BlockBlastGymEnv — Gym API, ödül + regret birleşimi
    │   ├── observation.py        # encode_observation, OBS_SIZE=142
    │   ├── action_mapper.py      # 192 aksiyon ↔ (piece, row, col), get_valid_action_mask
    │   ├── callbacks.py          # TrainingMetricsCallback, HeuristicComparisonCallback
    │   └── __init__.py
    │
    ├── agents/
    │   ├── base_agent.py         # Soyut taban
    │   ├── heuristic_agent.py    # HeuristicAgent — composite_score ile hamle seçimi
    │   └── random_agent.py       # Rastgele geçerli hamle
    │
    ├── utils/
    │   └── metrics.py            # compute_reward, compute_regret, composite_score, board metrikleri
    │
    ├── dashboard/                # Streamlit yardımcıları
    │   ├── board_renderer.py
    │   ├── metrics_renderer.py
    │   └── state_bridge.py
    │
    ├── vision/                   # Ekran görüntüsü → tahta/parça algılama (agent_loop ile)
    │   ├── screen_capture.py
    │   ├── calibration.py
    │   ├── board_detector.py
    │   ├── piece_detector.py
    │   └── debug_overlay.py
    │
    ├── control/                  # fiziksel cihaz / koordinat (ADB)
    │   ├── adb_controller.py
    │   ├── coordinate_mapper.py
    │   └── action_executor.py
    │
    └── tests/                    # pytest
        ├── test_board.py
        ├── test_pieces.py
        ├── test_heuristic.py
        ├── test_vision.py
        └── ...
```

---

## 3. Observation Space (142 boyut) — tam indeks haritası

### Genel düzen

`rl/observation.encode_observation(obs_dict)` çıktısı:

**`[ board_flat (64) | piece0 (25) | piece1 (25) | piece2 (25) | blocks_remaining_onehot (3) ]`**  
Toplam: **`OBS_SIZE = 142`** (`64 + 25×3 + 3`), **`dtype=float32`**, değerler **0.0–1.0** aralığında (tahta ve parça hücreleri 0/1).

`obs_dict` kaynağı: `GameEnv.get_observation()` (`env/game_env.py`).

### 3.1 Tahta — indeks `0 .. 63`

- `board_flat = obs_dict["board"].flatten()`
- `Board.grid` şekli **`(8, 8)`**, tipik olarak **`row-major (C order)`** flatten:
  - **`flat_index = row * 8 + col`**
  - **`row ∈ [0,7]`** üstten alta, **`col ∈ [0,7]`** soldan sağa
- Örnek: `(row=0, col=0)` → indeks **0**; `(7,7)` → indeks **63**

### 3.2 Üç parça yuvası — indeks `64 .. 138`

`obs_dict["pieces"]` **üç elemanlı** liste; sıra **`current_pieces[0], [1], [2]`** ile aynıdır.

Her parça:

- Parça yoksa (`None`): **25 sıfır**
- Parça varsa: `get_piece_cells(name)` matrisi **`5×5`** sol-üst köşeye yerleştirilir (kalan padding 0), sonra **`flatten()`** → **25 boyut**
- Parça içi indeks (her 25’lik blokta): **`local = r * 5 + c`**, `r,c ∈ [0,4]`

| Blok | Boyut | Global indeks aralığı (dahil) |
|------|-------|--------------------------------|
| Parça slot 0 | 25 | **64 – 88** |
| Parça slot 1 | 25 | **89 – 113** |
| Parça slot 2 | 25 | **114 – 138** |

### 3.3 Meta: kalan blok sayısı — indeks `139 .. 141`

`blocks_remaining` = `obs_dict["blocks_remaining"]` = halihazırda **yerleştirilmemiş** parça sayısı (**0–3**).

One-hot (`encode_observation`):

| `blocks_remaining` | `139` | `140` | `141` |
|--------------------|-------|-------|-------|
| 3 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 1 | 0 | 0 | 1 |
| 0 | **Hepsi 0** | **Hepsi 0** | **Hepsi 0** |

Not: `0` kalan durumunda vektörün son üç bileşeni sıfır kalır (tek-hot “yok” durumu ayrı kodlanmamıştır).

### `obs_dict` içindeki ek alanlar (vektöre girmez)

`get_observation()` ayrıca `"pieces_remaining"` (bool liste), `"score"`, `"turn"` döndürür; **`encode_observation` bunları kullanmaz**.

---

## 4. Action Space & Masking

### 4.1 Toplam aksiyon sayısı

`rl/action_mapper.py`:

- **`TOTAL_ACTIONS = 192`** = **3 parça indeksi × 8 satır × 8 sütun**

`BlockBlastGymEnv.action_space`: `gymnasium.spaces.Discrete(192)`.

### 4.2 İndeks kodlama

```text
piece_index = action // 64      # 0, 1 veya 2
row         = (action % 64) // 8   # 0..7
col         = action % 8           # 0..7
```

Ters dönüşüm: `tuple_to_action(piece_index, row, col) = piece_index * 64 + row * 8 + col`.

Semantik: **`(row, col)`** parçanın **sol üst köşesi**nin tahta üzerindeki konumudur (`Board.can_place` ile uyumlu).

### 4.3 Geçerlilik maskesi

- **`get_valid_action_mask(env)`** → **`(192,)` `bool`**
- `env` burada **`GameEnv`** örneğidir (`BlockBlastGymEnv.game_env`).
- `GameEnv.get_valid_actions()` tüm **`(piece_index, row, col)`** üçlülerini döndürür; her biri `tuple_to_action` ile maskede **`True`** yapılır.

**Gym sarmalayıcıda:**

- `BlockBlastGymEnv.get_action_mask()` / **`action_masks()`** → SB3 **Maskable** konvansiyonu için alias.
- `reset`/`step` dönen `info["action_mask"]` güncel maskayı taşır.

### 4.4 MaskablePPO ile veri akışı notu

Eğitimde `MaskablePPO` **`use_masking=True`** ile eval callback kullanır. Tek ortamda çıkarım yaparken doğru kullanım için `action_masks=env.action_masks()` (veya `model.predict(obs, action_masks=...)`) verilmesi gerekir; aksi halde geçersiz discrete indeks riski doğar (`rl/gym_env.py` geçersiz aksiyonda rastgele geçerli hamleye düşer veya özel ceza verir — bkz. bölüm 6).

---

## 5. Reward Logic (“v4” / güncel `utils.metrics`)

Projede sürüm etiketi yok; kullanıcıya referans olan **güncel ödül**, `utils/metrics.py` içindeki **`compute_reward`** ve (yalnızca Gym katmanında) ek **`compute_regret`** ile tanımlıdır.

### 5.1 Temel katsayılar (`utils/metrics.py`)

| Sabit | Değer | Rol |
|-------|-------|-----|
| `LINE_CLEAR_COEF` | **20.0** | Temizlenen satır+sütun sayısı (`lines_cleared`) ile çarpılır |
| `AGG_HEIGHT_COEF` | **0.35** | Aggregate height (`board_after`) cezası |
| `BUMPINESS_COEF` | **0.25** | Bumpiness cezası |
| `HOLE_COEF` | **5.0** | Delik sayısı cezası |
| `SURVIVAL_BONUS_PER_STEP` | **4.0** | **`game_over == False`** ise **eklenir** |
| `GAME_OVER_PENALTY` | **35.0** | **`game_over == True`** ise **çıkarılır** |
| `REGRET_PENALTY_PER_BLOCKED_PIECE` | **15.0** | Regret içinde kullanılır (aşağıda) |

### 5.2 Metrik tanımları (aynı dosya)

- **`board_aggregate_height(board)`**: Her sütunda üstten ilk dolu hücreye kadar; yükseklik = `8 - row`; sütunların toplamı.
- **`board_bumpiness(board)`**: Komşu sütun yükseklikleri farklarının mutlak toplamı.
- **`board_holes(board)`**: Bir sütunda dolu hücre görüldükten sonra gelen boş hücre sayısı (klasik “hole” tanımı).

### 5.3 `compute_reward(board_before, board_after, lines_cleared, game_over)`

Formül (kodla uyumlu):

```text
R_base = LINE_CLEAR_COEF * S
       - AGG_HEIGHT_COEF * A
       - BUMPINESS_COEF * B
       - HOLE_COEF * H
```

Burada **`S = lines_cleared`**, **`A,B,H`** sırasıyla **`board_after`** üzerinden hesaplanır.

Ardından:

- Oyun bitmediyse: **`R = R_base + SURVIVAL_BONUS_PER_STEP`**
- Oyun bittiyse: **`R = R_base - GAME_OVER_PENALTY`**

Not: `board_before` parametresi imzada vardır; **bu fonksiyon içinde kullanılmaz** (delta ödülü yok, mutlak `board_after` metrikleri var).

### 5.4 `compute_regret(board, upcoming_pieces)`

- `upcoming_pieces`: yerleştirilemeyecek şekilde sıradaki parça **isimleri** listesi (`None` elemanlar atlanır).
- Tahtada **hiçbir yere** yerleştirilemeyen her parça için **`-REGRET_PENALTY_PER_BLOCKED_PIECE`** (`-15.0`) birikir.

### 5.5 Hangi katman ne kullanıyor?

| Çağrı yeri | `compute_reward` | `compute_regret` |
|------------|------------------|------------------|
| `GameEnv.step` | Evet | Hayır |
| `BlockBlastGymEnv.step` | Evet | Evet — yerleştirme sonrası kalan parçalar üzerinden |

Yani **aynı `compute_reward`** hem ham `GameEnv` hem Gym’de kullanılır; **regret yalnızca `BlockBlastGymEnv`** ile RL trajectories’a eklenir.

### 5.6 Geçersiz aksiyon / edge case (`BlockBlastGymEnv.step`)

- Maskede **`False`** olan aksiyon: mümkünse rastgele **geçerli** hamle uygulanır.
- Geçerli hamle yoksa: `done=True`, **`reward = -100.0`** sabit, özel `info["invalid_action"]`.

### 5.7 Ölçekleme

`train.py` içinde **`VecNormalize(..., norm_reward=True, clip_reward=10.0)`** ham ödülü normalize eder; **`norm_obs=False`** — gözlem normalize edilmez.

---

## 6. Heuristic Logic

### Sınıf

**`agents.heuristic_agent.HeuristicAgent`** (`BaseAgent` alt sınıfı).

### Prensip

1. `GameEnv.get_valid_actions()` ile tüm geçerli **`(piece_index, row, col)`** üretilir.
2. Her aday için **`env.clone()`** ile kopya ortamda **`clone.step(...)`** çalıştırılır.
3. Yerleştirme sonrası:
   - `board_array = clone.board.get_grid()`
   - `remaining = [p for p in clone.current_pieces if p is not None]`
   - `lines_cleared = info.get("lines_cleared", 0)`
4. Skor: **`utils.metrics.composite_score(board_array, remaining, lines_cleared)`**

**En yüksek** `composite_score` veren hamle seçilir; eşitlikte döngü sırası ilk kazanır.

### `composite_score` formülü

```text
score = LINE_CLEAR_COEF * lines_cleared
      - AGG_HEIGHT_COEF * A
      - BUMPINESS_COEF * B
      - HOLE_COEF * H
      + compute_regret(board, upcoming_pieces)
```

**`SURVIVAL_BONUS_PER_STEP` / `GAME_OVER_PENALTY` burada yok** — heuristik yalnızca yerleşim sonrası tahta geometrisi + satır temizliği + regret ile skorlar.

### Önemli uyarı: `HeuristicAgent.__init__(weights=...)`

Constructor’da varsayılan bir **`weights`** sözlüğü (`lines`, `height`, vb.) tanımlıdır fakat **`evaluate_move` içinde `composite_score` çağrılırken bu sözlük kullanılmaz**. Fiili ağırlıklar **`utils/metrics.py` sabitleri**dir. Özel ağırlık isteniyorsa ya `metrics.py` sabitleri ya da `composite_score`/agent kodu genişletilmelidir.

---

## 7. Training Specs (`train.py`)

### 7.1 Varsayılan CLI hiperparametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--n-envs` | **64** | Paralel ortam |
| `--n-steps` | **2048** | Ortam başına rollout uzunluğu; buffer = **`n_steps * n_envs`** |
| `--batch-size` | **4096** | PPO minibatch; CUDA otomatik ayarında yükseltilebilir (≤ rollout) |
| `--n-epochs` | **4** | PPO epoch; CUDA otomatik ayarında taban yükseltilebilir |
| `--ent-coef` | **0.12** | Entropy katsayısı |
| `--timesteps` | **2_000_000** | Toplam öğrenme adımı |
| `--vec-env` | **subproc** | `DummyVecEnv` veya `SubprocVecEnv` |
| `--learning-rate` | **3e-4** | Kod içinde sabit (schedule değil) |

### 7.2 Politika mimarisi (`policy_kwargs`)

`MaskablePPO` **`MlpPolicy`** ile:

```python
policy_kwargs = dict(net_arch=[dict(pi=policy_arch_pi, vf=policy_arch_vf)])
```

**Cihaza göre seçilen varsayılanlar:**

| Profil | Koşul | `pi` / `vf` |
|--------|--------|-------------|
| CPU veya `--light-gpu` | `device != "cuda"` veya flag | **`[256, 256]`** |
| CUDA (varsayılan) | `cuda` ve `--light-gpu` yok, `--heavy-gpu` yok | **`[512, 512, 256]`** |
| CUDA `--heavy-gpu` | | **`[1024, 1024, 512]`** |
| Manuel | `--policy-net-arch 512,512,256` vb. | CLI ile verilen liste (hem pi hem vf) |

### 7.3 CUDA “otomatik batch/epoch tabanı”

`--no-gpu-auto-tune` kapalıyken ve CUDA’da:

- Normal: `floor_batch = min(32768, rollout_size)`, `epoch_floor = 8`
- Heavy: `floor_batch = min(65536, rollout_size)`, `epoch_floor = 10`
- `eff_batch = max(args.batch_size, floor_batch)` sonra `min(..., rollout_size)`
- `eff_epochs = max(args.n_epochs, epoch_floor)`

### 7.4 Diğer PPO sabitleri (kod içi)

- `gamma=0.99`, `gae_lambda=0.95`
- `clip_range=linear_schedule(0.2, 0.1)`
- `vf_coef=0.5`, `max_grad_norm=0.5`, `target_kl=0.02`
- `tensorboard_log`, model kayıtları `models/`, `logs/`

---

## 8. Data Flow — tek `step` zinciri

### 8.1 Özet şema

```text
MaskablePPO (sb3-contrib)
    ↓ gözlem vektörü (142,) + action_masks (192,)
BlockBlastGymEnv.step(action: int)
    ↓ decode: action_to_tuple → (piece_idx, row, col)
    ↓ get_valid_action_mask(game_env) — geçersizse düzeltme veya -100 çıkışı
GameEnv.step(piece_idx, row, col)
    ↓ Board.place → lines_cleared, grid güncellenir
    ↓ skor (oyun içi `score`) güncellenir; tur/parça slotları yönetilir
    ↓ compute_reward(...) → hamilton Gym öncesi GameEnv reward’u da üretilir
BlockBlastGymEnv
    ↓ compute_reward + compute_regret → RL reward
    ↓ encode_observation(get_observation()) → obs'
    ↓ info (+ action_mask)
VecNormalize (eğitimde) → normalize edilmiş reward PPO'ya
```

### 8.2 SB3 vektör ortamı

- `make_vec_env(BlockBlastGymEnv, n_envs=..., wrapper_class=Monitor, vec_env_cls=SubprocVecEnv|DummyVecEnv)`
- Her alt ortam bağımsız `GameEnv` taşır; rollout paralel toplanır.

### 8.3 Çıkarım (`evaluate.py` deseni)

Tipik döngü: `reset()` → `obs` → **`predict`** → `step`. Maskeli politika için **`action_masks`** sağlanması önerilir (tek env’de `env.action_masks()`).

---

## 9. İsimler ve hızlı referans

| Kavram | Dosya / sembol |
|--------|----------------|
| Gym ortamı | `rl.gym_env.BlockBlastGymEnv` |
| Ham oyun | `env.game_env.GameEnv` |
| Tahta | `env.board.Board` |
| Parça şekilleri | `env.pieces.PIECES`, `get_piece_cells` |
| Gözlem kodlama | `rl.observation.encode_observation`, `OBS_SIZE` |
| Aksiyon kodlama | `rl.action_mapper.action_to_tuple`, `TOTAL_ACTIONS` |
| Ödül | `utils.metrics.compute_reward`, `compute_regret` |
| Heuristik | `agents.heuristic_agent.HeuristicAgent`, `composite_score` |
| Eğitim girişi | `train.train()`, CLI `argparse` |

---

*Bu blueprint, depodaki mevcut kodla senkron tutulmalıdır; ödül veya gözlem şeması değişirse ilgili bölümler güncellenmelidir.*
