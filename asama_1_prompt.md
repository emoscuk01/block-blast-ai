# Block Blast AI — Aşama 0 + 1 Cursor Prompt
# Oyun Simülatörü ve Heuristik Baseline

---

## BAĞLAM VE AMAÇ

Bu proje, Block Blast mobil oyunu için otonom çalışan bir yapay zeka sistemi geliştirmenin ilk adımıdır. Nihai hedef, bir telefona bağlı bilgisayarda çalışan, ekranı kamera ile okuyan ve ADB üzerinden hamleleri gerçekleştiren tam otonom bir AI pipeline'ı kurmaktır. Ancak bu dosyada yalnızca **temel altyapı** istenmektedir:

1. Gerçek oyunun davranışını birebir taklit eden bir **Python simülatörü** (ortam / environment)
2. Hiç makine öğrenmesi kullanmadan, matematiksel skor fonksiyonuyla oynayan bir **heuristik baseline bot**
3. Bu iki bileşeni test eden ve baseline'ı ölçen **benchmark scripti**

Bu altyapı, ileride DQN / PPO gibi RL modellerinin üzerinde eğitileceği "dünya" olacaktır. Simülatör ne kadar sağlam ve hızlı olursa, RL eğitimi o kadar verimli ilerler.

---

## OYUN KURALLARI (KESİN, DEĞİŞTİRME)

Block Blast şu kurallara göre çalışır:

- Tahta **8 satır × 8 sütun**'dan oluşur. `board[row][col]` gösterimi kullanılır. `0` = boş, `1` = dolu.
- Her turda oyuncuya **3 adet blok** verilir. Bu bloklar sabit şekilli parçalardır, Tetris'teki gibi döndürülemez.
- Oyuncu bu 3 bloğu **istediği sırayla** tahtaya yerleştirir. Yerleştirme sırası stratejik önem taşır.
- Bir blok yalnızca tüm kareleri boş hücrelere denk geliyorsa yerleştirilebilir.
- Bir veya birden fazla **tam satır** ya da **tam sütun** dolduğunda o satır/sütun **silinir** ve puan kazanılır.
- Aynı hamle sonucunda birden fazla satır/sütun silinebilir; bu durum bonus puan getirir.
- 3 bloğun hiçbiri tahtanın herhangi bir yerine sığmıyorsa **oyun biter**.
- Blok şekilleri sabit bir listeden seçilir (aşağıda tanımlanmıştır). Gerçek oyundaki tüm yaygın şekiller dahil edilmelidir.

---

## DOSYA YAPISI (TAM OLARAK BU ŞEKİLDE OLUŞTUR)

```
block_blast_ai/
├── env/
│   ├── __init__.py
│   ├── board.py          # Tahta mantığı: yerleştirme, satır silme, kopya alma
│   ├── pieces.py         # Tüm blok şekillerinin tanımı ve rastgele seçim
│   └── game_env.py       # Ana ortam sınıfı: step(), reset(), render()
├── agents/
│   ├── __init__.py
│   ├── base_agent.py     # Soyut temel sınıf
│   ├── random_agent.py   # Tamamen rastgele hamle yapan bot (alt kıyas)
│   └── heuristic_agent.py# Matematiksel skor fonksiyonuyla oynayan baseline bot
├── utils/
│   ├── __init__.py
│   └── metrics.py        # Skor hesaplama yardımcı fonksiyonları
├── benchmark.py          # İki agent'ı karşılaştıran test scripti
├── play_demo.py          # Terminalde ASCII görselleştirmeyle tek oyun
└── tests/
    ├── test_board.py
    ├── test_pieces.py
    └── test_heuristic.py
```

---

## MODÜL 1: `env/pieces.py`

### Görev
Oyunda kullanılacak tüm blok şekillerini tanımla. Her şekil bir 2D liste olarak ifade edilir; `1` olan hücre dolu, `0` olan boş.

### Blok listesi (minimum, eksik kalmasın)

```python
# Her blok: (isim, şekil_matrisi) şeklinde tanımla
PIECES = {
    "tek":        [[1]],
    "yatay_2":    [[1, 1]],
    "dikey_2":    [[1], [1]],
    "yatay_3":    [[1, 1, 1]],
    "dikey_3":    [[1], [1], [1]],
    "yatay_4":    [[1, 1, 1, 1]],
    "dikey_4":    [[1], [1], [1], [1]],
    "yatay_5":    [[1, 1, 1, 1, 1]],
    "dikey_5":    [[1], [1], [1], [1], [1]],
    "kare_2x2":   [[1, 1], [1, 1]],
    "kare_3x3":   [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "L_sag":      [[1, 0], [1, 0], [1, 1]],
    "L_sol":      [[0, 1], [0, 1], [1, 1]],
    "L_ust":      [[1, 1, 1], [1, 0, 0]],
    "L_alt":      [[1, 0, 0], [1, 1, 1]],
    "J_sag":      [[1, 1], [1, 0], [1, 0]],
    "J_sol":      [[1, 1], [0, 1], [0, 1]],
    "T_sag":      [[1, 1, 1], [0, 1, 0]],
    "T_sol":      [[0, 1, 0], [1, 1, 1]],
    "T_dikey":    [[1, 0], [1, 1], [1, 0]],
    "S_yatay":    [[0, 1, 1], [1, 1, 0]],
    "S_dikey":    [[1, 0], [1, 1], [0, 1]],
    "Z_yatay":    [[1, 1, 0], [0, 1, 1]],
    "Z_dikey":    [[0, 1], [1, 1], [1, 0]],
    "kose_sol_ust": [[1, 1], [1, 0]],
    "kose_sag_ust": [[1, 1], [0, 1]],
    "kose_sol_alt": [[1, 0], [1, 1]],
    "kose_sag_alt": [[0, 1], [1, 1]],
}
```

### `pieces.py` içinde şu fonksiyonlar olacak:

```python
def get_piece_cells(piece_name: str) -> list[list[int]]:
    """Verilen isimli bloğun 2D matrisini döndür."""

def get_piece_size(piece_name: str) -> tuple[int, int]:
    """(satır_sayısı, sütun_sayısı) döndür."""

def get_random_pieces(n: int = 3, seed: int = None) -> list[str]:
    """n adet rastgele blok ismi döndür. Seed verilirse tekrarlanabilir."""

def get_all_piece_names() -> list[str]:
    """Tüm blok isimlerini liste olarak döndür."""
```

---

## MODÜL 2: `env/board.py`

### Görev
8×8 tahtanın tüm temel operasyonlarını gerçekleştiren sınıf. Saf mantık, görselleştirme veya ödül hesabı içermez.

### `Board` sınıfı tam arayüzü:

```python
class Board:
    ROWS: int = 8
    COLS: int = 8

    def __init__(self):
        # self.grid: np.ndarray, shape=(8,8), dtype=np.int8
        # Tüm hücreler 0 ile başlar

    def reset(self) -> np.ndarray:
        """Tahtayı sıfırla, boş grid döndür."""

    def copy(self) -> "Board":
        """Derin kopya döndür. Simülasyon için kritik."""

    def can_place(self, piece_name: str, row: int, col: int) -> bool:
        """
        Bloğu (row, col) sol üst köşesine yerleştirmek mümkün mü?
        Tahta sınırları ve çakışma kontrolü yap.
        """

    def place(self, piece_name: str, row: int, col: int) -> int:
        """
        Bloğu yerleştir. Doldurulan satır/sütunları sil.
        Silinen satır+sütun toplamını döndür.
        UYARI: can_place() False ise ValueError fırlat.
        """

    def get_valid_placements(self, piece_name: str) -> list[tuple[int, int]]:
        """
        Bu blok için geçerli tüm (row, col) konumlarını döndür.
        Liste boşsa blok hiçbir yere sığmıyor demektir.
        """

    def get_grid(self) -> np.ndarray:
        """Mevcut grid'in kopyasını döndür (dışarıdan değiştirilemez)."""

    def is_full_row(self, row: int) -> bool:
        """Satır tamamen dolu mu?"""

    def is_full_col(self, col: int) -> bool:
        """Sütun tamamen dolu mu?"""

    def count_holes(self) -> int:
        """
        Hole: üzerinde en az bir dolu hücre olan boş kare.
        Bu metrik hiçbir zaman temizlenemeyen ölü alanları ölçer.
        """

    def aggregate_height(self) -> int:
        """
        Her sütundaki en yüksek dolu hücrenin tahtanın altından
        uzaklığını topla. Yüksek değer = tahta dolmaya yakın.
        """

    def bumpiness(self) -> int:
        """
        Komşu sütun yükseklikleri arasındaki mutlak fark toplamı.
        Yüksek değer = pürüzlü yüzey = yerleştirme zorluğu.
        """

    def count_filled(self) -> int:
        """Tahtadaki toplam dolu hücre sayısı."""

    def count_empty(self) -> int:
        """Tahtadaki toplam boş hücre sayısı."""
```

### Satır/sütun silme mantığı (kesin kurallar):
- `place()` çağrıldıktan sonra tüm satırları tara; doluysa sil ve yukarıdakileri aşağı kaydır.
- Ardından tüm sütunları tara; doluysa sil ve sağdakileri sola kaydır.
- Satır ve sütun silme **aynı hamle sonucunda** gerçekleşebilir; ikisi ayrı ayrı hesaplanır ve toplanır.
- Silinen satır sayısı ile sütun sayısı ayrı tutulmalı (bonus hesabı için).

---

## MODÜL 3: `env/game_env.py`

### Görev
Simülatörün dışa açık arayüzü. RL framework'leriyle uyumlu `step()` / `reset()` yapısı.

### Aksiyon Uzayı Tasarımı (KRİTİK)

3 bloğu aynı anda yerleştirmeye çalışmak aksiyon uzayını patlattır. Bunun yerine şu mimari kullanılır:

- Her `step()` çağrısı **tek bir bloğu** tek bir konuma yerleştirir.
- Bir "tur" içinde 3 adet step çağrılır; `blocks_remaining` state'i bunu takip eder.
- Model, `blocks_remaining` değerini görerek sıralama stratejisi öğrenir.
- 3. bloğun yerleştirilmesiyle tur kapanır ve yeni 3 blok gelir.

### `GameEnv` sınıfı tam arayüzü:

```python
class GameEnv:
    def __init__(self, seed: int = None):
        """
        seed: tekrarlanabilirlik için.
        self.board: Board örneği
        self.current_pieces: list[str], uzunluk 3 (mevcut tur blokları)
        self.pieces_placed: int, bu turda kaç blok yerleştirildi (0,1,2)
        self.score: int, toplam oyun skoru
        self.turn: int, kaçıncı turda olduğumuz
        self.done: bool
        """

    def reset(self) -> dict:
        """
        Oyunu sıfırla. Observation dict döndür.
        Observation formatı aşağıda tanımlı.
        """

    def step(self, piece_index: int, row: int, col: int) -> tuple[dict, float, bool, dict]:
        """
        Parametreler:
            piece_index: 0, 1 veya 2 — current_pieces listesinden hangisi yerleştiriliyor
            row, col: tahtada sol üst köşe koordinatı

        Döndürür: (observation, reward, done, info)

        Kurallar:
        - Geçersiz hamle (sığmayan konum) -> ValueError fırlat
        - Yerleştirilen blok current_pieces'tan çıkarılır (None ile işaretlenir)
        - 3 blok da yerleştirildikten sonra yeni tur başlar: yeni 3 blok gelir
        - Hiçbir kalan blok tahtaya sığmıyorsa done=True
        - done=True olunca oyun biter, -100 ceza reward'a eklenir
        """

    def get_valid_actions(self) -> list[tuple[int, int, int]]:
        """
        Mevcut state'te geçerli tüm (piece_index, row, col) üçlülerini döndür.
        Bu liste boşsa oyun bitmeli demektir.
        """

    def render(self, mode: str = "ascii") -> str:
        """
        mode="ascii": Terminalde yazdırılabilir string döndür.
        Tahta, mevcut parçalar, skor, tur bilgisi gösterilsin.
        """

    def get_observation(self) -> dict:
        """
        State'in tam temsili. Format:
        {
            "board": np.ndarray shape=(8,8) dtype=np.float32,  # 0.0 veya 1.0
            "pieces": list[np.ndarray],  # Her biri max 5x5, padding 0, uzunluk 3
            "pieces_remaining": list[bool],  # [True, False, True] gibi
            "blocks_remaining": int,  # Bu turda kaç blok kaldı (0-3)
            "score": int,
            "turn": int,
        }
        """

    def clone(self) -> "GameEnv":
        """
        Mevcut state'in derin kopyasını döndür.
        Heuristik agent'ın lookahead hesabı için zorunlu.
        """
```

---

## MODÜL 4: `utils/metrics.py`

### Görev
Reward ve heuristik hesaplamalarında kullanılan saf (side-effect'siz) fonksiyonlar.

```python
import numpy as np

def compute_reward(
    board_before: np.ndarray,
    board_after: np.ndarray,
    lines_cleared: int,
    game_over: bool,
) -> float:
    """
    Ana reward fonksiyonu. Formül:

        R = (10 * S) - (0.5 * A) - (0.3 * B) - (8 * H) - (100 * game_over)

    S = silinen satır + sütun toplamı (lines_cleared)
    A = board_after üzerinde aggregate_height()
    B = board_after üzerinde bumpiness()
    H = board_after üzerinde count_holes()

    Döndürür: float
    """

def compute_regret(board: np.ndarray, upcoming_pieces: list[str]) -> float:
    """
    Pişmanlık skoru: mevcut tahtada upcoming_pieces içinden
    kaçı hiçbir yere yerleştirilemiyor?

    Her sığmayan parça için 50 ceza puanı uygula.
    Döndürür: float (negatif veya sıfır)
    """

def board_aggregate_height(board: np.ndarray) -> int:
    """Board numpy array'i alır, aggregate height hesaplar."""

def board_bumpiness(board: np.ndarray) -> int:
    """Board numpy array'i alır, bumpiness hesaplar."""

def board_holes(board: np.ndarray) -> int:
    """
    Hole tanımı: dolu bir hücrenin altında kalan boş hücre.
    Sütun bazlı tara: üstünden ilk dolu hücreyi bul,
    onun altındaki tüm boşları say.
    """

def composite_score(board: np.ndarray, upcoming_pieces: list[str]) -> float:
    """
    Heuristik agent'ın hamle değerlendirmesinde kullandığı
    tek sayı. compute_reward + compute_regret kombinasyonu.
    Hamle simülasyonu yapıldıktan sonra bu fonksiyon çağrılır.
    """
```

---

## MODÜL 5: `agents/heuristic_agent.py`

### Görev
Makine öğrenmesi kullanmadan, tüm geçerli hamleleri simüle ederek en yüksek `composite_score` döndüren hamleyi seçen deterministik bot.

### Algoritma (adım adım):

```
1. get_valid_actions() ile tüm geçerli (piece_index, row, col) üçlülerini al.
2. Her üçlü için:
   a. env.clone() ile ortamın kopyasını al.
   b. Kopyada step() ile hamleyi uygula.
   c. composite_score(board_after, remaining_pieces) hesapla.
   d. (üçlü, skor) çiftini kaydet.
3. En yüksek skoru veren üçlüyü seç ve döndür.
```

### `HeuristicAgent` sınıfı:

```python
class HeuristicAgent(BaseAgent):
    def __init__(self, weights: dict = None):
        """
        weights: reward fonksiyonu katsayıları.
        Varsayılan: {"lines": 10, "height": 0.5, "bump": 0.3, "holes": 8, "regret": 50}
        Dışarıdan geçilebilir → ileride grid search için.
        """

    def select_action(self, env: GameEnv) -> tuple[int, int, int]:
        """
        (piece_index, row, col) döndür.
        Yukarıdaki algoritmayı uygula.
        """

    def evaluate_move(self, env: GameEnv, piece_index: int, row: int, col: int) -> float:
        """
        Tek bir hamlenin composite_score'unu döndür.
        Dışarıdan test için erişilebilir.
        """
```

---

## MODÜL 6: `agents/random_agent.py`

```python
class RandomAgent(BaseAgent):
    def select_action(self, env: GameEnv) -> tuple[int, int, int]:
        """
        get_valid_actions() listesinden uniform rastgele bir üçlü seç.
        Geçerli aksiyon yoksa None döndür (oyun bitmeli).
        """
```

---

## MODÜL 7: `benchmark.py`

### Görev
İki agent'ı belirli sayıda oyun üzerinde karşılaştır ve istatistiksel sonuç ver.

### Çıktı formatı (terminale yazdır):

```
=== BENCHMARK SONUÇLARI (N=100 oyun) ===

Agent              | Ort. Skor | Maks Skor | Min Skor | Ort. Tur | Kazanma %
-------------------|-----------|-----------|----------|----------|----------
HeuristicAgent     |    4821.3 |    12450  |     340  |    48.2  |   —
RandomAgent        |     312.7 |     890   |      20  |     8.1  |   —

Heuristik / Random oran: 15.4x
```

### `benchmark.py` içinde olması gerekenler:

```python
def run_episode(agent, env: GameEnv) -> dict:
    """
    Tek bir oyunu baştan sona oynat.
    Döndür: {"score": int, "turns": int, "lines_cleared": int}
    """

def benchmark(agent, n_games: int = 100, seed_start: int = 0) -> dict:
    """
    n_games kadar oyun oynat, istatistikleri hesapla.
    Her oyun için seed = seed_start + i kullan (tekrarlanabilirlik).
    """

def compare(agents: dict, n_games: int = 100):
    """
    {"isim": agent} sözlüğü al, hepsini benchmark et, tabloyu yazdır.
    """

if __name__ == "__main__":
    from env.game_env import GameEnv
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent

    agents = {
        "HeuristicAgent": HeuristicAgent(),
        "RandomAgent": RandomAgent(),
    }
    compare(agents, n_games=100)
```

---

## MODÜL 8: `play_demo.py`

Terminalde ASCII görselleştirmeyle tek bir oyun oynat. Her adımda tahtayı, mevcut parçaları ve seçilen hamleyi göster. `HeuristicAgent` kullanılsın.

ASCII tahta formatı:
```
Tur: 12  |  Skor: 3240  |  Parçalar kaldı: 2

  0 1 2 3 4 5 6 7
0 . . . . . . . .
1 . . . . . . . .
2 . . . # # . . .
3 . # # # # # . .
4 # # # # # # # .
5 # # . # # # # #
6 # # # # # # # #
7 # # # # # # # #

Mevcut parçalar: [kare_2x2, yatay_3, None]
Seçilen hamle: parça=0, konum=(2,5)
Bekleniyor... (Enter'a bas)
```

---

## TESTLER: `tests/`

### `test_board.py` içinde şu senaryolar test edilmeli:

```python
def test_place_single_cell():
    """Tek hücrelik bloğu yerleştir, grid'de 1 adet 1 olsun."""

def test_place_horizontal_3():
    """Yatay 3'lü bloğu bir satıra yerleştir, doğru hücreler dolsun."""

def test_place_out_of_bounds():
    """Sınır dışı koordinata yerleştirme ValueError fırlatmalı."""

def test_place_overlap():
    """Dolu hücrenin üzerine yerleştirme ValueError fırlatmalı."""

def test_row_clear():
    """Bir satırı tamamen doldur, place() sonrasında silinsin ve 1 döndersin."""

def test_col_clear():
    """Bir sütunu tamamen doldur, place() sonrasında silinsin ve 1 döndürsün."""

def test_simultaneous_row_col_clear():
    """Aynı hamleyle hem satır hem sütun silinebilmeli."""

def test_holes_count():
    """Manuel tahta kur, count_holes() beklenen değeri döndürsün."""

def test_aggregate_height():
    """Manuel tahta kur, aggregate_height() beklenen değeri döndürsün."""

def test_bumpiness():
    """Manuel tahta kur, bumpiness() beklenen değeri döndürsün."""

def test_board_copy_independence():
    """copy() sonrası orijinali değiştirmek kopyayı etkilememeli."""
```

### `test_pieces.py`:

```python
def test_all_pieces_valid():
    """Tüm parça isimleri için get_piece_cells() çalışmalı, liste dönmeli."""

def test_piece_sizes_correct():
    """Bilinen parçalar için get_piece_size() doğru (rows, cols) döndürmeli."""

def test_random_pieces_count():
    """get_random_pieces(3) her zaman uzunluğu 3 olan liste döndürmeli."""

def test_random_pieces_reproducible():
    """Aynı seed ile get_random_pieces() aynı sonucu vermeli."""
```

### `test_heuristic.py`:

```python
def test_heuristic_beats_random():
    """50 oyunda HeuristicAgent ortalama skoru RandomAgent'ı geçmeli."""

def test_heuristic_no_invalid_moves():
    """HeuristicAgent hiçbir zaman geçersiz hamle seçmemeli."""

def test_evaluate_move_returns_float():
    """evaluate_move() her zaman float döndürmeli."""

def test_regret_penalizes_blocked_pieces():
    """Bloğu tamamen dolu tahtada regret skoru maksimum olmalı."""
```

---

## KOD STİLİ VE TEKNİK KISITLAMALAR

- **Dil:** Python 3.10+
- **Bağımlılıklar:** Sadece `numpy`, `pytest`. Başka kütüphane kullanma.
- **Tüm yorum satırları ve docstring'ler Türkçe yazılacak.**
- **Type hint'ler zorunlu:** Her fonksiyon parametresi ve dönüş değeri annotate edilmeli.
- **NumPy dizisi kopyalama:** `board.copy()` içinde `np.copy()` kullan, referans atama yapma.
- **Hiçbir global state:** Tüm state `Board` ve `GameEnv` örneklerinde tutulacak.
- **Performans:** `get_valid_actions()` mümkün olduğunca hızlı olmalı; `HeuristicAgent` her hamle için tüm valid action'ları simüle eder, bu yüzden `Board.copy()` çok çağrılacak — gereksiz deep copy'den kaçın.
- **Hata yönetimi:** Geçersiz hamlelerde açıklayıcı mesajla `ValueError` fırlat.
- **`__repr__` ve `__str__`:** `Board` ve `GameEnv` için tanımla, debug kolaylığı için.

---

## BAŞLANGIÇ NOKTASI

İlk olarak şu sırayla yaz:

1. `env/pieces.py` — tüm parça tanımları + fonksiyonlar
2. `env/board.py` — `Board` sınıfı tam implementasyon
3. `utils/metrics.py` — reward + regret + composite score
4. `env/game_env.py` — `GameEnv` sınıfı
5. `agents/random_agent.py`
6. `agents/heuristic_agent.py`
7. `benchmark.py`
8. `play_demo.py`
9. `tests/` — tüm test dosyaları

Her modülü bitirdikten sonra en az bir `assert` veya `pytest` ile çalıştığını doğrula.

---

## KONTROL LİSTESİ (bitince işaretle)

- [ ] `Board.place()` satır ve sütun silmeyi doğru yapıyor
- [ ] `Board.copy()` bağımsız kopya döndürüyor
- [ ] `GameEnv.step()` tur geçişini doğru yönetiyor
- [ ] `GameEnv.clone()` lookahead için kullanılabilir durumda
- [ ] `HeuristicAgent` her zaman geçerli hamle seçiyor
- [ ] `benchmark.py` çalıştırıldığında tablo çıktısı veriyor
- [ ] Tüm testler yeşil (`pytest tests/`)
- [ ] Heuristik bot random botu en az 5x skorla geçiyor
