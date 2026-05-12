# Block Blast AI — Aşama 4: Otonom Kontrol
# ADB ile Telefona Hamle Yaptır

---

## BAĞLAM

Aşama 0+1+2+3 tamamlandı. Elimizde şunlar var:
- `GameEnv` + eğitilmiş DQN modeli (`models/dqn_v1.zip`)
- `VisionPipeline` → ekran görüntüsünden board + pieces döndürüyor
- `calibration_data/config.json` → koordinat bilgileri

Bu aşamada tüm parçaları birbirine bağlıyoruz:
**Ekranı oku → Modele sor → ADB ile hamle yap → Tekrar et**

Mevcut hiçbir dosyaya dokunma, sadece yeni dosyalar ekle.

---

## DOSYA YAPISI (SADECE YENİ DOSYALAR)

```
block_blast_ai/
├── control/
│   ├── __init__.py
│   ├── adb_controller.py   # ADB swipe/tap komutları
│   ├── coordinate_mapper.py# Tahta koordinatı → piksel koordinatı
│   └── action_executor.py  # Model kararını ADB hareketine çevir
├── agent_loop.py           # Ana otonom döngü
└── requirements_control.txt # Ek bağımlılık yok, sadece belgeleme
```

---

## MODÜL 1: `control/adb_controller.py`

### Görev
ADB komutlarını Python'dan çalıştır. Saf bir wrapper sınıfı —
oyun mantığı bilmez, sadece dokunma hareketleri gerçekleştirir.

```python
import subprocess
import time
import logging

logger = logging.getLogger(__name__)

class ADBController:
    # Blok sürüklemek için bekleme süresi (saniye)
    SWIPE_DURATION_MS = 300   # Çok hızlı olursa oyun tanımaz
    TAP_DURATION_MS   = 50

    def __init__(self, device_id: str = None):
        """
        device_id: None ise otomatik ilk cihaz seçilir.
        """

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = None) -> bool:
        """
        Parmak sürükleme hareketi.
        Komut: adb shell input swipe x1 y1 x2 y2 duration
        Başarılıysa True, hata varsa False döndür ve logla.
        """

    def tap(self, x: int, y: int) -> bool:
        """
        Tek dokunma.
        Komut: adb shell input tap x y
        """

    def drag_piece(
        self,
        piece_center_x: int,
        piece_center_y: int,
        target_x: int,
        target_y: int
    ) -> bool:
        """
        Bir parçayı sürükle-bırak ile tahtaya yerleştir.

        Adımlar:
        1. Parçanın üzerine bas (tap) — parçayı seç
        2. 100ms bekle
        3. Parça merkezinden hedef konuma swipe yap
        4. 200ms bekle — animasyon tamamlansın
        Başarılıysa True döndür.
        """

    def is_connected(self) -> bool:
        """adb devices ile bağlantıyı kontrol et."""

    def _run(self, command: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
        """
        ADB komutunu çalıştır. Timeout ve hata yönetimi burada.
        Tüm public metodlar bu private metodu kullanır.
        """
```

---

## MODÜL 2: `control/coordinate_mapper.py`

### Görev
"Parça 0'ı tahta konumu (3, 5)'e yerleştir" bilgisini
piksel koordinatlarına çevir.

```python
import json
from pathlib import Path

class CoordinateMapper:
    def __init__(self, config: dict):
        """config: calibration_data/config.json içeriği."""

    def board_cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """
        Tahta hücresinin merkez piksel koordinatını döndür.
        Örnek: (3, 5) → (567, 823) (telefona göre değişir)

        Hesaplama:
        cell_x = board_left + col * cell_width  + cell_width  / 2
        cell_y = board_top  + row * cell_height + cell_height / 2
        """

    def piece_slot_to_pixel(self, slot_index: int) -> tuple[int, int]:
        """
        Parça slotunun (0, 1, 2) merkez piksel koordinatını döndür.
        Parçalar genellikle alt kısımda yatay sıralıdır.
        """

    def calculate_drag_target(
        self,
        piece_name: str,
        target_row: int,
        target_col: int
    ) -> tuple[int, int]:
        """
        Parçanın hangi koordinata sürükleneceğini hesapla.

        Block Blast'ta sürükleme hedefi parçanın SOL ÜST köşesi değil,
        parçanın MERKEZİ ile hücrenin merkezinin hizalanacağı noktadır.

        Hesaplama:
        1. Parçanın boyutunu al (piece_rows, piece_cols)
        2. Parçanın merkez offsetini hesapla
        3. Hedef hücrenin piksel merkezini bul
        4. Sürükleme hedefini ayarla

        UYARI: Bu hesap telefona ve oyun versiyonuna göre ince ayar
        gerektirebilir. `calibration_data/config.json`'a
        `drag_offset_x` ve `drag_offset_y` ekleyerek ayarla.
        """

    def validate_coordinates(self, x: int, y: int) -> bool:
        """
        Koordinatlar ekran sınırları içinde mi?
        Sınır dışı koordinatlar telefona zarar verebilir.
        """
```

---

## MODÜL 3: `control/action_executor.py`

### Görev
Model kararını `(piece_index, row, col)` üçlüsünden
gerçek dokunma hareketine çevir.

```python
import time
import logging
from control.adb_controller import ADBController
from control.coordinate_mapper import CoordinateMapper
from env.pieces import get_piece_cells

logger = logging.getLogger(__name__)

class ActionExecutor:
    MOVE_DELAY = 1.5      # Hamleler arası bekleme (saniye)
    TURN_DELAY = 2.0      # Tur geçişinde bekleme (saniye)
    RETRY_COUNT = 2       # Başarısız hamlede kaç kez tekrar dene

    def __init__(self, adb: ADBController, mapper: CoordinateMapper):
        ...

    def execute_action(
        self,
        piece_index: int,
        row: int,
        col: int,
        piece_name: str
    ) -> bool:
        """
        Tek bir hamleyi gerçekleştir.

        Adımlar:
        1. piece_index → slot piksel koordinatı (mapper)
        2. (row, col) → tahta piksel koordinatı (mapper)
        3. Sürükleme hedefini hesapla (mapper.calculate_drag_target)
        4. ADB drag_piece() ile hareketi gerçekleştir
        5. MOVE_DELAY kadar bekle (oyun animasyonu tamamlansın)
        6. Başarısızsa RETRY_COUNT kadar tekrar dene
        7. Başarı/başarısızlık logla

        Döndür: başarılıysa True
        """

    def execute_turn(
        self,
        actions: list[tuple[int, int, int]],
        piece_names: list[str]
    ) -> bool:
        """
        Bir turdaki 3 hamlenin tamamını sırayla gerçekleştir.
        actions: [(piece_index, row, col), ...] — en fazla 3 eleman
        piece_names: her aksiyon için parça ismi

        Herhangi bir hamle başarısız olursa False döndür.
        """
```

---

## MODÜL 4: `agent_loop.py`

### Görev
Tüm sistemi birleştiren ana döngü.
Bu dosyayı çalıştırmak otomasyonu başlatır.

```python
"""
Kullanım:
    python agent_loop.py                     # Varsayılan model
    python agent_loop.py --model models/dqn_v2
    python agent_loop.py --agent heuristic   # Model yerine heuristik kullan
    python agent_loop.py --dry-run           # ADB olmadan simüle et (test için)
"""

import time
import argparse
import logging
from pathlib import Path

# Kendi modüllerimiz
from stable_baselines3 import DQN
from vision import VisionPipeline
from control.adb_controller import ADBController
from control.coordinate_mapper import CoordinateMapper
from control.action_executor import ActionExecutor
from rl.gym_env import BlockBlastGymEnv
from rl.action_mapper import action_to_tuple, get_valid_action_mask
from rl.observation import encode_observation
from agents.heuristic_agent import HeuristicAgent

logger = logging.getLogger(__name__)

# --- DÖNGÜ PARAMETRELERİ ---
LOOP_INTERVAL   = 2.5   # Her tur arasında bekle (saniye)
MAX_ERRORS      = 5     # Arka arkaya bu kadar hata olursa dur
VISION_RETRIES  = 3     # Tespit başarısızsa kaç kez tekrar dene

def run_loop(args):
    """
    Ana otonom döngü. Sonsuz döngüde çalışır, Ctrl+C ile durdurulur.

    Her iterasyon:

    1. VisionPipeline.get_game_state() → mevcut tahta + parçalar
    2. Tespit güveni < 0.7 ise atla, tekrar dene
    3. Oyun bitti mi? (VisionPipeline.is_game_over()) → logla ve yeni oyun bekle
    4. game_state'i observation vektörüne çevir (encode_observation)
    5. Model.predict(observation, action_masks=valid_mask) → aksiyon
    6. Aksiyon geçerli mi? Geçersizse heuristik agent'a yedekle
    7. ActionExecutor.execute_action() → hamleyi yap
    8. Sonucu logla: tur no, seçilen parça, koordinat, güven skoru
    9. LOOP_INTERVAL kadar bekle
    10. MAX_ERRORS kadar arka arkaya hata olursa dur ve uyar
    """

def build_observation_from_vision(game_state: dict) -> tuple:
    """
    VisionPipeline çıktısını GameEnv formatına çevir,
    ardından encode_observation ile vektöre dönüştür.

    Döndür: (obs_vector, valid_action_mask)
    """

def wait_for_game_start():
    """
    'Game Over' ekranı görünüyorsa veya menüdeyse
    oyun başlayana kadar 3 saniyede bir kontrol et.
    """

def log_action(turn: int, piece_index: int, piece_name: str, row: int, col: int, confidence: float):
    """Her hamleyi yapılandırılmış formatta logla."""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/dqn_v1")
    parser.add_argument("--agent",  choices=["dqn", "heuristic"], default="dqn")
    parser.add_argument("--dry-run", action="store_true",
                        help="ADB komutlarını gerçekten gönderme, sadece logla")
    args = parser.parse_args()
    run_loop(args)
```

---

## LOGLAMA FORMATI

Her hamle için standart log satırı:

```
[TUR 047] Parça: kare_2x2 (slot 0) → Konum (3,5) | Güven: 0.94 | Süre: 1.2s
[TUR 047] Hamle başarılı ✓
[TUR 048] UYARI: Tespit güveni düşük (0.61), tekrar deneniyor...
[TUR 048] Hamle başarılı ✓
[OYUN BİTTİ] Toplam tur: 48 | Süre: 2m 14s
```

Log dosyası: `logs/agent_YYYYMMDD_HHMMSS.log`

---

## DRY-RUN MODU

`--dry-run` flag'i ile ADB komutları gerçekten gönderilmez,
sadece loglanır. Telefon bağlı olmadan sistemi test etmek için kullan.

```
[DRY-RUN] adb shell input swipe 234 1680 567 823 300
[DRY-RUN] Hamle simüle edildi: kare_2x2 → (3,5)
```

---

## HATA SENARYOLARI VE ÇÖZÜMLER

```python
# Tüm hata senaryolarını ele al:

# 1. ADB bağlantısı kesildi
#    → 10 saniye bekle, yeniden bağlan, MAX_ERRORS'a saymayacak

# 2. Tespit güveni sürekli düşük
#    → debug_frames/ klasörüne görüntü kaydet, döngüyü yavaşlat

# 3. Model geçersiz aksiyon üretiyor
#    → HeuristicAgent'a yedekle, olayı logla

# 4. Oyun dondu / menüye düştü
#    → wait_for_game_start() çağır

# 5. Koordinat ekran dışına çıkıyor
#    → CoordinateMapper.validate_coordinates() False dönerse hamleyi atla
```

---

## KONTROL LİSTESİ

- [ ] `ADBController.is_connected()` True dönüyor
- [ ] `ADBController.tap(x, y)` telefonda görünür bir tap yapıyor
- [ ] `ADBController.swipe()` telefonda sürükleme yapıyor
- [ ] `CoordinateMapper` test için bilinen bir hücrenin pikselini doğru veriyor
- [ ] `--dry-run` modda telefon olmadan `agent_loop.py` çalışıyor
- [ ] `agent_loop.py` ilk turu başarıyla tamamlıyor
- [ ] Log dosyası oluşuyor ve okunabilir formatta
- [ ] MAX_ERRORS aşılınca döngü düzgün kapanıyor
