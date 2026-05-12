# Block Blast AI — Aşama 3: Bilgisayarlı Görü
# Ekran Okuma → 8×8 Matris

---

## BAĞLAM

Aşama 0+1+2 tamamlandı. Elimizde şunlar var:
- Çalışan `GameEnv` simülatörü
- Eğitilmiş DQN modeli (`models/dqn_v1.zip`)
- `evaluate.py` ile doğrulanmış model performansı

Bu aşamada modeli gerçek oyuna bağlıyoruz. Telefon ekranını okuyup
8×8 matrise çeviren bir bilgisayarlı görü pipeline'ı yazıyoruz.
Model henüz telefona hamle yapmıyor — bu Aşama 4'ün işi.
Bu aşamanın tek çıktısı: **ekran görüntüsü → `GameEnv` formatında observation dict.**

---

## KURULUM GEREKSİNİMLERİ

### scrcpy kurulumu (Windows)
```
https://github.com/Genymobile/scrcpy adresinden Windows zip indir
PATH'e ekle
```

### ADB kurulumu
```
Android Studio SDK Tools içinde geliyor
ya da: https://developer.android.com/tools/releases/platform-tools
```

### Telefon hazırlığı
```
Ayarlar → Geliştirici seçenekleri → USB hata ayıklama: AÇ
Bilgisayara USB ile bağla
adb devices → cihaz görünmeli
```

### Python bağımlılıkları
```
opencv-python>=4.9.0
Pillow>=10.0.0
numpy (zaten var)
```

---

## DOSYA YAPISI (SADECE YENİ DOSYALAR)

```
block_blast_ai/
├── vision/
│   ├── __init__.py
│   ├── screen_capture.py   # ADB ile ekran görüntüsü al
│   ├── board_detector.py   # 8×8 tahtayı tespit et ve matrise çevir
│   ├── piece_detector.py   # 3 bloğu tespit et ve isimlendirmedir
│   ├── calibration.py      # İlk kurulumda koordinatları ayarla
│   └── debug_overlay.py    # Tespiti görselleştir (debug için)
├── calibration_data/
│   └── config.json         # Kalibre edilmiş koordinatlar burada saklanır
└── requirements_vision.txt
```

---

## MODÜL 1: `vision/screen_capture.py`

### Görev
ADB üzerinden telefon ekran görüntüsü al ve PIL Image olarak döndür.

```python
import subprocess
import numpy as np
from PIL import Image
import io

class ScreenCapture:
    def __init__(self, device_id: str = None):
        """
        device_id: birden fazla cihaz bağlıysa belirt.
        None ise otomatik seç.
        """

    def capture(self) -> Image.Image:
        """
        ADB screencap komutuyla ekran görüntüsü al.
        PIL Image (RGB) döndür.

        Komut:
            adb [-s device_id] exec-out screencap -p

        Bu yöntem PNG binary'sini stdout'tan okur — dosya yazmaz.
        Gecikme ~200-400ms, kabul edilebilir.
        """

    def capture_region(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """
        Tam ekranı al, belirtilen bölgeyi kırp ve döndür.
        Crop koordinatları piksel cinsinden.
        """

    def get_screen_size(self) -> tuple[int, int]:
        """
        adb shell wm size çıktısından (genişlik, yükseklik) döndür.
        """

    def test_connection(self) -> bool:
        """
        adb devices çalıştır, cihaz bağlı mı kontrol et.
        Bağlı değilse açıklayıcı hata mesajı ver.
        """
```

---

## MODÜL 2: `vision/calibration.py`

### Görev
Block Blast ekranındaki tahta ve parça bölgelerinin koordinatlarını
bir kez belirle ve `calibration_data/config.json`'a kaydet.
Sonraki çalışmalarda bu dosyadan oku.

### Kalibasyon mantığı

Block Blast ekranı her telefonda farklı boyutta olur.
Koordinatları hardcode etmek yerine bir kez interaktif kalibre et.

```python
import json
from pathlib import Path

CONFIG_PATH = Path("calibration_data/config.json")

class Calibrator:
    def __init__(self, screen_capture: ScreenCapture):
        ...

    def run_interactive(self):
        """
        Adım adım kalibasyon sihirbazı:

        1. Ekran görüntüsü al, dosyaya kaydet (calibration_data/screenshot.png)
        2. Kullanıcıya: "screenshot.png'yi aç, şu koordinatları gir:" mesajı ver:
           - Tahtanın sol üst köşesi (x, y)
           - Tahtanın sağ alt köşesi (x, y)
           - Parça alanının sol üst köşesi (x, y)
           - Parça alanının sağ alt köşesi (x, y)
        3. Girilen değerleri doğrula (mantıklı aralıkta mı?)
        4. config.json'a kaydet
        5. Doğrulama: tekrar ekran al, tespiti debug_overlay ile göster
        """

    def load_config(self) -> dict:
        """
        config.json varsa yükle, yoksa run_interactive() çağır.
        """

    def get_cell_coordinates(self, config: dict) -> list[list[tuple[int,int,int,int]]]:
        """
        Tahta koordinatlarından 8×8 = 64 hücrenin (x, y, w, h) değerlerini hesapla.
        Her hücre eşit boyutta varsayılır.
        """
```

### `calibration_data/config.json` formatı:
```json
{
  "screen_size": [1080, 2340],
  "board": {
    "top_left": [45, 380],
    "bottom_right": [1035, 1370],
    "cell_size": [123, 123]
  },
  "pieces_area": {
    "top_left": [45, 1450],
    "bottom_right": [1035, 1850]
  },
  "calibrated_at": "2025-01-01T12:00:00"
}
```

---

## MODÜL 3: `vision/board_detector.py`

### Görev
Ekran görüntüsünden 8×8 board matrisini çıkar.

### Renk bazlı tespit stratejisi

Block Blast'ta her hücre ya dolu (renkli) ya da boş (koyu/siyah).
Boş hücreler çok koyu, dolu hücreler parlak renkli.

```python
import cv2
import numpy as np
from PIL import Image

class BoardDetector:
    def __init__(self, config: dict):
        """config: Calibrator'dan gelen ayarlar."""

    def detect(self, screenshot: Image.Image) -> np.ndarray:
        """
        8×8 binary matris döndür (0=boş, 1=dolu).
        dtype=np.int8

        Algoritma:
        1. screenshot'ı numpy array'e çevir (RGB)
        2. Her hücre için merkez piksel rengini al
           (hücrenin tam ortası, kenar etkilerinden kaçın)
        3. Pikselin parlaklığını hesapla:
           brightness = 0.299*R + 0.587*G + 0.114*B
        4. brightness > THRESHOLD ise dolu (1), değilse boş (0)
        5. THRESHOLD varsayılan: 60 (kalibrasyonla ayarlanabilir)

        NOT: Sadece merkez piksele değil, hücrenin %40-%60 bölgesinin
        medyan parlaklığına bak. Tek piksel yanıltıcı olabilir.
        """

    def detect_with_confidence(self, screenshot: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        (board_matrix, confidence_matrix) döndür.
        confidence: her hücre için 0.0-1.0 arası güven skoru.
        Düşük güvenli hücreler debug için işaretlenebilir.
        """

    def set_threshold(self, threshold: int):
        """Parlaklık eşiğini runtime'da değiştir."""
```

---

## MODÜL 4: `vision/piece_detector.py`

### Görev
Ekrandaki 3 bloğu tespit et ve `pieces.py`'daki isimlere eşle.

### Strateji

Bu modülde iki yaklaşım dene, ikisini de implement et:

**Yaklaşım A — Şablon eşleme (basit, hızlı)**
Her parçanın küçük görüntüsünü önceden kaydet,
`cv2.matchTemplate` ile en yüksek eşleşmeyi bul.

**Yaklaşım B — Şekil tespiti (sağlam, tercih edilen)**
Renk filtresiyle parça piksellerini bul, kontur analizi yap,
dolu/boş hücre grid'ine çevir, `pieces.py` şekilleriyle karşılaştır.

```python
class PieceDetector:
    def __init__(self, config: dict):
        ...

    def detect_pieces(self, screenshot: Image.Image) -> list[str | None]:
        """
        3 elemanlı liste döndür.
        Her eleman: pieces.py'daki parça ismi veya None (boş slot).
        Örnek: ["kare_2x2", "yatay_3", None]

        Önce Yaklaşım B'yi dene, güven düşükse Yaklaşım A'ya geç.
        """

    def _extract_piece_region(self, screenshot: Image.Image, slot_index: int) -> np.ndarray:
        """
        Config'den parça alanını al, 3 slot'a böl, slot_index'teki bölgeyi döndür.
        """

    def _shape_to_piece_name(self, shape_matrix: np.ndarray) -> str | None:
        """
        Tespit edilen binary matrixi pieces.py şekilleriyle karşılaştır.
        En yüksek IoU (Intersection over Union) skorunu veren ismi döndür.
        Hiçbiri yeterince benzemiyorsa None döndür ve uyarı logla.
        """

    def _match_template(self, piece_region: np.ndarray) -> str | None:
        """Yaklaşım A implementasyonu."""

    def _detect_by_shape(self, piece_region: np.ndarray) -> str | None:
        """Yaklaşım B implementasyonu."""
```

---

## MODÜL 5: `vision/debug_overlay.py`

### Görev
Tespiti görselleştir. Bu modül üretim kodunda kullanılmaz,
sadece geliştirme ve debug sürecinde.

```python
import cv2
import numpy as np
from PIL import Image

def draw_board_overlay(
    screenshot: Image.Image,
    board_matrix: np.ndarray,
    config: dict,
    confidence: np.ndarray = None
) -> Image.Image:
    """
    Ekran görüntüsü üzerine:
    - Her hücrenin sınırını çiz (yeşil kare)
    - Dolu hücreleri kırmızı ile işaretle
    - confidence verilmişse düşük güvenli hücreleri sarı ile işaretle
    Sonucu debug_overlay.png olarak kaydet ve döndür.
    """

def draw_pieces_overlay(
    screenshot: Image.Image,
    detected_pieces: list[str | None],
    config: dict
) -> Image.Image:
    """
    Parça alanı üzerine tespit edilen parça isimlerini yaz.
    """

def save_debug_frame(screenshot: Image.Image, board: np.ndarray, pieces: list, frame_id: int):
    """
    debug_frames/ klasörüne tam debug görüntüsünü kaydet.
    Tespit hatalarını analiz etmek için kullanılır.
    """
```

---

## ENTEGRASYON SINIFI

### `vision/__init__.py`

```python
class VisionPipeline:
    """
    Tüm vision modüllerini bir araya getiren ana sınıf.
    Aşama 4'te bu sınıf kullanılacak.
    """
    def __init__(self, debug: bool = False):
        self.capture = ScreenCapture()
        config = Calibrator(self.capture).load_config()
        self.board_detector = BoardDetector(config)
        self.piece_detector = PieceDetector(config)
        self.debug = debug

    def get_game_state(self) -> dict | None:
        """
        Tek çağrıyla tam oyun durumunu döndür.

        Döndür:
        {
            "board": np.ndarray (8×8),
            "pieces": list[str | None],  # 3 parça
            "confidence": float,          # Ortalama tespit güveni
            "screenshot": PIL.Image,      # Ham görüntü
        }

        Tespit başarısızsa None döndür ve hatayı logla.
        """

    def is_game_over(self, screenshot: Image.Image) -> bool:
        """
        Oyun bitti ekranını tespit et.
        "Game Over" veya skor ekranı görünüyor mu?
        Renk histogramı veya basit template matching ile kontrol et.
        """
```

---

## TEST VE DOĞRULAMA

### `tests/test_vision.py`

```python
def test_screen_capture_returns_image():
    """capture() PIL Image döndürmeli, boyut telefon çözünürlüğüyle eşleşmeli."""

def test_board_detector_empty_board():
    """Tüm hücreler boş ekranda board_detector 8×8 sıfır matrisi döndürmeli."""

def test_board_detector_full_row():
    """Test görüntüsüyle bilinen bir satırı doğru tespit etmeli."""

def test_piece_detector_known_piece():
    """Kaydedilmiş test ekran görüntüsünde bilinen parçayı doğru tanımalı."""

def test_calibration_loads():
    """config.json varsa Calibrator hatasız yüklemeli."""
```

### Manuel test scripti: `vision_test.py`

```python
"""
Çalıştır: python vision_test.py
Telefon bağlı ve Block Blast açık olmalı.

Yapacakları:
1. Ekran görüntüsü al
2. Tahta ve parçaları tespit et
3. debug_overlay.png kaydet
4. Terminale tespit sonuçlarını yazdır
"""
```

---

## KOD STİLİ

- Tüm yorum ve docstring'ler Türkçe
- Type hint'ler zorunlu
- `cv2.imencode` / `cv2.imdecode` ile PIL ↔ numpy dönüşümlerini bir yerde topla
- ADB komutları `subprocess.run(..., capture_output=True, timeout=5)` ile çağrıl
- Timeout: ADB bağlantısı kesilirse 5 saniye bekleyip hata ver
- Hiçbir `print()` → her şey `logging` modülü ile

---

## KONTROL LİSTESİ

- [ ] `adb devices` cihazı gösteriyor
- [ ] `ScreenCapture.capture()` hatasız çalışıyor, görüntü mantıklı görünüyor
- [ ] `Calibrator.run_interactive()` config.json oluşturuyor
- [ ] `BoardDetector.detect()` boş tahtada 64 sıfır döndürüyor
- [ ] `BoardDetector.detect()` dolu satırda doğru 1'leri döndürüyor
- [ ] `PieceDetector.detect_pieces()` en az 2/3 parçayı doğru tanıyor
- [ ] `debug_overlay.png` görsel olarak mantıklı görünüyor
- [ ] `VisionPipeline.get_game_state()` tam dict döndürüyor
