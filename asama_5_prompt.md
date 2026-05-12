# Block Blast AI — Aşama 5: Dashboard
# Gerçek Zamanlı Görselleştirme (Streamlit)

---

## BAĞLAM

Tüm aşamalar tamamlandı. Elimizde şunlar var:
- Çalışan otonom döngü (`agent_loop.py`)
- Her hamle için yapılandırılmış log dosyaları (`logs/agent_*.log`)
- Eğitilmiş DQN modeli
- `VisionPipeline` → canlı ekran görüntüleri

Bu aşamada otonom ajan çalışırken bilgisayar ekranında ne olduğunu
gösteren bir dashboard hazırlıyoruz.

**Hedef görünüm:**
Sol tarafta canlı tahta + seçilen hamle, sağ tarafta AI metrikleri
ve skor grafiği. Bu ekranın videosu GitHub README'ye koyulacak.

---

## DOSYA YAPISI (SADECE YENİ DOSYALAR)

```
block_blast_ai/
├── dashboard/
│   ├── __init__.py
│   ├── state_bridge.py     # agent_loop ↔ dashboard arası veri köprüsü
│   ├── board_renderer.py   # 8×8 tahtayı Streamlit'te çiz
│   └── metrics_renderer.py # Grafikler ve metrik kartları
├── dashboard_app.py        # Ana Streamlit uygulaması
└── requirements_dashboard.txt
```

---

## MİMARİ: İKİ AYRI PROCESS

Dashboard ve agent_loop **aynı anda** çalışır, farklı terminal penceresinde:

```
Terminal 1: python agent_loop.py
Terminal 2: streamlit run dashboard_app.py
```

Bu iki process haberleşmek için **JSON dosya köprüsü** kullanır:
`dashboard/live_state.json` — agent_loop her turda bu dosyayı günceller,
dashboard her N saniyede bir okur.

Bu yaklaşım basit, debug kolaylığı sağlar ve websocket/socket gerektirmez.

---

## MODÜL 1: `dashboard/state_bridge.py`

### Görev
`agent_loop.py`'ın dashboard'a veri yazmasını ve
dashboard'ın bu veriyi okumasını sağla.

```python
import json
import time
from pathlib import Path
import numpy as np

STATE_FILE = Path("dashboard/live_state.json")

class StateBridge:
    """agent_loop tarafında kullanılır — veri yazar."""

    def update(
        self,
        turn: int,
        board: np.ndarray,
        pieces: list[str | None],
        last_action: tuple[int, int, int] | None,
        last_reward: float,
        score: int,
        confidence: float,
        q_values: list[float] | None = None,
    ):
        """
        Mevcut state'i JSON dosyasına yaz.
        Atomic write kullan: önce .tmp yaz, sonra rename.
        Bu sayede dashboard yarım dosya okumaz.

        JSON formatı:
        {
            "turn": 47,
            "timestamp": 1735000000.0,
            "board": [[0,1,0,...], ...],  # 8×8 liste
            "pieces": ["kare_2x2", "yatay_3", null],
            "last_action": {"piece_index": 0, "row": 3, "col": 5, "piece_name": "kare_2x2"},
            "last_reward": 12.5,
            "score": 3240,
            "confidence": 0.94,
            "q_values": [0.12, -0.34, ...],  # En iyi 10 aksiyon Q değeri, null olabilir
            "is_game_over": false
        }
        """

    def mark_game_over(self, final_score: int, total_turns: int):
        """Oyun bitince is_game_over=True yaz."""

class StateReader:
    """Dashboard tarafında kullanılır — veri okur."""

    def read(self) -> dict | None:
        """
        JSON dosyasını oku ve dict döndür.
        Dosya yoksa veya bozuksa None döndür.
        """

    def is_fresh(self, max_age_seconds: float = 10.0) -> bool:
        """
        Son güncelleme max_age_seconds'dan yeni mi?
        Eski ise agent_loop çalışmıyor olabilir.
        """
```

---

## MODÜL 2: `dashboard/board_renderer.py`

### Görev
8×8 board matrisini Streamlit'te görsel olarak çiz.
Son yapılan hamleyi vurgula.

```python
import streamlit as st
import numpy as np

# Renk paleti
EMPTY_COLOR  = "#1a1a2e"   # Koyu lacivert — boş hücre
FILLED_COLOR = "#4CAF50"   # Yeşil — dolu hücre
ACTION_COLOR = "#FF6B35"   # Turuncu — son hamlenin hücreleri
GRID_COLOR   = "#2d2d44"   # Çizgi rengi

def render_board(
    board: np.ndarray,
    last_action: dict | None = None,
    piece_shape: list[list[int]] | None = None
):
    """
    8×8 board'u HTML tablo olarak Streamlit'te göster.
    st.markdown() ile HTML/CSS enjekte et.

    Son hamle varsa (last_action + piece_shape):
    - Yerleştirilen hücreleri ACTION_COLOR ile vurgula

    Hücre boyutu: 42×42 px, kenarlık: 2px
    """

def render_pieces(pieces: list[str | None]):
    """
    3 parçayı yan yana göster.
    Her parça küçük bir grid olarak çizilir (aynı HTML tablo yaklaşımı).
    None olan parçalar için gri "✓ yerleştirildi" göster.
    """
```

---

## MODÜL 3: `dashboard/metrics_renderer.py`

### Görev
Skor, tur, reward ve Q değeri grafiklerini göster.

```python
import streamlit as st
import pandas as pd

def render_score_card(score: int, turn: int, confidence: float):
    """
    Üç adet büyük metrik kartı:
    [ SKOR: 3240 ]  [ TUR: 47 ]  [ GÜVEN: %94 ]
    st.metric() kullan.
    """

def render_reward_chart(reward_history: list[float]):
    """
    Son 50 hamlenin reward değerlerini çizgi grafiği olarak göster.
    st.line_chart() kullan.
    Pozitif reward yeşil, negatif kırmızı çizgi.
    """

def render_q_values(q_values: list[float] | None):
    """
    En iyi 10 aksiyonun Q değerini yatay bar chart olarak göster.
    q_values None ise "Q değeri mevcut değil" mesajı göster.
    """

def render_status_badge(is_fresh: bool, is_game_over: bool):
    """
    Bağlantı durumu:
    - Taze veri: 🟢 Çalışıyor
    - Eski veri: 🟡 Bekliyor
    - Oyun bitti: 🔴 Oyun Bitti
    """
```

---

## MODÜL 4: `dashboard_app.py`

### Ana Streamlit uygulaması

```python
"""
Çalıştır: streamlit run dashboard_app.py
Tarayıcıda: http://localhost:8501
"""
import streamlit as st
import time
from dashboard.state_bridge import StateReader
from dashboard.board_renderer import render_board, render_pieces
from dashboard.metrics_renderer import (
    render_score_card, render_reward_chart,
    render_q_values, render_status_badge
)

# Sayfa yapılandırması
st.set_page_config(
    page_title="Block Blast AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Özel CSS — koyu tema, neon vurgular
CUSTOM_CSS = """
<style>
    .main { background-color: #0f0f1a; color: #e0e0e0; }
    .stMetric label { color: #888; font-size: 12px; }
    .stMetric [data-testid="metric-container"] { background: #1a1a2e; border-radius: 8px; padding: 12px; }
    h1 { color: #4CAF50; font-family: monospace; }
</style>
"""

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("⬛ Block Blast AI — Canlı Dashboard")

    reader = StateReader()

    # Reward geçmişi — session_state'te tut
    if "reward_history" not in st.session_state:
        st.session_state.reward_history = []

    # Otomatik yenileme: 2 saniyede bir
    refresh_placeholder = st.empty()

    while True:
        state = reader.read()

        with refresh_placeholder.container():
            # Durum badge'i
            render_status_badge(
                is_fresh=reader.is_fresh(),
                is_game_over=state.get("is_game_over", False) if state else False
            )

            if state is None:
                st.warning("⏳ agent_loop.py bekleniyor... `python agent_loop.py` çalıştır.")
            else:
                # Reward geçmişini güncelle
                if state.get("last_reward") is not None:
                    st.session_state.reward_history.append(state["last_reward"])
                    # Son 200 hamlede tut
                    st.session_state.reward_history = st.session_state.reward_history[-200:]

                # Düzen: sol sütun (tahta) | sağ sütun (metrikler)
                col_board, col_metrics = st.columns([1, 1])

                with col_board:
                    st.subheader("🎯 Tahta")
                    render_board(
                        board=state["board"],
                        last_action=state.get("last_action"),
                    )
                    st.subheader("🧩 Mevcut Parçalar")
                    render_pieces(state["pieces"])

                with col_metrics:
                    st.subheader("📊 Metrikler")
                    render_score_card(
                        score=state["score"],
                        turn=state["turn"],
                        confidence=state["confidence"]
                    )
                    st.subheader("📈 Reward Geçmişi")
                    render_reward_chart(st.session_state.reward_history)
                    st.subheader("🧠 Q Değerleri")
                    render_q_values(state.get("q_values"))

        time.sleep(2)  # 2 saniyede bir yenile
```

---

## `agent_loop.py`'A EKLENECEK KOD

Aşama 4'te yazdığın `agent_loop.py`'ın `run_loop()` fonksiyonuna
şu satırları ekle (mevcut kodu değiştirme, sadece `StateBridge` çağrısı ekle):

```python
# agent_loop.py içinde, import bloğuna ekle:
from dashboard.state_bridge import StateBridge

# run_loop() içinde, StateBridge örneği oluştur:
bridge = StateBridge()

# Her hamle sonrası bridge.update() çağır:
bridge.update(
    turn=turn_number,
    board=current_board,
    pieces=current_pieces,
    last_action=(piece_index, row, col),
    last_reward=reward,
    score=game_score,
    confidence=vision_confidence,
    q_values=None,  # Model Q değeri döndürüyorsa buraya ekle
)
```

---

## `requirements_dashboard.txt`

```
streamlit>=1.32.0
pandas>=2.0.0
watchdog>=4.0.0
```

---

## KONTROL LİSTESİ

- [ ] `streamlit run dashboard_app.py` tarayıcıda açılıyor
- [ ] `agent_loop.py` çalışmıyorken uyarı mesajı gösteriyor
- [ ] `agent_loop.py` çalışınca tahta otomatik güncelleniyor
- [ ] Son hamle turuncu ile vurgulanıyor
- [ ] Reward grafiği canlı güncelleniyor
- [ ] Sayfa `layout="wide"` ile geniş ekranda iyi görünüyor
- [ ] Ekran kaydı alındığında dashboard okunabilir ve temiz görünüyor
- [ ] `live_state.json` her tur güncelleniyor (dosyayı terminal'de kontrol et)

---

## GITHUB README İÇİN

Dashboard tamamlanınca şunu yap:

1. `agent_loop.py` + `dashboard_app.py`'ı aynı anda çalıştır
2. OBS veya Windows Game Bar (Win+G) ile ekran kaydı al
3. GIF'e çevir (30 saniye yeterli): `ffmpeg -i kayit.mp4 -vf fps=10 demo.gif`
4. `README.md`'ye ekle

Bu GIF projenin en güçlü anlatım aracı olur.
