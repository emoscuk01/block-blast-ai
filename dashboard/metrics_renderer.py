"""Skor, tur, reward ve Q değeri grafiklerini gösteren modül."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def render_score_card(score: int, turn: int, confidence: float) -> None:
    """Üç adet büyük metrik kartı gösterir."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="SKOR", value=f"{score:,}")
    with c2:
        st.metric(label="TUR", value=str(turn))
    with c3:
        st.metric(label="GÜVEN", value=f"%{int(confidence * 100)}")


def render_reward_chart(reward_history: list[float]) -> None:
    """Son hamlelerin reward değerlerini çizgi grafiği olarak gösterir."""
    if not reward_history:
        st.info("Henüz reward verisi yok.")
        return

    df = pd.DataFrame({
        "Hamle": list(range(1, len(reward_history) + 1)),
        "Reward": reward_history,
    })
    st.line_chart(df, x="Hamle", y="Reward")


def render_q_values(q_values: Optional[list[float]]) -> None:
    """En iyi aksiyonların Q değerini yatay bar chart olarak gösterir."""
    if q_values is None or len(q_values) == 0:
        st.info("Q değeri mevcut değil.")
        return

    top_n = min(10, len(q_values))
    sorted_vals = sorted(enumerate(q_values), key=lambda x: x[1], reverse=True)[:top_n]

    df = pd.DataFrame({
        "Aksiyon": [f"#{idx}" for idx, _ in sorted_vals],
        "Q Değeri": [val for _, val in sorted_vals],
    })
    st.bar_chart(df, x="Aksiyon", y="Q Değeri")


def render_status_badge(is_fresh: bool, is_game_over: bool) -> None:
    """Bağlantı durumunu renkli badge olarak gösterir."""
    if is_game_over:
        st.markdown(
            '<span style="color:#FF4444; font-size:16px; font-weight:bold;">'
            '🔴 Oyun Bitti</span>',
            unsafe_allow_html=True,
        )
    elif is_fresh:
        st.markdown(
            '<span style="color:#4CAF50; font-size:16px; font-weight:bold;">'
            '🟢 Çalışıyor</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="color:#FFD700; font-size:16px; font-weight:bold;">'
            '🟡 Bekliyor</span>',
            unsafe_allow_html=True,
        )
