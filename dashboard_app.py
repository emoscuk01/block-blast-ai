"""
Block Blast AI — Canlı Dashboard

Çalıştır: streamlit run dashboard_app.py
Tarayıcıda: http://localhost:8501
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from dashboard.state_bridge import StateReader
from dashboard.board_renderer import render_board, render_pieces
from dashboard.metrics_renderer import (
    render_score_card,
    render_reward_chart,
    render_q_values,
    render_status_badge,
)

st.set_page_config(
    page_title="Block Blast AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0f0f1a; color: #e0e0e0; }
    .stMetric label { color: #888; font-size: 12px; }
    .stMetric [data-testid="metric-container"] {
        background: #1a1a2e; border-radius: 8px; padding: 12px;
    }
    h1 { color: #4CAF50; font-family: monospace; }
    h2, h3 { color: #ccc; }
    .stSubheader { color: #aaa; }
</style>
"""


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("⬛ Block Blast AI — Canlı Dashboard")

    reader = StateReader()

    if "reward_history" not in st.session_state:
        st.session_state.reward_history = []
    if "last_turn" not in st.session_state:
        st.session_state.last_turn = 0

    state = reader.read()

    is_fresh = reader.is_fresh()
    is_game_over = state.get("is_game_over", False) if state else False
    render_status_badge(is_fresh=is_fresh, is_game_over=is_game_over)

    if state is None:
        st.warning("⏳ agent_loop.py bekleniyor... `python agent_loop.py` çalıştır.")
        st.info(
            "Dashboard otomatik olarak her 2 saniyede bir yenilenir. "
            "agent_loop başlatıldığında veriler otomatik gösterilecek."
        )
        time.sleep(2)
        st.rerun()
        return

    current_turn = state.get("turn", 0)
    if current_turn != st.session_state.last_turn:
        reward = state.get("last_reward")
        if reward is not None:
            st.session_state.reward_history.append(reward)
            st.session_state.reward_history = st.session_state.reward_history[-200:]
        st.session_state.last_turn = current_turn

    col_board, col_metrics = st.columns([1, 1])

    with col_board:
        st.subheader("🎯 Tahta")
        board_data = state.get("board", np.zeros((8, 8)).tolist())
        board = np.array(board_data)
        render_board(
            board=board,
            last_action=state.get("last_action"),
        )

        st.subheader("🧩 Mevcut Parçalar")
        pieces = state.get("pieces", [None, None, None])
        render_pieces(pieces)

    with col_metrics:
        st.subheader("📊 Metrikler")
        render_score_card(
            score=state.get("score", 0),
            turn=state.get("turn", 0),
            confidence=state.get("confidence", 0.0),
        )

        st.subheader("📈 Reward Geçmişi")
        render_reward_chart(st.session_state.reward_history)

        st.subheader("🧠 Q Değerleri")
        render_q_values(state.get("q_values"))

    st.divider()
    with st.expander("🔧 Ham Veri (Debug)"):
        st.json(state)

    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
