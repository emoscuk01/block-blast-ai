"""8×8 board matrisini Streamlit'te görsel olarak çizen modül."""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from env.pieces import get_piece_cells

EMPTY_COLOR = "#1a1a2e"
FILLED_COLOR = "#4CAF50"
ACTION_COLOR = "#FF6B35"
GRID_COLOR = "#2d2d44"
CELL_SIZE = 42
BORDER = 2


def render_board(
    board: np.ndarray | list,
    last_action: Optional[dict] = None,
    piece_shape: Optional[list[list[int]]] = None,
) -> None:
    """8×8 board'u HTML tablo olarak Streamlit'te gösterir."""
    if isinstance(board, list):
        board = np.array(board)

    action_cells: set[tuple[int, int]] = set()
    if last_action is not None:
        piece_name = last_action.get("piece_name")
        row = last_action.get("row", 0)
        col = last_action.get("col", 0)
        if piece_name:
            try:
                cells = get_piece_cells(piece_name)
                for r, row_data in enumerate(cells):
                    for c, val in enumerate(row_data):
                        if val == 1:
                            action_cells.add((row + r, col + c))
            except ValueError:
                pass

    html = f'<table style="border-collapse:collapse; margin:auto;">'

    # Sütun numaraları
    html += '<tr><td style="width:20px;"></td>'
    for c in range(8):
        html += (
            f'<td style="width:{CELL_SIZE}px; text-align:center; '
            f'color:#888; font-size:11px; font-family:monospace;">{c}</td>'
        )
    html += '</tr>'

    for r in range(8):
        html += '<tr>'
        html += (
            f'<td style="width:20px; text-align:center; color:#888; '
            f'font-size:11px; font-family:monospace; vertical-align:middle;">{r}</td>'
        )
        for c in range(8):
            if (r, c) in action_cells:
                bg = ACTION_COLOR
            elif board[r, c] == 1:
                bg = FILLED_COLOR
            else:
                bg = EMPTY_COLOR

            html += (
                f'<td style="width:{CELL_SIZE}px; height:{CELL_SIZE}px; '
                f'background:{bg}; border:{BORDER}px solid {GRID_COLOR};"></td>'
            )
        html += '</tr>'
    html += '</table>'

    st.markdown(html, unsafe_allow_html=True)


def render_pieces(pieces: list[Optional[str]]) -> None:
    """3 parçayı yan yana gösterir."""
    cols = st.columns(3)

    for i, piece_name in enumerate(pieces):
        with cols[i]:
            if piece_name is None:
                st.markdown(
                    f'<div style="text-align:center; color:#888; padding:10px;">'
                    f'<span style="color:#4CAF50;">✓</span> yerleştirildi</div>',
                    unsafe_allow_html=True,
                )
                continue

            st.caption(piece_name)
            try:
                cells = get_piece_cells(piece_name)
            except ValueError:
                st.text("?")
                continue

            html = '<table style="border-collapse:collapse; margin:auto;">'
            for row in cells:
                html += '<tr>'
                for val in row:
                    bg = FILLED_COLOR if val == 1 else EMPTY_COLOR
                    html += (
                        f'<td style="width:18px; height:18px; '
                        f'background:{bg}; border:1px solid {GRID_COLOR};"></td>'
                    )
                html += '</tr>'
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)
