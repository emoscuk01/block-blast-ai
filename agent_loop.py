"""
Block Blast AI — Ana Otonom Döngü

Kullanım:
    python agent_loop.py                     # Varsayılan DQN modeli
    python agent_loop.py --model models/dqn_v2
    python agent_loop.py --agent heuristic   # Model yerine heuristik kullan
    python agent_loop.py --dry-run           # ADB olmadan simüle et (test için)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- DÖNGÜ PARAMETRELERİ ---
LOOP_INTERVAL: float = 2.5
MAX_ERRORS: int = 5
VISION_RETRIES: int = 3
MIN_CONFIDENCE: float = 0.7


def setup_logging() -> None:
    """Log dosyası ve terminal çıktısını yapılandırır."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"agent_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Log dosyası: %s", log_file)


def build_observation_from_vision(game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    VisionPipeline çıktısını encode_observation formatına çevirir.
    Döndürür: (obs_vector, valid_action_mask)
    """
    from rl.observation import encode_observation
    from rl.action_mapper import TOTAL_ACTIONS, tuple_to_action
    from env.pieces import get_piece_cells, get_piece_size

    board = game_state["board"]
    pieces = game_state["pieces"]

    pieces_arrays = []
    pieces_remaining = []
    for piece_name in pieces:
        if piece_name is not None:
            cells = get_piece_cells(piece_name)
            padded = np.zeros((5, 5), dtype=np.float32)
            for r, row in enumerate(cells):
                for c, val in enumerate(row):
                    padded[r, c] = float(val)
            pieces_arrays.append(padded)
            pieces_remaining.append(True)
        else:
            pieces_arrays.append(np.zeros((5, 5), dtype=np.float32))
            pieces_remaining.append(False)

    blocks_remaining = sum(1 for p in pieces if p is not None)

    obs_dict = {
        "board": board.astype(np.float32),
        "pieces": pieces_arrays,
        "pieces_remaining": pieces_remaining,
        "blocks_remaining": blocks_remaining,
        "score": 0,
        "turn": 0,
    }

    obs_vector = encode_observation(obs_dict)

    # Valid action mask hesapla
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    for idx, piece_name in enumerate(pieces):
        if piece_name is None:
            continue
        p_rows, p_cols = get_piece_size(piece_name)
        cells = get_piece_cells(piece_name)
        for r in range(8 - p_rows + 1):
            for c in range(8 - p_cols + 1):
                can_place = True
                for pr in range(p_rows):
                    for pc in range(p_cols):
                        if cells[pr][pc] == 1 and board[r + pr, c + pc] != 0:
                            can_place = False
                            break
                    if not can_place:
                        break
                if can_place:
                    action_id = tuple_to_action(idx, r, c)
                    mask[action_id] = True

    return obs_vector, mask


def wait_for_game_start(vision) -> None:
    """Oyun başlayana kadar bekler."""
    logger.info("Oyun başlaması bekleniyor...")
    while True:
        try:
            screenshot = vision.capture.capture()
            if not vision.is_game_over(screenshot):
                logger.info("Oyun tespit edildi, başlıyoruz!")
                return
        except Exception:
            pass
        time.sleep(3)


def log_action(
    turn: int,
    piece_index: int,
    piece_name: str,
    row: int,
    col: int,
    confidence: float,
) -> None:
    """Her hamleyi yapılandırılmış formatta loglar."""
    logger.info(
        "[TUR %03d] Parça: %s (slot %d) → Konum (%d,%d) | Güven: %.2f",
        turn, piece_name, piece_index, row, col, confidence,
    )


def run_loop(args: argparse.Namespace) -> None:
    """
    Ana otonom döngü. Sonsuz döngüde çalışır, Ctrl+C ile durdurulur.
    """
    from vision import VisionPipeline
    from control.adb_controller import ADBController
    from control.coordinate_mapper import CoordinateMapper
    from control.action_executor import ActionExecutor
    from rl.action_mapper import action_to_tuple
    from agents.heuristic_agent import HeuristicAgent

    # StateBridge (Aşama 5 ile birlikte çalışır, yoksa atla)
    bridge = None
    try:
        from dashboard.state_bridge import StateBridge
        bridge = StateBridge()
    except ImportError:
        pass

    dry_run = args.dry_run
    vision = VisionPipeline(debug=True)

    adb = ADBController(dry_run=dry_run)
    if not dry_run and not adb.is_connected():
        logger.error("ADB bağlantısı kurulamadı! --dry-run ile deneyin.")
        return

    config = vision.config
    mapper = CoordinateMapper(config)
    executor = ActionExecutor(adb, mapper)

    # Agent seçimi
    use_heuristic = args.agent == "heuristic"
    model = None
    heuristic = HeuristicAgent()

    if not use_heuristic:
        try:
            from stable_baselines3 import DQN
            model = DQN.load(args.model)
            logger.info("DQN modeli yüklendi: %s", args.model)
        except Exception as e:
            logger.warning("DQN modeli yüklenemedi (%s), heuristik agent'a geçiliyor.", e)
            use_heuristic = True

    logger.info("Agent: %s", "heuristic" if use_heuristic else "dqn")
    logger.info("Dry-run: %s", dry_run)

    consecutive_errors = 0
    turn_number = 0
    start_time = time.time()

    try:
        while True:
            # 1. Oyun durumunu al
            game_state = None
            for retry in range(VISION_RETRIES):
                game_state = vision.get_game_state()
                if game_state is not None and game_state["confidence"] >= MIN_CONFIDENCE:
                    break
                if game_state is not None:
                    logger.warning(
                        "[TUR %03d] UYARI: Tespit güveni düşük (%.2f), tekrar deneniyor...",
                        turn_number, game_state["confidence"],
                    )
                time.sleep(1)

            if game_state is None:
                consecutive_errors += 1
                logger.error("Oyun durumu tespit edilemedi (hata %d/%d)", consecutive_errors, MAX_ERRORS)
                if consecutive_errors >= MAX_ERRORS:
                    logger.critical("Arka arkaya %d hata, döngü durduruluyor.", MAX_ERRORS)
                    break
                time.sleep(LOOP_INTERVAL)
                continue

            # 2. Oyun bitti mi?
            if vision.is_game_over(game_state["screenshot"]):
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                logger.info(
                    "[OYUN BİTTİ] Toplam tur: %d | Süre: %dm %ds",
                    turn_number, minutes, seconds,
                )
                if bridge:
                    bridge.mark_game_over(final_score=0, total_turns=turn_number)
                wait_for_game_start(vision)
                turn_number = 0
                start_time = time.time()
                consecutive_errors = 0
                continue

            board = game_state["board"]
            pieces = game_state["pieces"]
            confidence = game_state["confidence"]
            turn_number += 1

            # 3. Aksiyon seç
            piece_index: Optional[int] = None
            row: Optional[int] = None
            col: Optional[int] = None
            piece_name: Optional[str] = None

            if use_heuristic or model is None:
                # Heuristik fallback — basit greedy seçim
                from env.board import Board
                from env.pieces import get_piece_cells, get_piece_size
                temp_board = Board()
                temp_board.grid = board.copy()

                best_score = float("-inf")
                for idx, pname in enumerate(pieces):
                    if pname is None:
                        continue
                    placements = temp_board.get_valid_placements(pname)
                    for r, c in placements:
                        from utils.metrics import composite_score
                        test_board = temp_board.copy()
                        lines = test_board.place(pname, r, c)
                        remaining = [p for j, p in enumerate(pieces) if p is not None and j != idx]
                        score = composite_score(test_board.get_grid(), remaining, lines)
                        if score > best_score:
                            best_score = score
                            piece_index, row, col, piece_name = idx, r, c, pname
            else:
                obs_vector, mask = build_observation_from_vision(game_state)
                if not np.any(mask):
                    logger.info("Geçerli hamle yok, oyun bitmeli.")
                    continue

                raw_action, _ = model.predict(obs_vector, deterministic=True)
                action_int = int(raw_action)
                pi, r_act, c_act = action_to_tuple(action_int)

                if mask[action_int] and pi < len(pieces) and pieces[pi] is not None:
                    piece_index, row, col = pi, r_act, c_act
                    piece_name = pieces[pi]
                else:
                    # Model geçersiz aksiyon üretti → heuristik yedek
                    logger.warning("Model geçersiz aksiyon üretti, heuristik yedek kullanılıyor.")
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) > 0:
                        fallback_action = int(valid_indices[0])
                        pi, r_act, c_act = action_to_tuple(fallback_action)
                        piece_index, row, col = pi, r_act, c_act
                        piece_name = pieces[pi]

            if piece_index is None or piece_name is None:
                logger.warning("Hamle seçilemedi, atlanıyor.")
                consecutive_errors += 1
                time.sleep(LOOP_INTERVAL)
                continue

            # 4. Hamleyi logla
            log_action(turn_number, piece_index, piece_name, row, col, confidence)

            # 5. Hamleyi uygula
            t0 = time.time()
            success = executor.execute_action(piece_index, row, col, piece_name)
            elapsed_action = time.time() - t0

            if success:
                logger.info("[TUR %03d] Hamle başarılı ✓ (%.1fs)", turn_number, elapsed_action)
                consecutive_errors = 0
            else:
                logger.warning("[TUR %03d] Hamle başarısız ✗", turn_number)
                consecutive_errors += 1

            # 6. Dashboard güncelle
            if bridge:
                bridge.update(
                    turn=turn_number,
                    board=board,
                    pieces=pieces,
                    last_action=(piece_index, row, col),
                    last_reward=0.0,
                    score=0,
                    confidence=confidence,
                    q_values=None,
                )

            # 7. Hata limiti kontrolü
            if consecutive_errors >= MAX_ERRORS:
                logger.critical("Arka arkaya %d hata, döngü durduruluyor.", MAX_ERRORS)
                break

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        logger.info(
            "\nOtonom döngü durduruldu (Ctrl+C). Toplam tur: %d, süre: %.0fs",
            turn_number, elapsed,
        )


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Block Blast AI — Otonom Döngü")
    parser.add_argument("--model", default="models/dqn_v1", help="DQN model yolu")
    parser.add_argument(
        "--agent", choices=["dqn", "heuristic"], default="dqn", help="Agent tipi"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="ADB komutlarını gerçekten gönderme"
    )
    args = parser.parse_args()
    run_loop(args)
