"""GPU Ortam Doğrulama Testi.

GPU-vectorized ortamın temel işlevlerini test eder.

Kullanım:
    python -m tests.test_gpu_env
"""

from __future__ import annotations

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch


def test_pieces_registry():
    """Parça tensör registry'sinin doğru oluşturulduğunu test et."""
    print("\n[TEST 1] Pieces Registry...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from env.pieces_gpu import PieceRegistry, NUM_PIECE_TYPES, PIECE_NAMES
    from env.pieces import PIECES

    reg = PieceRegistry.get(device)

    assert reg.tensors_5x5.shape == (NUM_PIECE_TYPES, 5, 5), \
        f"tensors_5x5 shape: {reg.tensors_5x5.shape}"
    assert len(reg.kernels) == NUM_PIECE_TYPES

    for pid, name in enumerate(PIECE_NAMES):
        cells = PIECES[name]
        expected_n = sum(c for row in cells for c in row)
        actual_n = reg.n_cells[pid].item()
        assert actual_n == expected_n, f"{name}: expected {expected_n} cells, got {actual_n}"

    print(f"  OK: {NUM_PIECE_TYPES} parça doğrulandı ({device})")


def test_board_operations():
    """Board GPU operasyonlarını test et."""
    print("\n[TEST 2] Board Operations...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from env.pieces_gpu import PieceRegistry, PIECE_NAMES
    from env.board_gpu import BatchBoard

    PieceRegistry.get(device)
    N = 4
    board = BatchBoard(N, device)

    assert board.grids.sum().item() == 0, "Tahtalar boş olmalı"

    tek_id = PIECE_NAMES.index("tek")
    piece_ids = torch.full((N,), tek_id, dtype=torch.long, device=device)
    rows = torch.zeros(N, dtype=torch.long, device=device)
    cols = torch.zeros(N, dtype=torch.long, device=device)

    lines = board.place_batch(piece_ids, rows, cols)
    assert board.grids[:, 0, 0].sum().item() == N, "Her tahtada (0,0) dolu olmalı"
    assert lines.sum().item() == 0, "Satır silinmemeli"

    board.reset()
    assert board.grids.sum().item() == 0, "Reset sonrası boş olmalı"

    print(f"  OK: Board ops ({device})")


def test_valid_mask():
    """Valid action mask hesaplamasını test et."""
    print("\n[TEST 3] Valid Action Mask...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from env.pieces_gpu import PieceRegistry, PIECE_NAMES
    from env.board_gpu import BatchBoard

    PieceRegistry.get(device)
    N = 2
    board = BatchBoard(N, device)

    tek_id = PIECE_NAMES.index("tek")
    piece_ids = torch.full((N, 3), tek_id, dtype=torch.long, device=device)

    mask = board.compute_valid_mask(piece_ids)
    assert mask.shape == (N, 192), f"Mask shape: {mask.shape}"
    assert mask.all(), f"Boş tahtada tek parça: {mask.sum().item()}/192 geçerli"

    piece_ids[:, 2] = -1
    mask2 = board.compute_valid_mask(piece_ids)
    slot2_mask = mask2[:, 128:192]
    assert not slot2_mask.any(), "Kapalı slot'ta aksiyon olmamalı"

    print(f"  OK: Valid mask doğrulandı ({device})")


def test_batch_game_env():
    """BatchGameEnv step/reset döngüsünü test et."""
    print("\n[TEST 4] BatchGameEnv Step/Reset...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from env.game_env_gpu import BatchGameEnv

    N = 8
    env = BatchGameEnv(N, device)
    obs, masks = env.reset_all()

    assert obs.shape == (N, 142), f"Obs shape: {obs.shape}"
    assert masks.shape == (N, 192), f"Masks shape: {masks.shape}"
    assert masks.any(dim=1).all(), "Tüm ortamlarda geçerli aksiyon olmalı"

    total_dones = 0
    for step_i in range(100):
        valid_indices = []
        for i in range(N):
            valid = masks[i].nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                chosen = valid[torch.randint(len(valid), (1,), device=device)].item()
            else:
                chosen = 0
            valid_indices.append(chosen)

        actions = torch.tensor(valid_indices, dtype=torch.long, device=device)
        obs, rewards, dones, infos = env.step(actions)
        masks = infos["action_masks"]
        total_dones += dones.sum().item()

        assert obs.shape == (N, 142)
        assert rewards.shape == (N,)

    print(f"  OK: 100 step, {total_dones} episode bitti ({device})")


def test_gpu_vec_env():
    """GpuVecEnv SB3 uyumluluğunu test et."""
    print("\n[TEST 5] GpuVecEnv SB3 Uyumluluğu...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from rl.gpu_vec_env import GpuVecEnv

    N = 4
    env = GpuVecEnv(n_envs=N, device=device)

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (N, 142)

    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.shape == (N, 192)

    for _ in range(50):
        actions = []
        for i in range(N):
            valid = np.where(masks[i])[0]
            if len(valid) > 0:
                actions.append(np.random.choice(valid))
            else:
                actions.append(0)

        obs, rewards, dones, infos = env.step(np.array(actions))
        assert obs.shape == (N, 142)
        assert rewards.shape == (N,)
        masks = env.action_masks()

    env.close()
    print(f"  OK: 50 step SB3 uyumlu ({device})")


def test_observation_consistency():
    """GPU ve CPU observation encoding tutarlılığı."""
    print("\n[TEST 6] Observation Tutarlılığı (CPU vs GPU)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from env.pieces_gpu import PieceRegistry, PIECE_NAMES
    from rl.observation_gpu import encode_observation_batch
    from rl.observation import encode_observation

    reg = PieceRegistry.get(device)

    board_np = np.zeros((8, 8), dtype=np.float32)
    board_np[0, 0] = 1
    board_np[3, 5] = 1

    pieces_names = ["tek", "yatay_2", "kare_2x2"]
    pieces_arrays = []
    for name in pieces_names:
        idx = PIECE_NAMES.index(name)
        arr = reg.tensors_5x5[idx].cpu().numpy()
        pieces_arrays.append(arr)

    obs_dict = {
        "board": board_np,
        "pieces": pieces_arrays,
        "pieces_remaining": [True, True, True],
        "blocks_remaining": 3,
        "score": 0,
        "turn": 1,
    }
    cpu_obs = encode_observation(obs_dict)

    board_gpu = torch.tensor(board_np, device=device).unsqueeze(0)
    pieces_gpu = torch.stack([
        reg.tensors_5x5[PIECE_NAMES.index(n)] for n in pieces_names
    ]).unsqueeze(0)
    remaining_gpu = torch.ones(1, 3, dtype=torch.bool, device=device)

    gpu_obs = encode_observation_batch(board_gpu, pieces_gpu, remaining_gpu)
    gpu_obs_np = gpu_obs[0].cpu().numpy()

    max_diff = np.abs(cpu_obs - gpu_obs_np).max()
    assert max_diff < 1e-5, f"CPU vs GPU obs farkı: {max_diff}"

    print(f"  OK: CPU ve GPU obs tutarlı (max fark: {max_diff:.2e})")


def run_all_tests():
    """Tüm testleri çalıştır."""
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n{'='*60}")
    print(f"  GPU Ortam Doğrulama Testleri ({device_name})")
    print(f"{'='*60}")

    tests = [
        test_pieces_registry,
        test_board_operations,
        test_valid_mask,
        test_batch_game_env,
        test_gpu_vec_env,
        test_observation_consistency,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Sonuç: {passed}/{passed + failed} test geçti")
    if failed == 0:
        print(f"  TUM TESTLER BASARILI [OK]")
    else:
        print(f"  {failed} TEST BASARISIZ [FAIL]")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
