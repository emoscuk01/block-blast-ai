"""GPU Batched Game Environment.

N paralel Block Blast oyununu tek GPU tensörü olarak yönetir.
Tüm oyun mantığı (parça yerleştirme, satır silme, skor, game over)
vektörize GPU operasyonlarıyla çalışır.

Kullanım:
    env = BatchGameEnv(n_envs=2048, device=torch.device("cuda"))
    obs, masks = env.reset_all()
    obs, rewards, dones, infos = env.step(actions)
"""

from __future__ import annotations

import torch

from env.board_gpu import BatchBoard, BOARD_SIZE
from env.pieces_gpu import PieceRegistry, NUM_PIECE_TYPES, get_random_piece_ids
from rl.observation_gpu import encode_observation_batch, OBS_SIZE
from utils.metrics_gpu import compute_reward_batch


class BatchGameEnv:
    """N paralel Block Blast oyununu GPU'da yönetir."""

    def __init__(self, n_envs: int, device: torch.device) -> None:
        self.n_envs = n_envs
        self.device = device

        # Parça registry'sini oluştur
        self.reg = PieceRegistry.get(device)

        # Board
        self.board = BatchBoard(n_envs, device)

        # Parçalar
        self.piece_ids = torch.full(
            (n_envs, 3), -1, dtype=torch.long, device=device,
        )
        self.pieces_5x5 = torch.zeros(
            n_envs, 3, 5, 5, dtype=torch.float32, device=device,
        )
        self.pieces_placed = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Skor ve durum
        self.scores = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.turns = torch.ones(n_envs, dtype=torch.long, device=device)
        self.dones = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Cache: action decode tablosu (sabit)
        _all = torch.arange(192, device=device)
        self._act_slots = _all // 64
        self._act_rows = (_all % 64) // 8
        self._act_cols = _all % 8

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Tüm ortamları sıfırla. Returns: (obs [N,142], masks [N,192])."""
        self.board.reset()
        self.scores.zero_()
        self.turns.fill_(1)
        self.pieces_placed.zero_()
        self.dones.zero_()
        self._deal_new_pieces()

        obs = self._get_obs()
        masks = self.board.compute_valid_mask(self.piece_ids)
        return obs, masks

    def reset_envs(self, env_mask: torch.BoolTensor) -> None:
        """Belirtilen ortamları sıfırla (auto-reset için)."""
        if not env_mask.any():
            return
        self.board.reset(env_mask)
        self.scores[env_mask] = 0.0
        self.turns[env_mask] = 1
        self.pieces_placed[env_mask] = 0
        self.dones[env_mask] = False
        self._deal_new_pieces(env_mask)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, dict]:
        """Batched step: N ortamda eşzamanlı hamle.

        Returns: (obs [N,142], rewards [N], dones [N], infos dict)
        """
        N = self.n_envs
        device = self.device
        arange_N = torch.arange(N, device=device)

        # Action decode
        piece_slots = self._act_slots[actions]
        rows = self._act_rows[actions]
        cols = self._act_cols[actions]

        # Aktif ortamlar
        active = ~self.dones

        # Seçilen slot'un parça tipi
        piece_ids_for_action = self.piece_ids[arange_N, piece_slots]

        # --- Parça yerleştir ---
        lines_cleared = torch.zeros(N, dtype=torch.long, device=device)
        if active.any():
            lines = self.board.place_batch(
                piece_ids=piece_ids_for_action[active],
                rows=rows[active],
                cols=cols[active],
                env_mask=active,
            )
            lines_cleared[active] = lines

        # --- Skor güncelle ---
        line_score = lines_cleared.float() * 10.0
        multi_bonus = (lines_cleared.float() - 1.0).clamp(min=0) * 5.0
        self.scores += (line_score + multi_bonus) * active.float()

        # --- Parça slot'unu kapat ---
        # Sadece aktif ortamlardaki slot'ları -1 yap
        slot_update = self.piece_ids[arange_N, piece_slots].clone()
        slot_update[active] = -1
        self.piece_ids[arange_N, piece_slots] = slot_update

        # Pieces 5x5 güncelle
        p5_update = self.pieces_5x5[arange_N, piece_slots].clone()
        p5_update[active] = 0.0
        self.pieces_5x5[arange_N, piece_slots] = p5_update

        self.pieces_placed += active.long()

        # --- Yeni tur kontrolü ---
        need_new_turn = active & (self.pieces_placed >= 3)
        if need_new_turn.any():
            self.turns[need_new_turn] += 1
            self.pieces_placed[need_new_turn] = 0
            self._deal_new_pieces(need_new_turn)

        # --- Game Over kontrolü ---
        action_mask = self.board.compute_valid_mask(self.piece_ids)
        has_valid = action_mask.any(dim=1)
        new_dones = active & (~has_valid)
        self.dones = self.dones | new_dones

        # --- Reward hesapla ---
        rewards = compute_reward_batch(
            boards_after=self.board.grids,
            lines_cleared=lines_cleared,
            dones=new_dones,
        )
        rewards = rewards * active.float()

        # --- Terminal obs ---
        terminal_obs = self._get_obs()

        # --- Auto-reset ---
        if new_dones.any():
            self.reset_envs(new_dones)
            new_mask = self.board.compute_valid_mask(self.piece_ids)
            action_mask[new_dones] = new_mask[new_dones]

        obs = self._get_obs()

        infos = {
            "scores": self.scores.clone(),
            "turns": self.turns.clone(),
            "lines_cleared": lines_cleared,
            "terminal_dones": new_dones,
            "action_masks": action_mask,
            "terminal_observation": terminal_obs,
        }

        return obs, rewards, new_dones, infos

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> torch.Tensor:
        """[N, 142] observation — GPU'da."""
        pieces_remaining = (self.piece_ids >= 0)
        return encode_observation_batch(
            boards=self.board.grids,
            pieces_5x5=self.pieces_5x5,
            pieces_remaining=pieces_remaining,
        )

    # ------------------------------------------------------------------
    # Parça Yönetimi
    # ------------------------------------------------------------------

    def _deal_new_pieces(self, env_mask: torch.BoolTensor | None = None) -> None:
        """Rastgele 3 parça dağıt."""
        reg = self.reg
        if env_mask is None:
            K = self.n_envs
            new_ids = get_random_piece_ids(K, 3, self.device)
            self.piece_ids[:] = new_ids
            for slot in range(3):
                self.pieces_5x5[:, slot] = reg.tensors_5x5[new_ids[:, slot]]
        else:
            K = env_mask.sum().item()
            if K == 0:
                return
            new_ids = get_random_piece_ids(K, 3, self.device)
            self.piece_ids[env_mask] = new_ids
            for slot in range(3):
                self.pieces_5x5[env_mask, slot] = reg.tensors_5x5[new_ids[:, slot]]
