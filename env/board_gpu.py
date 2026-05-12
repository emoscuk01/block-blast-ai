"""GPU Batched Board Operations.

N adet 8×8 tahtayı tek torch.Tensor olarak yönetir.
Tüm operasyonlar (yerleştirme, satır/sütun silme, geçerlilik kontrolü)
vektörize GPU kernel'ları ile çalışır — Python for döngüsü yoktur.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from env.pieces_gpu import PieceRegistry, MAX_CELLS

BOARD_SIZE: int = 8


class BatchBoard:
    """N adet 8×8 tahtayı GPU tensörü olarak yönetir."""

    def __init__(self, n_envs: int, device: torch.device) -> None:
        self.n_envs = n_envs
        self.device = device
        self.reg = PieceRegistry.get(device)
        self.grids = torch.zeros(
            n_envs, BOARD_SIZE, BOARD_SIZE,
            dtype=torch.float32, device=device,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_mask: torch.BoolTensor | None = None) -> None:
        """Belirtilen ortamların tahtalarını sıfırla."""
        if env_mask is None:
            self.grids.zero_()
        else:
            self.grids[env_mask] = 0.0

    # ------------------------------------------------------------------
    # Yerleştirme
    # ------------------------------------------------------------------

    def place_batch(
        self,
        piece_ids: torch.LongTensor,
        rows: torch.LongTensor,
        cols: torch.LongTensor,
        env_mask: torch.BoolTensor | None = None,
    ) -> torch.LongTensor:
        """Parçaları tahtaya yerleştir ve dolu satır/sütunları temizle.

        Args:
            piece_ids: [K] — her ortam için parça tipi ID'si
            rows:      [K] — yerleştirme satırı
            cols:      [K] — yerleştirme sütunu
            env_mask:  [N] bool — sadece True olan ortamlara uygula (None → hepsi)

        Returns:
            lines_cleared: [K] — her ortamda silinen satır+sütun sayısı
        """
        if env_mask is not None:
            grids = self.grids[env_mask]
            K = grids.shape[0]
        else:
            grids = self.grids
            K = self.n_envs

        if K == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        reg = self.reg

        # --- Parça hücrelerini tahtaya yaz ---
        offsets = reg.offsets[piece_ids]    # [K, MAX_CELLS, 2]
        n_cells = reg.n_cells[piece_ids]   # [K]

        # Mutlak pozisyonlar
        abs_rows = rows.unsqueeze(1) + offsets[:, :, 0]  # [K, MAX_CELLS]
        abs_cols = cols.unsqueeze(1) + offsets[:, :, 1]   # [K, MAX_CELLS]

        # Geçerli hücre mask'ı (padding değil)
        cell_range = torch.arange(MAX_CELLS, device=self.device).unsqueeze(0)
        valid_cell = cell_range < n_cells.unsqueeze(1)  # [K, MAX_CELLS]

        # Sınır kontrolü
        in_bounds = (
            (abs_rows >= 0) & (abs_rows < BOARD_SIZE) &
            (abs_cols >= 0) & (abs_cols < BOARD_SIZE)
        )
        valid = valid_cell & in_bounds

        # Scatter ile tahtaya yaz
        env_idx = torch.arange(K, device=self.device).unsqueeze(1).expand_as(abs_rows)

        safe_rows = abs_rows.clamp(0, BOARD_SIZE - 1)
        safe_cols = abs_cols.clamp(0, BOARD_SIZE - 1)

        grids[env_idx[valid], safe_rows[valid], safe_cols[valid]] = 1.0

        # --- Dolu satır/sütun silme ---
        row_full = (grids.sum(dim=2) == BOARD_SIZE)  # [K, 8]
        col_full = (grids.sum(dim=1) == BOARD_SIZE)  # [K, 8]

        row_clear = row_full.unsqueeze(2).expand_as(grids)
        col_clear = col_full.unsqueeze(1).expand_as(grids)
        clear_mask = row_clear | col_clear
        grids = grids * (~clear_mask).float()

        lines_cleared = row_full.long().sum(dim=1) + col_full.long().sum(dim=1)

        if env_mask is not None:
            self.grids[env_mask] = grids
        else:
            self.grids = grids

        return lines_cleared

    # ------------------------------------------------------------------
    # Geçerli Aksiyon Mask'ı — Conv2d ile
    # ------------------------------------------------------------------

    def compute_valid_mask(
        self,
        piece_ids: torch.LongTensor,
    ) -> torch.BoolTensor:
        """Tüm ortamlar için geçerli aksiyon mask'ı hesapla.

        Args:
            piece_ids: [N, 3] — her ortamın 3 parça slot'undaki parça tipi
                       -1 = slot kullanıldı (geçersiz)

        Returns:
            mask: [N, 192] bool — geçerli aksiyonlar True
        """
        N = self.n_envs
        reg = self.reg
        mask = torch.zeros(N, 192, dtype=torch.bool, device=self.device)
        boards = self.grids

        for slot in range(3):
            slot_offset = slot * 64
            pids = piece_ids[:, slot]
            available = pids >= 0

            if not available.any():
                continue

            unique_pids = pids[available].unique()

            for pid_val in unique_pids:
                pid = pid_val.item()
                env_sel = available & (pids == pid)
                if not env_sel.any():
                    continue

                sel_boards = boards[env_sel]
                K = sel_boards.shape[0]

                kernel = reg.kernels[pid]
                pH = reg.sizes[pid, 0].item()
                pW = reg.sizes[pid, 1].item()

                overlap = F.conv2d(
                    sel_boards.unsqueeze(1),
                    kernel,
                    padding=0,
                )
                overlap = overlap.squeeze(1)

                valid_positions = (overlap == 0)

                outH, outW = valid_positions.shape[1], valid_positions.shape[2]
                valid_8x8 = torch.zeros(K, BOARD_SIZE, BOARD_SIZE, dtype=torch.bool, device=self.device)
                valid_8x8[:, :outH, :outW] = valid_positions

                mask[env_sel, slot_offset:slot_offset + 64] = valid_8x8.reshape(K, 64)

        return mask

    # ------------------------------------------------------------------
    # Heuristik Metrikleri (reward hesabı için)
    # ------------------------------------------------------------------

    def aggregate_height_batch(self) -> torch.Tensor:
        """Her ortam için aggregate height: [N]."""
        filled = (self.grids > 0).float()
        has_any = filled.any(dim=1)
        first_filled_row = filled.argmax(dim=1)
        heights = (BOARD_SIZE - first_filled_row) * has_any.long()
        return heights.sum(dim=1).float()

    def bumpiness_batch(self) -> torch.Tensor:
        """Her ortam için bumpiness: [N]."""
        filled = (self.grids > 0).float()
        has_any = filled.any(dim=1)
        first_filled_row = filled.argmax(dim=1)
        heights = (BOARD_SIZE - first_filled_row) * has_any.long()
        diffs = (heights[:, 1:] - heights[:, :-1]).abs()
        return diffs.sum(dim=1).float()

    def holes_batch(self) -> torch.Tensor:
        """Her ortam için delik sayısı: [N]."""
        filled = (self.grids > 0)
        cum_filled, _ = filled.cummax(dim=1)
        holes = cum_filled & (~filled)
        return holes.sum(dim=(1, 2)).float()
