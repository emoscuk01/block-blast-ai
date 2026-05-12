"""GPU Batched Board Operations.

N adet 8x8 tahtayi tek torch.Tensor olarak yonetir.
Tum operasyonlar (yerlestirme, satir/sutun silme, gecerlilik kontrolu)
vektorize GPU kernel'lari ile calisir — Python for dongusu yoktur.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from env.pieces_gpu import PieceRegistry, MAX_CELLS

BOARD_SIZE: int = 8


class BatchBoard:
    """N adet 8x8 tahtayi GPU tensoru olarak yonetir."""

    def __init__(self, n_envs: int, device: torch.device) -> None:
        self.n_envs = n_envs
        self.device = device
        self.reg = PieceRegistry.get(device)
        self.grids = torch.zeros(
            n_envs, BOARD_SIZE, BOARD_SIZE,
            dtype=torch.float32, device=device,
        )

        # Pre-compute position grid (sabit, bir kere olustur)
        self._pos_r = torch.arange(BOARD_SIZE, device=device).repeat_interleave(BOARD_SIZE)  # [64]
        self._pos_c = torch.arange(BOARD_SIZE, device=device).repeat(BOARD_SIZE)              # [64]
        self._cell_range = torch.arange(MAX_CELLS, device=device)  # [9]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_mask: torch.BoolTensor | None = None) -> None:
        """Belirtilen ortamlarin tahtalarini sifirla."""
        if env_mask is None:
            self.grids.zero_()
        else:
            self.grids[env_mask] = 0.0

    # ------------------------------------------------------------------
    # Yerlestirme
    # ------------------------------------------------------------------

    def place_batch(
        self,
        piece_ids: torch.LongTensor,
        rows: torch.LongTensor,
        cols: torch.LongTensor,
        env_mask: torch.BoolTensor | None = None,
    ) -> torch.LongTensor:
        """Parcalari tahtaya yerlestir ve dolu satir/sutunlari temizle.

        Args:
            piece_ids: [K] — her ortam icin parca tipi ID'si
            rows:      [K] — yerlestirme satiri
            cols:      [K] — yerlestirme sutunu
            env_mask:  [N] bool — sadece True olan ortamlara uygula (None -> hepsi)

        Returns:
            lines_cleared: [K] — her ortamda silinen satir+sutun sayisi
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

        # --- Parca hucrelerini tahtaya yaz ---
        offsets = reg.offsets[piece_ids]    # [K, MAX_CELLS, 2]
        n_cells = reg.n_cells[piece_ids]   # [K]

        # Mutlak pozisyonlar
        abs_rows = rows.unsqueeze(1) + offsets[:, :, 0]  # [K, MAX_CELLS]
        abs_cols = cols.unsqueeze(1) + offsets[:, :, 1]   # [K, MAX_CELLS]

        # Gecerli hucre mask'i (padding degil)
        cell_range = self._cell_range.unsqueeze(0)
        valid_cell = cell_range < n_cells.unsqueeze(1)  # [K, MAX_CELLS]

        # Sinir kontrolu
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

        # --- Dolu satir/sutun silme ---
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
    # Gecerli Aksiyon Mask'i — Tamamen Vektorize (Python loop yok)
    # ------------------------------------------------------------------

    def compute_valid_mask(
        self,
        piece_ids: torch.LongTensor,
    ) -> torch.BoolTensor:
        """Tum ortamlar icin gecerli aksiyon mask'i hesapla.

        Tamamen vektorize: Python loop yok, tek GPU kernel batch'i.

        Args:
            piece_ids: [N, 3] — her ortamin 3 parca slot'undaki parca tipi
                       -1 = slot kullanildi (gecersiz)

        Returns:
            mask: [N, 192] bool — gecerli aksiyonlar True
        """
        N = self.n_envs
        reg = self.reg
        device = self.device
        boards = self.grids  # [N, 8, 8]

        pos_r = self._pos_r  # [64]
        pos_c = self._pos_c  # [64]
        cell_range = self._cell_range  # [9]

        mask = torch.zeros(N, 192, dtype=torch.bool, device=device)

        for slot in range(3):
            slot_offset = slot * 64
            pids = piece_ids[:, slot]  # [N]
            available = pids >= 0      # [N]

            if not available.any():
                continue

            # -1 pid'leri 0'a clamp (sonra available mask ile filtrelenecek)
            safe_pids = pids.clamp(min=0)  # [N]

            # Her ortamin parcasinin hucre offset'leri
            off_r = reg.offsets[safe_pids, :, 0]   # [N, 9]
            off_c = reg.offsets[safe_pids, :, 1]   # [N, 9]
            n_cells = reg.n_cells[safe_pids]        # [N]
            piece_h = reg.sizes[safe_pids, 0]       # [N]
            piece_w = reg.sizes[safe_pids, 1]       # [N]

            # Mutlak hucre pozisyonlari: [N, 64, 9]
            # abs_r[n, p, c] = pos_r[p] + off_r[n, c]
            abs_r = pos_r[None, :, None] + off_r[:, None, :]  # [N, 64, 9]
            abs_c = pos_c[None, :, None] + off_c[:, None, :]  # [N, 64, 9]

            # Hangi hucreler gercek (padding degil)
            valid_cell = cell_range[None, None, :] < n_cells[:, None, None]  # [N, 1, 9] -> broadcast

            # Sinir kontrolu
            in_bounds = (
                (abs_r >= 0) & (abs_r < BOARD_SIZE) &
                (abs_c >= 0) & (abs_c < BOARD_SIZE)
            )  # [N, 64, 9]

            # Guvenli indeksleme icin clamp
            safe_r = abs_r.clamp(0, BOARD_SIZE - 1)
            safe_c = abs_c.clamp(0, BOARD_SIZE - 1)

            # Tahtadan hucre degerlerini oku: boards[n, r, c]
            env_idx = torch.arange(N, device=device)[:, None, None].expand_as(safe_r)
            cell_values = boards[env_idx, safe_r, safe_c]  # [N, 64, 9]
            cell_empty = (cell_values == 0)

            # Parca tahtaya sigiyor mu kontrolu
            pos_fits = (
                (pos_r[None, :] + piece_h[:, None] <= BOARD_SIZE) &
                (pos_c[None, :] + piece_w[:, None] <= BOARD_SIZE)
            )  # [N, 64]

            # Bir pozisyon gecerli eger:
            # 1. Tum gercek hucreler sinir icinde VE bos
            # 2. Parca tahtaya sigiyor
            # 3. Slot kullanilabilir
            cell_ok = (~valid_cell) | (in_bounds & cell_empty)  # padding hucreleri OK
            all_cells_ok = cell_ok.all(dim=2)  # [N, 64]

            position_valid = all_cells_ok & pos_fits & available[:, None]

            mask[:, slot_offset:slot_offset + 64] = position_valid

        return mask

    # ------------------------------------------------------------------
    # Heuristik Metrikleri (reward hesabi icin)
    # ------------------------------------------------------------------

    def aggregate_height_batch(self) -> torch.Tensor:
        """Her ortam icin aggregate height: [N]."""
        filled = (self.grids > 0).float()
        has_any = filled.any(dim=1)
        first_filled_row = filled.argmax(dim=1)
        heights = (BOARD_SIZE - first_filled_row) * has_any.long()
        return heights.sum(dim=1).float()

    def bumpiness_batch(self) -> torch.Tensor:
        """Her ortam icin bumpiness: [N]."""
        filled = (self.grids > 0).float()
        has_any = filled.any(dim=1)
        first_filled_row = filled.argmax(dim=1)
        heights = (BOARD_SIZE - first_filled_row) * has_any.long()
        diffs = (heights[:, 1:] - heights[:, :-1]).abs()
        return diffs.sum(dim=1).float()

    def holes_batch(self) -> torch.Tensor:
        """Her ortam icin delik sayisi: [N]."""
        filled = (self.grids > 0)
        cum_filled, _ = filled.cummax(dim=1)
        holes = cum_filled & (~filled)
        return holes.sum(dim=(1, 2)).float()
