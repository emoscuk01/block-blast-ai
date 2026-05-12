"""GPU Parça Tensör Registry.

Tüm 28 blok şeklini PyTorch tensörleri olarak ön-yükler.
board_gpu ve game_env_gpu tarafından kullanılır.

Kullanım:
    from env.pieces_gpu import PieceRegistry
    reg = PieceRegistry.get(device)
    kernel = reg.kernels[piece_id]
"""

from __future__ import annotations

from typing import Optional

import torch

from env.pieces import PIECES, _PIECE_NAMES

# Parça isimleri sabit sıralı liste (indeks = piece_id)
PIECE_NAMES: list[str] = list(_PIECE_NAMES)
NUM_PIECE_TYPES: int = len(PIECE_NAMES)
MAX_CELLS: int = 9  # kare_3x3 = 9 hücre (en fazla)


class PieceRegistry:
    """Tüm parça tensörlerini tutan singleton registry.

    Attributes:
        tensors_5x5 : [28, 5, 5]  — 5×5 padded (observation için)
        sizes        : [28, 2]     — (rows, cols) gerçek boyut
        n_cells      : [28]        — dolu hücre sayısı
        offsets      : [28, 9, 2]  — (dr, dc) hücre offset'leri, -1 ile padded
        kernels      : list[Tensor] — conv2d kernel'ları (farklı boyutlar)
    """

    _instance: Optional["PieceRegistry"] = None
    _device: Optional[torch.device] = None

    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.tensors_5x5 = torch.zeros(NUM_PIECE_TYPES, 5, 5, dtype=torch.float32)
        self.sizes = torch.zeros(NUM_PIECE_TYPES, 2, dtype=torch.long)
        self.n_cells = torch.zeros(NUM_PIECE_TYPES, dtype=torch.long)
        self.offsets = torch.full((NUM_PIECE_TYPES, MAX_CELLS, 2), -1, dtype=torch.long)
        self.kernels: list[torch.Tensor] = []

        self._build()

    def _build(self) -> None:
        """Tüm parça tensörlerini oluştur."""
        for pid, name in enumerate(PIECE_NAMES):
            cells = PIECES[name]
            h = len(cells)
            w = len(cells[0])
            self.sizes[pid, 0] = h
            self.sizes[pid, 1] = w

            cell_idx = 0
            for r in range(h):
                for c in range(w):
                    if cells[r][c] == 1:
                        self.tensors_5x5[pid, r, c] = 1.0
                        self.offsets[pid, cell_idx, 0] = r
                        self.offsets[pid, cell_idx, 1] = c
                        cell_idx += 1
            self.n_cells[pid] = cell_idx

            # Conv2d kernel — gerçek boyutunda (padded değil)
            kernel = torch.zeros(1, 1, h, w, dtype=torch.float32)
            for r in range(h):
                for c in range(w):
                    kernel[0, 0, r, c] = float(cells[r][c])
            self.kernels.append(kernel.to(self.device))

        self.tensors_5x5 = self.tensors_5x5.to(self.device)
        self.sizes = self.sizes.to(self.device)
        self.n_cells = self.n_cells.to(self.device)
        self.offsets = self.offsets.to(self.device)

    @classmethod
    def get(cls, device: torch.device) -> "PieceRegistry":
        """Singleton erişim. İlk çağrıda oluşturur, sonra cache'ten döner."""
        if cls._instance is None or cls._device != device:
            cls._instance = cls(device)
            cls._device = device
        return cls._instance


def get_random_piece_ids(
    n_envs: int, n_pieces: int, device: torch.device
) -> torch.LongTensor:
    """GPU'da rastgele parça ID'leri üret: [n_envs, n_pieces]."""
    return torch.randint(
        0, NUM_PIECE_TYPES, (n_envs, n_pieces),
        dtype=torch.long, device=device,
    )
