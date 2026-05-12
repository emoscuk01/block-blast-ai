"""SB3-Uyumlu GPU VecEnv Wrapper.

GPU BatchGameEnv'i Stable-Baselines3'ün VecEnv API'sine sarar.
MaskablePPO ile doğrudan kullanılabilir.

GPU'da hesaplayıp numpy'a çevirerek SB3'e verir.
Ortam hesaplamaları GPU'da olduğu için devasa hız artışı sağlar;
sadece obs/action/mask transferi (küçük) CPU↔GPU overhead oluşturur.

Kullanım:
    from rl.gpu_vec_env import GpuVecEnv
    env = GpuVecEnv(n_envs=2048, device=torch.device("cuda"))
    model = MaskablePPO("MlpPolicy", env=env, ...)
    model.learn(total_timesteps=50_000_000)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from env.game_env_gpu import BatchGameEnv
from rl.observation_gpu import OBS_SIZE
from rl.action_mapper import TOTAL_ACTIONS


class GpuVecEnv(VecEnv):
    """GPU-vectorized Block Blast ortamı — SB3 VecEnv API uyumlu.

    Tüm ortam mantığı GPU'da çalışır. SB3'e numpy array'ler döndürülür.

    MaskablePPO action_masks() metodunu destekler.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_envs: int,
        device: torch.device | str = "cuda",
    ) -> None:
        self.gpu_device = torch.device(device) if isinstance(device, str) else device
        self.batch_env = BatchGameEnv(n_envs, self.gpu_device)

        # SB3 VecEnv gereksinimleri
        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32,
        )
        action_space = spaces.Discrete(TOTAL_ACTIONS)

        super().__init__(
            num_envs=n_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Async step için geçici değişkenler
        self._actions: Optional[torch.LongTensor] = None
        self._cached_masks: Optional[np.ndarray] = None
        self._terminal_obs: Optional[np.ndarray] = None

        # İlk reset
        self._initial_reset_done = False

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Tüm ortamları sıfırla, observation döndür."""
        obs_gpu, masks_gpu = self.batch_env.reset_all()
        self._cached_masks = masks_gpu.cpu().numpy()
        self._initial_reset_done = True
        return obs_gpu.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """Asenkron step: aksiyonları kaydet."""
        self._actions = torch.tensor(
            actions, dtype=torch.long, device=self.gpu_device,
        )

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step sonuclarini dondur.

        Returns:
            (obs, rewards, dones, infos) — hepsi numpy
        """
        assert self._actions is not None, "step_async cagrilmadan step_wait cagirildi"

        obs_gpu, rewards_gpu, dones_gpu, gpu_infos = self.batch_env.step(self._actions)

        # GPU -> CPU transfer (kucuk boyut, minimal overhead)
        obs = obs_gpu.cpu().numpy()
        rewards = rewards_gpu.cpu().numpy()
        dones = dones_gpu.cpu().numpy()

        # Action mask'i cache'le
        action_masks = gpu_infos["action_masks"]
        self._cached_masks = action_masks.cpu().numpy()

        # Terminal obs'lari kaydet
        terminal_obs_gpu = gpu_infos.get("terminal_observation")
        if terminal_obs_gpu is not None:
            self._terminal_obs = terminal_obs_gpu.cpu().numpy()

        # SB3 info dict'leri -- LAZY: sadece biten env'lere detay ekle
        terminal_dones = gpu_infos["terminal_dones"].cpu().numpy()

        # Hizli yol: bos dict listesi + sadece terminal env'lere doldur
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        terminal_indices = np.where(terminal_dones)[0]
        if len(terminal_indices) > 0:
            scores = gpu_infos["scores"].cpu().numpy()
            turns = gpu_infos["turns"].cpu().numpy()

            for i in terminal_indices:
                infos[i] = {
                    "score": int(scores[i]),
                    "turn": int(turns[i]),
                    "episode": {
                        "r": float(rewards[i]),
                        "l": int(turns[i]),
                        "t": 0.0,
                    },
                }
                if self._terminal_obs is not None:
                    infos[i]["terminal_observation"] = self._terminal_obs[i]

        self._actions = None
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Temizlik."""
        pass  # GPU tensörler garbage-collected olur

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed — GPU ortamda global torch seed kullanılır."""
        if seed is not None:
            torch.manual_seed(seed)
        return [seed] * self.num_envs

    def env_is_wrapped(self, wrapper_class: type, indices: Sequence[int] | None = None) -> list[bool]:
        """Wrapper kontrolü — bu env hiçbir wrapper ile sarılmamış."""
        if indices is None:
            return [False] * self.num_envs
        return [False] * len(indices)

    def env_method(self, method_name: str, *args: Any, indices: Sequence[int] | None = None, **kwargs: Any) -> list:
        """Env metodu çağrısı — action_masks desteklenir."""
        if method_name == "action_masks":
            masks = self.action_masks()  # [N, 192] numpy
            if indices is None:
                return [masks[i] for i in range(self.num_envs)]
            return [masks[i] for i in indices]
        raise NotImplementedError(
            f"GpuVecEnv.env_method('{method_name}') desteklenmiyor. "
            "GPU ortam bireysel ortam metodlarını desteklemez."
        )

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None) -> list:
        """Attribute erişimi."""
        if attr_name == "action_masks":
            return [self.action_masks]
        raise AttributeError(f"GpuVecEnv.{attr_name} bulunamadı")

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None) -> None:
        """Attribute ayarlama."""
        raise NotImplementedError("GpuVecEnv.set_attr desteklenmiyor")

    # ------------------------------------------------------------------
    # MaskablePPO Uyumluluğu
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """MaskablePPO için aksiyon mask'ı: [N, 192] bool numpy.

        SB3 MaskablePPO bu metodu her step'te çağırır.
        Mask, son step veya reset'ten cache'lenmiş olarak döner.
        """
        if self._cached_masks is None:
            # İlk çağrı — henüz step/reset yapılmamış
            if not self._initial_reset_done:
                self.reset()
            masks_gpu = self.batch_env.board.compute_valid_mask(
                self.batch_env.piece_ids
            )
            self._cached_masks = masks_gpu.cpu().numpy()
        return self._cached_masks

    # ------------------------------------------------------------------
    # Render (opsiyonel, debug için)
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[str]:
        """İlk ortamın tahtasını ASCII olarak yazdır."""
        grid = self.batch_env.board.grids[0].cpu().numpy()
        lines = []
        header = "  " + " ".join(str(c) for c in range(8))
        lines.append(header)
        for r in range(8):
            row_str = " ".join("#" if grid[r, c] > 0 else "." for c in range(8))
            lines.append(f"{r} {row_str}")
        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output

    def get_images(self) -> Sequence[np.ndarray]:
        """Render görüntüleri — desteklenmiyor."""
        raise NotImplementedError
