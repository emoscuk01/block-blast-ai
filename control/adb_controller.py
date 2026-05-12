"""ADB komutlarını Python'dan çalıştıran wrapper sınıfı.

Oyun mantığı bilmez, sadece dokunma hareketleri gerçekleştirir.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ADBController:
    """ADB üzerinden telefona dokunma ve sürükleme hareketleri gönderir."""

    SWIPE_DURATION_MS: int = 300
    TAP_DURATION_MS: int = 50

    def __init__(self, device_id: Optional[str] = None, dry_run: bool = False) -> None:
        self.device_id = device_id
        self.dry_run = dry_run

    def _adb_cmd(self, args: list[str]) -> list[str]:
        """device_id varsa '-s device_id' ekleyerek komut oluşturur."""
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        return cmd

    def _run(self, command: list[str], timeout: int = 5) -> subprocess.CompletedProcess:
        """ADB komutunu çalıştırır. Timeout ve hata yönetimi burada."""
        if self.dry_run:
            cmd_str = " ".join(command)
            logger.info("[DRY-RUN] %s", cmd_str)
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        try:
            result = subprocess.run(
                command, capture_output=True, timeout=timeout
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.warning("ADB komut hatası (kod %d): %s", result.returncode, stderr)
            return result
        except subprocess.TimeoutExpired:
            logger.error("ADB komutu zaman aşımına uğradı (%ds): %s", timeout, " ".join(command))
            raise
        except FileNotFoundError:
            logger.error("ADB bulunamadı. PATH'e eklendiğinden emin olun.")
            raise

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: Optional[int] = None
    ) -> bool:
        """Parmak sürükleme hareketi yapar."""
        dur = duration_ms or self.SWIPE_DURATION_MS
        cmd = self._adb_cmd(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur)]
        )
        try:
            result = self._run(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error("Swipe başarısız: %s", e)
            return False

    def tap(self, x: int, y: int) -> bool:
        """Tek dokunma hareketi yapar."""
        cmd = self._adb_cmd(["shell", "input", "tap", str(x), str(y)])
        try:
            result = self._run(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error("Tap başarısız: %s", e)
            return False

    def drag_piece(
        self,
        piece_center_x: int,
        piece_center_y: int,
        target_x: int,
        target_y: int,
    ) -> bool:
        """
        Parçayı sürükle-bırak ile tahtaya yerleştirir.
        1. Parçanın üzerine bas
        2. 100ms bekle
        3. Hedefe swipe
        4. 200ms bekle
        """
        if not self.tap(piece_center_x, piece_center_y):
            return False

        time.sleep(0.1)

        if not self.swipe(piece_center_x, piece_center_y, target_x, target_y):
            return False

        time.sleep(0.2)
        return True

    def is_connected(self) -> bool:
        """ADB ile bağlantıyı kontrol eder."""
        if self.dry_run:
            logger.info("[DRY-RUN] ADB bağlantı kontrolü atlandı")
            return True

        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, timeout=5, text=True
            )
            lines = result.stdout.strip().split("\n")
            devices = [l for l in lines[1:] if l.strip() and "device" in l]
            connected = len(devices) > 0
            if not connected:
                logger.warning("ADB cihaz bulunamadı")
            return connected
        except Exception as e:
            logger.error("ADB bağlantı kontrolü başarısız: %s", e)
            return False
