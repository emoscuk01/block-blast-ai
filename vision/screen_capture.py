"""ADB üzerinden telefon ekran görüntüsü alma modülü."""

from __future__ import annotations

import io
import logging
import subprocess
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """ADB screencap komutuyla ekran görüntüsü alan sınıf."""

    def __init__(self, device_id: Optional[str] = None) -> None:
        self.device_id = device_id

    def _adb_cmd(self, args: list[str]) -> list[str]:
        """device_id varsa '-s device_id' ekleyerek ADB komut listesi oluşturur."""
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        return cmd

    def capture(self) -> Image.Image:
        """
        ADB screencap komutuyla ekran görüntüsü alır.
        PIL Image (RGB) döndürür.
        Gecikme ~200-400ms.
        """
        cmd = self._adb_cmd(["exec-out", "screencap", "-p"])
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ADB screencap başarısız (kod {result.returncode}): "
                    f"{result.stderr.decode(errors='replace')}"
                )
            img = Image.open(io.BytesIO(result.stdout))
            return img.convert("RGB")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ADB screencap zaman aşımına uğradı (5s)")
        except Exception as e:
            logger.error("Ekran görüntüsü alınamadı: %s", e)
            raise

    def capture_region(self, x: int, y: int, w: int, h: int) -> Image.Image:
        """Tam ekranı alır, belirtilen bölgeyi kırpıp döndürür."""
        full = self.capture()
        return full.crop((x, y, x + w, y + h))

    def get_screen_size(self) -> tuple[int, int]:
        """adb shell wm size çıktısından (genişlik, yükseklik) döndürür."""
        cmd = self._adb_cmd(["shell", "wm", "size"])
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=5, text=True
            )
            output = result.stdout.strip()
            # "Physical size: 1080x2340" formatını parse et
            size_str = output.split(":")[-1].strip()
            w, h = size_str.split("x")
            return (int(w), int(h))
        except Exception as e:
            logger.error("Ekran boyutu alınamadı: %s", e)
            raise

    def test_connection(self) -> bool:
        """ADB bağlantısını kontrol eder."""
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, timeout=5, text=True
            )
            lines = result.stdout.strip().split("\n")
            devices = [
                l for l in lines[1:] if l.strip() and "device" in l
            ]
            if not devices:
                logger.warning("Hiçbir ADB cihazı bulunamadı. USB hata ayıklama açık mı?")
                return False
            if self.device_id:
                found = any(self.device_id in d for d in devices)
                if not found:
                    logger.warning("Belirtilen cihaz bulunamadı: %s", self.device_id)
                    return False
            logger.info("ADB bağlantısı başarılı. Cihaz sayısı: %d", len(devices))
            return True
        except Exception as e:
            logger.error("ADB bağlantı testi başarısız: %s", e)
            return False
