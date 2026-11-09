"""Configuración persistente de BlinkEye (se guarda en config.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Settings:
    camera_index: int = 0
    timeout_seconds: float = 5.0
    alarm_enabled: bool = True
    show_landmarks: bool = True
    alarm_message: str = "Parpadea, parpadea, parpadea"
    sensitivity: int = 50       # 0-100: mayor = detecta parpadeos más sutiles
    brightness: int = 0         # 0-100: brillo virtual para poca luz

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        target = path or CONFIG_PATH
        if target.is_file():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                valid = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in valid})
            except (OSError, ValueError, TypeError):
                pass
        return cls()

    def save(self, path: Path | None = None) -> None:
        target = path or CONFIG_PATH
        try:
            target.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # No es crítico: si no se puede guardar, la app sigue funcionando.
            pass
