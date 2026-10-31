"""Einstellungen, die für alle Nutzer gelten (z. B. das Embedding-Modell der gemeinsamen Wissensbasis).

Gespeichert in memory/app_settings.json; Einstellungen pro Nutzer liegen in memory/user_settings.json.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from . import config

_lock = threading.Lock()


def _file():
    return config.MEMORY_DIR / "app_settings.json"


def _read() -> dict:
    try:
        return json.loads(_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get(key: str, default: Any = None) -> Any:
    return _read().get(key, default)


def put(key: str, value: Any) -> None:
    with _lock:
        data = _read()
        data[key] = value
        _file().parent.mkdir(parents=True, exist_ok=True)
        _file().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
