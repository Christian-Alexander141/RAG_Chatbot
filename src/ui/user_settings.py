"""Gewählte Modelle pro Nutzer, gespeichert in memory/user_settings.json."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, fields

from Agents.models import ModelChoice
from rag_chatpipline import config

SETTINGS_FILE = config.MEMORY_DIR / "user_settings.json"
FIELDS = [f.name for f in fields(ModelChoice)]
_lock = threading.Lock()


def _read_all() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load(user: str) -> dict[str, str]:
    saved = _read_all().get(user, {})
    return {k: v for k, v in saved.items() if k in FIELDS and isinstance(v, str)}


def save(user: str, choice: ModelChoice) -> None:
    with _lock:
        data = _read_all()
        data[user] = asdict(choice)
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve(saved: dict[str, str], available: dict[str, list[str]]) -> ModelChoice:
    """Gespeicherte Wahl, sonst Standard. Ist ein Modell nicht mehr vorhanden, das erste verfügbare."""
    defaults = ModelChoice.defaults()
    values = {}
    for name in FIELDS:
        options = available.get(name) or []
        candidates = [saved.get(name), getattr(defaults, name)]
        values[name] = next((c for c in candidates if c and (not options or c in options)), options[0] if options else getattr(defaults, name))
    return ModelChoice(**values)
