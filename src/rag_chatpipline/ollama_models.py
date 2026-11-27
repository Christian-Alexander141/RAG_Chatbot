"""Lokal installierte Ollama-Modelle und was sie können (completion, tools, embedding, ...)."""
from __future__ import annotations

import json
import logging
import urllib.request

from . import config

logger = logging.getLogger(__name__)


def _ollama(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(config.OLLAMA_BASE_URL + path, data=data), timeout=5) as r:
        return json.load(r)


def _models(cloud: bool) -> dict[str, list[str]]:
    try:
        names = [m["name"] for m in _ollama("/api/tags")["models"] if bool(m.get("remote_host")) == cloud]
    except Exception:
        logger.warning("Ollama-Modelle konnten nicht abgefragt werden", exc_info=True)
        return {}
    models = {}
    for name in names:
        try:
            models[name] = _ollama("/api/show", {"model": name}).get("capabilities", ["completion"])
        except Exception as exc:  # z. B. ein Cloud-Modell, das es nicht mehr gibt (HTTP 410): nur dieses weglassen
            logger.warning("Ollama-Modell %s nicht abfragbar, wird nicht angeboten: %s", name, exc)
    return models


def local_models() -> dict[str, list[str]]:
    """{Modellname: Fähigkeiten} aller lokal installierten Modelle. Leer, wenn Ollama nicht erreichbar ist.

    Cloud-Modelle fehlen absichtlich, sie würden Daten an fremde Server schicken.
    """
    return _models(cloud=False)


def cloud_models() -> dict[str, list[str]]:
    """{Modellname: Fähigkeiten} der in Ollama eingetragenen Cloud-Modelle (laufen auf fremden Servern)."""
    return _models(cloud=True)


def with_capability(capability: str) -> list[str]:
    return sorted(name for name, capabilities in local_models().items() if capability in capabilities)
