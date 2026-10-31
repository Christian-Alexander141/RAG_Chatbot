"""Welche Modelle Chat, Agenten und Bildgenerierung nutzen, und welche zur Auswahl stehen."""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass

from rag_chatpipline import config

from . import image_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelChoice:
    chat: str  # formuliert die Antworten
    agent: str  # Werkzeug-Agent, braucht Tool-Calling
    image_prompt: str  # Bild-Agent, schreibt die FLUX-Prompts
    image: str  # Bildmodell, Ordnername unter models/

    @classmethod
    def defaults(cls) -> ModelChoice:
        return cls(config.CHAT_MODEL, config.AGENT_MODEL, config.IMAGE_PROMPT_MODEL, config.FLUX_MODEL_DIR.name)


def _ollama(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(config.OLLAMA_BASE_URL + path, data=data), timeout=5) as r:
        return json.load(r)


def language_models() -> dict[str, list[str]]:
    """Lokal installierte Ollama-Sprachmodelle: {"chat": alle, "tools": nur mit Tool-Calling}.

    Cloud-Modelle fehlen absichtlich, sie würden Daten an fremde Server schicken. Embedding-Modelle auch.
    Ist Ollama nicht erreichbar, sind beide Listen leer.
    """
    chat, tools = [], []
    try:
        for model in _ollama("/api/tags")["models"]:
            if model.get("remote_host"):
                continue
            name = model["name"]
            capabilities = _ollama("/api/show", {"model": name}).get("capabilities", ["completion"])
            if "completion" in capabilities:
                chat.append(name)
                if "tools" in capabilities:
                    tools.append(name)
    except Exception:
        logger.warning("Ollama-Modelle konnten nicht abgefragt werden", exc_info=True)
    return {"chat": sorted(chat), "tools": sorted(tools)}


def image_models() -> list[str]:
    return image_backend.available_image_models()
