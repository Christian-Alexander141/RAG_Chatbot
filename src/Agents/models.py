"""Welche Modelle Chat, Agenten und Bildgenerierung nutzen, und welche zur Auswahl stehen."""
from __future__ import annotations

from dataclasses import dataclass

from rag_chatpipline import config, ollama_models

from . import image_backend


@dataclass(frozen=True)
class ModelChoice:
    chat: str  # formuliert die Antworten
    agent: str  # Werkzeug-Agent, braucht Tool-Calling
    image_prompt: str  # Bild-Agent, schreibt die FLUX-Prompts
    image: str  # Bildmodell, Ordnername unter models/

    @classmethod
    def defaults(cls) -> ModelChoice:
        return cls(config.CHAT_MODEL, config.AGENT_MODEL, config.IMAGE_PROMPT_MODEL, config.FLUX_MODEL_DIR.name)


def language_models() -> dict[str, list[str]]:
    """Lokal installierte Ollama-Modelle: {"chat": Sprachmodelle, "tools": nur mit Tool-Calling,
    "embedding": Embedding-Modelle}. Ohne Cloud-Modelle. Ist Ollama nicht erreichbar, sind alle Listen leer."""
    models = ollama_models.local_models()
    return {
        "chat": sorted(n for n, caps in models.items() if "completion" in caps),
        "tools": sorted(n for n, caps in models.items() if "completion" in caps and "tools" in caps),
        "embedding": sorted(n for n, caps in models.items() if "embedding" in caps),
    }


def image_models() -> list[str]:
    return image_backend.available_image_models()
