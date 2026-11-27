"""Welche Modelle Chat, Agenten, Bild- und Videogenerierung nutzen, und welche zur Auswahl stehen."""
from __future__ import annotations

from dataclasses import dataclass

from rag_chatpipline import config, ollama_models

from . import image_backend, video_backend


@dataclass(frozen=True)
class ModelChoice:
    chat: str  # formuliert die Antworten
    agent: str  # Werkzeug-Agent, braucht Tool-Calling
    image_prompt: str  # Bild- und Video-Agent, schreibt die Prompts für FLUX und das Videomodell
    image: str  # Bildmodell, Ordnername unter models/
    video: str = ""  # Videomodell, Ordnername unter models/video/ (leer = Standard)

    @classmethod
    def defaults(cls) -> ModelChoice:
        return cls(config.CHAT_MODEL, config.AGENT_MODEL, config.IMAGE_PROMPT_MODEL, config.FLUX_MODEL_DIR.name,
                   config.VIDEO_MODEL_DIR.name)


def language_models(include_cloud: bool = False) -> dict[str, list[str]]:
    """Ollama-Modelle: {"chat": Sprachmodelle, "tools": nur mit Tool-Calling, "embedding": Embedding-Modelle}.

    Nur lokale Modelle, mit include_cloud (für Admins) auch Cloud-Modelle als Sprachmodelle. Embedding-Modelle sind
    immer lokal, sonst gingen alle Dokumente der Wissensbasis an fremde Server. Ohne Ollama sind alle Listen leer.
    """
    local = ollama_models.local_models()
    llms = {**local, **(ollama_models.cloud_models() if include_cloud else {})}
    return {
        "chat": sorted(n for n, caps in llms.items() if "completion" in caps),
        "tools": sorted(n for n, caps in llms.items() if "completion" in caps and "tools" in caps),
        "embedding": sorted(n for n, caps in local.items() if "embedding" in caps),
    }


def image_models() -> list[str]:
    return image_backend.available_image_models()


def video_models() -> list[str]:
    return video_backend.available_video_models()
