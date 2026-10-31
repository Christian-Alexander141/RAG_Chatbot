"""Bild-Agent: macht aus einem Bildwunsch einen englischen FLUX-Prompt und erzeugt das Bild.

Bildwünsche ("Erstelle ein Bild von ...") leitet chat_agent direkt hierher, ohne den Werkzeug-Agenten.
Das Werkzeug generate_image nutzt denselben Weg, falls der Werkzeug-Agent einen Wunsch selbst erkennt.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_chatpipline import config

from . import image_backend

logger = logging.getLogger(__name__)

# Ein Verb fürs Erzeugen und ein Bild-Wort, in beliebiger Reihenfolge ("Erstelle ein Bild ...", "Ein Foto von ... generieren").
_VERB = r"\b(erstell|generier|erzeug|zeichne|male?\b|mach|gestalte|create|generate|draw|paint|make)"
_NOUN = r"(bild(es|er|ern)?|foto(s)?|grafik(en)?|illustration(en)?|zeichnung(en)?|gemälde|logo(s)?|image(s)?|picture(s)?|photo(s)?)\b"
IMAGE_REQUEST = re.compile(rf"(?=.*{_VERB})(?=.*{_NOUN})", re.IGNORECASE | re.DOTALL)

PROMPT_WRITER = """Du schreibst Prompts für das Bildmodell FLUX.
Mache aus dem Bildwunsch des Nutzers einen einzigen englischen Prompt: Motiv, Umgebung, Stil, Licht,
Perspektive und Stimmung, höchstens 80 Wörter. Bezieht sich der Wunsch auf den bisherigen Chat
("davon", "noch eins"), nimm das Motiv von dort. Gib nur den Prompt aus, ohne Anführungszeichen oder Erklärung."""

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 500

_prompt_models: dict[str, Any] = {}


def is_image_request(text: str) -> bool:
    return bool(IMAGE_REQUEST.search(text))


def get_prompt_model(name: Optional[str] = None):
    """Modell, das die Prompts schreibt (Standard: IMAGE_PROMPT_MODEL), oder None (dann geht der Wunsch unverändert an FLUX)."""
    name = name or config.IMAGE_PROMPT_MODEL
    if name not in _prompt_models:
        from langchain_ollama import ChatOllama

        _prompt_models[name] = ChatOllama(model=name, base_url=config.OLLAMA_BASE_URL, temperature=0.7, reasoning=False)
    return _prompt_models[name]


def write_prompt(request: str, history: Iterable[BaseMessage] = (), model_name: Optional[str] = None,
                 model: Any = None) -> str:
    """Formuliert den Bildwunsch als englischen FLUX-Prompt. Klappt das nicht, bleibt der Wunsch wie er ist."""
    model = model or get_prompt_model(model_name)
    if model is None:
        return request
    recent = [m for m in history if isinstance(m, (HumanMessage, AIMessage)) and str(m.content).strip()]
    context = [type(m)(str(m.content)[:MAX_HISTORY_CHARS]) for m in recent[-MAX_HISTORY_MESSAGES:]]
    try:
        reply = model.invoke([SystemMessage(PROMPT_WRITER), *context, HumanMessage(f"Bildwunsch: {request}")])
    except Exception:
        logger.warning("Prompt für FLUX konnte nicht formuliert werden, nutze den Wunsch direkt", exc_info=True)
        return request
    prompt = re.sub(r"<think>.*?</think>", "", str(reply.content), flags=re.DOTALL)
    prompt = re.sub(r"^\s*prompt:\s*", "", prompt.strip(), flags=re.IGNORECASE).strip().strip('"')
    return prompt or request


def generate(prompt: str, image_model: Optional[str] = None) -> tuple[str, dict]:
    """Erzeugt das Bild. Gibt (Meldung, Artefakt) zurück wie ein Tool; das Artefakt ist bei Fehlern leer."""
    logger.info("Bild-Agent (%s): %s", image_model or "Standardmodell", prompt)
    try:
        path = image_backend.generate_image(prompt, image_model)
    except Exception as exc:
        logger.warning("Bildgenerierung fehlgeschlagen", exc_info=True)
        return f"Bildgenerierung fehlgeschlagen: {exc}", {}
    return "Das Bild wurde erzeugt und wird dem Nutzer angezeigt.", {"name": path.name, "prompt": prompt}
