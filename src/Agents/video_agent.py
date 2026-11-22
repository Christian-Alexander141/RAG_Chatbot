"""Video-Agent: macht aus einem Videowunsch einen englischen Prompt und erzeugt das Video (mit Ton).

Videowünsche ("Erstelle ein Video von ...", "Animiere das Bild") leitet chat_agent direkt hierher, ohne den
Werkzeug-Agenten. Das Werkzeug generate_video nutzt denselben Weg, falls der Werkzeug-Agent einen Wunsch selbst
erkennt. Den Prompt schreibt dasselbe Sprachmodell wie beim Bild-Agenten (Einstellung "Bild- und Video-Agent").
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.messages import BaseMessage

from rag_chatpipline import config

from . import image_agent, video_backend

logger = logging.getLogger(__name__)

_VERB = image_agent._VERB
_NOUN = r"(video(s)?|clip(s)?|film(e|chen)?|animation(en)?|movie(s)?)\b"
# Ein Verb fürs Erzeugen und ein Video-Wort, oder "animiere ..." allein ("Animiere das Bild").
VIDEO_REQUEST = re.compile(rf"(?=.*{_VERB})(?=.*\b{_NOUN})|\b(animier|animate)", re.IGNORECASE | re.DOTALL)
# Bezug auf ein Bild: "aus dem Bild", "animiere das Foto", "this image"
_IMAGE_REFERENCE = re.compile(r"\b(bild(es)?|foto(s)?|grafik|image|picture|photo|daraus|davon)\b|\banimier|\banimate",
                              re.IGNORECASE)
# Pfad einer Bilddatei im Workspace: in Backticks (so nennt die Oberfläche angehängte Dateien, auch mit
# Leerzeichen im Namen) oder ohne Leerzeichen direkt im Text
_IMAGE_PATH = re.compile(r"`([^`]+\.(?:png|jpe?g|webp|bmp))`|([\w./-]+\.(?:png|jpe?g|webp|bmp))\b", re.IGNORECASE)

PROMPT_WRITER = """Du schreibst Prompts für ein Videomodell.
Mache aus dem Videowunsch des Nutzers einen einzigen englischen Prompt, höchstens 120 Wörter: Schauplatz, Motiv,
was nacheinander passiert, Bewegung, Kameraführung (Schwenk, Zoom, Perspektive), Stil und Licht.
Beschreibe danach die Geräusche und Musik zwischen <AUDCAP> und <ENDAUDCAP>. Gesprochene Sätze stehen wörtlich
zwischen <S> und <E>. Bezieht sich der Wunsch auf den bisherigen Chat oder ein Bild ("davon", "das Bild"),
beschreibe, wie sich die Szene aus diesem Bild heraus bewegt. Gib nur den Prompt aus, ohne Erklärung."""


def is_video_request(text: str) -> bool:
    return bool(VIDEO_REQUEST.search(text))


def start_image(message: str, last_image: Optional[str] = None) -> Optional[str]:
    """Startbild für Bild-zu-Video als Pfad im Workspace, oder None für ein Video nur aus Text.

    1. Ein Bildpfad in der Nachricht, der im Workspace existiert (z. B. ein angehängtes Bild unter Uploads/)
    2. Sonst das zuletzt im Chat erzeugte Bild, wenn die Nachricht sich auf ein Bild bezieht ("Animiere das Bild")
    """
    root = config.WORKSPACE_DIR.resolve()
    for match in _IMAGE_PATH.finditer(message):
        candidate = (match.group(1) or match.group(2)).strip(" ./")
        path = (root / candidate).resolve()
        if root in path.parents and path.is_file():
            return str(path.relative_to(root))
    if last_image and _IMAGE_REFERENCE.search(message):
        path = config.IMAGES_DIR / last_image
        if path.is_file():
            return str(path.resolve().relative_to(root))
    return None


def write_prompt(request: str, history: Iterable[BaseMessage] = (), model_name: Optional[str] = None) -> str:
    """Formuliert den Videowunsch als englischen Prompt. Klappt das nicht, bleibt der Wunsch wie er ist."""
    return image_agent.write_prompt(request, history, model_name, instructions=PROMPT_WRITER, label="Videowunsch")


def generate(prompt: str, video_model: Optional[str] = None, image: Optional[str] = None) -> tuple[str, dict]:
    """Erzeugt das Video. Gibt (Meldung, Artefakt) zurück wie ein Tool; das Artefakt ist bei Fehlern leer.

    image ist ein Pfad im Workspace (Startbild) oder None.
    """
    logger.info("Video-Agent (%s, Startbild %s): %s", video_model or "Standardmodell", image, prompt)
    try:
        start = None
        if image:
            start = (config.WORKSPACE_DIR / image).resolve()
            if config.WORKSPACE_DIR.resolve() not in start.parents or not start.is_file():
                raise FileNotFoundError(f"Startbild '{image}' gibt es im Workspace nicht.")
        path = video_backend.generate_video(prompt, video_model, start)
    except Exception as exc:
        logger.warning("Videogenerierung fehlgeschlagen", exc_info=True)
        return f"Videogenerierung fehlgeschlagen: {exc}", {}
    return "Das Video wurde erzeugt und wird dem Nutzer angezeigt.", {"video": path.name, "prompt": prompt}
