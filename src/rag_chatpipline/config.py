"""Zentrale Konfiguration. Alle Werte lassen sich per Umgebungsvariable überschreiben."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Chat-Modell: formuliert die Antworten an die Nutzer.
CHAT_MODEL = os.getenv("CHAT_MODEL", "cas/teuken-7b-q4km:latest")
# Agent-Modell: ruft Tools auf (braucht gutes Tool-Calling). Gleicher Name wie CHAT_MODEL = ein Modell für beides.
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3:1.7b")
# Bild-Agent: schreibt aus dem Bildwunsch den englischen FLUX-Prompt.
IMAGE_PROMPT_MODEL = os.getenv("IMAGE_PROMPT_MODEL", AGENT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# "true"/"false" schaltet das Denken (Thinking) des Agent-Modells ein/aus, leer = Modell-Standard.
AGENT_REASONING = {"true": True, "false": False}.get(os.getenv("AGENT_REASONING", "").lower())

# Alles, worauf der Agent Zugriff hat, liegt in diesem Verzeichnis.
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")).resolve()
DOCUMENTS_DIR = WORKSPACE_DIR / "Dateien"
IMAGES_DIR = WORKSPACE_DIR / "AIbilder"
MEMORY_DIR = ROOT / "memory"
QDRANT_PATH = MEMORY_DIR / "qdrant"
COLLECTION_NAME = "documents"

# Bildgenerierung: "flux" (lokal via diffusers) oder "mock" (Platzhalterbild zum Testen)
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "flux")
FLUX_MODEL = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
# Lokale Modelle (lädt der Installer einmalig herunter)
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models")).resolve()
FLUX_MODEL_DIR = MODELS_DIR / FLUX_MODEL.split("/")[-1]
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "768"))
# Sekunden, die das Bildmodell nach dem letzten Bild im Arbeitsspeicher bleibt (0 = sofort freigeben).
IMAGE_KEEP_ALIVE = int(os.getenv("IMAGE_KEEP_ALIVE", "60"))

CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o]


def ensure_dirs() -> None:
    for path in (WORKSPACE_DIR, DOCUMENTS_DIR, IMAGES_DIR, MEMORY_DIR):
        path.mkdir(parents=True, exist_ok=True)
