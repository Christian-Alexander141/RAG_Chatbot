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
# Suche in der Wissensbasis: "hybrid" (Embeddings + BM25) oder "dense" (nur Embeddings)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# "true"/"false" schaltet das Denken (Thinking) des Agent-Modells ein/aus, leer = Modell-Standard.
AGENT_REASONING = {"true": True, "false": False}.get(os.getenv("AGENT_REASONING", "").lower())

# Alles, worauf der Agent Zugriff hat, liegt in diesem Verzeichnis.
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")).resolve()
DOCUMENTS_DIR = WORKSPACE_DIR / "Dateien"
IMAGES_DIR = WORKSPACE_DIR / "AIbilder"
VIDEOS_DIR = WORKSPACE_DIR / "AIvideos"
# Bilder, die an eine Chat-Nachricht angehängt werden (z. B. für eigene neuronale Netze oder Bild-zu-Video)
UPLOADS_DIR = WORKSPACE_DIR / "Uploads"
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

# Videogenerierung: "diffusers" (lokal, Standard Kandinsky 6.0 Video Lite) oder "mock" (Platzhaltervideo zum Testen)
VIDEO_BACKEND = os.getenv("VIDEO_BACKEND", "diffusers")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers")
# Videomodelle liegen in einem eigenen Unterordner, damit sie nicht als Bildmodelle erscheinen.
VIDEO_MODELS_DIR = MODELS_DIR / "video"
VIDEO_MODEL_DIR = VIDEO_MODELS_DIR / VIDEO_MODEL.split("/")[-1]
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "864"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))
# Bilder pro Video bei 24 Bildern pro Sekunde: 49 = 2 Sekunden, 121 = 5 Sekunden (Maximum des Modells)
VIDEO_FRAMES = int(os.getenv("VIDEO_FRAMES", "49"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "24"))
VIDEO_STEPS = int(os.getenv("VIDEO_STEPS", "16"))  # die distill-Fassung ist auf 16 Schritte ausgelegt
VIDEO_GUIDANCE = float(os.getenv("VIDEO_GUIDANCE", "1.0"))  # distill-Fassung: 1.0, andere Modelle meist 5.0

# Eigene neuronale Netze (je Netz ein Unterordner mit network.py), siehe docs/eigene-netze.md
NETWORKS_DIR = Path(os.getenv("NETWORKS_DIR", ROOT / "networks")).resolve()

CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o]


def ensure_dirs() -> None:
    for path in (WORKSPACE_DIR, DOCUMENTS_DIR, IMAGES_DIR, VIDEOS_DIR, UPLOADS_DIR, MEMORY_DIR):
        path.mkdir(parents=True, exist_ok=True)
