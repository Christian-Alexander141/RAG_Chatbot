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

# Spracheingabe (Speech-to-Text): Whisper lokal über transformers
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "openai/whisper-large-v3-turbo")
SPEECH_MODELS_DIR = MODELS_DIR / "speech"
SPEECH_MODEL_DIR = SPEECH_MODELS_DIR / SPEECH_MODEL.split("/")[-1]
# Sprache der Aufnahmen, z. B. "de" oder "en". Leer = automatisch erkennen (bei kurzen Sätzen unzuverlässiger).
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "de")
# "auto" (GPU, wenn genug Grafikspeicher frei ist, sonst CPU), "cuda" oder "cpu"
SPEECH_DEVICE = os.getenv("SPEECH_DEVICE", "auto")
# Sekunden, die das Modell nach der letzten Aufnahme geladen bleibt (0 = sofort freigeben)
SPEECH_KEEP_ALIVE = int(os.getenv("SPEECH_KEEP_ALIVE", "300"))
# Längere Aufnahmen werden abgeschnitten
SPEECH_MAX_SECONDS = int(os.getenv("SPEECH_MAX_SECONDS", "120"))

# Eigene neuronale Netze (je Netz ein Unterordner mit network.py), siehe docs/eigene-netze.md
NETWORKS_DIR = Path(os.getenv("NETWORKS_DIR", ROOT / "networks")).resolve()

CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o]

# Anmeldung mit Microsoft Active Directory (optional, siehe README). Aus: keine Anmeldeseite, wie ohne AD.
AD_ENABLED = os.getenv("AD_ENABLED", "false").strip().lower() in ("1", "true", "yes", "ja", "on")
# Domänencontroller, kommagetrennt; antwortet einer nicht, wird der nächste versucht. ldaps:// oder ldap:// (StartTLS)
AD_SERVER = os.getenv("AD_SERVER", "")
# DNS-Name der Domäne: ergänzt Benutzernamen ohne @ und ergibt den Suchbereich (firma.local -> DC=firma,DC=local)
AD_DOMAIN = os.getenv("AD_DOMAIN", "")
AD_BASE_DN = os.getenv("AD_BASE_DN", "")  # leer = aus AD_DOMAIN
# Nur Mitglieder dieser Gruppen dürfen sich anmelden (Name oder DN, mit ";" getrennt). Leer = alle Konten der Domäne.
AD_ALLOWED_GROUPS = [g.strip() for g in os.getenv("AD_ALLOWED_GROUPS", "").split(";") if g.strip()]
# Admins (Gruppen wie oben): wählen das Embedding-Modell für alle und dürfen Cloud-Modelle nutzen. Leer = niemand.
AD_ADMIN_GROUPS = [g.strip() for g in os.getenv("AD_ADMIN_GROUPS", "").split(";") if g.strip()]
# Dienstkonto (normales Domänenkonto ohne besondere Rechte): prüft nach der Anmeldung laufend, ob ein Konto noch
# aktiv ist, damit gesperrte oder deaktivierte Konten sofort keinen Zugriff mehr haben. Passwort direkt oder als Datei.
AD_BIND_USER = os.getenv("AD_BIND_USER", "")
AD_BIND_PASSWORD = os.getenv("AD_BIND_PASSWORD", "")
AD_BIND_PASSWORD_FILE = os.getenv("AD_BIND_PASSWORD_FILE", "")
# So viele Sekunden gilt das Ergebnis dieser Prüfung, bevor das AD erneut gefragt wird (0 = bei jeder Anfrage).
AD_CHECK_SECONDS = int(os.getenv("AD_CHECK_SECONDS", "10"))
# Zertifikat der Zertifizierungsstelle (PEM), die das Zertifikat der Domänencontroller ausgestellt hat.
# Leer = Zertifikate des Betriebssystems.
AD_CA_FILE = os.getenv("AD_CA_FILE", "")
AD_TIMEOUT = int(os.getenv("AD_TIMEOUT", "5"))  # Sekunden je Domänencontroller


def ensure_dirs() -> None:
    for path in (WORKSPACE_DIR, DOCUMENTS_DIR, IMAGES_DIR, VIDEOS_DIR, UPLOADS_DIR, MEMORY_DIR):
        path.mkdir(parents=True, exist_ok=True)
