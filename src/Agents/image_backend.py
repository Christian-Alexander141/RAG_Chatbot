"""Lokale Bildgenerierung mit diffusers. Standard ist FLUX.2 [klein] 4B; "mock" erzeugt ein Platzhalterbild zum Testen.

Bildmodelle liegen als Ordner unter models/. Das Standardmodell lädt der Installer einmalig herunter
(download_flux_model), die App selbst lädt nichts herunter.
"""
from __future__ import annotations

import gc
import logging
import struct
import threading
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from rag_chatpipline import config

logger = logging.getLogger(__name__)

_pipeline = None
_pipeline_dir: Optional[Path] = None
_lock = threading.Lock()

# Gleiche Gewichte liegen im Repo zusätzlich als eine große Einzeldatei; diffusers braucht nur die Unterordner.
_SKIP_FILES = ["*.jpg", "flux-2-klein-4b.safetensors"]
_DONE_MARKER = ".download-complete"
_DOWNLOAD_GB = 16


class ImageBackendError(RuntimeError):
    pass


def available_image_models() -> list[str]:
    """Vollständig heruntergeladene Bildmodelle (Ordnernamen unter MODELS_DIR)."""
    if not config.MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in config.MODELS_DIR.iterdir() if (p / _DONE_MARKER).exists())


def _model_dir(name: Optional[str]) -> Path:
    name = name or config.FLUX_MODEL_DIR.name
    if name not in available_image_models():
        raise ImageBackendError(
            f"Bildmodell '{name}' fehlt in {config.MODELS_DIR}. Herunterladen mit ./install.sh (Windows: install.bat)"
        )
    return config.MODELS_DIR / name


def download_flux_model() -> None:
    """Lädt das FLUX-Modell nach FLUX_MODEL_DIR, falls es dort noch nicht vollständig liegt."""
    if (config.FLUX_MODEL_DIR / _DONE_MARKER).exists():
        print(f"FLUX-Modell vorhanden: {config.FLUX_MODEL_DIR}")
        return
    import shutil

    from huggingface_hub import snapshot_download

    config.FLUX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(config.FLUX_MODEL_DIR).free / 1e9
    if free_gb < _DOWNLOAD_GB + 2:
        raise ImageBackendError(f"Zu wenig Speicherplatz: {free_gb:.0f} GB frei, gebraucht werden etwa {_DOWNLOAD_GB} GB.")
    print(f"Lade {config.FLUX_MODEL} nach {config.FLUX_MODEL_DIR} (etwa {_DOWNLOAD_GB} GB, nur beim ersten Mal) ...")
    # Abgebrochene Downloads setzt snapshot_download beim nächsten Aufruf fort.
    snapshot_download(config.FLUX_MODEL, local_dir=config.FLUX_MODEL_DIR, ignore_patterns=_SKIP_FILES)
    (config.FLUX_MODEL_DIR / _DONE_MARKER).touch()
    print("FLUX-Modell heruntergeladen.")


def _free_vram_from_ollama() -> None:
    """Ollama hält Sprachmodelle im Grafikspeicher. Für FLUX wird der Platz gebraucht.
    Ollama lädt sie bei der nächsten Anfrage von selbst wieder."""
    try:
        import ollama

        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        for running in client.ps().models:
            client.generate(model=running.model, keep_alive=0)
    except Exception:
        logger.debug("Ollama-Modelle konnten nicht entladen werden", exc_info=True)


def _load_pipeline(model_dir: Path):
    global _pipeline, _pipeline_dir
    if _pipeline is not None and _pipeline_dir == model_dir:
        return _pipeline
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise ImportError("Für FLUX fehlen Pakete. Installation: ./install.sh (Windows: install.bat)") from exc

    if _pipeline is not None:  # anderes Modell gewählt: altes freigeben, zwei passen nicht in den Speicher
        _pipeline = _pipeline_dir = None
        gc.collect()
        torch.cuda.empty_cache()

    gpu = torch.cuda.is_available()
    # Ältere GPUs (z. B. GTX 10xx) können kein bfloat16.
    dtype = torch.float16 if gpu and not torch.cuda.is_bf16_supported(including_emulation=False) else torch.bfloat16
    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    if gpu:
        # Holt immer nur den gerade nötigen Modellteil in den Grafikspeicher, damit auch 8-12 GB reichen.
        pipe.enable_model_cpu_offload()
    logger.info("Bildmodell %s geladen (%s, %s)", model_dir.name, "GPU" if gpu else "CPU", dtype)
    _pipeline, _pipeline_dir = pipe, model_dir
    return pipe


def _write_mock_png(path: Path, size: int = 256) -> None:
    """Schreibt ein graues PNG ohne Fremdabhängigkeiten."""
    row = b"\x00" + b"\x80" * (size * 3)
    raw = row * size

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def generate_image(prompt: str, model: Optional[str] = None) -> Path:
    """Erzeugt ein Bild zum Prompt mit dem Bildmodell model (Standard: FLUX_MODEL) und speichert es in IMAGES_DIR."""
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target = config.IMAGES_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.png"

    if config.IMAGE_BACKEND == "mock":
        _write_mock_png(target)
        return target
    if config.IMAGE_BACKEND != "flux":
        raise ImageBackendError(f"Unbekanntes IMAGE_BACKEND '{config.IMAGE_BACKEND}' (erlaubt: flux, mock).")

    model_dir = _model_dir(model)
    with _lock:  # eine Generierung gleichzeitig, sonst geht der Speicher aus
        _free_vram_from_ollama()
        pipe = _load_pipeline(model_dir)
        image = pipe(
            prompt=prompt,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            num_inference_steps=4,  # FLUX.2 [klein] und FLUX.1-schnell sind auf 4 Schritte ausgelegt
            guidance_scale=1.0,
        ).images[0]
    image.save(target)
    return target
