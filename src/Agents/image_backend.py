"""Lokale Bildgenerierung. Standard ist FLUX (diffusers); "mock" erzeugt ein Platzhalterbild zum Testen."""
from __future__ import annotations

import struct
import threading
import zlib
from datetime import datetime
from pathlib import Path

from rag_chatpipline import config

_pipeline = None
_lock = threading.Lock()


class ImageBackendError(RuntimeError):
    pass


def _load_flux():
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImportError(
                "Für FLUX fehlen Pakete. Installation: pip install -r requirements-image.txt"
            ) from exc

        pipe = FluxPipeline.from_pretrained(config.FLUX_MODEL, torch_dtype=torch.bfloat16)
        # Lagert Modellteile bei Bedarf in den RAM aus, damit auch kleinere GPUs reichen.
        pipe.enable_model_cpu_offload()
        _pipeline = pipe
    return _pipeline


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


def generate_image(prompt: str) -> Path:
    """Erzeugt ein Bild zum Prompt und speichert es in IMAGES_DIR."""
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target = config.IMAGES_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.png"

    if config.IMAGE_BACKEND == "mock":
        _write_mock_png(target)
        return target
    if config.IMAGE_BACKEND != "flux":
        raise ImageBackendError(f"Unbekanntes IMAGE_BACKEND '{config.IMAGE_BACKEND}' (erlaubt: flux, mock).")

    with _lock:  # eine Generierung gleichzeitig, sonst geht der Speicher aus
        pipe = _load_flux()
        image = pipe(
            prompt,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            num_inference_steps=4,  # FLUX.1-schnell ist auf 4 Schritte ohne Guidance ausgelegt
            guidance_scale=0.0,
            max_sequence_length=256,
        ).images[0]
    image.save(target)
    return target
