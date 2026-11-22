"""Lokale Videogenerierung mit diffusers. Standard ist Kandinsky 6.0 Video Lite (distill, MIT-Lizenz), das Videos
mit passendem Ton erzeugt, aus Text oder aus Text und Startbild. "mock" erzeugt ein Platzhaltervideo zum Testen.

Videomodelle liegen als Ordner unter models/video/. Das Standardmodell lädt der Installer mit --video einmalig
herunter (download_video_model), die App selbst lädt nichts herunter.

Speicher: Kandinsky 6 bringt einen 16 GB großen Textencoder (Qwen2.5-VL 7B) mit, das ganze Modell hat 27 GB.
Damit nicht alles gleichzeitig im Arbeitsspeicher liegt, wird der Prompt zuerst allein mit den Textencodern
kodiert. Diese werden danach freigegeben, erst dann wird das eigentliche Videomodell geladen. Weil ein Video
ohnehin Minuten dauert, bleibt nach dem Video nichts im Speicher.
"""
from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rag_chatpipline import config

from . import image_backend

logger = logging.getLogger(__name__)

_DONE_MARKER = ".download-complete"
_DOWNLOAD_GB = 28
_TEXT_ENCODERS = ("text_encoder", "text_encoder_2")
# Pipelines, bei denen der Prompt vorab mit den Textencodern allein kodiert wird (siehe oben)
_SPLIT_PIPELINES = {"Kandinsky6TI2VAPipeline"}
# Wie bei FLUX gemessen: Beim Laden belegt ein Modell etwa das 1,5-Fache seiner Dateien im Arbeitsspeicher.
_RAM_FACTOR = 1.5
# Belegt die größte Komponente mehr als diesen Anteil des Grafikspeichers, wird schichtweise ausgelagert.
_VRAM_SHARE = 0.7


class VideoBackendError(RuntimeError):
    pass


def available_video_models() -> list[str]:
    """Vollständig heruntergeladene Videomodelle (Ordnernamen unter models/video/)."""
    if not config.VIDEO_MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in config.VIDEO_MODELS_DIR.iterdir() if (p / _DONE_MARKER).exists())


def _model_dir(name: Optional[str]) -> Path:
    name = name or config.VIDEO_MODEL_DIR.name
    if name not in available_video_models():
        raise VideoBackendError(
            f"Videomodell '{name}' fehlt in {config.VIDEO_MODELS_DIR}. "
            "Herunterladen mit ./install.sh --video (Windows: install.bat --video)"
        )
    return config.VIDEO_MODELS_DIR / name


def download_video_model() -> None:
    """Lädt das Videomodell nach VIDEO_MODEL_DIR, falls es dort noch nicht vollständig liegt."""
    target = config.VIDEO_MODEL_DIR
    if (target / _DONE_MARKER).exists():
        print(f"Videomodell vorhanden: {target}")
        return
    import shutil

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 1e9
    if free_gb < _DOWNLOAD_GB + 2:
        raise VideoBackendError(f"Zu wenig Speicherplatz: {free_gb:.0f} GB frei, gebraucht werden etwa {_DOWNLOAD_GB} GB.")
    print(f"Lade {config.VIDEO_MODEL} nach {target} (etwa {_DOWNLOAD_GB} GB, nur beim ersten Mal) ...")
    snapshot_download(config.VIDEO_MODEL, local_dir=target)  # setzt abgebrochene Downloads fort
    (target / _DONE_MARKER).touch()
    print("Videomodell heruntergeladen.")


def _model_index(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))


def _check_free_ram(model_dir: Path, components: list[str]) -> None:
    """Bricht mit einer Meldung ab, statt den Rechner in Speichermangel zu treiben (siehe image_backend)."""
    try:
        import psutil
    except ImportError:
        return
    need = sum(f.stat().st_size for c in components for f in (model_dir / c).glob("*.safetensors")) * _RAM_FACTOR
    available = psutil.virtual_memory().available
    if available < need:
        raise VideoBackendError(
            f"Zu wenig freier Arbeitsspeicher für {model_dir.name}: {available / 1e9:.0f} GB frei, gebraucht werden "
            f"etwa {need / 1e9:.0f} GB. Bitte andere Programme oder eine zweite Instanz des Chatbots schließen."
        )


def _dtype(torch: Any) -> Any:
    gpu = torch.cuda.is_available()
    # Ältere GPUs (z. B. GTX 10xx) können kein bfloat16.
    return torch.float16 if gpu and not torch.cuda.is_bf16_supported(including_emulation=False) else torch.bfloat16


def _load(model_dir: Path, components: list[str], skip: list[str], torch: Any) -> Any:
    from diffusers import DiffusionPipeline

    _check_free_ram(model_dir, components)
    dtype = _dtype(torch)
    # Im gespeicherten Format laden und danach umwandeln, wie bei FLUX (spart Arbeitsspeicher beim Laden).
    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **{name: None for name in skip})
    if dtype != torch.bfloat16:
        pipe.to(dtype=dtype)
    if torch.cuda.is_available():
        sizes = [
            sum(p.numel() * p.element_size() for p in module.parameters())
            for module in pipe.components.values() if isinstance(module, torch.nn.Module)
        ]
        if max(sizes, default=0) > _VRAM_SHARE * torch.cuda.get_device_properties(0).total_memory:
            # Passt ein Teil nicht in den Grafikspeicher: Schicht für Schicht auf die GPU holen. Langsamer, geht aber.
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()  # ganze Modellteile nacheinander auf die GPU
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()  # dekodiert das Video in Kacheln, spart Grafikspeicher
    logger.info("Videomodell %s geladen (%s, %s, ohne %s)", model_dir.name,
                "GPU" if torch.cuda.is_available() else "CPU", dtype, ", ".join(skip) or "nichts")
    return pipe


def _encode_prompt(model_dir: Path, prompt: str, components: list[str], torch: Any) -> dict[str, Any]:
    """Stufe 1: nur Textencoder und Tokenizer laden, den Prompt kodieren und alles wieder freigeben."""
    text_parts = [c for c in components if c in _TEXT_ENCODERS or c.startswith("tokenizer") or c == "scheduler"]
    pipe = _load(model_dir, text_parts, [c for c in components if c not in text_parts], torch)
    try:
        with torch.no_grad():
            embeds = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False,
                                        device=pipe._execution_device, dtype=_dtype(torch))
    finally:
        del pipe
        image_backend._free_memory()
    prompt_embeds, pooled = embeds[0].cpu(), embeds[1].cpu()
    if torch.isnan(prompt_embeds).any() or torch.isnan(pooled).any():
        raise VideoBackendError(
            "Der Textencoder hat in float16 ungültige Werte geliefert. Das kann auf GPUs ohne bfloat16 "
            "(z. B. GTX 10xx) passieren. Ein anderes Videomodell oder eine neuere GPU verwenden."
        )
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled}


def _write_mock_video(path: Path, size: int = 64, frames: int = 12) -> None:
    """Graues Platzhaltervideo (1 Sekunde), braucht nur PyAV."""
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=frames)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        grey = np.full((size, size, 3), 128, dtype=np.uint8)
        for _ in range(frames):
            container.mux(stream.encode(av.VideoFrame.from_ndarray(grey, format="rgb24")))
        container.mux(stream.encode())


def generate_video(prompt: str, model: Optional[str] = None, image: Optional[Path] = None) -> Path:
    """Erzeugt ein Video zum Prompt, optional ausgehend von einem Startbild, und speichert es in VIDEOS_DIR."""
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.VIDEOS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.mp4"

    if config.VIDEO_BACKEND == "mock":
        _write_mock_video(target)
        return target
    if config.VIDEO_BACKEND != "diffusers":
        raise VideoBackendError(f"Unbekanntes VIDEO_BACKEND '{config.VIDEO_BACKEND}' (erlaubt: diffusers, mock).")

    model_dir = _model_dir(model)
    try:
        import torch
        from diffusers.utils import encode_video
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Für Videos fehlen Pakete. Installation: ./install.sh --video (Windows: install.bat --video)") from exc

    index = _model_index(model_dir)
    components = [name for name, value in index.items() if not name.startswith("_") and isinstance(value, list) and value[0]]
    split = index.get("_class_name") in _SPLIT_PIPELINES
    call: dict[str, Any] = {
        "height": config.VIDEO_HEIGHT,
        "width": config.VIDEO_WIDTH,
        "num_frames": config.VIDEO_FRAMES,
        "num_inference_steps": config.VIDEO_STEPS,
        "guidance_scale": config.VIDEO_GUIDANCE,
    }
    pipe = None
    with image_backend._lock:  # immer nur ein Bild oder Video gleichzeitig, sonst geht der Speicher aus
        try:
            image_backend.release()  # FLUX belegt rund 16 GB Arbeitsspeicher
            image_backend.free_gpu()  # Ollama-Modelle und Spracherkennung aus dem Grafikspeicher
            if split:
                call.update(_encode_prompt(model_dir, prompt, components, torch))
                pipe = _load(model_dir, [c for c in components if c not in _TEXT_ENCODERS], list(_TEXT_ENCODERS), torch)
            else:
                call["prompt"] = prompt
                pipe = _load(model_dir, components, [], torch)

            parameters = inspect.signature(pipe.__call__).parameters
            if "frame_rate" in parameters:
                call["frame_rate"] = float(config.VIDEO_FPS)
            if image is not None:
                if "image" not in parameters:
                    raise VideoBackendError(f"Das Videomodell {model_dir.name} kann kein Startbild verwenden.")
                call["image"] = Image.open(image).convert("RGB")
            logger.info("Erzeuge Video mit %s: %s", model_dir.name, {k: v for k, v in call.items() if "embeds" not in k})
            output = pipe(**call)

            extra: dict[str, Any] = {}
            audio = getattr(output, "audio", None)
            rate = getattr(pipe, "audio_sample_rate", None)
            if audio is not None and rate:
                track = torch.as_tensor(audio[0]).float().cpu()
                extra = {"audio": track[None] if track.ndim == 1 else track, "audio_sample_rate": rate}
            encode_video(output.frames[0], fps=config.VIDEO_FPS, output_path=str(target), **extra)
        finally:
            del pipe
            image_backend._free_memory()
    return target
