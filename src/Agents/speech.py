"""Spracheingabe: wandelt Sprachaufnahmen lokal in Text um (Speech-to-Text mit Whisper).

Standardmodell ist Whisper large-v3-turbo von OpenAI (MIT-Lizenz, 99 Sprachen, gut in Deutsch). Es liegt unter
models/speech/ und wird vom Installer einmalig heruntergeladen (download_speech_model). Die App selbst lädt nichts.

Eingänge:
- transcribe_pcm16: Aufnahme des Mikrofon-Knopfs der Oberfläche (16-Bit-PCM, mono)
- transcribe_file: Audiodatei (WAV, MP3, OGG, M4A, WebM, ...), für Kommandozeile und API

Die Aufnahme verlässt den Rechner nicht. Das Modell bleibt nach der letzten Aufnahme SPEECH_KEEP_ALIVE Sekunden
geladen. Vor Bild- und Videogenerierung wird es freigegeben (release), damit der Grafikspeicher reicht.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rag_chatpipline import config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Whisper erwartet 16 kHz mono
_DONE_MARKER = ".download-complete"
_MIN_SECONDS = 0.3
# Einfache Sprachaktivitätsprüfung in Blöcken von 30 ms. Sprache wechselt zwischen lauten Silben und leisen
# Pausen, Stille und gleichmäßiges Rauschen nicht. Ohne diese Prüfung macht Whisper aus Rauschen Sätze wie "Amen."
_FRAME = 480
_MIN_LEVEL = 0.005  # Effektivwert der lauteren Blöcke (Vollaussteuerung = 1.0)
_MIN_CONTRAST = 3.0  # laute Blöcke mindestens 3-mal (knapp 10 dB) lauter als die leisen
# Whisper "hört" in Stille oder Rauschen gern Sätze, die in seinen Trainingsdaten häufig waren (Untertitel-Abspann).
_HALLUCINATIONS = re.compile(
    r"\b(untertitel(ung)?( im auftrag)? (des|der|von) (zdf|ard|swr|br|wdr|ndr|mdr|hr|funk)\b[^.!?]*"
    r"|untertitel der amara\.org-community"
    r"|vielen dank f(ü|ue)r(s|'s)? zu(sehen|schauen)"
    r"|thanks? (you )?for watching)[.!?]*",
    re.IGNORECASE,
)

_pipeline: Any = None
_device: Optional[str] = None
_lock = threading.Lock()
_last_used = 0.0
_release_timer: Optional[threading.Timer] = None


class SpeechError(RuntimeError):
    pass


# === Modell ===
def is_installed() -> bool:
    return (config.SPEECH_MODEL_DIR / _DONE_MARKER).exists()


def _model_dir() -> Path:
    if not is_installed():
        raise SpeechError(
            f"Das Spracherkennungsmodell fehlt in {config.SPEECH_MODELS_DIR}. "
            "Herunterladen mit ./install.sh (Windows: install.bat)"
        )
    return config.SPEECH_MODEL_DIR


def download_speech_model() -> None:
    """Lädt das Spracherkennungsmodell nach SPEECH_MODEL_DIR, falls es dort noch nicht vollständig liegt."""
    target = config.SPEECH_MODEL_DIR
    if (target / _DONE_MARKER).exists():
        print(f"Spracherkennungsmodell vorhanden: {target}")
        return
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"Lade {config.SPEECH_MODEL} nach {target} (etwa 1,6 GB, nur beim ersten Mal) ...")
    snapshot_download(config.SPEECH_MODEL, local_dir=target)  # setzt abgebrochene Downloads fort
    (target / _DONE_MARKER).touch()
    print("Spracherkennungsmodell heruntergeladen.")


def _choose_device(torch: Any) -> str:
    if config.SPEECH_DEVICE == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if config.SPEECH_DEVICE == "cuda":
        return "cuda"
    free, _ = torch.cuda.mem_get_info()
    # Ollama oder ein Bildmodell können den Grafikspeicher belegen. Dann lieber auf der CPU rechnen.
    return "cuda" if free > 4e9 else "cpu"


def _dtype(torch: Any, device: str) -> Any:
    # float16 nur auf GPUs, die damit schnell rechnen (ab Volta). Ältere GPUs und die CPU nehmen float32.
    if device == "cuda" and torch.cuda.get_device_capability(0) >= (7, 0):
        return torch.float16
    return torch.float32


def _load(device: Optional[str] = None) -> Any:
    global _pipeline, _device
    import torch
    from transformers import pipeline

    if _pipeline is not None and device in (None, _device):
        return _pipeline  # geladenes Modell weiter nutzen (der freie Grafikspeicher zählt es sonst selbst mit)
    device = device or _choose_device(torch)
    _pipeline = None
    dtype = _dtype(torch, device)
    _pipeline = pipeline("automatic-speech-recognition", model=str(_model_dir()), dtype=dtype, device=device)
    _device = device
    logger.info("Spracherkennung %s geladen (%s, %s)", config.SPEECH_MODEL_DIR.name, device, dtype)
    return _pipeline


def release() -> None:
    """Gibt das Spracherkennungsmodell frei (z. B. bevor ein Bild oder Video erzeugt wird)."""
    global _pipeline, _device
    with _lock:
        if _pipeline is None:
            return
        _pipeline = _device = None
    from . import image_backend

    image_backend._free_memory()
    logger.info("Spracherkennung freigegeben")


def _release_if_idle() -> None:
    if time.monotonic() - _last_used >= config.SPEECH_KEEP_ALIVE - 1:
        release()


def _schedule_release() -> None:
    global _release_timer
    if _release_timer is not None:
        _release_timer.cancel()
    _release_timer = threading.Timer(max(config.SPEECH_KEEP_ALIVE, 0), _release_if_idle)
    _release_timer.daemon = True
    _release_timer.start()


# === Audio vorbereiten ===
def _resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or len(audio) == 0:
        return audio.astype(np.float32)
    count = max(1, round(len(audio) * SAMPLE_RATE / rate))
    return np.interp(np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio).astype(np.float32)


def pcm16_to_audio(data: bytes, sample_rate: int) -> np.ndarray:
    """16-Bit-PCM (little endian, mono) wie vom Mikrofon der Oberfläche -> float32, 16 kHz, Werte -1 bis 1."""
    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return _resample(samples, sample_rate)


def load_audio_file(path: Path) -> np.ndarray:
    """Liest eine Audiodatei (alles, was FFmpeg kennt) als float32, mono, 16 kHz, höchstens SPEECH_MAX_SECONDS."""
    import av

    parts: list[np.ndarray] = []
    limit = config.SPEECH_MAX_SECONDS * SAMPLE_RATE
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise SpeechError(f"'{Path(path).name}' enthält keine Tonspur.")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(container.streams.audio[0]):
                parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
                if sum(len(p) for p in parts) >= limit:
                    break  # der Rest würde ohnehin abgeschnitten
            else:
                parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    except av.error.FFmpegError as exc:
        raise SpeechError(f"'{Path(path).name}' ist keine lesbare Audiodatei: {exc}") from exc
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)


def has_speech(audio: np.ndarray) -> bool:
    """Grob: enthält die Aufnahme Sprache? Gemessen an Lautstärke und Unterschied zwischen lauten und leisen Blöcken."""
    if len(audio) < _MIN_SECONDS * SAMPLE_RATE:
        return False
    frames = audio[: len(audio) // _FRAME * _FRAME].reshape(-1, _FRAME)
    levels = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    loud, quiet = np.percentile(levels, 90), np.percentile(levels, 10)
    return bool(loud >= _MIN_LEVEL and loud / quiet >= _MIN_CONTRAST)


def _drop_hallucinations(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _HALLUCINATIONS.sub("", text)).strip(" ,")


# === Erkennen ===
def transcribe(audio: np.ndarray) -> str:
    """Wandelt Sprache (float32, mono, 16 kHz) in Text um. Leer bei Stille, Rauschen oder zu kurzen Aufnahmen."""
    global _last_used
    audio = np.asarray(audio, dtype=np.float32)[: config.SPEECH_MAX_SECONDS * SAMPLE_RATE]
    seconds = len(audio) / SAMPLE_RATE
    if not has_speech(audio):
        return ""

    options: dict[str, Any] = {"task": "transcribe"}
    if config.SPEECH_LANGUAGE:
        options["language"] = config.SPEECH_LANGUAGE
    started = time.monotonic()
    with _lock:
        try:
            pipe = _load()
            # Whisper verarbeitet 30 Sekunden am Stück, längere Aufnahmen brauchen Zeitstempel.
            result = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=options,
                          return_timestamps=seconds > 30)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or _device == "cpu":
                raise
            logger.warning("Zu wenig Grafikspeicher für die Spracherkennung, rechne auf der CPU")
            pipe = _load("cpu")
            result = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=options,
                          return_timestamps=seconds > 30)
        finally:
            _last_used = time.monotonic()
    _schedule_release()
    text = _drop_hallucinations(str(result.get("text", "")))
    logger.info("Spracheingabe (%.1f s Audio, %.1f s Rechenzeit, %s): %s", seconds, time.monotonic() - started, _device, text)
    return text


def transcribe_pcm16(data: bytes, sample_rate: int) -> str:
    return transcribe(pcm16_to_audio(data, sample_rate))


def transcribe_file(path: Path) -> str:
    return transcribe(load_audio_file(path))
