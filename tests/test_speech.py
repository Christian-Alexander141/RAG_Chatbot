"""Spracheingabe: Audio vorbereiten, Stille erkennen, Halluzinationen entfernen, Whisper-Aufruf und API.

Das echte Whisper-Modell wird nie geladen, eine nachgebaute Pipeline steht dafür.
"""
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from Agents import image_backend, speech
from rag_chatpipline import api_server, config

RATE = speech.SAMPLE_RATE


def speech_like(seconds: float = 2.0, rate: int = RATE) -> np.ndarray:
    """Silben (Töne) mit Pausen dazwischen: laut und leise im Wechsel wie Sprache."""
    t = np.arange(int(seconds * rate)) / rate
    syllables = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)
    return (0.3 * np.sin(2 * np.pi * 220 * t) * syllables).astype(np.float32)


class FakePipeline:
    def __init__(self, text=" Wie spät ist es? "):
        self.text, self.calls = text, []

    def __call__(self, inputs, generate_kwargs, return_timestamps):
        self.calls.append({"samples": len(inputs["raw"]), "rate": inputs["sampling_rate"],
                           "options": generate_kwargs, "timestamps": return_timestamps})
        return {"text": self.text}


@pytest.fixture
def fake_whisper(monkeypatch):
    pipe = FakePipeline()
    monkeypatch.setattr(speech, "_load", lambda device=None: pipe)
    monkeypatch.setattr(speech, "_schedule_release", lambda: None)
    return pipe


def test_pcm16_from_the_browser_becomes_float_audio_at_16_khz():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    assert speech.pcm16_to_audio(pcm, RATE).tolist() == pytest.approx([0, 0.5, -1.0, 0.99997], abs=1e-4)

    one_second_24k = np.zeros(24_000, dtype="<i2").tobytes()
    assert len(speech.pcm16_to_audio(one_second_24k, 24_000)) == RATE
    assert len(speech.pcm16_to_audio(b"\x01", RATE)) == 0  # halbe Abtastwerte werden verworfen


def test_speech_is_told_apart_from_silence_and_noise():
    noise = np.random.default_rng(0).normal(0, 0.02, 3 * RATE).astype(np.float32)
    hum = (0.05 * np.sin(2 * np.pi * 50 * np.arange(3 * RATE) / RATE)).astype(np.float32)

    assert speech.has_speech(speech_like())
    assert speech.has_speech(speech_like() * 0.05)  # leise gesprochen
    assert not speech.has_speech(np.zeros(3 * RATE, dtype=np.float32))
    assert not speech.has_speech(noise)
    assert not speech.has_speech(hum)
    assert not speech.has_speech(speech_like(0.2))  # zu kurz


@pytest.mark.parametrize("raw, cleaned", [
    ("Untertitel im Auftrag des ZDF, 2017", ""),
    ("Vielen Dank fürs Zuschauen!", ""),
    ("Wie spät ist es? Untertitel der Amara.org-Community", "Wie spät ist es?"),
    ("Erstelle ein Video von einem Zug.", "Erstelle ein Video von einem Zug."),
])
def test_typical_whisper_hallucinations_are_removed(raw, cleaned):
    assert speech._drop_hallucinations(raw) == cleaned


def test_transcribe_passes_language_and_skips_silence(fake_whisper, monkeypatch):
    monkeypatch.setattr(config, "SPEECH_LANGUAGE", "de")

    assert speech.transcribe(speech_like()) == "Wie spät ist es?"
    call = fake_whisper.calls[0]
    assert call["options"] == {"task": "transcribe", "language": "de"} and call["rate"] == RATE
    assert call["timestamps"] is False

    assert speech.transcribe(np.zeros(2 * RATE, dtype=np.float32)) == ""
    assert len(fake_whisper.calls) == 1  # Stille erreicht das Modell gar nicht


def test_long_recordings_use_timestamps_and_are_capped(fake_whisper, monkeypatch):
    monkeypatch.setattr(config, "SPEECH_MAX_SECONDS", 40)
    monkeypatch.setattr(config, "SPEECH_LANGUAGE", "")

    speech.transcribe(speech_like(50))
    call = fake_whisper.calls[0]
    assert call["samples"] == 40 * RATE and call["timestamps"] is True
    assert "language" not in call["options"]  # leer = Sprache automatisch erkennen


def test_audio_files_are_decoded_to_mono_16_khz(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "aufnahme.wav"
    stereo = np.repeat((speech_like(1.5, 44_100) * 32767).astype("<i2")[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44_100)
        f.writeframes(stereo.tobytes())

    audio = speech.load_audio_file(path)
    assert abs(len(audio) - 1.5 * RATE) < 0.01 * RATE and audio.dtype == np.float32
    assert speech.has_speech(audio)

    broken = tmp_path / "kaputt.mp3"
    broken.write_bytes(b"keine Audiodatei")
    with pytest.raises(speech.SpeechError):
        speech.load_audio_file(broken)


def test_transcribe_api(tmp_path, monkeypatch):
    client = TestClient(api_server.app)
    monkeypatch.setattr(speech, "is_installed", lambda: False)
    assert client.post("/transcribe", files={"file": ("a.wav", b"x")}).status_code == 503

    monkeypatch.setattr(speech, "is_installed", lambda: True)
    monkeypatch.setattr(speech, "transcribe_file", lambda path: f"Text aus {path.name}")
    reply = client.post("/transcribe", files={"file": ("../../frage.ogg", b"x")})
    assert reply.json() == {"text": "Text aus frage.ogg"}


def test_missing_model_gives_install_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPEECH_MODEL_DIR", tmp_path / "fehlt")
    assert not speech.is_installed()
    with pytest.raises(speech.SpeechError, match="install"):
        speech._model_dir()


def test_image_and_video_generation_release_the_speech_model(monkeypatch):
    released = []
    monkeypatch.setattr(speech, "release", lambda: released.append(True))
    monkeypatch.setattr(image_backend, "_free_vram_from_ollama", lambda: None)
    image_backend.free_gpu()
    assert released == [True]
