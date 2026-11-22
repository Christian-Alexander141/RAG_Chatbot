"""Video-Agent: Erkennen von Videowünschen, Startbild, Ablauf im Chat und zweistufiges Laden des Videomodells."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from Agents import chat_agent, image_backend, video_agent, video_backend
from rag_chatpipline import api_server, config

from test_agent_api import ScriptedModel

pytest.importorskip("av")  # Platzhaltervideos schreibt PyAV (Extra "image")


@pytest.fixture(autouse=True)
def no_real_chat_model(monkeypatch):
    monkeypatch.setattr(chat_agent, "get_chat_model", lambda *args: None)


@pytest.mark.parametrize("text", [
    "Erstelle ein Video von einer Katze am Strand",
    "Generiere einen kurzen Clip, in dem ein Zahnrad rotiert",
    "Mach ein Video aus dem Bild",
    "Animiere das Bild",
    "create a video of a sunset",
])
def test_video_requests_are_recognised(text):
    assert video_agent.is_video_request(text)


@pytest.mark.parametrize("text", [
    "Erstelle ein Bild von einer Katze",
    "Wie lade ich ein Video in die Videokonferenz hoch?",
    "Fasse die Reisekostenrichtlinie zusammen",
])
def test_other_messages_are_not_video_requests(text):
    assert not video_agent.is_video_request(text)


def test_start_image_from_attachment_or_last_image(workspace):
    (config.UPLOADS_DIR / "Mein Foto.png").write_bytes(b"x")
    (config.IMAGES_DIR / "letztes.png").write_bytes(b"x")

    attached = "Mach daraus ein Video\n\n[Angehängte Dateien im Workspace: `Uploads/Mein Foto.png`]"
    assert video_agent.start_image(attached, "letztes.png") == "Uploads/Mein Foto.png"
    assert video_agent.start_image("Animiere das Bild", "letztes.png") == "AIbilder/letztes.png"
    assert video_agent.start_image("Erstelle ein Video von einem Hund", "letztes.png") is None
    assert video_agent.start_image("Animiere ../../etc/passwd.png", None) is None


def test_video_request_goes_to_video_agent_and_uses_last_image(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(config, "VIDEO_BACKEND", "mock")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    agent = chat_agent.build_agent(model=ScriptedModel(responses=[AIMessage(content="unbenutzt")]))

    first = chat_agent.chat("Erstelle ein Bild von einer Katze", agent=agent)
    assert len(first["images"]) == 1 and first["videos"] == []

    events = list(chat_agent.chat_stream("Animiere das Bild", chat_id=first["chat_id"], agent=agent))
    kinds = [e["event"] for e in events]
    assert "video_agent_start" in kinds and "agent_start" not in kinds
    call = next(e for e in events if e["event"] == "tool_start")
    assert call["name"] == "generate_video" and call["args"]["start_image"] == f"AIbilder/{first['images'][0]}"

    result = events[-1]["result"]
    assert result["text"] == "Hier ist das Video." and result["images"] == []
    assert (config.VIDEOS_DIR / result["videos"][0]).read_bytes()[4:8] == b"ftyp"  # MP4


def test_video_errors_are_reported_instead_of_crashing(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "VIDEO_BACKEND", "diffusers")
    monkeypatch.setattr(config, "VIDEO_MODELS_DIR", workspace / "keine_modelle")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    agent = chat_agent.build_agent(model=ScriptedModel(responses=[AIMessage(content="unbenutzt")]))

    result = chat_agent.chat("Erstelle ein Video von einem Hund", agent=agent)
    assert "fehlgeschlagen" in result["text"] and "--video" in result["text"] and result["videos"] == []


def test_tool_agent_can_call_generate_video_and_api_returns_urls(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "VIDEO_BACKEND", "mock")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "generate_video", "args": {"prompt": "a dog"}, "id": "v1"}]),
        AIMessage(content="Hier ist dein Video."),
    ])
    agent = chat_agent.build_agent(model=model)
    monkeypatch.setattr(chat_agent, "get_agent", lambda *args: agent)

    # kein Videowunsch nach dem Muster: Der Werkzeug-Agent entscheidet selbst und ruft generate_video auf
    reply = TestClient(api_server.app).post("/chat", json={"message": "Einen rennenden Hund als Bewegtbild, bitte"}).json()
    assert reply["text"] == "Hier ist dein Video." and reply["images"] == []
    assert reply["videos"][0].startswith("/videos/") and reply["videos"][0].endswith(".mp4")


class FakePipeline:
    """Steht für Kandinsky6TI2VAPipeline und merkt sich, was geladen und aufgerufen wurde."""

    loaded: list = []
    calls: list = []

    def __init__(self, skipped):
        self.skipped = sorted(skipped)
        self.components = {}
        self._execution_device = "cpu"
        self.audio_sample_rate = 16000
        FakePipeline.loaded.append(self.skipped)

    @classmethod
    def from_pretrained(cls, model_dir, torch_dtype=None, **skip):
        assert all(value is None for value in skip.values())
        return cls(skip)

    def to(self, **kwargs):
        return self

    def enable_model_cpu_offload(self):
        pass

    def enable_sequential_cpu_offload(self):
        pass

    def encode_prompt(self, prompt, do_classifier_free_guidance, device, dtype):
        import torch

        assert not do_classifier_free_guidance
        return torch.ones(1, 3, 8), torch.ones(1, 8), None, None, None, None

    def __call__(self, height, width, num_frames, num_inference_steps, guidance_scale, frame_rate=24.0,
                 image=None, prompt=None, prompt_embeds=None, pooled_prompt_embeds=None):
        import torch
        from PIL import Image

        FakePipeline.calls.append({"prompt": prompt, "embeds": prompt_embeds is not None, "image": image is not None})
        frame = Image.new("RGB", (32, 32), "gray")
        return SimpleNamespace(frames=[[frame] * 5], audio=torch.zeros(1, 1600))


def test_kandinsky_prompt_is_encoded_before_the_video_model_is_loaded(workspace, monkeypatch):
    import diffusers

    model_dir = workspace / "modelle" / "Kandinsky-Test"
    model_dir.mkdir(parents=True)
    (model_dir / ".download-complete").touch()
    components = ["transformer", "vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2", "scheduler",
                  "audio_vae", "vocoder"]
    index = {"_class_name": "Kandinsky6TI2VAPipeline", **{c: ["diffusers", "X"] for c in components}}
    (model_dir / "model_index.json").write_text(json.dumps(index))
    monkeypatch.setattr(config, "VIDEO_BACKEND", "diffusers")
    monkeypatch.setattr(config, "VIDEO_MODELS_DIR", model_dir.parent)
    monkeypatch.setattr(config, "VIDEO_FRAMES", 5)
    monkeypatch.setattr(diffusers, "DiffusionPipeline", FakePipeline)
    monkeypatch.setattr(image_backend, "_free_vram_from_ollama", lambda: None)
    FakePipeline.loaded, FakePipeline.calls = [], []
    from PIL import Image

    Image.new("RGB", (8, 8)).save(config.UPLOADS_DIR / "start.png")

    path = video_backend.generate_video("a cat", "Kandinsky-Test", config.UPLOADS_DIR / "start.png")

    # Stufe 1 ohne Videomodell, Stufe 2 ohne Textencoder: nie beide gleichzeitig im Speicher
    assert FakePipeline.loaded == [["audio_vae", "transformer", "vae", "vocoder"], ["text_encoder", "text_encoder_2"]]
    assert FakePipeline.calls == [{"prompt": None, "embeds": True, "image": True}]
    import av

    with av.open(str(path)) as container:
        assert sorted(s.type for s in container.streams) == ["audio", "video"]
