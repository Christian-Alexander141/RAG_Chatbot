import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))

import user_settings  # noqa: E402
from Agents import models  # noqa: E402
from Agents.models import ModelChoice  # noqa: E402
from rag_chatpipline import config, ollama_models  # noqa: E402


def test_settings_are_saved_per_user(monkeypatch, tmp_path):
    monkeypatch.setattr(user_settings, "SETTINGS_FILE", tmp_path / "user_settings.json")
    user_settings.save("anna", ModelChoice("chat-a", "agent-a", "prompt-a", "bild-a"))
    user_settings.save("ben", ModelChoice("chat-b", "agent-b", "prompt-b", "bild-b"))
    assert user_settings.load("anna")["chat"] == "chat-a"
    assert user_settings.load("ben")["image"] == "bild-b"
    assert user_settings.load("neu") == {}


def test_resolve_falls_back_when_model_is_gone(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "standard-chat")
    available = {"chat": ["mistral:7b", "standard-chat"], "agent": ["qwen3.5:2b"], "image_prompt": [], "image": []}
    choice = user_settings.resolve({"chat": "geloescht:1b", "agent": "geloescht:1b"}, available)
    assert choice.chat == "standard-chat"  # gespeichertes fehlt: Standard
    assert choice.agent == "qwen3.5:2b"  # Standard fehlt auch: erstes verfügbares
    assert choice.image == config.FLUX_MODEL_DIR.name  # keine Liste: Standard bleibt


def test_language_models_are_sorted_by_capability_without_cloud_models(monkeypatch):
    tags = {"models": [
        {"name": "qwen3.5:2b"}, {"name": "mistral:7b"}, {"name": "granite-embedding:278m"},
        {"name": "gpt-oss:120b-cloud", "remote_host": "https://ollama.com:443"},
    ]}
    caps = {"qwen3.5:2b": ["completion", "tools"], "mistral:7b": ["completion"], "granite-embedding:278m": ["embedding"]}
    monkeypatch.setattr(ollama_models, "_ollama",
                        lambda path, payload=None: tags if path == "/api/tags" else {"capabilities": caps[payload["model"]]})
    assert models.language_models() == {
        "chat": ["mistral:7b", "qwen3.5:2b"], "tools": ["qwen3.5:2b"], "embedding": ["granite-embedding:278m"],
    }


def test_language_models_without_ollama(monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("aus")

    monkeypatch.setattr(ollama_models, "_ollama", offline)
    assert models.language_models() == {"chat": [], "tools": [], "embedding": []}
