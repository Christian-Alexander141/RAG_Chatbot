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


def test_cloud_models_only_on_request_and_never_for_embeddings(monkeypatch):
    tags = {"models": [
        {"name": "qwen3.5:2b"}, {"name": "granite-embedding:278m"},
        {"name": "gpt-oss:120b-cloud", "remote_host": "https://ollama.com:443"},
        {"name": "embed-cloud", "remote_host": "https://ollama.com:443"},
    ]}
    caps = {"qwen3.5:2b": ["completion", "tools"], "granite-embedding:278m": ["embedding"],
            "gpt-oss:120b-cloud": ["completion", "tools"], "embed-cloud": ["embedding"]}
    monkeypatch.setattr(ollama_models, "_ollama",
                        lambda path, payload=None: tags if path == "/api/tags" else {"capabilities": caps[payload["model"]]})
    assert "gpt-oss:120b-cloud" not in models.language_models()["chat"]
    admin = models.language_models(include_cloud=True)
    assert admin["chat"] == admin["tools"] == ["gpt-oss:120b-cloud", "qwen3.5:2b"]
    assert admin["embedding"] == ["granite-embedding:278m"]  # sonst gingen alle Dokumente an fremde Server


def test_validated_keeps_only_offered_models():
    current = ModelChoice("mistral:7b", "qwen3.5:2b", "qwen3.5:2b", "FLUX.2-klein-4B")
    available = {"chat": ["mistral:7b", "qwen3.5:9b"], "agent": ["qwen3.5:2b"], "image_prompt": ["mistral:7b"],
                 "image": [], "video": []}
    # Gefälscht: Cloud-Modell und unbekanntes Bildmodell, obwohl nicht angeboten
    settings = {"chat": "qwen3.5:9b", "agent": "gpt-oss:120b-cloud", "image_prompt": "", "image": "anderes-bild"}
    choice = user_settings.validated(settings, available, current)
    assert choice == ModelChoice("qwen3.5:9b", "qwen3.5:2b", "qwen3.5:2b", "FLUX.2-klein-4B", current.video)


def test_a_broken_model_does_not_hide_the_others(monkeypatch):
    tags = {"models": [{"name": "qwen3.5:2b"}, {"name": "alt-cloud", "remote_host": "https://ollama.com:443"},
                       {"name": "gpt-oss:120b-cloud", "remote_host": "https://ollama.com:443"}]}

    def ollama(path, payload=None):
        if path == "/api/tags":
            return tags
        if payload["model"] == "alt-cloud":
            raise OSError("HTTP Error 410: Gone")  # Cloud-Modell gibt es nicht mehr
        return {"capabilities": ["completion"]}

    monkeypatch.setattr(ollama_models, "_ollama", ollama)
    assert models.language_models(include_cloud=True)["chat"] == ["gpt-oss:120b-cloud", "qwen3.5:2b"]
