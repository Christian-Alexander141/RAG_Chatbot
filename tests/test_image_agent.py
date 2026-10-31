import pytest
from langchain_core.messages import AIMessage

from Agents import chat_agent, image_agent, image_backend
from Agents.models import ModelChoice
from rag_chatpipline import config

from test_agent_api import ScriptedModel


@pytest.mark.parametrize("text", [
    "Erstelle ein Bild von einem Hund am Strand",
    "kannst du mir bitte ein Foto von Berlin generieren?",
    "Ein Logo für unsere Firma erzeugen",
    "Zeichne eine Illustration eines Leuchtturms",
    "Mach mir ein Bild davon",
    "create an image of a red car",
])
def test_image_requests_are_detected(text):
    assert image_agent.is_image_request(text)


@pytest.mark.parametrize("text", [
    "Wer bucht den Firmenwagen?",
    "Erstelle eine Zusammenfassung von faq.txt",
    "Wie groß ist der Bildschirm im Besprechungsraum?",
    "Ich habe einmal ein Bild gesehen",
])
def test_other_messages_are_not_image_requests(text):
    assert not image_agent.is_image_request(text)


def test_write_prompt_uses_model_and_strips_extras():
    model = ScriptedModel(responses=[AIMessage(content='<think>hmm</think>Prompt: "a dog on a beach, golden hour"')])
    assert image_agent.write_prompt("Hund am Strand", model=model) == "a dog on a beach, golden hour"


def test_write_prompt_falls_back_to_request_on_error():
    class Broken(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("Ollama aus")

    assert image_agent.write_prompt("Hund am Strand", model=Broken(responses=[])) == "Hund am Strand"


def test_image_request_goes_to_image_agent_with_chosen_models(workspace, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(chat_agent, "get_chat_model", lambda *args: None)
    seen = {}
    monkeypatch.setattr(image_agent, "get_prompt_model",
                        lambda name=None: seen.setdefault("prompt_model", name) and None)
    real_generate = image_backend.generate_image
    monkeypatch.setattr(image_backend, "generate_image",
                        lambda prompt, model=None: seen.setdefault("image_model", model) and real_generate(prompt))

    # Der Werkzeug-Agent darf nicht antworten: Ein Bildwunsch läuft ausschließlich über den Bild-Agenten.
    agent = chat_agent.build_agent(model=ScriptedModel(responses=[]))
    models = ModelChoice(chat="c", agent="a", image_prompt="prompter", image="FLUX.2-klein-4B")
    events = list(chat_agent.chat_stream("Erstelle ein Bild von einem Hund", agent=agent, models=models))
    kinds = [e["event"] for e in events]

    assert "retrieval_start" not in kinds and "agent_start" not in kinds
    assert kinds[:4] == ["start", "image_agent_start", "tool_start", "tool_end"]
    assert seen == {"prompt_model": "prompter", "image_model": "FLUX.2-klein-4B"}
    result = events[-1]["result"]
    assert len(result["images"]) == 1 and (config.IMAGES_DIR / result["images"][0]).exists()
    # Der Werkzeug-Agent kennt den Zug für das weitere Gespräch.
    history = agent.get_state({"configurable": {"thread_id": result["chat_id"]}}).values["messages"]
    assert history[-2].content == "Erstelle ein Bild von einem Hund"


def test_missing_image_model_is_reported(workspace, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    content, artifact = image_agent.generate("a dog", "gibtsnicht")
    assert "fehlgeschlagen" in content and "gibtsnicht" in content and artifact == {}


def test_available_image_models_only_lists_complete_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "fertig").mkdir()
    (tmp_path / "fertig" / ".download-complete").touch()
    (tmp_path / "halb").mkdir()
    assert image_backend.available_image_models() == ["fertig"]
