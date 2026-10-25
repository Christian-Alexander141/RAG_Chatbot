from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from Agents import chat_agent
from rag_chatpipline import api_server, config


@pytest.fixture(autouse=True)
def no_real_chat_model(monkeypatch):
    """Tests laden nie echte Ollama-Modelle; ohne Chat-Modell antwortet der Agent direkt."""
    monkeypatch.setattr(chat_agent, "_chat_model", False)


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted_agent():
    model = ScriptedModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "generate_image", "args": {"prompt": "a cat"}, "id": "c1"}]),
            AIMessage(content="Hier ist dein Bild."),
            AIMessage(content="Gern geschehen."),
        ]
    )
    return chat_agent.build_agent(model=model)


def test_agent_runs_tool_and_collects_images_per_turn(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    agent = scripted_agent()

    first = chat_agent.chat("Mal eine Katze", agent=agent)
    assert first["text"] == "Hier ist dein Bild."
    assert len(first["images"]) == 1 and (config.IMAGES_DIR / first["images"][0]).exists()

    # zweiter Zug im selben Chat: kein altes Bild wird erneut gemeldet
    second = chat_agent.chat("Danke", chat_id=first["chat_id"], agent=agent)
    assert second["chat_id"] == first["chat_id"] and second["images"] == []


def test_api_chat_and_documents(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(chat_agent, "_agent", scripted_agent())
    monkeypatch.setattr(api_server.knowledge_base, "_vector_store", vector_store)
    client = TestClient(api_server.app)

    reply = client.post("/chat", json={"message": "Mal eine Katze"}).json()
    assert reply["text"] == "Hier ist dein Bild." and reply["chatId"]
    # /images ist beim Import fest auf den echten Workspace gemountet, daher nur URL und Datei prüfen
    assert reply["images"][0].startswith("/images/")
    assert (config.IMAGES_DIR / Path(reply["images"][0]).name).exists()

    upload = client.post("/documents", files=[("files", ("../../evil/faq.txt", b"Kaffee ist kostenlos.", "text/plain"))])
    assert upload.json() == [{"name": "faq.txt", "chunks": 1}]
    assert (config.DOCUMENTS_DIR / "faq.txt").exists()
    assert client.get("/documents").json() == [{"name": "faq.txt", "chunks": 1}]
    assert client.post("/documents", files=[("files", ("x.exe", b"MZ"))]).status_code == 400

    assert client.delete("/documents/faq.txt").status_code == 204
    assert client.get("/documents").json() == []
    assert not (config.DOCUMENTS_DIR / "faq.txt").exists()
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307 and root.headers["location"] == "/ui"
    assert client.get("/ui/").status_code == 200  # Chainlit ist eingebunden


def test_retrieved_documents_are_injected_into_system_prompt(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    doc = config.DOCUMENTS_DIR / "fuhrpark.txt"
    doc.write_text("Firmenwagen buchen nur Abteilungsleiter.", encoding="utf-8")
    chat_agent.knowledge_base.ingest_file(doc)

    seen = []

    class RecordingModel(ScriptedModel):
        def _generate(self, messages, *args, **kwargs):
            seen.append(messages[0].content)
            return super()._generate(messages, *args, **kwargs)

    agent = chat_agent.build_agent(model=RecordingModel(responses=[AIMessage(content="ok")]))
    chat_agent.chat("Wer bucht den Firmenwagen?", agent=agent)

    assert "Firmenwagen buchen nur Abteilungsleiter." in seen[0]
    assert "fuhrpark.txt" in seen[0]


def test_chat_model_writes_final_answer_from_documents_and_tool_results(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    doc = config.DOCUMENTS_DIR / "fuhrpark.txt"
    doc.write_text("Firmenwagen buchen nur Abteilungsleiter.", encoding="utf-8")
    chat_agent.knowledge_base.ingest_file(doc)

    prompts = []

    class Writer(ScriptedModel):
        def _generate(self, messages, *args, **kwargs):
            prompts.append(messages)
            return super()._generate(messages, *args, **kwargs)

    writer = Writer(responses=[AIMessage(content="Teuken-Antwort 1"), AIMessage(content="Teuken-Antwort 2")])
    agent = scripted_agent()

    first = chat_agent.chat("Mal eine Katze und wer bucht den Firmenwagen?", agent=agent, chat_model=writer)
    assert first["text"] == "Teuken-Antwort 1" and len(first["images"]) == 1
    material = prompts[0][-1].content
    assert "Abteilungsleiter" in material and "generate_image" in material and "Hier ist dein Bild." in material

    chat_agent.chat("Danke", chat_id=first["chat_id"], agent=agent, chat_model=writer)
    # der zweite Aufruf sieht den bisherigen Verlauf (Frage + Teuken-Antwort)
    assert any(getattr(m, "content", "") == "Teuken-Antwort 1" for m in prompts[1])


def test_chat_stream_reports_progress_and_streams_tokens(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    writer = ScriptedModel(responses=[AIMessage(content="Fertig.")])

    events = list(chat_agent.chat_stream("Mal eine Katze", agent=scripted_agent(), chat_model=writer))
    kinds = [e["event"] for e in events]

    assert kinds[:4] == ["start", "retrieval_start", "retrieval_end", "agent_start"]
    assert kinds.index("tool_start") < kinds.index("tool_end") < kinds.index("write_start") < kinds.index("token")
    assert next(e for e in events if e["event"] == "tool_start")["name"] == "generate_image"
    assert kinds[-1] == "done" and events[-1]["result"]["text"] == "Fertig."
    assert len(events[-1]["result"]["images"]) == 1


def test_seed_history_restores_context_once(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    seen = []

    class Recorder(ScriptedModel):
        def _generate(self, messages, *args, **kwargs):
            seen.append([m.content for m in messages])
            return super()._generate(messages, *args, **kwargs)

    agent = chat_agent.build_agent(model=Recorder(responses=[AIMessage(content="Alex.")]))
    chat_agent.seed_history("alt", [("user", "Ich heiße Alex."), ("assistant", "Freut mich.")], agent=agent)
    chat_agent.seed_history("alt", [("user", "NICHT ÜBERSCHREIBEN"), ("assistant", "x")], agent=agent)
    chat_agent.chat("Wie heiße ich?", chat_id="alt", agent=agent)

    assert "Ich heiße Alex." in seen[0] and "Freut mich." in seen[0]
    assert "NICHT ÜBERSCHREIBEN" not in seen[0]
