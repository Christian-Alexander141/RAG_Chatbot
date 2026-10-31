import pytest
from fastapi.testclient import TestClient

from Agents import chat_agent, document_lookup
from rag_chatpipline import api_server, config
from rag_chatpipline import knowledge_base as kb

from test_agent_api import ScriptedModel

SOURCES = ["Reisekostenrichtlinie_2023.pdf", "Reisekostenrichtlinie_2025.pdf", "Urlaubsantrag.docx",
           "IT-Sicherheitsrichtlinie.pdf", "Homeoffice-Vereinbarung.docx", "faq.txt"]


@pytest.mark.parametrize("message, expected", [
    ("Wo finde ich den Urlaubsantrag?", ["Urlaubsantrag.docx"]),
    ("Zeig mir die IT Sicherheitsrichtlinie", ["IT-Sicherheitsrichtlinie.pdf"]),
    ("Kannst du mir die Homeoffice Vereinbarung schicken?", ["Homeoffice-Vereinbarung.docx"]),
    ("Gib mir die Reisekostenrichtlinie von 2023", ["Reisekostenrichtlinie_2023.pdf"]),
    # zwei Fassungen und keine Jahreszahl in der Frage: beide anbieten
    ("Ich brauche die Reisekosten-Richtlinie", ["Reisekostenrichtlinie_2023.pdf", "Reisekostenrichtlinie_2025.pdf"]),
    ("gib mir die reisekostenrichtline", ["Reisekostenrichtlinie_2023.pdf", "Reisekostenrichtlinie_2025.pdf"]),  # Tippfehler
    ("Zeig mir die FAQ", ["faq.txt"]),
])
def test_document_requests_find_the_document(message, expected):
    assert document_lookup.find_documents(message, SOURCES) == expected


@pytest.mark.parametrize("message", [
    "Was steht in der Reisekostenrichtlinie zu Hotels?",  # Inhaltsfrage: an die RAG-Suche
    "Wie hoch ist die Kilometerpauschale laut Reisekostenrichtlinie?",
    "Wer bucht den Firmenwagen?",
    "Zeig mir die Datenschutzrichtlinie",  # gibt es nicht
    "Gib mir die Richtlinie",  # zu ungenau: "Richtlinie" deckt keinen Dateinamen ab
])
def test_other_messages_are_left_to_the_agent(message):
    assert document_lookup.find_documents(message, SOURCES) == []


def test_document_request_is_answered_without_language_model(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(kb, "_vector_store", vector_store)
    doc = config.DOCUMENTS_DIR / "Urlaubsantrag.txt"
    doc.write_text("Antrag auf Erholungsurlaub", encoding="utf-8")
    kb.ingest_file(doc, vector_store=vector_store)

    # Weder Werkzeug-Agent noch Chat-Modell dürfen antworten.
    agent = chat_agent.build_agent(model=ScriptedModel(responses=[]))
    writer = ScriptedModel(responses=[])
    events = list(chat_agent.chat_stream("Gib mir bitte den Urlaubsantrag", agent=agent, chat_model=writer))
    kinds = [e["event"] for e in events]

    assert kinds == ["start", "lookup_start", "tool_start", "tool_end", "token", "done"]
    assert "[Urlaubsantrag.txt](/files/Urlaubsantrag.txt)" in events[-1]["result"]["text"]


def test_files_route_serves_documents_but_nothing_outside(workspace):
    (config.DOCUMENTS_DIR / "faq.txt").write_text("Kaffee ist kostenlos.", encoding="utf-8")
    (workspace / "geheim.txt").write_text("nicht ausliefern", encoding="utf-8")
    client = TestClient(api_server.app)

    response = client.get("/files/faq.txt")
    assert response.status_code == 200 and response.text == "Kaffee ist kostenlos."
    assert response.headers["content-disposition"].startswith("inline")
    assert client.get("/files/..%2Fgeheim.txt").status_code == 404
    assert client.get("/files/fehlt.pdf").status_code == 404


def test_hybrid_search_finds_exact_terms(tmp_path, vector_store, monkeypatch):
    """Die Test-Embeddings sind Zufallsvektoren. Den richtigen Abschnitt findet hier nur BM25: Er landet sicher unter
    den ersten zwei (Platz 1 bei BM25; im ungünstigsten Fall gleichauf mit dem zufälligen Platz 1 der Embeddings)."""
    for i in range(30):
        filler = tmp_path / f"notiz{i}.txt"
        filler.write_text(f"Allgemeine Notiz Nummer {i} über Besprechungen und Termine.", encoding="utf-8")
        kb.ingest_file(filler, vector_store=vector_store)
    target = tmp_path / "wartung.txt"
    target.write_text("Die Pumpe ABX-4711 wird alle sechs Monate gewartet.", encoding="utf-8")
    kb.ingest_file(target, vector_store=vector_store)

    top = kb.search("Wartung ABX-4711", k=2, vector_store=vector_store)
    assert "wartung.txt" in [d.metadata["source"] for d in top]

    # Neue oder entfernte Dokumente sind sofort in der BM25-Suche berücksichtigt.
    kb.delete_source("wartung.txt", vector_store=vector_store)
    assert all(d.metadata["source"] != "wartung.txt" for d in kb.search("ABX-4711", k=5, vector_store=vector_store))


def test_bm25_ranks_rare_terms_and_ignores_stopwords():
    from rag_chatpipline.bm25 import BM25, tokenize

    assert tokenize("Wer bucht den Firmenwagen?") == ["bucht", "firmenwagen"]
    index = BM25({"a": "Firmenwagen buchen nur Abteilungsleiter", "b": "Urlaub beantragen", "c": "Firmenwagen Firmenwagen Tankkarte"})
    assert [doc_id for doc_id, _ in index.search("Tankkarte Firmenwagen", k=3)] == ["c", "a"]
