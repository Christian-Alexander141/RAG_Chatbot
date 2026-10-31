import pytest

from rag_chatpipline import knowledge_base as kb


def test_ingest_search_replace_and_delete(tmp_path, vector_store):
    doc = tmp_path / "urlaub.txt"
    doc.write_text("Urlaub muss vier Wochen vorher beantragt werden.", encoding="utf-8")

    assert kb.ingest_file(doc, vector_store=vector_store) == 1
    assert kb.list_sources(vector_store) == {"urlaub.txt": 1}
    assert "vier Wochen" in kb.search("Urlaub", vector_store=vector_store)[0].page_content

    # erneutes Indexieren ersetzt statt zu duplizieren
    kb.ingest_file(doc, vector_store=vector_store)
    assert kb.list_sources(vector_store) == {"urlaub.txt": 1}

    kb.delete_source("urlaub.txt", vector_store=vector_store)
    assert kb.list_sources(vector_store) == {}


def test_docx_and_pdf_are_loaded(tmp_path):
    from docx import Document as Docx

    path = tmp_path / "richtlinie.docx"
    word = Docx()
    word.add_paragraph("Passwörter müssen 14 Zeichen lang sein.")
    word.save(path)

    docs = kb.load_file(path)
    assert "14 Zeichen" in docs[0].page_content


def test_unsupported_and_empty_files_raise(tmp_path, vector_store):
    import pytest

    exe = tmp_path / "x.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(kb.KnowledgeBaseError):
        kb.load_file(exe)

    empty = tmp_path / "leer.txt"
    empty.write_text("  \n")
    with pytest.raises(kb.KnowledgeBaseError):
        kb.ingest_file(empty, vector_store=vector_store)


@pytest.fixture
def fake_ollama(monkeypatch, workspace):
    """Wissensbasis mit Embedding-Modellen ohne Ollama: jedes "Modell" liefert eine andere Vektorgröße."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from qdrant_client import QdrantClient

    sizes = {"modell-a": 16, "modell-b": 24}
    monkeypatch.setattr(kb, "_client", QdrantClient(":memory:"))
    monkeypatch.setattr(kb, "_vector_store", None)
    monkeypatch.setattr(kb, "_make_embeddings", lambda model: DeterministicFakeEmbedding(size=sizes[model]))
    monkeypatch.setattr(kb.ollama_models, "with_capability", lambda capability: sorted(sizes))
    monkeypatch.setattr(kb.config, "EMBEDDING_MODEL", "nicht-installiert")
    return sizes


def test_missing_default_embedding_model_falls_back_to_installed_one(fake_ollama):
    from rag_chatpipline import app_settings

    store = kb.get_vector_store()
    assert store.collection_name == "documents-modell-a"
    assert app_settings.get("embedding_model") == "modell-a"  # Wahl gemerkt, damit sie stabil bleibt


def test_switching_embedding_model_reads_documents_into_new_collection(fake_ollama):
    (kb.config.DOCUMENTS_DIR / "faq.txt").write_text("Kaffee ist kostenlos.", encoding="utf-8")
    kb.set_embedding_model("modell-a")
    assert kb.list_sources() == {"faq.txt": 1}

    assert kb.set_embedding_model("modell-b") == 1  # mit dem neuen Modell eingelesen
    assert kb.embedding_model() == "modell-b" and kb.get_vector_store().collection_name == "documents-modell-b"
    assert "kostenlos" in kb.search("Kaffee", k=1)[0].page_content

    # Hochladen und Entfernen wirken auf alle Modelle: zurück bei modell-a fehlt nichts und es bleibt nichts übrig.
    neu = kb.config.DOCUMENTS_DIR / "neu.txt"
    neu.write_text("Neues Dokument.", encoding="utf-8")
    kb.add_document(neu, "neu.txt")
    kb.remove_document("faq.txt")
    assert kb.set_embedding_model("modell-a") == 1  # nur neu.txt fehlte dort
    assert kb.list_sources() == {"neu.txt": 1}


def test_same_model_ignores_latest_tag():
    assert kb.same_model("nomic-embed-text", "nomic-embed-text:latest")
    assert not kb.same_model("nomic-embed-text", "granite-embedding:278m")
