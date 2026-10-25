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
