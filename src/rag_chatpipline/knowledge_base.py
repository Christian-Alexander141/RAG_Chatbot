"""Wissensbasis: eigene Dokumente einlesen, in Qdrant indexieren und durchsuchen."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

TEXT_SUFFIXES = {".txt", ".md", ".mdx", ".json", ".csv", ".html", ".xml", ".yml", ".yaml", ".log"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}

_embeddings = None
_vector_store = None


class KnowledgeBaseError(RuntimeError):
    pass


def require_ollama_model(model: str) -> None:
    """Prüft, ob das Modell lokal vorhanden ist. Lädt bewusst nichts herunter."""
    import ollama

    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    try:
        client.show(model)
    except ollama.ResponseError as exc:
        raise KnowledgeBaseError(
            f"Das Ollama-Modell '{model}' ist nicht installiert. Installieren mit: ollama pull {model}"
        ) from exc
    except Exception as exc:
        raise KnowledgeBaseError(f"Ollama ist unter {config.OLLAMA_BASE_URL} nicht erreichbar: {exc}") from exc


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings

        require_ollama_model(config.EMBEDDING_MODEL)
        _embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        embeddings = get_embeddings()
        config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(config.QDRANT_PATH))
        size = len(embeddings.embed_query("dimension probe"))

        if not client.collection_exists(config.COLLECTION_NAME):
            client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
        else:
            existing = client.get_collection(config.COLLECTION_NAME).config.params.vectors.size
            if existing != size:
                raise KnowledgeBaseError(
                    f"Die Collection '{config.COLLECTION_NAME}' hat Vektorgröße {existing}, das Embedding-Modell "
                    f"'{config.EMBEDDING_MODEL}' liefert {size}. Bitte {config.QDRANT_PATH} löschen und neu indexieren."
                )

        _vector_store = QdrantVectorStore(client=client, collection_name=config.COLLECTION_NAME, embedding=embeddings)
    return _vector_store


def load_file(path: Path, source: Optional[str] = None) -> list[Document]:
    """Liest eine Datei und gibt sie als Dokumente zurück (PDF: eine Seite pro Dokument)."""
    source = source or path.name
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages)]
        return [
            Document(page_content=text, metadata={"source": source, "page": number})
            for number, text in pages
            if text.strip()
        ]

    if suffix == ".docx":
        from docx import Document as DocxDocument

        text = "\n".join(p.text for p in DocxDocument(path).paragraphs)
    elif suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise KnowledgeBaseError(f"Dateityp '{suffix}' wird nicht unterstützt (erlaubt: {sorted(SUPPORTED_SUFFIXES)}).")

    return [Document(page_content=text, metadata={"source": source})] if text.strip() else []


def ingest_file(path: Path, source: Optional[str] = None, vector_store: Any = None) -> int:
    """Indexiert eine Datei neu (vorhandene Chunks derselben Quelle werden ersetzt). Gibt die Chunk-Anzahl zurück."""
    source = source or path.name
    store = vector_store or get_vector_store()
    documents = load_file(path, source)
    if not documents:
        raise KnowledgeBaseError(f"'{source}' enthält keinen lesbaren Text.")

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(documents)
    delete_source(source, vector_store=store)
    store.add_documents(chunks)
    return len(chunks)


def add_document(source_path: Path, name: str, vector_store: Any = None) -> int:
    """Legt eine Datei im Dokumentenordner ab und indexiert sie. Bei Fehlern wird nichts zurückgelassen."""
    name = Path(name).name  # nie Pfadanteile aus einem Upload übernehmen
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise KnowledgeBaseError(f"'{name}': Dateityp nicht unterstützt (erlaubt: {sorted(SUPPORTED_SUFFIXES)}).")
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.DOCUMENTS_DIR / name
    if source_path.resolve() != target.resolve():
        shutil.copyfile(source_path, target)
    try:
        return ingest_file(target, name, vector_store=vector_store)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def remove_document(name: str, vector_store: Any = None) -> None:
    name = Path(name).name
    delete_source(name, vector_store=vector_store)
    (config.DOCUMENTS_DIR / name).unlink(missing_ok=True)


def ingest_directory(directory: Path = config.DOCUMENTS_DIR, vector_store: Any = None) -> dict[str, int]:
    """Indexiert alle unterstützten Dateien eines Ordners. Gibt {Quelle: Chunks} zurück."""
    result: dict[str, int] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            source = str(path.relative_to(directory))
            try:
                result[source] = ingest_file(path, source, vector_store=vector_store)
            except KnowledgeBaseError:
                continue
    return result


def _source_filter(source: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))])


def delete_source(source: str, vector_store: Any = None) -> None:
    store = vector_store or get_vector_store()
    store.client.delete(collection_name=store.collection_name, points_selector=_source_filter(source))


def list_sources(vector_store: Any = None) -> dict[str, int]:
    """Gibt {Quelle: Anzahl Chunks} aller indexierten Dokumente zurück."""
    store = vector_store or get_vector_store()
    counts: dict[str, int] = {}
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name, limit=256, offset=offset, with_payload=["metadata"]
        )
        for point in points:
            source = (point.payload or {}).get("metadata", {}).get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
        if offset is None:
            return counts


def search(query: str, k: int = 4, vector_store: Any = None) -> list[Document]:
    store = vector_store or get_vector_store()
    return store.similarity_search(query, k=k)
