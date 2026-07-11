from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import shutil
import subprocess
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "ollama:qwen3.5:2b"
Embeddings_MODEL = "jina-embeddings-v5-text-small"

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Dateien"
DEFAULT_QDRANT_PATH = DEFAULT_MEMORY_DIR / "qdrant"

_embeddings = None
_vector_store = None
_llm_model_ready = False
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".mdx", ".json", ".csv", ".html", ".xml", ".yml", ".yaml", ".log"}


def _resolve_memory_dir(memory_dir: Optional[str | Path] = None) -> Path:
    path = Path(memory_dir) if memory_dir is not None else DEFAULT_MEMORY_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_chat_session(
    chat_id: Optional[str] = None,
    memory_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Erstellt eine neue Chat-Sitzung mit eigenem Markdown-Gedächtnis."""
    base_dir = _resolve_memory_dir(memory_dir)
    resolved_chat_id = chat_id or f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    chat_dir = base_dir / resolved_chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)

    soul_path = chat_dir / "soul.md"
    if not soul_path.exists():
        soul_path.write_text(
            f"# Soul Memory\n\nChat-ID: {resolved_chat_id}\n\n",
            encoding="utf-8",
        )

    return {
        "chat_id": resolved_chat_id,
        "memory_dir": str(base_dir),
        "memory_file": str(soul_path),
    }


def append_exchange(
    chat_id: str,
    user_message: str,
    assistant_message: str,
    memory_dir: Optional[str | Path] = None,
) -> Path:
    """Hängt einen neuen Austausch an die Markdown-Gedächtnisdatei an."""
    session = create_chat_session(chat_id=chat_id, memory_dir=memory_dir)
    soul_path = Path(session["memory_file"])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = (
        f"\n## {timestamp}\n\n"
        f"**User:** {user_message.strip()}\n\n"
        f"**Assistant:** {assistant_message.strip()}\n"
    )
    with soul_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return soul_path


def read_memory(chat_id: str, memory_dir: Optional[str | Path] = None) -> str:
    """Liest das bestehende Gedächtnis einer Chat-Sitzung ein."""
    session = create_chat_session(chat_id=chat_id, memory_dir=memory_dir)
    memory_file = Path(session["memory_file"])
    if memory_file.exists():
        return memory_file.read_text(encoding="utf-8")
    return ""


def ensure_ollama_model(model_name: Optional[str] = None) -> bool:
    """Stellt sicher, dass das Ollama-Modell lokal vorhanden ist."""
    global _llm_model_ready
    if _llm_model_ready:
        return True

    resolved_model = model_name or (LLM_MODEL.split(":", 1)[1] if LLM_MODEL.startswith("ollama:") else LLM_MODEL)
    if shutil.which("ollama") is None:
        return False

    try:
        list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if list_result.returncode == 0 and resolved_model in list_result.stdout:
            _llm_model_ready = True
            return True

        pull_result = subprocess.run(["ollama", "pull", resolved_model], capture_output=True, text=True, check=False)
        _llm_model_ready = pull_result.returncode == 0
        return _llm_model_ready
    except Exception:
        return False


def _load_documents_from_directory(data_dir: Optional[str | Path] = None) -> list[Document]:
    """Liest Textdateien aus einem Ordner und erzeugt Dokumente."""
    source_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not source_dir.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
        except Exception:
            continue

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(file_path.relative_to(source_dir))},
            )
        )

    return documents


def ensure_embeddings_index(
    data_dir: Optional[str | Path] = None,
    vector_store: Any = None,
    force_rebuild: bool = False,
) -> bool:
    """Prüft, ob bereits Embeddings vorhanden sind, und ingested die Daten aus Dateien andernfalls."""
    store = vector_store or get_vector_store()
    if store is None:
        return False

    if not force_rebuild:
        try:
            docs = store.similarity_search("test", k=1)
            if docs:
                return True
        except Exception:
            pass

    documents = _load_documents_from_directory(data_dir)
    if not documents:
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return False

    try:
        store.add_documents(chunks)
        return True
    except Exception:
        return False


def get_embeddings():
    """Lädt die Embeddings nur bei Bedarf und cachet sie."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            return None

        for candidate_model in [Embeddings_MODEL, "sentence-transformers/all-MiniLM-L6-v2"]:
            try:
                _embeddings = HuggingFaceEmbeddings(
                    model_name=candidate_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
                break
            except Exception:
                continue

    return _embeddings


def get_vector_store():
    """Erzeugt einen persistenten Qdrant-Store, falls möglich."""
    global _vector_store
    if _vector_store is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from langchain_qdrant import QdrantVectorStore
        except Exception:  # pragma: no cover - defensive fallback
            return None

        embeddings = get_embeddings()
        if embeddings is None:
            return None

        try:
            DEFAULT_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(str(DEFAULT_QDRANT_PATH))
            vector_size = len(embeddings.embed_query("sample text"))

            if not client.collection_exists("test"):
                client.create_collection(
                    collection_name="test",
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

            _vector_store = QdrantVectorStore(
                client=client,
                collection_name="test",
                embedding=embeddings,
            )
        except Exception:
            _vector_store = None

    return _vector_store


def _retrieve_context(query: str, vector_store: Any = None, top_k: int = 3) -> list[str]:
    """Ruft relevante Kontextstücke aus dem Vektor-Store ab."""
    store = vector_store or get_vector_store()
    if store is None:
        return []

    try:
        docs = store.similarity_search(query, k=top_k)
    except Exception:
        return []

    return [doc.page_content for doc in docs if getattr(doc, "page_content", None)]


def _build_prompt(query: str, memory_text: str, context_text: str) -> str:
    """Baut den Prompt für die LLM-Antwort aus Gedächtnis und Kontext zusammen."""
    return f"""Du bist ein hilfreicher Assistent. Nutze das verfügbare Gedächtnis und den relevanten Kontext, um die Frage zu beantworten. Wenn etwas fehlt, ignoriere es einfach.\n\nGedächtnis:\n{memory_text or 'Kein vorheriges Gedächtnis verfügbar.'}\n\nKontext:\n{context_text or 'Kein weiterer Kontext verfügbar.'}\n\nNutzerfrage: {query}\n\nAntworte auf Deutsch, präzise und hilfreich."""


def _default_llm(prompt: str) -> str:
    """Fallback-LLM, falls kein externer Provider verfügbar ist."""
    ensure_ollama_model()
    try:
        from langchain_ollama import Ollama
    except Exception:
        return f"LLM-Fallback: {prompt}"

    llm = Ollama(model=LLM_MODEL, temperature=0.2)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def call_rag_pipeline(
    query: str,
    chat_id: Optional[str] = None,
    memory_dir: Optional[str | Path] = None,
    llm: Optional[Callable[[str], str]] = None,
    top_k: int = 3,
    vector_store: Any = None,
    data_dir: Optional[str | Path] = None,
) -> str:
    """Führt die RAG-Pipeline aus und speichert die Interaktion im Chat-Gedächtnis."""
    session = create_chat_session(chat_id=chat_id, memory_dir=memory_dir)
    ensure_ollama_model()
    ensure_embeddings_index(data_dir=data_dir, vector_store=vector_store)

    memory_text = read_memory(session["chat_id"], memory_dir=memory_dir)
    context_text = "\n\n".join(_retrieve_context(query, vector_store=vector_store, top_k=top_k))
    prompt = _build_prompt(query, memory_text, context_text)

    answer = llm(prompt) if llm is not None else _default_llm(prompt)
    answer = str(answer)
    append_exchange(session["chat_id"], query, answer, memory_dir=memory_dir)
    return answer
