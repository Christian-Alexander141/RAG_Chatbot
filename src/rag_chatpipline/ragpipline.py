from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import uuid

from . import config, knowledge_base

DEFAULT_MEMORY_DIR = config.MEMORY_DIR


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


def _retrieve_context(query: str, vector_store: Any = None, top_k: int = 3) -> list[str]:
    """Ruft relevante Kontextstücke aus der Wissensbasis ab."""
    try:
        docs = knowledge_base.search(query, k=top_k, vector_store=vector_store)
    except Exception:
        return []
    return [doc.page_content for doc in docs if getattr(doc, "page_content", None)]


def _build_prompt(query: str, memory_text: str, context_text: str) -> str:
    """Baut den Prompt für die LLM-Antwort aus Gedächtnis und Kontext zusammen."""
    return f"""Du bist ein hilfreicher Assistent. Nutze das verfügbare Gedächtnis und den relevanten Kontext, um die Frage zu beantworten. Wenn etwas fehlt, ignoriere es einfach.\n\nGedächtnis:\n{memory_text or 'Kein vorheriges Gedächtnis verfügbar.'}\n\nKontext:\n{context_text or 'Kein weiterer Kontext verfügbar.'}\n\nNutzerfrage: {query}\n\nAntworte auf Deutsch, präzise und hilfreich."""


def _default_llm(prompt: str) -> str:
    """Beantwortet den Prompt direkt mit dem konfigurierten Ollama-Modell."""
    from langchain_ollama import ChatOllama

    knowledge_base.require_ollama_model(config.CHAT_MODEL)
    llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0.2)
    return str(llm.invoke(prompt).content)


def call_rag_pipeline(
    query: str,
    chat_id: Optional[str] = None,
    memory_dir: Optional[str | Path] = None,
    llm: Optional[Callable[[str], str]] = None,
    top_k: int = 3,
    vector_store: Any = None,
) -> str:
    """Führt die RAG-Pipeline aus und speichert die Interaktion im Chat-Gedächtnis."""
    session = create_chat_session(chat_id=chat_id, memory_dir=memory_dir)

    memory_text = read_memory(session["chat_id"], memory_dir=memory_dir)
    context_text = "\n\n".join(_retrieve_context(query, vector_store=vector_store, top_k=top_k))
    prompt = _build_prompt(query, memory_text, context_text)

    answer = llm(prompt) if llm is not None else _default_llm(prompt)
    answer = str(answer)
    append_exchange(session["chat_id"], query, answer, memory_dir=memory_dir)
    return answer
