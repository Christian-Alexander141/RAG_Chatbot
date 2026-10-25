"""Chat-Agent mit zwei Modellrollen, beide über Ollama.

- Agent-Modell (AGENT_MODEL, z. B. qwen3:1.7b): ruft die Tools auf (Dateisystem, Bild, Suche).
- Chat-Modell (CHAT_MODEL, z. B. Teuken): formuliert aus Frage, Dokumenttreffern und Tool-Ergebnissen
  die endgültige Antwort. Sind beide Modelle gleich, entfällt dieser zweite Schritt.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from rag_chatpipline import config, knowledge_base

from .tools import TOOLS, retrieve_context

logger = logging.getLogger(__name__)

AGENT_PROMPT = """Du bist der Werkzeug-Agent des internen Firmenassistenten und läufst vollständig lokal.
Antworte auf Deutsch, kurz und sachlich. Deine Antwort geht an einen Redakteur, der daraus die Antwort
für den Nutzer schreibt.

Werkzeuge:
- Firmenwissen: Passende Dokumentauszüge stehen unten im Abschnitt "Treffer aus den Firmendokumenten".
  Beantworte Wissensfragen auf Basis dieser Treffer und nenne die Quelle (Dateiname). Passt nichts,
  sage das offen, statt zu raten. Mit search_documents kannst du gezielt nachsuchen.
- Dateisystem-Tools (list_files, read_file, write_file, make_directory, search_files): Du arbeitest nur
  im Workspace. Überschreibe Dateien nur, wenn der Nutzer es ausdrücklich verlangt.
- generate_image: Für Bildwünsche. Schreibe den Prompt auf Englisch. Das Bild wird dem Nutzer
  automatisch angezeigt."""

CHAT_PROMPT = """Du bist der interne Assistent des Unternehmens und läufst vollständig lokal.
Antworte immer auf Deutsch, freundlich, sachlich und präzise.

Zur aktuellen Nutzerfrage bekommst du Material: Treffer aus den Firmendokumenten, Ergebnisse von
Werkzeugen (Dateien, Bilder) und den Entwurf eines Hilfsagenten. Schreibe daraus die endgültige Antwort.
Erfinde nichts dazu. Nenne bei Firmenwissen die Quelle (Dateiname). Gibt das Material nichts her, sage
das offen. Wurde ein Bild erzeugt, erwähne nur kurz, dass es unten angezeigt wird."""

MAX_TOOL_OUTPUT = 1500
MAX_HISTORY_MESSAGES = 20

_agent = None
_chat_model = None
_histories: dict[str, list[BaseMessage]] = {}


@dataclass
class TurnContext:
    """Pro Anfrage einmal gesucht und von Agent und Chat-Modell gemeinsam genutzt."""

    documents: str = ""


@dynamic_prompt
def add_retrieved_context(request: ModelRequest) -> str:
    """Hängt die Dokumenttreffer an den Prompt. Kleine Modelle rufen ein Suchtool oft nicht selbst auf."""
    context = getattr(request.runtime, "context", None)
    documents = getattr(context, "documents", "")
    if not documents:
        return AGENT_PROMPT
    return f"{AGENT_PROMPT}\n\nTreffer aus den Firmendokumenten:\n{documents}"


def build_agent(model: Any = None, checkpointer: Any = None, tools: Optional[list] = None):
    if model is None:
        from langchain_ollama import ChatOllama

        knowledge_base.require_ollama_model(config.AGENT_MODEL)
        model = ChatOllama(
            model=config.AGENT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.3,
            reasoning=config.AGENT_REASONING,
        )
    return create_agent(
        model=model,
        tools=TOOLS if tools is None else tools,
        middleware=[add_retrieved_context],
        context_schema=TurnContext,
        checkpointer=checkpointer or InMemorySaver(),
    )


def build_chat_model():
    """Chat-Modell für die Schlussantwort, oder None, wenn das Agent-Modell selbst antwortet."""
    if config.CHAT_MODEL == config.AGENT_MODEL:
        return None
    from langchain_ollama import ChatOllama

    knowledge_base.require_ollama_model(config.CHAT_MODEL)
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0.3)


def get_agent():
    global _agent
    if _agent is None:
        config.ensure_dirs()
        _agent = build_agent()
    return _agent


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        _chat_model = build_chat_model() or False  # False = bewusst kein zweites Modell
    return _chat_model or None


def _writer_messages(history: list[BaseMessage], question: str, documents: str,
                     tool_outputs: list[str], draft: str) -> list[BaseMessage]:
    sections = [f"Nutzerfrage: {question}"]
    if documents:
        sections.append(f"Treffer aus den Firmendokumenten:\n{documents}")
    if tool_outputs:
        sections.append("Ergebnisse der Werkzeuge:\n" + "\n".join(tool_outputs))
    sections.append(f"Entwurf des Hilfsagenten:\n{draft}")
    sections.append("Schreibe jetzt die endgültige Antwort an den Nutzer.")
    return [SystemMessage(CHAT_PROMPT), *history, HumanMessage("\n\n".join(sections))]


def seed_history(chat_id: str, exchanges: list[tuple[str, str]], agent: Any = None) -> None:
    """Stellt den Verlauf eines gespeicherten Chats wieder her (nach Neustart oder beim Öffnen eines alten Chats).

    exchanges sind (Rolle, Text) mit Rolle "user" oder "assistant". Ein laufender Verlauf wird nicht überschrieben.
    """
    agent = agent or get_agent()
    thread = {"configurable": {"thread_id": chat_id}}
    if agent.get_state(thread).values.get("messages"):
        return
    messages = [HumanMessage(text) if role == "user" else AIMessage(text) for role, text in exchanges if text.strip()]
    if not messages:
        return
    agent.update_state(thread, {"messages": messages}, as_node="model")
    _histories.setdefault(chat_id, messages[-MAX_HISTORY_MESSAGES:])


def chat_stream(message: str, chat_id: Optional[str] = None, agent: Any = None,
                chat_model: Any = None) -> Iterator[dict[str, Any]]:
    """Wie chat(), liefert aber Ereignisse, sobald sie eintreten (Fortschritt und Antwort-Tokens).

    Ereignisse (Schlüssel "event"): start, retrieval_start, retrieval_end, agent_start, tool_start, tool_end,
    write_start, token, done. "done" enthält unter "result" das Ergebnis wie chat().
    """
    chat_id = chat_id or f"chat-{uuid.uuid4().hex[:12]}"
    agent = agent or get_agent()
    chat_model = chat_model or get_chat_model()
    yield {"event": "start", "chat_id": chat_id}

    yield {"event": "retrieval_start"}
    try:
        documents = retrieve_context(message, k=3)
    except Exception:
        logger.warning("Wissensbasis nicht verfügbar", exc_info=True)
        documents = ""
    yield {"event": "retrieval_end", "documents": documents}

    yield {"event": "agent_start"}
    turn: list[BaseMessage] = []
    for update in agent.stream(
        {"messages": [HumanMessage(message)]},
        config={"configurable": {"thread_id": chat_id}},
        context=TurnContext(documents=documents),
        stream_mode="updates",
    ):
        for node_output in update.values():
            new_messages = node_output.get("messages") if isinstance(node_output, dict) else None
            for m in new_messages if isinstance(new_messages, list) else []:
                turn.append(m)
                if isinstance(m, AIMessage):
                    for call in m.tool_calls:
                        yield {"event": "tool_start", "id": call["id"], "name": call["name"], "args": call["args"]}
                elif isinstance(m, ToolMessage):
                    yield {"event": "tool_end", "id": m.tool_call_id, "name": m.name, "output": str(m.content)}

    tool_messages = [m for m in turn if isinstance(m, ToolMessage)]
    images = [m.artifact["name"] for m in tool_messages if isinstance(m.artifact, dict) and "name" in m.artifact]
    draft = next((str(m.content) for m in reversed(turn) if isinstance(m, AIMessage)), "")

    if chat_model is None:
        text = draft
        yield {"event": "token", "text": text}
    else:
        yield {"event": "write_start"}
        tool_outputs = [f"- {m.name}: {str(m.content)[:MAX_TOOL_OUTPUT]}" for m in tool_messages]
        history = _histories.setdefault(chat_id, [])
        parts: list[str] = []
        for chunk in chat_model.stream(_writer_messages(history, message, documents, tool_outputs, draft)):
            token = str(chunk.content)
            if token:
                parts.append(token)
                yield {"event": "token", "text": token}
        text = "".join(parts).strip()
        if not text:  # Schreibmodell hat nichts geliefert: Entwurf des Agenten anzeigen
            text = draft
            yield {"event": "token", "text": text}
        history += [HumanMessage(message), AIMessage(text)]
        del history[:-MAX_HISTORY_MESSAGES]

    yield {"event": "done", "result": {"chat_id": chat_id, "text": text, "images": images}}


def chat(message: str, chat_id: Optional[str] = None, agent: Any = None, chat_model: Any = None) -> dict[str, Any]:
    """Schickt eine Nachricht durch Agent und Chat-Modell. Der Verlauf wird pro chat_id gehalten.

    Gibt {"chat_id", "text", "images"} zurück; images sind die in diesem Zug erzeugten Bilder.
    """
    for event in chat_stream(message, chat_id, agent, chat_model):
        if event["event"] == "done":
            return event["result"]
    raise RuntimeError("chat_stream ist ohne Ergebnis beendet worden")
