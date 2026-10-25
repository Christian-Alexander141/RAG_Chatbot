"""Chainlit-Oberfläche für den Firmen-Chatbot.

- Linke Seitenleiste: alte Chats (Chainlit-Historie, gespeichert in SQLite)
- Rechte Seitenleiste "Wissensquellen": Dokumente hochladen, ansehen und entfernen
- Antworten werden gestreamt, die Arbeitsschritte des Agenten erscheinen als Schritte
"""
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict

from Agents.chat_agent import chat_stream, seed_history
from history_db import create_data_layer
from rag_chatpipline import knowledge_base

HELP = (
    "Ich beantworte Fragen zu euren Dokumenten, arbeite mit Dateien im Workspace und erzeuge Bilder.\n\n"
    "- **Dokumente** verwaltest du in der Seitenleiste *Wissensquellen* (rechts). Du kannst auch eine Datei "
    "an eine Nachricht anhängen.\n"
    "- **Alte Chats** findest du in der linken Seitenleiste."
)

TOOL_LABELS = {
    "search_documents": "Dokumente durchsuchen",
    "list_files": "Dateien auflisten",
    "read_file": "Datei lesen",
    "write_file": "Datei schreiben",
    "make_directory": "Ordner anlegen",
    "search_files": "Dateien suchen",
    "generate_image": "Bild erzeugen",
}


@cl.data_layer
def data_layer():
    return create_data_layer()


@cl.header_auth_callback
async def header_auth(headers) -> Optional[cl.User]:
    """Meldet ohne Login an. Hinter einem Reverse-Proxy mit SSO liefert dieser den Nutzer im Header
    X-Forwarded-User, jeder sieht dann nur seine eigenen Chats. Ohne Proxy teilen sich alle den Nutzer "lokal"."""
    return cl.User(identifier=headers.get("x-forwarded-user") or "lokal", metadata={"provider": "header"})


async def _open_documents_panel() -> None:
    await cl.ElementSidebar.set_title("Wissensquellen")
    await cl.ElementSidebar.set_elements([cl.CustomElement(name="DocumentsPanel", props={})])


@cl.on_chat_start
async def start() -> None:
    await _open_documents_panel()
    await cl.Message(HELP, actions=[cl.Action(name="open_documents", payload={}, label="Wissensquellen öffnen")]).send()


@cl.action_callback("open_documents")
async def open_documents(action: cl.Action) -> None:
    await _open_documents_panel()


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    """Alter Chat wird geöffnet: Gesprächsverlauf für Agent und Chat-Modell wiederherstellen."""
    exchanges = [
        (step["type"].removesuffix("_message"), step.get("output") or "")
        for step in thread.get("steps", [])
        if step.get("type") in ("user_message", "assistant_message")
    ]
    await cl.make_async(seed_history)(thread["id"], exchanges)
    await _open_documents_panel()


async def _ingest_attachments(message: cl.Message) -> None:
    for element in message.elements or []:
        if not getattr(element, "path", None):
            continue
        try:
            chunks = await cl.make_async(knowledge_base.add_document)(Path(element.path), element.name)
            await cl.Message(f"✅ **{element.name}** zur Wissensbasis hinzugefügt ({chunks} Abschnitte).").send()
        except Exception as exc:
            await cl.Message(f"⚠️ **{element.name}** konnte nicht aufgenommen werden: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _ingest_attachments(message)
    text = message.content.strip()
    if not text:
        return

    answer = cl.Message(content="")
    steps: dict[str, cl.Step] = {}
    current: Optional[cl.Step] = None
    result: Optional[dict] = None

    events = chat_stream(text, cl.context.session.thread_id)
    done = object()
    try:
        while (event := await cl.make_async(next)(events, done)) is not done:
            kind = event["event"]
            if kind == "retrieval_start":
                current = cl.Step(name="Wissensbasis durchsuchen", type="retrieval")
                await current.send()
            elif kind == "retrieval_end":
                current.output = "Dokumentauszüge als Kontext bereitgestellt." if event["documents"] else "Wissensbasis ist leer."
                await current.update()
            elif kind == "agent_start":
                current = cl.Step(name="Agent arbeitet", type="run")
                await current.send()
            elif kind == "tool_start":
                step = cl.Step(name=TOOL_LABELS.get(event["name"], event["name"]), type="tool", parent_id=current.id)
                step.input = str(event["args"])
                steps[event["id"]] = step
                await step.send()
            elif kind == "tool_end":
                step = steps[event["id"]]
                step.output = event["output"]
                await step.update()
            elif kind == "write_start":
                current.output = "Fertig."
                await current.update()
                current = cl.Step(name="Antwort formulieren", type="llm")
                await current.send()
            elif kind == "token":
                await answer.stream_token(event["text"])
            elif kind == "done":
                result = event["result"]
    except Exception as exc:
        await cl.Message(f"⚠️ Der Agent ist fehlgeschlagen: {exc}").send()
        return

    if current is not None:
        current.output = "Fertig."
        await current.update()

    # Bilder als Markdown in den Text: so bleiben sie auch in gespeicherten Chats erhalten.
    answer.content = result["text"] + "".join(f"\n\n![Bild](/images/{name})" for name in result["images"])
    answer.elements = [cl.CustomElement(name="SpeakButton", props={"text": result["text"]}, display="inline")]
    await answer.send()
