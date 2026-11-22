"""Chainlit-Oberfläche für den Firmen-Chatbot.

- Linke Seitenleiste: alte Chats (Chainlit-Historie, gespeichert in SQLite)
- Rechte Seitenleiste "Wissensquellen": Dokumente hochladen, ansehen und entfernen
- Antworten werden gestreamt, die Arbeitsschritte des Agenten erscheinen als Schritte
- Benutzermenü > Einstellungen: Sprach-, Bild- und Videomodelle für Chat und Agenten wählen (pro Nutzer gespeichert)
- Angehängte Dateien: Dokumente gehen in die Wissensbasis, alles andere (z. B. Bilder) nach workspace/Uploads/.
  Der Agent bekommt die Pfade mitgeteilt, damit eigene neuronale Netze oder der Video-Agent sie nutzen können.
"""
import shutil
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.input_widget import Select
from chainlit.types import ThreadDict

import user_settings
from Agents import neural_networks
from Agents.chat_agent import chat_stream, seed_history
from Agents.models import ModelChoice, image_models, language_models, video_models
from history_db import create_data_layer
from rag_chatpipline import config, knowledge_base

HELP = (
    "Ich beantworte Fragen zu euren Dokumenten, arbeite mit Dateien im Workspace und erzeuge Bilder und Videos.\n\n"
    "- **Dokumente** verwaltest du in der Seitenleiste *Wissensquellen* (rechts). Du kannst auch eine Datei "
    "an eine Nachricht anhängen. Ein Dokument direkt bekommst du z. B. mit *„Gib mir die Reisekostenrichtlinie“*.\n"
    "- **Videos** mit Ton: *„Erstelle ein Video von …“* oder nach einem Bild *„Animiere das Bild“*. "
    "Das dauert einige Minuten.\n"
    "- **Alte Chats** findest du in der linken Seitenleiste.\n"
    "- **Modelle** wählst du im Benutzermenü (oben rechts) unter *Einstellungen*."
)

SETTINGS_LABELS = {
    "chat": ("Chat-Modell", "Formuliert die Antworten an dich."),
    "agent": ("Werkzeug-Agent", "Durchsucht Dokumente und arbeitet mit Dateien. Nur Modelle mit Tool-Calling."),
    "image_prompt": ("Bild- und Video-Agent", "Schreibt aus deinem Bild- oder Videowunsch den Prompt für das Modell."),
    "image": ("Bildmodell", "Erzeugt die Bilder. Weitere Modelle legt man unter models/ ab."),
    "video": ("Videomodell", "Erzeugt die Videos mit Ton. Weitere Modelle legt man unter models/video/ ab."),
}
EMBEDDING_LABEL = (
    "Embedding-Modell (für alle Nutzer)",
    "Sucht in der Wissensbasis nach Bedeutung. Ein Wechsel liest alle Dokumente neu ein.",
)

TOOL_LABELS = {
    "search_documents": "Dokumente durchsuchen",
    "list_files": "Dateien auflisten",
    "read_file": "Datei lesen",
    "write_file": "Datei schreiben",
    "make_directory": "Ordner anlegen",
    "search_files": "Dateien suchen",
    "generate_image": "Bild erzeugen",
    "generate_video": "Video erzeugen",
    "find_document": "Dokument suchen",
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


def _user_id() -> str:
    user = cl.user_session.get("user")
    return user.identifier if user else "lokal"


async def _send_settings() -> None:
    """Zeigt die Modellwahl an (Dialog "Einstellungen") und merkt sich die Wahl für diese Sitzung."""
    llms = await cl.make_async(language_models)()
    available = {"chat": llms["chat"], "agent": llms["tools"], "image_prompt": llms["chat"], "image": image_models(),
                 "video": video_models()}
    choice = user_settings.resolve(user_settings.load(_user_id()), available)
    cl.user_session.set("models", choice)
    widgets = []
    for key, (label, description) in SETTINGS_LABELS.items():
        current = getattr(choice, key)
        values = available[key] or [current]  # ohne Ollama bzw. ohne heruntergeladenes Modell: nur die aktuelle Wahl
        widgets.append(Select(id=key, label=label, values=values, initial_value=current, description=description))

    # Das Embedding-Modell gehört zur gemeinsamen Wissensbasis, nicht zum Nutzer.
    current = knowledge_base.embedding_model()
    values = llms["embedding"] or [current]
    initial = next((v for v in values if knowledge_base.same_model(v, current)), values[0])
    label, description = EMBEDDING_LABEL
    widgets.append(Select(id="embedding", label=label, values=values, initial_value=initial, description=description))
    await cl.ChatSettings(widgets).send()


@cl.on_settings_update
async def settings_update(settings: dict) -> None:
    current = cl.user_session.get("models") or ModelChoice.defaults()
    choice = ModelChoice(**{key: settings.get(key) or getattr(current, key) for key in SETTINGS_LABELS})
    cl.user_session.set("models", choice)
    await cl.make_async(user_settings.save)(_user_id(), choice)

    embedding = settings.get("embedding")
    if embedding and not knowledge_base.same_model(embedding, knowledge_base.embedding_model()):
        note = cl.Message(f"Die Wissensbasis wird mit **{embedding}** eingelesen. Das gilt für alle Nutzer und kann "
                          "bei vielen Dokumenten etwas dauern …")
        await note.send()
        try:
            added = await cl.make_async(knowledge_base.set_embedding_model)(embedding)
            note.content = f"✅ Die Wissensbasis nutzt jetzt **{embedding}** ({added} Dokumente neu eingelesen)."
        except Exception as exc:
            note.content = f"⚠️ Das Embedding-Modell konnte nicht gewechselt werden: {exc}"
        await note.update()
        await _open_documents_panel()


@cl.on_chat_start
async def start() -> None:
    await _send_settings()
    await _open_documents_panel()
    text = HELP
    networks = await cl.make_async(neural_networks.labels)()
    if networks:
        text += ("\n- **Eigene neuronale Netze:** " + ", ".join(networks.values())
                 + ". Bilder oder Dateien dafür hängst du einfach an die Nachricht an.")
    await cl.Message(text, actions=[cl.Action(name="open_documents", payload={}, label="Wissensquellen öffnen")]).send()


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
    await _send_settings()
    models = cl.user_session.get("models")
    await cl.make_async(seed_history)(thread["id"], exchanges, None, models.agent)
    await _open_documents_panel()


def _save_upload(source: Path, name: str) -> Path:
    """Legt eine angehängte Datei unter workspace/Uploads/ ab, ohne eine vorhandene zu überschreiben."""
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.UPLOADS_DIR / name
    number = 1
    while target.exists():
        target = config.UPLOADS_DIR / f"{Path(name).stem}_{number}{Path(name).suffix}"
        number += 1
    shutil.copyfile(source, target)
    return target


async def _ingest_attachments(message: cl.Message) -> list[str]:
    """Nimmt angehängte Dateien auf und gibt ihre Pfade im Workspace zurück.

    Dokumente (PDF, Word, Text, CSV, ...) kommen in die Wissensbasis, alles andere (Bilder, Audio, ...) nach Uploads/.
    """
    paths = []
    for element in message.elements or []:
        if not getattr(element, "path", None):
            continue
        name = Path(element.name).name
        try:
            if Path(name).suffix.lower() in knowledge_base.SUPPORTED_SUFFIXES:
                chunks = await cl.make_async(knowledge_base.add_document)(Path(element.path), name)
                await cl.Message(f"✅ **{name}** zur Wissensbasis hinzugefügt ({chunks} Abschnitte).").send()
                target = config.DOCUMENTS_DIR / name
            else:
                target = await cl.make_async(_save_upload)(Path(element.path), name)
            paths.append(str(target.resolve().relative_to(config.WORKSPACE_DIR.resolve())))
        except Exception as exc:
            await cl.Message(f"⚠️ **{name}** konnte nicht aufgenommen werden: {exc}").send()
    return paths


@cl.on_message
async def on_message(message: cl.Message) -> None:
    attachments = await _ingest_attachments(message)
    text = message.content.strip()
    if not text:
        return
    if attachments:  # So weiß der Agent, unter welchem Pfad er die Dateien an Tools übergeben kann.
        text += "\n\n[Angehängte Dateien im Workspace: " + ", ".join(f"`{p}`" for p in attachments) + "]"

    answer = cl.Message(content="")
    steps: dict[str, cl.Step] = {}
    current: Optional[cl.Step] = None
    result: Optional[dict] = None

    events = chat_stream(text, cl.context.session.thread_id, models=cl.user_session.get("models"))
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
            elif kind == "image_agent_start":
                current = cl.Step(name="Bild-Agent arbeitet", type="run")
                await current.send()
            elif kind == "video_agent_start":
                current = cl.Step(name="Video-Agent arbeitet (dauert einige Minuten)", type="run")
                await current.send()
            elif kind == "lookup_start":
                current = cl.Step(name="Dokumentsuche", type="run")
                await current.send()
            elif kind == "tool_start":
                label = TOOL_LABELS.get(event["name"]) or neural_networks.labels().get(event["name"], event["name"])
                step = cl.Step(name=label, type="tool", parent_id=current.id)
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

    # Bilder als Markdown in den Text: so bleiben sie auch in gespeicherten Chats erhalten. Videos lassen sich in
    # Markdown nicht einbetten: Player als Element (nur in diesem Chat) und ein Link, der auch gespeichert bleibt.
    answer.content = (
        result["text"]
        + "".join(f"\n\n![Bild](/images/{name})" for name in result["images"])
        + "".join(f"\n\n[🎬 Video öffnen](/videos/{name})" for name in result.get("videos", []))
    )
    answer.elements = [cl.CustomElement(name="SpeakButton", props={"text": result["text"]}, display="inline")]
    answer.elements += [cl.Video(name=name, url=f"/videos/{name}", display="inline") for name in result.get("videos", [])]
    await answer.send()
