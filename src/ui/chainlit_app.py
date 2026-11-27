"""Chainlit-Oberfläche für den Firmen-Chatbot.

- Linke Seitenleiste: alte Chats (Chainlit-Historie, gespeichert in SQLite)
- Rechte Seitenleiste "Wissensquellen": Dokumente hochladen, ansehen und entfernen
- Antworten werden gestreamt, die Arbeitsschritte des Agenten erscheinen als Schritte
- Benutzermenü > Einstellungen: Sprach-, Bild- und Videomodelle für Chat und Agenten wählen (pro Nutzer gespeichert)
- Angehängte Dateien: Dokumente gehen in die Wissensbasis, alles andere (z. B. Bilder) nach workspace/Uploads/.
  Der Agent bekommt die Pfade mitgeteilt, damit eigene neuronale Netze oder der Video-Agent sie nutzen können.
- Spracheingabe: Mikrofon-Knopf neben dem Eingabefeld. Die Aufnahme wird lokal mit Whisper in Text umgewandelt
  und dann wie eine getippte Nachricht beantwortet.
- Anmeldung: ohne Login (Nutzer "lokal" oder aus dem Header eines Reverse-Proxys) oder, mit AD_ENABLED=true,
  über eine Anmeldeseite mit dem Konto aus dem Active Directory. Dann wird vor jeder Aktion geprüft, ob das Konto
  noch Zugang hat, und nur Admins wählen das Embedding-Modell und Cloud-Modelle und löschen oder ersetzen Dokumente.
"""
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.config import config as chainlit_config
from chainlit.data import get_data_layer
from chainlit.input_widget import Select
from chainlit.types import InputAudioChunk, ThreadDict

import user_settings
from Agents import neural_networks, speech
from Agents.chat_agent import chat_stream, seed_history
from Agents.models import ModelChoice, image_models, language_models, video_models
from history_db import create_data_layer
from rag_chatpipline import ad_auth, config, knowledge_base

logger = logging.getLogger(__name__)

HELP = (
    "Ich beantworte Fragen zu euren Dokumenten, arbeite mit Dateien im Workspace und erzeuge Bilder und Videos.\n\n"
    "- **Dokumente** verwaltest du in der Seitenleiste *Wissensquellen* (rechts). Du kannst auch eine Datei "
    "an eine Nachricht anhängen. Ein Dokument direkt bekommst du z. B. mit *„Gib mir die Reisekostenrichtlinie“*.\n"
    "- **Videos** mit Ton: *„Erstelle ein Video von …“* oder nach einem Bild *„Animiere das Bild“*. "
    "Das dauert einige Minuten.\n"
    "- **Spracheingabe:** Mikrofon-Knopf neben dem Eingabefeld drücken, sprechen, zum Beenden erneut drücken.\n"
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
CLOUD_NOTE = " Cloud-Modelle (Name mit „cloud“) schicken deine Texte an fremde Server."
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


# Chainlit entscheidet beim Start, welche Anmeldung die Oberfläche nutzt: Es wird nur eine der beiden registriert.
if ad_auth.enabled():
    @cl.password_auth_callback
    async def ad_login(username: str, password: str) -> Optional[cl.User]:
        """Anmeldeseite: Benutzername und Passwort werden am Active Directory geprüft."""
        try:
            user = await asyncio.to_thread(ad_auth.authenticate, username, password)
        except ad_auth.ADError as exc:
            logger.error("AD-Anmeldung nicht möglich: %s", exc)
            return None
        if user is None:
            return None
        return cl.User(identifier=user.username, display_name=user.display_name,
                       metadata={"provider": "ad", "email": user.email})
else:
    @cl.header_auth_callback
    async def header_auth(headers) -> Optional[cl.User]:
        """Meldet ohne Login an. Hinter einem Reverse-Proxy mit SSO liefert dieser den Nutzer im Header
        X-Forwarded-User, jeder sieht dann nur seine eigenen Chats. Ohne Proxy teilen sich alle den Nutzer "lokal"."""
        return cl.User(identifier=headers.get("x-forwarded-user") or "lokal", metadata={"provider": "header"})


async def _open_documents_panel() -> None:
    await cl.ElementSidebar.set_title("Wissensquellen")
    # Nur eine Anzeige: Die API prüft selbst, wer löschen darf.
    panel = cl.CustomElement(name="DocumentsPanel", props={"canDelete": _admin_rights()})
    await cl.ElementSidebar.set_elements([panel])


def _user_id() -> str:
    user = cl.user_session.get("user")
    return user.identifier if user else "lokal"


def _admin_rights() -> bool:
    """Embedding-Modell wählen (gilt für alle), Dokumente löschen oder ersetzen: mit AD nur Admins, ohne AD jeder."""
    return ad_auth.admin_rights(cl.user_session.get("admin"))


def _may_use_cloud() -> bool:
    """Cloud-Modelle schicken die Daten an fremde Server: nur für Admins, also nur mit AD."""
    return ad_auth.enabled() and bool(cl.user_session.get("admin"))


async def _has_access() -> bool:
    """Mit AD: Hat das Konto noch Zugang (nicht gesperrt, deaktiviert, ...)? Sonst ein Hinweis statt einer Antwort.

    Das braucht es für Chats, die schon offen sind: Neue Verbindungen weist der LoginGuard im API-Server ab.
    Ändern sich die Admin-Rechte, passt sich die Modellauswahl an.
    """
    if not ad_auth.enabled():
        return True
    access = await asyncio.to_thread(ad_auth.check_access, _user_id())
    if not access.allowed:
        if access.reason == ad_auth.AD_UNREACHABLE:
            text = ("🔒 Dein Zugang lässt sich gerade nicht prüfen, das Active Directory antwortet nicht. "
                    "Bitte später erneut versuchen.")
        else:
            text = "🔒 Dein Zugang zum Chatbot ist beendet. Bei Fragen wende dich bitte an die IT."
        await cl.Message(text).send()
        return False
    previous = cl.user_session.get("admin")
    cl.user_session.set("admin", access.admin)
    if previous is not None and previous != access.admin:
        await _send_settings()  # z. B. keine Cloud-Modelle mehr: Auswahl und gewählte Modelle neu
        await _open_documents_panel()  # Löschknöpfe passend zur Rolle
    return True


async def _available_models() -> tuple[dict[str, list[str]], list[str]]:
    """Modelle, die dieser Nutzer wählen darf, und die Embedding-Modelle."""
    llms = await cl.make_async(language_models)(_may_use_cloud())
    available = {"chat": llms["chat"], "agent": llms["tools"], "image_prompt": llms["chat"], "image": image_models(),
                 "video": video_models()}
    return available, llms["embedding"]


async def _send_settings() -> None:
    """Zeigt die Modellwahl an (Dialog "Einstellungen") und merkt sich die Wahl für diese Sitzung."""
    available, embeddings = await _available_models()
    choice = user_settings.resolve(user_settings.load(_user_id()), available)
    cl.user_session.set("models", choice)
    widgets = []
    for key, (label, description) in SETTINGS_LABELS.items():
        current = getattr(choice, key)
        values = available[key] or [current]  # ohne Ollama bzw. ohne heruntergeladenes Modell: nur die aktuelle Wahl
        if _may_use_cloud() and key in ("chat", "agent", "image_prompt"):
            description += CLOUD_NOTE
        widgets.append(Select(id=key, label=label, values=values, initial_value=current, description=description))

    # Das Embedding-Modell gehört zur gemeinsamen Wissensbasis, nicht zum Nutzer.
    if _admin_rights():
        current = knowledge_base.embedding_model()
        values = embeddings or [current]
        initial = next((v for v in values if knowledge_base.same_model(v, current)), values[0])
        label, description = EMBEDDING_LABEL
        widgets.append(Select(id="embedding", label=label, values=values, initial_value=initial,
                              description=description))
    await cl.ChatSettings(widgets).send()


@cl.on_settings_update
async def settings_update(settings: dict) -> None:
    if not await _has_access():
        return
    available, embeddings = await _available_models()
    choice = user_settings.validated(settings, available, cl.user_session.get("models") or ModelChoice.defaults())
    cl.user_session.set("models", choice)
    await cl.make_async(user_settings.save)(_user_id(), choice)

    embedding = settings.get("embedding")
    # Nur wer darf, und nur ein angebotenes Modell: Auch dieser Wert kommt vom Browser.
    if (embedding and _admin_rights() and embedding in embeddings
            and not knowledge_base.same_model(embedding, knowledge_base.embedding_model())):
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
    if not await _has_access():
        return
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
    if not await _has_access():
        return
    await _send_settings()
    models = cl.user_session.get("models")
    await cl.make_async(seed_history)(thread["id"], exchanges, None, models.agent)
    cl.user_session.set("thread_named", True)
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
                chunks = await cl.make_async(knowledge_base.add_document)(Path(element.path), name,
                                                                          replace=_admin_rights())
                await cl.Message(f"✅ **{name}** zur Wissensbasis hinzugefügt ({chunks} Abschnitte).").send()
                target = config.DOCUMENTS_DIR / name
            else:
                target = await cl.make_async(_save_upload)(Path(element.path), name)
            paths.append(str(target.resolve().relative_to(config.WORKSPACE_DIR.resolve())))
        except knowledge_base.DocumentExistsError:
            await cl.Message(f"⚠️ **{name}** gibt es in der Wissensbasis schon. {ad_auth.REPLACE_DENIED}").send()
        except Exception as exc:
            await cl.Message(f"⚠️ **{name}** konnte nicht aufgenommen werden: {exc}").send()
    return paths


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if not await _has_access():
        return
    attachments = await _ingest_attachments(message)
    text = message.content.strip()
    if not text:
        return
    if attachments:  # So weiß der Agent, unter welchem Pfad er die Dateien an Tools übergeben kann.
        text += "\n\n[Angehängte Dateien im Workspace: " + ", ".join(f"`{p}`" for p in attachments) + "]"
    cl.user_session.set("thread_named", True)  # Chainlit benennt den Chat nach der ersten getippten Nachricht
    await _answer(text)


# === Spracheingabe ===
# Der Browser schickt die Aufnahme in kleinen Blöcken (16-Bit-PCM, mono, Abtastrate aus .chainlit/config.toml).
MAX_AUDIO_SECONDS = 300  # Puffer-Obergrenze; erkannt werden höchstens SPEECH_MAX_SECONDS


@cl.on_audio_start
async def audio_start() -> bool:
    """Mikrofon-Knopf gedrückt. True startet die Aufnahme im Browser."""
    if not await _has_access():
        return False
    if not speech.is_installed():
        await cl.Message("⚠️ Die Spracheingabe ist nicht eingerichtet: Das Spracherkennungsmodell fehlt. "
                         "Installation mit `./install.sh` (Windows: `install.bat`).").send()
        return False
    cl.user_session.set("audio", bytearray())
    return True


@cl.on_audio_chunk
async def audio_chunk(chunk: InputAudioChunk) -> None:
    buffer = cl.user_session.get("audio")
    if buffer is None:
        return
    if chunk.isStart:
        buffer.clear()
    if len(buffer) < MAX_AUDIO_SECONDS * chainlit_config.features.audio.sample_rate * 2:
        buffer.extend(chunk.data)


@cl.on_audio_end
async def audio_end() -> None:
    """Mikrofon-Knopf erneut gedrückt: Aufnahme in Text umwandeln und wie eine getippte Nachricht beantworten."""
    try:
        await asyncio.sleep(0.2)  # die letzten Audioblöcke werden parallel verarbeitet
        buffer = cl.user_session.get("audio")
        cl.user_session.set("audio", None)
        if not buffer or not await _has_access():
            return
        try:
            text = await cl.make_async(speech.transcribe_pcm16)(bytes(buffer), chainlit_config.features.audio.sample_rate)
        except Exception as exc:
            await cl.Message(f"⚠️ Die Spracheingabe ist fehlgeschlagen: {exc}").send()
            return
        if not text:
            await cl.Message("🎙️ Ich habe keine Sprache erkannt. Bitte noch einmal versuchen.").send()
            return
        # Erkannten Text als Nachricht des Nutzers anzeigen (wird mit dem Chat gespeichert)
        await cl.Message(content=text, type="user_message", author=_user_id()).send()
        if not cl.user_session.get("thread_named"):
            # Chainlit nennt einen Chat, der per Sprache beginnt, nur "audio". Besser: der erkannte Text.
            cl.user_session.set("thread_named", True)
            if (layer := get_data_layer()) is not None:
                await layer.update_thread(thread_id=cl.context.session.thread_id, name=text[:100])
        await _answer(text)
    finally:
        await cl.context.emitter.task_end()


async def _answer(text: str) -> None:
    """Beantwortet eine Nachricht: Ereignisse des Agenten als Arbeitsschritte anzeigen, Antwort streamen."""
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
