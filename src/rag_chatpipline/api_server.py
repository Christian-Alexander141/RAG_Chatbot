import asyncio
import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import ad_auth, config, knowledge_base

logger = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
# Muss vor dem ersten Chainlit-Import stehen: Chainlit liest Konfiguration und public/ von hier.
os.environ.setdefault("CHAINLIT_APP_ROOT", str(UI_DIR))


def _ensure_auth_secret() -> None:
    """Chainlit braucht für die Chat-Historie ein Signier-Geheimnis. Wird einmalig erzeugt und lokal gespeichert."""
    if os.environ.get("CHAINLIT_AUTH_SECRET"):
        return
    secret_file = config.MEMORY_DIR / ".chainlit_auth_secret"
    if not secret_file.exists():
        config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        secret_file.write_text(secrets.token_urlsafe(48))
        secret_file.chmod(0o600)
    os.environ["CHAINLIT_AUTH_SECRET"] = secret_file.read_text().strip()


_ensure_auth_secret()
if ad_auth.enabled():
    ad_auth.check_config()  # AD-Anmeldung eingeschaltet, aber nicht eingerichtet: lieber gar nicht erst starten
from chainlit.auth import get_token_from_cookies  # noqa: E402
from chainlit.auth.jwt import decode_jwt  # noqa: E402
from chainlit.user import User  # noqa: E402
from chainlit.utils import mount_chainlit  # noqa: E402
UI_PATH = "/ui"
AUTH_COOKIE = "access_token"  # Name des Anmelde-Cookies von Chainlit (CHAINLIT_AUTH_COOKIE_NAME nicht gesetzt)

config.ensure_dirs()
app = FastAPI(title="RAG Chat API")


def _token_user(connection: HTTPConnection) -> Optional[User]:
    """Angemeldeter Nutzer: Cookie aus dem Browser oder "Authorization: Bearer" für andere Programme."""
    token = get_token_from_cookies(connection.cookies)
    if not token:
        scheme, _, value = connection.headers.get("authorization", "").partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else ""
    if not token:
        return None
    try:
        return decode_jwt(token)  # prüft Signatur und Ablauf
    except Exception:
        return None


def _without_credentials(scope: Scope) -> Scope:
    """Dieselbe Anfrage ohne Anmeldung (Cookie access_token und Authorization entfernt)."""
    headers = []
    for name, value in scope["headers"]:
        if name == b"authorization":
            continue
        if name == b"cookie":
            kept = [c for c in value.decode("latin-1").split(";") if not c.strip().startswith(AUTH_COOKIE)]
            if not kept:
                continue
            value = ";".join(kept).encode("latin-1")
        headers.append((name, value))
    return {**scope, "headers": headers}


class LoginGuard:
    """Mit AD-Anmeldung: Zugang nur für angemeldete Konten, die im AD noch aktiv sind.

    Ist ein Konto inzwischen gesperrt, deaktiviert oder aus der Gruppe entfernt, gilt die Anfrage als nicht angemeldet
    und der Browser bekommt die Anweisung, das Cookie zu löschen. Chainlit zeigt dann die Anmeldeseite und baut keine
    Chat-Verbindung (Websocket) mehr auf. Chat, Dokumente, Bilder und Videos gibt es nur angemeldet, die Oberfläche
    unter /ui prüft selbst und zeigt die Anmeldeseite.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or not ad_auth.enabled():
            await self.app(scope, receive, send)
            return
        user = _token_user(HTTPConnection(scope))
        access = await asyncio.to_thread(ad_auth.check_access, user.identifier) if user is not None else None
        allowed = access is not None and access.allowed
        # Für die Routen: Admin-Rechte dieses Kontos (request.state.ad_access), siehe _admin_rights
        scope = {**scope, "state": {**scope.get("state", {}), "ad_access": access if allowed else None}}
        if user is not None and not allowed:
            scope = _without_credentials(scope)
            if scope["type"] == "http":
                send = _clear_cookie(send)
        path = scope["path"]
        public = path in ("/", "/health") or path == UI_PATH or path.startswith(UI_PATH + "/")
        if scope["type"] == "http" and not public and scope["method"] != "OPTIONS" and not allowed:
            await JSONResponse({"detail": "Bitte zuerst anmelden."}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _admin_rights(request: Request) -> bool:
    access = request.scope.get("state", {}).get("ad_access")
    return ad_auth.admin_rights(access.admin if access else False)


def _clear_cookie(send: Send) -> Send:
    async def clearing(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = message.get("headers", [])
            # Setzt die Antwort selbst ein neues Cookie (erneute Anmeldung), bleibt es stehen.
            if not any(n == b"set-cookie" and v.startswith(AUTH_COOKIE.encode() + b"=") for n, v in headers):
                cookie = f"{AUTH_COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=lax".encode()
                message = {**message, "headers": [*headers, (b"set-cookie", cookie)]}
        await send(message)

    return clearing


class IgnoreAbortedRequests:
    """Bricht der Browser eine Anfrage ab (z. B. bei der Weiterleitung nach der Anmeldung), meldet Chainlits
    Middleware "RuntimeError: No response returned" mit langem Traceback, obwohl nichts kaputt ist. Nur genau
    dieser Fall wird leise übergangen: Der Browser hat die Verbindung getrennt, eine Antwort braucht niemand mehr.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        disconnected = False

        async def watching_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            disconnected = disconnected or message["type"] == "http.disconnect"
            return message

        try:
            await self.app(scope, watching_receive, send)
        except RuntimeError as exc:
            if not (disconnected and str(exc) == "No response returned."):
                raise
            logger.debug("Anfrage vom Browser abgebrochen: %s", scope["path"])


app.add_middleware(IgnoreAbortedRequests)
app.add_middleware(LoginGuard)
# Nach LoginGuard hinzugefügt und damit außen: auch Antworten mit 401 bekommen die CORS-Header.
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    chatId: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    chatId: str
    images: list[str] = []  # URLs der in dieser Antwort erzeugten Bilder
    videos: list[str] = []  # URLs der in dieser Antwort erzeugten Videos


class TranscriptionResponse(BaseModel):
    text: str  # leer, wenn keine Sprache erkannt wurde


class DocumentInfo(BaseModel):
    name: str
    chunks: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@app.post("/rag", response_model=ChatResponse, include_in_schema=False)  # alter Pfad des Frontends
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    from Agents.chat_agent import chat

    try:
        result = chat(payload.message, payload.chatId)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Der Agent ist fehlgeschlagen: {exc}") from exc
    return ChatResponse(
        text=result["text"],
        chatId=result["chat_id"],
        images=[f"/images/{name}" for name in result["images"]],
        videos=[f"/videos/{name}" for name in result.get("videos", [])],
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Spracheingabe: wandelt eine Audiodatei (WAV, MP3, OGG, WebM, ...) lokal mit Whisper in Text um."""
    from Agents import speech

    if not speech.is_installed():
        raise HTTPException(status_code=503, detail="Das Spracherkennungsmodell fehlt. Installation mit ./install.sh")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(file.filename or "").name or "aufnahme")
        path.write_bytes(file.file.read())
        try:
            text = speech.transcribe_file(path)
        except speech.SpeechError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TranscriptionResponse(text=text)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    try:
        sources = knowledge_base.list_sources()
    except knowledge_base.KnowledgeBaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [DocumentInfo(name=name, chunks=count) for name, count in sorted(sources.items())]


@app.post("/documents", response_model=list[DocumentInfo])
def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> list[DocumentInfo]:
    """Hochladen darf jeder. Ein vorhandenes Dokument ersetzen nur, wer auch löschen darf (mit AD: Admins)."""
    replace = _admin_rights(request)
    saved: list[DocumentInfo] = []
    for upload in files:
        name = Path(upload.filename or "").name
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / name
            staged.write_bytes(upload.file.read())
            try:
                chunks = knowledge_base.add_document(staged, name, replace=replace)
            except knowledge_base.DocumentExistsError as exc:
                raise HTTPException(status_code=409, detail=f"{exc} {ad_auth.REPLACE_DENIED}") from exc
            except knowledge_base.KnowledgeBaseError as exc:
                unsupported = "nicht unterstützt" in str(exc)
                raise HTTPException(status_code=400 if unsupported else 422, detail=str(exc)) from exc
        saved.append(DocumentInfo(name=name, chunks=chunks))
    return saved


@app.delete("/documents/{name}", status_code=204)
def delete_document(name: str, request: Request) -> None:
    if not _admin_rights(request):
        raise HTTPException(status_code=403, detail="Dokumente löschen dürfen nur Admins.")
    knowledge_base.remove_document(name)


@app.get("/files/{name:path}")
def document_file(name: str) -> FileResponse:
    """Liefert ein Dokument der Wissensbasis zum Ansehen im Browser (Links der Dokumentsuche und der Seitenleiste)."""
    root = config.DOCUMENTS_DIR.resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden.")
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/videos", StaticFiles(directory=config.VIDEOS_DIR), name="videos")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(UI_PATH)


# Chat-Oberfläche (Chainlit) im selben Prozess, weil der lokale Qdrant-Speicher nur einen Prozess zulässt.
mount_chainlit(app=app, target=str(UI_DIR / "chainlit_app.py"), path=UI_PATH)
