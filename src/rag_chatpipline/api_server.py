import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, knowledge_base

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
from chainlit.utils import mount_chainlit  # noqa: E402
UI_PATH = "/ui"

config.ensure_dirs()
app = FastAPI(title="RAG Chat API")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    chatId: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    chatId: str
    images: list[str] = []  # URLs der in dieser Antwort erzeugten Bilder


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
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    try:
        sources = knowledge_base.list_sources()
    except knowledge_base.KnowledgeBaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [DocumentInfo(name=name, chunks=count) for name, count in sorted(sources.items())]


@app.post("/documents", response_model=list[DocumentInfo])
def upload_documents(files: list[UploadFile] = File(...)) -> list[DocumentInfo]:
    saved: list[DocumentInfo] = []
    for upload in files:
        name = Path(upload.filename or "").name
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / name
            staged.write_bytes(upload.file.read())
            try:
                chunks = knowledge_base.add_document(staged, name)
            except knowledge_base.KnowledgeBaseError as exc:
                unsupported = "nicht unterstützt" in str(exc)
                raise HTTPException(status_code=400 if unsupported else 422, detail=str(exc)) from exc
        saved.append(DocumentInfo(name=name, chunks=chunks))
    return saved


@app.delete("/documents/{name}", status_code=204)
def delete_document(name: str) -> None:
    knowledge_base.remove_document(name)


app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(UI_PATH)


# Chat-Oberfläche (Chainlit) im selben Prozess, weil der lokale Qdrant-Speicher nur einen Prozess zulässt.
mount_chainlit(app=app, target=str(UI_DIR / "chainlit_app.py"), path=UI_PATH)
