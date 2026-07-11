from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from rag_chatpipline.ragpipline import call_rag_pipeline

app = FastAPI(title="RAG Chat API")


class RagRequest(BaseModel):
    message: str
    chatId: Optional[str] = None
    memoryDir: Optional[str] = None
    dataDir: Optional[str] = None


class RagResponse(BaseModel):
    text: str
    chatId: Optional[str] = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rag", response_model=RagResponse)
def rag_endpoint(payload: RagRequest) -> RagResponse:
    answer = call_rag_pipeline(
        query=payload.message,
        chat_id=payload.chatId,
        memory_dir=payload.memoryDir,
        data_dir=payload.dataDir,
    )
    return RagResponse(text=answer, chatId=payload.chatId)
