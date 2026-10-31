import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatpipline import config  # noqa: E402


@pytest.fixture(autouse=True)
def no_prompt_model(monkeypatch):
    """Der Bild-Agent fragt in Tests kein echtes Ollama-Modell; der Bildwunsch geht unverändert weiter."""
    from Agents import image_agent

    monkeypatch.setattr(image_agent, "get_prompt_model", lambda name=None: None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolierter Workspace, damit Tests nie echte Daten anfassen."""
    root = tmp_path / "workspace"
    monkeypatch.setattr(config, "WORKSPACE_DIR", root)
    monkeypatch.setattr(config, "DOCUMENTS_DIR", root / "Dateien")
    monkeypatch.setattr(config, "IMAGES_DIR", root / "AIbilder")
    config.ensure_dirs()
    return root


@pytest.fixture
def vector_store():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=VectorParams(size=32, distance=Distance.COSINE))
    return QdrantVectorStore(client=client, collection_name="t", embedding=DeterministicFakeEmbedding(size=32))
