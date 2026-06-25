from fastapi import FastAPI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ===Globale Pramter===
LLM_MODEL = "ollama:qwen3.5:2b"
Embeddings_MODEL = "jina-embeddings-v5-text-small"
RERANKING_MODEL = "ollama:qwen3.5:2b"



# ===Embeddings einrichten===
embeddings = HuggingFaceEmbeddings(
    model_name=Embeddings_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


# ===Vektor-Datenbank einrichten===

client = QdrantClient(":memory:")

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)
