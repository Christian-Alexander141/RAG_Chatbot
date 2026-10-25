"""SQLite-Speicher für die Chat-Historie von Chainlit (Seitenleiste mit alten Chats)."""
import sqlite3
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from rag_chatpipline import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY, "identifier" TEXT NOT NULL UNIQUE, "metadata" TEXT NOT NULL, "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY, "createdAt" TEXT, "name" TEXT, "userId" TEXT, "userIdentifier" TEXT,
    "tags" TEXT, "metadata" TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY, "name" TEXT NOT NULL, "type" TEXT NOT NULL, "threadId" TEXT NOT NULL,
    "parentId" TEXT, "streaming" INTEGER NOT NULL, "waitForAnswer" INTEGER, "isError" INTEGER,
    "metadata" TEXT, "tags" TEXT, "input" TEXT, "output" TEXT, "createdAt" TEXT, "command" TEXT,
    "start" TEXT, "end" TEXT, "generation" TEXT, "showInput" TEXT, "language" TEXT, "indent" INTEGER,
    "defaultOpen" INTEGER, "autoCollapse" INTEGER, "icon" TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY, "threadId" TEXT, "type" TEXT, "chainlitKey" TEXT, "url" TEXT, "objectKey" TEXT,
    "name" TEXT NOT NULL, "display" TEXT, "size" TEXT, "language" TEXT, "page" INTEGER, "autoPlay" INTEGER,
    "playerConfig" TEXT, "forId" TEXT, "mime" TEXT, "props" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY, "forId" TEXT NOT NULL, "threadId" TEXT NOT NULL, "value" INTEGER NOT NULL, "comment" TEXT
);
"""

DB_PATH = config.MEMORY_DIR / "chainlit.db"


def create_data_layer() -> SQLAlchemyDataLayer:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.executescript(SCHEMA)
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")
