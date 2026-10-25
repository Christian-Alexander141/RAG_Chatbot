"""Tools des Chat-Agenten: Wissensbasis, Dateisystem (auf den Workspace begrenzt) und Bildgenerierung."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from rag_chatpipline import config, knowledge_base

from . import image_backend

logger = logging.getLogger(__name__)

MAX_READ_CHARS = 20_000


# === Wissensbasis ===
def retrieve_context(query: str, k: int = 4) -> str:
    """Sucht passende Textstellen und formatiert sie mit Quellenangabe. Leer, wenn nichts gefunden wurde."""
    docs = knowledge_base.search(query, k=k)
    parts = []
    for doc in docs:
        page = f", Seite {doc.metadata['page']}" if "page" in doc.metadata else ""
        parts.append(f"[Quelle: {doc.metadata.get('source', 'unbekannt')}{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@tool
def search_documents(query: str) -> str:
    """Durchsucht die hochgeladenen Firmendokumente (Wissensbasis) gezielt nach einem Stichwort.
    Relevante Treffer zur Nutzerfrage liegen dir bereits im Prompt vor; nutze dieses Tool nur für
    zusätzliche oder präzisere Suchen.

    Args:
        query: Suchanfrage in natürlicher Sprache.
    """
    logger.info("search_documents: %s", query)
    try:
        return retrieve_context(query) or "Keine passenden Dokumente gefunden."
    except Exception as exc:
        return f"Die Wissensbasis ist nicht verfügbar: {exc}"


# === Dateisystem (nur innerhalb von WORKSPACE_DIR) ===
def _resolve(user_path: str) -> Path:
    """Löst einen Pfad relativ zum Workspace auf und verweigert alles außerhalb (auch per ../ oder Symlink)."""
    root = config.WORKSPACE_DIR.resolve()
    full = (root / user_path).resolve()
    if full != root and root not in full.parents:
        raise PermissionError(f"Zugriff verweigert: '{user_path}' liegt außerhalb des Workspace.")
    return full


def _relative(path: Path) -> str:
    return str(path.relative_to(config.WORKSPACE_DIR.resolve())) or "."


@tool
def list_files(path: str = ".") -> str:
    """Listet Dateien und Ordner eines Verzeichnisses im Workspace auf.

    Args:
        path: Verzeichnis relativ zum Workspace, z. B. "." oder "Dateien".
    """
    try:
        directory = _resolve(path)
        if not directory.is_dir():
            return f"'{path}' ist kein Verzeichnis."
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries) or "(leer)"
    except (PermissionError, OSError) as exc:
        return str(exc)


@tool
def read_file(path: str) -> str:
    """Liest eine Textdatei aus dem Workspace.

    Args:
        path: Dateipfad relativ zum Workspace.
    """
    try:
        file = _resolve(path)
        if not file.is_file():
            return f"'{path}' existiert nicht."
        text = file.read_text(encoding="utf-8", errors="replace")
    except (PermissionError, OSError) as exc:
        return str(exc)
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n\n[... gekürzt, {len(text) - MAX_READ_CHARS} weitere Zeichen]"
    return text


@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Schreibt Text in eine Datei im Workspace. Fehlende Ordner werden angelegt.
    Bestehende Dateien werden nur mit overwrite=True überschrieben.

    Args:
        path: Dateipfad relativ zum Workspace.
        content: Der Dateiinhalt.
        overwrite: Muss True sein, um eine vorhandene Datei zu ersetzen.
    """
    try:
        file = _resolve(path)
        if file.exists() and not overwrite:
            return f"'{path}' existiert bereits. Mit overwrite=True überschreiben."
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    except (PermissionError, OSError) as exc:
        return str(exc)
    logger.info("write_file: %s", file)
    return f"Datei geschrieben: {_relative(file)}"


@tool
def make_directory(path: str) -> str:
    """Legt einen Ordner im Workspace an.

    Args:
        path: Ordnerpfad relativ zum Workspace.
    """
    try:
        directory = _resolve(path)
        directory.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        return str(exc)
    return f"Ordner angelegt: {_relative(directory)}"


@tool
def search_files(pattern: str) -> str:
    """Sucht Dateien im Workspace per Glob-Muster, z. B. "*.pdf" oder "**/*.md".

    Args:
        pattern: Glob-Muster relativ zum Workspace.
    """
    root = config.WORKSPACE_DIR.resolve()
    try:
        matches = [_relative(p) for p in sorted(root.glob(pattern)) if p.resolve() == root or root in p.resolve().parents]
    except (ValueError, OSError) as exc:
        return f"Ungültiges Muster: {exc}"
    return "\n".join(matches[:100]) or "Keine Treffer."


# === Bildgenerierung ===
@tool(response_format="content_and_artifact")
def generate_image(prompt: str) -> tuple[str, dict]:
    """Erzeugt ein Bild mit FLUX und zeigt es dem Nutzer im Chat an. Nutze dieses Tool, wenn der
    Nutzer ein Bild, eine Illustration oder Grafik möchte. Der Prompt sollte auf Englisch und
    möglichst detailliert sein (Motiv, Stil, Licht, Stimmung).

    Args:
        prompt: Englische Bildbeschreibung.
    """
    logger.info("generate_image: %s", prompt)
    try:
        path = image_backend.generate_image(prompt)
    except Exception as exc:
        return f"Bildgenerierung fehlgeschlagen: {exc}", {}
    return "Das Bild wurde erzeugt und wird dem Nutzer angezeigt.", {"name": path.name, "prompt": prompt}


TOOLS = [search_documents, list_files, read_file, write_file, make_directory, search_files, generate_image]
