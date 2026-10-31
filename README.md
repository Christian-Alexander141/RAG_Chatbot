# RAG-Chatbot

Prototyp eines firmeninternen Chatbots, der lokal laufende Sprachmodelle über **Ollama** nutzbar macht.

- **Agent** auf Basis von LangChain (`create_agent`) mit Ollama als einzigem Modellprovider
- **Eigene Dokumente als Wissensquelle** (PDF, DOCX, TXT, MD, CSV, JSON, …), indexiert in Qdrant mit Ollama-Embeddings
- **Dateisystemzugriff**, begrenzt auf den Ordner `workspace/` (Lesen, Schreiben ohne stilles Überschreiben, Auflisten, Suchen; kein Löschen)
- **Bilder** lokal mit FLUX (diffusers)
- **Oberfläche** mit [Chainlit](https://github.com/Chainlit/chainlit), eingebunden in den FastAPI-Server
- **Vorlesen** im Browser (Web Speech API) über den Button "Vorlesen" an jeder Antwort

## Start

```bash
./install.sh --dev               # legt .venv an und installiert pyproject.toml (--dev: mit Test-Paketen)
source .venv/bin/activate
ollama pull qwen3:1.7b          # Agent-Modell (muss Tool-Calling können)
ollama pull cas/teuken-7b-q4km    # Chat-Modell (Teuken, formuliert die Antworten)
ollama pull nomic-embed-text    # Embedding-Modell
python main.py                  # Chainlit-Oberfläche auf http://localhost:8000 (leitet nach /ui), API unter /docs
```

Die App lädt **keine** Modelle selbst herunter. Fehlt eines, kommt eine Fehlermeldung mit dem passenden `ollama pull`-Befehl.

Unter **Windows** stattdessen `install.bat --dev` (oder Doppelklick) und `.venv\Scripts\activate`. Beide Starter rufen `install.py` auf, das geht auch direkt: `python install.py --dev`.

Der Installer überspringt alles, was schon erledigt ist, und installiert nur neu, wenn sich `pyproject.toml` oder die gewählten Extras geändert haben (`--force` legt `.venv` neu an). Windows und WSL können sich keine `.venv` teilen. Unter WSL das Projekt am besten im Linux-Dateisystem (`~/...`) klonen, nicht unter `/mnt/c`. Manuell geht es auch mit `pip install ".[dev]"`.

Tests: `pytest`

## Konfiguration (Umgebungsvariablen)

| Variable | Standard | Bedeutung |
|---|---|---|
| `CHAT_MODEL` | `cas/teuken-7b-q4km:latest` | Chat-Modell: formuliert die Antworten (Teuken) |
| `AGENT_MODEL` | `qwen3:1.7b` | Agent-Modell: ruft Tools auf, braucht gutes Tool-Calling |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama-Embeddingmodell |
| `AGENT_REASONING` | leer | `true`/`false` schaltet das Denken des Agent-Modells ein/aus |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama-Adresse |
| `WORKSPACE_DIR` | `./workspace` | Einziger Ordner, auf den der Agent zugreifen darf |
| `IMAGE_BACKEND` | `flux` | `flux` oder `mock` (graues Platzhalterbild zum Testen) |
| `FLUX_MODEL` / `IMAGE_SIZE` | `black-forest-labs/FLUX.1-schnell` / `768` | FLUX-Modell und Kantenlänge |
| `CORS_ORIGINS` | `http://localhost:3000` | Erlaubte Frontend-Origins, kommagetrennt |

Wechselt man das Embedding-Modell, ändert sich die Vektorgröße. Dann `memory/qdrant/` löschen und die Dokumente neu hochladen.

## Bildgenerierung mit FLUX

```bash
./install.sh --dev --image
```

FLUX.1-schnell braucht eine GPU oder sehr viel RAM (grob 24 GB und mehr) und lädt beim ersten Einsatz mehrere GB von Hugging Face. Ohne passende Hardware `IMAGE_BACKEND=mock` setzen. Fehlen die Pakete, meldet der Agent die fehlgeschlagene Generierung im Chat.

## Bedienung

- **Antworten** erscheinen Wort für Wort. Darüber zeigt die Oberfläche die Arbeitsschritte des Agenten (Wissensbasis, Tool-Aufrufe, Antwort formulieren).
- **Wissensquellen** (rechte Seitenleiste): Dokumente hochladen, ansehen und entfernen. Alternativ eine Datei an eine Nachricht anhängen.
- **Alte Chats** (linke Seitenleiste): Chats werden in `memory/chainlit.db` gespeichert und lassen sich wieder öffnen. Der Gesprächsverlauf wird dabei für das Modell wiederhergestellt, auch nach einem Neustart.
- **Vorlesen:** Button an jeder neuen Antwort (nicht in wieder geöffneten alten Chats).
- **Nutzer:** Es gibt keinen Login. Alle teilen sich den Nutzer `lokal`. Steht ein Reverse-Proxy mit SSO davor, der den Header `X-Forwarded-User` setzt, hat jeder seine eigenen Chats. Ohne vertrauenswürdigen Proxy lässt sich dieser Header fälschen.

## API

| Route | Zweck |
|---|---|
| `POST /chat` `{message, chatId?}` | Agent-Antwort `{text, chatId, images[]}` (`/rag` ist ein Alias) |
| `GET /documents`, `POST /documents`, `DELETE /documents/{name}` | Wissensbasis verwalten |
| `GET /images/{name}` | erzeugte Bilder |
| `GET /health` | Statusprüfung |

## Struktur

- `main.py`: Startpunkt (`--mode backend|chat|frontend|all`)
- `install.py` (+ Starter `install.sh` / `install.bat`): richtet `.venv` ein und installiert `pyproject.toml`
- `src/Agents/`: `chat_agent.py` (Agent), `tools.py` (Wissensbasis, Dateisystem, Bild), `image_backend.py` (FLUX/Mock)
- `src/rag_chatpipline/`: `api_server.py`, `knowledge_base.py`, `config.py`
- `src/ui/`: Chainlit-Oberfläche (`chainlit_app.py`, `history_db.py`, `.chainlit/config.toml`, `public/elements/` mit `SpeakButton.jsx` und `DocumentsPanel.jsx`)
- `tests/`

Die API (`POST /chat`) hält den Verlauf pro `chatId` nur im Arbeitsspeicher. Die Oberfläche stellt ihn aus der gespeicherten Historie wieder her.

## Verwendete Open-Source-Projekte

Lizenzen der Python-Pakete laut den installierten Paketen (Stand der Installation).

| Projekt | Verwendung | Lizenz |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | Lokaler Modellserver (Chat-, Agent- und Embedding-Modelle) | MIT |
| [LangChain](https://github.com/langchain-ai/langchain) | Agent (`create_agent`), Tools, Modellanbindung | MIT |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent-Laufzeit, Gesprächsverlauf | MIT |
| [langchain-ollama](https://github.com/langchain-ai/langchain) / [ollama-python](https://github.com/ollama/ollama-python) | Ollama-Anbindung | MIT |
| [langchain-qdrant](https://github.com/langchain-ai/langchain) / [langchain-text-splitters](https://github.com/langchain-ai/langchain) | Vektorsuche, Textaufteilung | MIT |
| [Qdrant](https://github.com/qdrant/qdrant-client) (qdrant-client, lokaler Modus) | Vektordatenbank der Wissensbasis | Apache-2.0 |
| [Chainlit](https://github.com/Chainlit/chainlit) | Chat-Oberfläche, Chat-Historie, Seitenleisten | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | API-Server | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | Webserver | BSD-3-Clause |
| [python-multipart](https://github.com/Kludex/python-multipart) | Datei-Uploads | Apache-2.0 |
| [pypdf](https://github.com/py-pdf/pypdf) | PDF-Text auslesen | BSD-3-Clause |
| [python-docx](https://github.com/python-openxml/python-docx) | Word-Dokumente auslesen | MIT |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) / [aiosqlite](https://github.com/omnilib/aiosqlite) | Speicher der Chat-Historie (SQLite) | MIT |
| [pytest](https://github.com/pytest-dev/pytest) / [HTTPX](https://github.com/encode/httpx) | Tests | MIT / BSD-3-Clause |
| [Diffusers](https://github.com/huggingface/diffusers) und [PyTorch](https://github.com/pytorch/pytorch) (optional) | Bildgenerierung mit FLUX | Apache-2.0 / BSD-3-Clause |

**Modelle** haben eigene Lizenzen, die nicht zu den Paketen gehören. Bitte vor dem Firmeneinsatz prüfen: Teuken (Chat-Modell), Qwen3 (Agent-Modell), nomic-embed-text (Embeddings) und FLUX (Bildmodell, die Lizenz unterscheidet sich je Variante, z. B. schnell und dev).
