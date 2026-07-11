import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_chatpipline.api_server import app as rag_app
from rag_chatpipline.ragpipline import call_rag_pipeline, create_chat_session


def run_backend(host: str = "0.0.0.0", port: int = 8000) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "uvicorn ist nicht in der aktiven Python-Umgebung installiert. "
            "Bitte aktivieren Sie die Projekt-Umgebung und installieren Sie das Paket dort, z. B.: "
            "source myenv/bin/activate && pip install uvicorn"
        ) from exc

    uvicorn.run("main:rag_app", host=host, port=port, reload=False)


def run_frontend() -> None:
    import subprocess
    import shutil

    frontend_dir = ROOT / "app" / "ai-chatbot"
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        raise SystemExit(
            "npm konnte nicht gefunden werden. Bitte installiere Node.js/npm oder verwende einen Node-Paketmanager."
        )

    if not frontend_dir.exists():
        raise SystemExit(f"Frontend-Verzeichnis nicht gefunden: {frontend_dir}")

    process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    ready_url = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if "Local:" in line or "http://localhost" in line:
                ready_url = line.strip().split()[-1]
                print(f"Frontend gestartet: {ready_url}")
    finally:
        process.terminate()
        process.wait(timeout=5)


def run_all(host: str = "0.0.0.0", port: int = 8000) -> None:
    import threading
    import subprocess

    frontend_dir = ROOT / "app" / "ai-chatbot"
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        raise SystemExit(
            "npm konnte nicht gefunden werden. Bitte installiere Node.js/npm oder verwende einen Node-Paketmanager."
        )

    if not frontend_dir.exists():
        raise SystemExit(f"Frontend-Verzeichnis nicht gefunden: {frontend_dir}")

    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def backend_thread() -> None:
        try:
            run_backend(host=host, port=port)
        finally:
            if frontend_proc.poll() is None:
                frontend_proc.terminate()

    thread = threading.Thread(target=backend_thread, daemon=True)
    thread.start()

    try:
        assert frontend_proc.stdout is not None
        for line in frontend_proc.stdout:
            print(line, end="")
    finally:
        if frontend_proc.poll() is None:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=5)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Zentraler Einstiegspunkt für die RAG-Chatbot-App")
    parser.add_argument("--mode", choices=["backend", "frontend", "all", "chat"], default="all")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--message", default="Hallo")
    parser.add_argument("--chat-id", default=None)
    args = parser.parse_args()

    if args.mode == "backend":
        run_backend(host=args.host, port=args.port)
        return

    if args.mode == "frontend":
        run_frontend()
        return

    if args.mode == "all":
        print("Starte Backend und Frontend. Backend: http://127.0.0.1:8000, Frontend: http://localhost:3000")
        run_all(host=args.host, port=args.port)
        return

    session = create_chat_session(chat_id=args.chat_id)
    answer = call_rag_pipeline(query=args.message, chat_id=session["chat_id"])
    print(f"Chat-ID: {session['chat_id']}")
    print(answer)


if __name__ == "__main__":
    run_cli()
