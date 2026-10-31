"""Richtet die virtuelle Umgebung .venv ein und installiert die Abhängigkeiten aus pyproject.toml.

Läuft unter Linux, WSL, macOS und Windows. Bereits erledigte Schritte werden übersprungen.
Erneut installiert wird nur, wenn sich pyproject.toml oder die gewählten Extras geändert haben.

Aufruf: python install.py [--dev] [--image] [--force]
  --dev     Test-Pakete (pytest, httpx)
  --image   FLUX-Bildgenerierung (torch, diffusers, ...; mehrere GB)
  --force   .venv neu anlegen und alles neu installieren
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
STAMP = VENV / ".install-stamp"
WINDOWS = os.name == "nt"


def venv_python(windows: bool = WINDOWS) -> Path:
    return VENV / ("Scripts/python.exe" if windows else "bin/python")


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.release().lower()


def venv_works() -> bool:
    python = venv_python()
    if not python.exists():
        return False
    result = subprocess.run([str(python), "-c", "import pip"], capture_output=True)
    return result.returncode == 0


def create_venv() -> None:
    print(f"Lege virtuelle Umgebung an: {VENV}")
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
    if result.returncode != 0 or not venv_works():
        shutil.rmtree(VENV, ignore_errors=True)
        hint = ""
        if sys.platform.startswith("linux"):
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            hint = (
                "\nUnter Debian/Ubuntu (auch WSL) fehlt oft das venv-Paket:\n"
                f"  sudo apt install python3-venv   (oder python{version}-venv)"
            )
        fail("Die virtuelle Umgebung konnte nicht angelegt werden." + hint)


def ensure_venv(force: bool) -> None:
    if force and VENV.exists():
        print("--force: lege .venv neu an")
        shutil.rmtree(VENV)

    if venv_works():
        print(f"Virtuelle Umgebung vorhanden: {VENV}")
        return

    if VENV.exists():
        # Windows und Linux/WSL können sich keine .venv teilen. Eine fremde löschen wir nicht ungefragt.
        if venv_python(not WINDOWS).exists():
            other = "Linux/WSL" if WINDOWS else "Windows"
            fail(
                f"{VENV} wurde unter {other} angelegt und passt nicht zu diesem System.\n"
                "Ordner löschen oder mit --force neu anlegen (die andere Umgebung geht dabei verloren)."
            )
        print("Virtuelle Umgebung ist unvollständig oder defekt, lege sie neu an.")
        shutil.rmtree(VENV)

    create_venv()


def install(extras: list[str], force: bool) -> None:
    target = f".[{','.join(extras)}]" if extras else "."
    # Zeilenenden vereinheitlichen, damit eine Windows-Kopie mit CRLF denselben Stempel ergibt.
    pyproject = (ROOT / "pyproject.toml").read_bytes().replace(b"\r\n", b"\n")
    stamp = f"{hashlib.sha256(pyproject).hexdigest()} {target}"

    if not force and STAMP.exists() and STAMP.read_text().strip() == stamp:
        print(f"Abhängigkeiten sind bereits installiert ({target}).")
        return

    print(f"Installiere Abhängigkeiten aus pyproject.toml ({target}) ...")
    python = str(venv_python())
    for command in (
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        [python, "-m", "pip", "install", target],
    ):
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            fail("Installation fehlgeschlagen, siehe Ausgabe oben.")
    STAMP.write_text(stamp + "\n")


def main() -> None:
    # Sonst erscheinen eigene Meldungen bei umgeleiteter Ausgabe erst nach denen von pip.
    sys.stdout.reconfigure(line_buffering=True)
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(__doc__)
        return
    unknown = [a for a in args if a not in ("--dev", "--image", "--force")]
    if unknown:
        fail(f"Unbekannte Option: {' '.join(unknown)} (siehe --help)")

    if sys.version_info < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} oder neuer wird benötigt, gefunden: {platform.python_version()}")

    if is_wsl() and str(ROOT).startswith("/mnt/"):
        print(
            "Hinweis: Das Projekt liegt im Windows-Dateisystem. Unter WSL ist das deutlich langsamer;\n"
            "besser ins Linux-Dateisystem klonen (z. B. ~/RAG_Chatbot)."
        )

    force = "--force" in args
    extras = [name for name in ("dev", "image") if f"--{name}" in args]
    ensure_venv(force)
    install(extras, force)

    activate = r".venv\Scripts\activate" if WINDOWS else "source .venv/bin/activate"
    print(f"\nFertig. Umgebung aktivieren mit:\n  {activate}\nDanach starten mit: python main.py")


if __name__ == "__main__":
    main()
