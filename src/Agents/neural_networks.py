"""Eigene neuronale Netze als Werkzeuge der LangChain-Agenten. Anleitung: docs/eigene-netze.md

Jedes Netz liegt in einem eigenen Ordner unter NETWORKS_DIR (Standard: networks/) und hat dort eine Datei
network.py. Darin steht eine Klasse, die von NeuralNetwork erbt. Beim ersten Aufbau eines Agenten werden alle
diese Klassen gefunden und als LangChain-Tools an create_agent(tools=...) übergeben. Das Sprachmodell liest
Name, Beschreibung und Parameter jedes Netzes und entscheidet selbst, wann es ein Netz aufruft.

Ablauf eines Aufrufs: Sprachmodell -> Tool-Aufruf mit Argumenten -> NeuralNetwork.predict(**Argumente)
-> Text (und optional ein Bild) -> zurück an das Sprachmodell, das daraus die Antwort schreibt.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import re
import shutil
import sys
import threading
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import create_schema_from_function

from rag_chatpipline import config

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
NETWORK_FILE = "network.py"
_PACKAGE = "user_networks"  # unter diesem Namen werden die Netz-Ordner importiert


@dataclass
class NetworkResult:
    """Ergebnis mit Bild. Statt eines einfachen Textes kann predict() auch das zurückgeben.

    text: geht an das Sprachmodell (wie ein normaler Rückgabetext)
    image: Bilddatei (PNG, JPG), die dem Nutzer im Chat angezeigt wird, z. B. ein Diagramm
    """

    text: str
    image: Optional[Path] = None


class NeuralNetwork:
    """Basisklasse für eigene Netze. Pflicht sind name, description und predict().

    Beispiel (networks/mein_netz/network.py):

        class MeinNetz(NeuralNetwork):
            name = "mein_netz"
            description = "Wofür das Netz da ist und wann der Agent es aufrufen soll."

            def load(self):
                self.model = ...  # Gewichte laden, einmal vor dem ersten Aufruf

            def predict(self, wert: float) -> str:
                '''Args:
                    wert: Was der Agent hier übergeben soll.
                '''
                return f"Ergebnis: {self.model(wert)}"
    """

    #: Name des Tools, so ruft das Sprachmodell das Netz auf. Nur a-z, 0-9 und _, z. B. "handschrift_erkennen".
    name: str = ""
    #: Anzeige in der Oberfläche, z. B. "Handschrift erkennen". Leer = name.
    label: str = ""
    #: Wofür das Netz da ist und wann es genutzt werden soll. Das liest das Sprachmodell.
    description: str = ""
    #: Dateien im Netz-Ordner, ohne die das Netz nicht angeboten wird, z. B. ["model.pt"].
    required_files: Sequence[str] = ()
    #: False = Netz wird nicht angeboten (zum vorübergehenden Abschalten).
    enabled: bool = True

    #: Ordner des Netzes. Setzt der Lader, bevor load() aufgerufen wird.
    folder: Path = Path(".")

    def load(self) -> None:
        """Lädt Gewichte und Modell. Wird genau einmal aufgerufen, vor dem ersten predict()."""

    def predict(self, *args: Any, **kwargs: Any) -> Union[str, NetworkResult]:
        """Führt das Netz aus. Parameter mit Typangaben und Docstring (Abschnitt Args) werden zu den
        Tool-Argumenten. Rückgabe: Text für das Sprachmodell oder NetworkResult (Text und Bild)."""
        raise NotImplementedError

    # === Hilfen für eigene Netze ===
    def file(self, name: str) -> Path:
        """Pfad einer Datei im Netz-Ordner, z. B. self.file("model.pt")."""
        return self.folder / name

    @staticmethod
    def workspace_file(path: str) -> Path:
        """Datei aus dem Workspace, z. B. "Uploads/ziffer.png". Pfade außerhalb des Workspace werden abgelehnt."""
        root = config.WORKSPACE_DIR.resolve()
        full = (root / path).resolve()
        if full != root and root not in full.parents:
            raise PermissionError(f"'{path}' liegt außerhalb des Workspace.")
        if not full.is_file():
            raise FileNotFoundError(f"'{path}' gibt es im Workspace nicht.")
        return full

    def new_image_path(self, suffix: str = ".png") -> Path:
        """Neuer Dateiname im Bilderordner, z. B. für ein Diagramm, das mit NetworkResult angezeigt wird."""
        config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        return config.IMAGES_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_{self.name}{suffix}"

    @staticmethod
    def device() -> str:
        """"cuda", wenn PyTorch eine NVIDIA-GPU findet, sonst "cpu"."""
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # === intern ===
    _lock: threading.Lock
    _loaded: bool = False

    def _prepare(self, folder: Path) -> None:
        self.folder = folder
        self.label = self.label or self.name
        self.description = self.description.strip() or inspect.getdoc(self.predict) or ""
        self._lock = threading.Lock()
        self._loaded = False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} aus {self.folder}>"


@dataclass
class NetworkStatus:
    """Was beim Einlesen eines Netz-Ordners herauskam (für Log und `python main.py --mode networks`)."""

    folder: str
    name: str
    ready: bool
    message: str


# === Netze finden ===
def _import_module(folder: Path) -> types.ModuleType:
    """Importiert network.py als Teil eines eigenen Pakets je Ordner.

    So funktionieren relative Importe ("from .model import MeinModell"), und zwei Netze mit gleich
    benannten Hilfsdateien (z. B. beide mit model.py) kommen sich nicht in die Quere.
    """
    if _PACKAGE not in sys.modules:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = []
        sys.modules[_PACKAGE] = root
    package_name = _PACKAGE + "." + re.sub(r"\W", "_", folder.name)
    for key in [k for k in sys.modules if k == package_name or k.startswith(package_name + ".")]:
        del sys.modules[key]  # beim erneuten Einlesen den aktuellen Stand der Dateien nehmen
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.network")


def _network_classes(module: types.ModuleType) -> list[type[NeuralNetwork]]:
    return [
        obj for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, NeuralNetwork) and obj is not NeuralNetwork
        and obj.__module__ == module.__name__
    ]


def _check(net: NeuralNetwork, taken: set[str]) -> Optional[str]:
    """Grund, warum das Netz nicht angeboten wird, oder None."""
    if not NAME_PATTERN.match(net.name or ""):
        return f"name {net.name!r} ist ungültig: nur Kleinbuchstaben a-z, Ziffern und _, Beginn mit Buchstabe"
    if net.name in taken:
        return f"name {net.name!r} ist schon vergeben"
    if type(net).predict is NeuralNetwork.predict:
        return "predict() fehlt"
    if not net.description:
        return "description fehlt (das Sprachmodell muss wissen, wofür das Netz da ist)"
    if not net.enabled:
        return "ausgeschaltet (enabled = False)"
    missing = [f for f in net.required_files if not net.file(f).exists()]
    if missing:
        return f"Datei fehlt im Netz-Ordner: {', '.join(missing)} (erst trainieren?)"
    return None


def discover(directory: Optional[Path] = None, reserved: Sequence[str] = ()) -> tuple[list[NeuralNetwork], list[NetworkStatus]]:
    """Liest alle Netz-Ordner ein. Gibt die einsatzbereiten Netze und den Status jedes Ordners zurück.

    Fehler in einem Netz (Syntaxfehler, fehlende Pakete, fehlende Gewichte) halten die App nicht auf,
    sie erscheinen nur im Status und im Log.
    """
    directory = directory or config.NETWORKS_DIR
    ready: list[NeuralNetwork] = []
    statuses: list[NetworkStatus] = []
    taken = set(reserved)
    if not directory.is_dir():
        return ready, statuses
    for folder in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if not (folder / NETWORK_FILE).is_file():
            continue
        try:
            module = _import_module(folder)
            classes = _network_classes(module)
        except Exception as exc:
            logger.warning("Netz-Ordner %s konnte nicht geladen werden", folder, exc_info=True)
            statuses.append(NetworkStatus(folder.name, "", False, f"{NETWORK_FILE} fehlerhaft: {type(exc).__name__}: {exc}"))
            continue
        if not classes:
            statuses.append(NetworkStatus(folder.name, "", False, f"keine Klasse, die von NeuralNetwork erbt, in {NETWORK_FILE}"))
            continue
        for cls in classes:
            try:
                net = cls()
                net._prepare(folder)
            except Exception as exc:
                logger.warning("Netz %s konnte nicht angelegt werden", cls.__name__, exc_info=True)
                statuses.append(NetworkStatus(folder.name, "", False, f"{cls.__name__}(): {type(exc).__name__}: {exc}"))
                continue
            problem = _check(net, taken)
            statuses.append(NetworkStatus(folder.name, net.name, problem is None, problem or "bereit"))
            if problem is None:
                taken.add(net.name)
                ready.append(net)
    for status in statuses:
        log = logger.info if status.ready else logger.warning
        log("Eigenes Netz %s (%s): %s", status.name or "?", status.folder, status.message)
    return ready, statuses


# === Netz als LangChain-Tool ===
def _publish_image(image: Path) -> str:
    """Legt ein Ergebnisbild in den Bilderordner (falls nötig) und gibt den Dateinamen zurück."""
    image = Path(image).resolve()
    images = config.IMAGES_DIR.resolve()
    if image.parent != images:
        images.mkdir(parents=True, exist_ok=True)
        target = images / f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_{image.name}"
        shutil.copyfile(image, target)
        image = target
    return image.name


def run(net: NeuralNetwork, arguments: dict[str, Any]) -> tuple[str, dict]:
    """Ruft das Netz auf wie ein Tool: lädt es beim ersten Mal, fängt Fehler ab.

    Gibt (Text für das Sprachmodell, Artefakt) zurück. Das Artefakt enthält ein Bild, falls das Netz eins liefert.
    """
    try:
        with net._lock:  # ein Aufruf je Netz gleichzeitig; load() läuft nur einmal
            if not net._loaded:
                logger.info("Lade eigenes Netz %s", net.name)
                net.load()
                net._loaded = True
            result = net.predict(**arguments)
    except Exception as exc:
        logger.warning("Eigenes Netz %s ist fehlgeschlagen", net.name, exc_info=True)
        return f"Das neuronale Netz '{net.name}' ist fehlgeschlagen: {type(exc).__name__}: {exc}", {}
    if isinstance(result, NetworkResult):
        if result.image is None:
            return result.text, {}
        return result.text, {"name": _publish_image(result.image), "network": net.name}
    return str(result), {}


def as_tool(net: NeuralNetwork) -> BaseTool:
    """Macht aus einem Netz ein LangChain-Tool. Die Argumente kommen aus der Signatur von predict()."""
    schema = create_schema_from_function(net.name, net.predict, parse_docstring=True, error_on_invalid_docstring=False)

    def call(**arguments: Any) -> tuple[str, dict]:
        return run(net, arguments)

    return StructuredTool.from_function(
        func=call,
        name=net.name,
        description=net.description,
        args_schema=schema,
        response_format="content_and_artifact",
    )


# === Zwischenspeicher für die laufende App ===
_cache: Optional[tuple[list[NeuralNetwork], list[NetworkStatus], list[BaseTool]]] = None
_cache_lock = threading.Lock()


def _loaded() -> tuple[list[NeuralNetwork], list[NetworkStatus], list[BaseTool]]:
    global _cache
    with _cache_lock:
        if _cache is None:
            from .tools import TOOLS  # Namen der eingebauten Tools sind für eigene Netze tabu

            nets, statuses = discover(reserved=[t.name for t in TOOLS])
            _cache = (nets, statuses, [as_tool(n) for n in nets])
        return _cache


def networks() -> list[NeuralNetwork]:
    return list(_loaded()[0])


def statuses() -> list[NetworkStatus]:
    return list(_loaded()[1])


def tools() -> list[BaseTool]:
    """Alle einsatzbereiten Netze als Tools, zum Anhängen an create_agent(tools=...)."""
    return list(_loaded()[2])


def labels() -> dict[str, str]:
    """{Tool-Name: Anzeigename} für die Arbeitsschritte in der Oberfläche."""
    return {n.name: n.label for n in networks()}


def prompt_section() -> str:
    """Abschnitt für den System-Prompt des Werkzeug-Agenten. Kleine Modelle finden Tools so leichter."""
    nets = networks()
    if not nets:
        return ""
    lines = [f"- {n.name}: {n.description.splitlines()[0]}" for n in nets]
    return (
        "- Eigene neuronale Netze (als Tools): Rufe sie auf, wenn die Frage zu ihrer Aufgabe passt, und erfinde "
        "keine Ergebnisse. Dateien (z. B. angehängte Bilder) übergibst du mit ihrem Pfad im Workspace.\n"
        + "\n".join(f"  {line}" for line in lines)
    )


def reset() -> None:
    """Vergisst die eingelesenen Netze. Beim nächsten Zugriff wird NETWORKS_DIR neu gelesen."""
    global _cache
    with _cache_lock:
        _cache = None


# === Kommandozeile: python main.py --mode networks [--network NAME --args JSON] ===
def cli(name: Optional[str] = None, arguments: str = "{}") -> int:
    """Zeigt den Status aller Netze. Mit name wird das Netz direkt aufgerufen, ohne Sprachmodell."""
    print(f"Netz-Ordner: {config.NETWORKS_DIR}")
    found = statuses()
    if not found:
        print("Keine Netze gefunden. Je Netz ein Unterordner mit network.py anlegen, siehe docs/eigene-netze.md")
    for status in found:
        mark = "OK  " if status.ready else "--  "
        print(f"  {mark}{status.name or '?':28} {status.folder + '/':16} {status.message}")
    if not name:
        return 0 if all(s.ready for s in found) else 1

    net = next((n for n in networks() if n.name == name), None)
    if net is None:
        print(f"\nNetz '{name}' ist nicht einsatzbereit oder unbekannt.")
        return 1
    tool = as_tool(net)
    print(f"\nTool, wie es das Sprachmodell sieht:\n  Name: {tool.name}\n  Beschreibung: {tool.description}")
    print(f"  Argumente: {json.dumps(tool.tool_call_schema.model_json_schema().get('properties', {}), ensure_ascii=False)}")
    try:
        parsed = json.loads(arguments)
        validated = tool.args_schema.model_validate(parsed).model_dump()
    except Exception as exc:
        print(f"\nUngültige Argumente {arguments}: {exc}")
        return 1
    text, artifact = run(net, validated)
    print(f"\nAufruf mit {validated}\nErgebnis:\n{text}")
    if artifact.get("name"):
        print(f"Bild: {config.IMAGES_DIR / artifact['name']}")
    return 0 if not text.startswith(f"Das neuronale Netz '{net.name}' ist fehlgeschlagen") else 1
