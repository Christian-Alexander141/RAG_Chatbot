"""Eigene neuronale Netze: Einlesen aus networks/, Tool-Schema, Fehlerbehandlung und Aufruf durch den Agenten."""
import textwrap
from pathlib import Path

from langchain_core.messages import AIMessage

from Agents import chat_agent, neural_networks
from rag_chatpipline import config

from test_agent_api import ScriptedModel

VERDOPPLER = '''
from Agents.neural_networks import NeuralNetwork

from .helfer import FAKTOR


class Verdoppler(NeuralNetwork):
    name = "verdoppeln"
    label = "Verdoppeln"
    description = "Verdoppelt eine Zahl."
    loads = 0

    def load(self):
        type(self).loads += 1

    def predict(self, zahl: float, runden: bool = False) -> str:
        """Verdoppelt.

        Args:
            zahl: Die Zahl, die verdoppelt wird.
            runden: Ergebnis auf ganze Zahlen runden.
        """
        if zahl < 0:
            raise ValueError("nur positive Zahlen")
        ergebnis = zahl * FAKTOR
        return f"Ergebnis: {round(ergebnis) if runden else ergebnis}"
'''


def write_network(root: Path, folder: str, code: str, **files: str) -> Path:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "network.py").write_text(textwrap.dedent(code), encoding="utf-8")
    for name, content in files.items():
        (path / name).write_text(content, encoding="utf-8")
    return path


def test_discover_reports_ready_and_broken_networks(workspace):
    root = config.NETWORKS_DIR
    write_network(root, "verdoppler", VERDOPPLER, **{"helfer.py": "FAKTOR = 2\n"})
    # zweites Netz mit gleich benannter Hilfsdatei: darf sich nicht mit dem ersten vermischen
    write_network(root, "verdreifacher", VERDOPPLER.replace('"verdoppeln"', '"verdreifachen"'), **{"helfer.py": "FAKTOR = 3\n"})
    write_network(root, "kaputt", "def oops(:\n")
    write_network(root, "ohne_gewichte", '''
        from Agents.neural_networks import NeuralNetwork

        class OhneGewichte(NeuralNetwork):
            name = "ohne_gewichte"
            description = "x"
            required_files = ["model.pt"]

            def predict(self) -> str:
                return "x"
    ''')
    write_network(root, "belegt", '''
        from Agents.neural_networks import NeuralNetwork

        class Belegt(NeuralNetwork):
            name = "read_file"
            description = "x"

            def predict(self) -> str:
                return "x"
    ''')
    (root / "nur_ein_ordner").mkdir()

    statuses = {s.folder: s for s in neural_networks.statuses()}
    assert set(statuses) == {"verdoppler", "verdreifacher", "kaputt", "ohne_gewichte", "belegt"}
    assert statuses["verdoppler"].ready and statuses["verdreifacher"].ready
    assert "fehlerhaft" in statuses["kaputt"].message
    assert "model.pt" in statuses["ohne_gewichte"].message
    assert "vergeben" in statuses["belegt"].message

    tools = {t.name: t for t in neural_networks.tools()}
    assert set(tools) == {"verdoppeln", "verdreifachen"}
    assert tools["verdoppeln"].invoke({"zahl": 2.5}) == "Ergebnis: 5.0"
    assert tools["verdreifachen"].invoke({"zahl": 2}) == "Ergebnis: 6.0"
    assert neural_networks.labels()["verdoppeln"] == "Verdoppeln"


def test_tool_schema_comes_from_predict_signature_and_docstring(workspace):
    write_network(config.NETWORKS_DIR, "verdoppler", VERDOPPLER, **{"helfer.py": "FAKTOR = 2\n"})
    tool = neural_networks.tools()[0]
    schema = tool.tool_call_schema.model_json_schema()

    assert tool.description == "Verdoppelt eine Zahl."
    assert schema["required"] == ["zahl"]
    assert schema["properties"]["zahl"]["type"] == "number"
    assert schema["properties"]["zahl"]["description"] == "Die Zahl, die verdoppelt wird."
    assert schema["properties"]["runden"]["default"] is False


def test_errors_go_back_to_the_agent_and_load_runs_once(workspace):
    write_network(config.NETWORKS_DIR, "verdoppler", VERDOPPLER, **{"helfer.py": "FAKTOR = 2\n"})
    net = neural_networks.networks()[0]

    text, artifact = neural_networks.run(net, {"zahl": -1})
    assert "fehlgeschlagen" in text and "nur positive Zahlen" in text and artifact == {}
    neural_networks.run(net, {"zahl": 1})
    assert type(net).loads == 1


def test_network_result_with_image_is_shown_in_chat(workspace):
    write_network(config.NETWORKS_DIR, "diagramm", '''
        from Agents.neural_networks import NetworkResult, NeuralNetwork

        class Diagramm(NeuralNetwork):
            name = "diagramm"
            description = "Zeichnet ein Diagramm."

            def predict(self, datei: str) -> NetworkResult:
                """Args:
                    datei: Pfad im Workspace.
                """
                quelle = self.workspace_file(datei)
                bild = self.new_image_path()
                bild.write_bytes(quelle.read_bytes())
                return NetworkResult("Diagramm erstellt.", bild)
    ''')
    (config.WORKSPACE_DIR / "Uploads" / "werte.png").write_bytes(b"\x89PNG")
    net = neural_networks.networks()[0]

    text, artifact = neural_networks.run(net, {"datei": "Uploads/werte.png"})
    assert text == "Diagramm erstellt."
    assert (config.IMAGES_DIR / artifact["name"]).read_bytes() == b"\x89PNG"

    text, artifact = neural_networks.run(net, {"datei": "../geheim.txt"})
    assert "außerhalb" in text and artifact == {}


def test_agent_calls_network_tool_and_sees_it_in_its_prompt(workspace, vector_store, monkeypatch):
    monkeypatch.setattr(chat_agent.knowledge_base, "_vector_store", vector_store)
    monkeypatch.setattr(chat_agent, "get_chat_model", lambda *args: None)
    write_network(config.NETWORKS_DIR, "verdoppler", VERDOPPLER, **{"helfer.py": "FAKTOR = 2\n"})
    prompts = []

    class Recorder(ScriptedModel):
        def _generate(self, messages, *args, **kwargs):
            prompts.append(messages[0].content)
            return super()._generate(messages, *args, **kwargs)

    model = Recorder(responses=[
        AIMessage(content="", tool_calls=[{"name": "verdoppeln", "args": {"zahl": 21}, "id": "n1"}]),
        AIMessage(content="Das Doppelte von 21 ist 42."),
    ])
    events = list(chat_agent.chat_stream("Was ist das Doppelte von 21?", agent=chat_agent.build_agent(model=model)))

    tool_end = next(e for e in events if e["event"] == "tool_end")
    assert tool_end["name"] == "verdoppeln" and tool_end["output"] == "Ergebnis: 42.0"
    assert events[-1]["result"]["text"] == "Das Doppelte von 21 ist 42."
    assert "verdoppeln: Verdoppelt eine Zahl." in prompts[0]


def test_cli_lists_networks_and_runs_one(workspace, capsys):
    write_network(config.NETWORKS_DIR, "verdoppler", VERDOPPLER, **{"helfer.py": "FAKTOR = 2\n"})

    assert neural_networks.cli("verdoppeln", '{"zahl": 4, "runden": true}') == 0
    output = capsys.readouterr().out
    assert "verdoppeln" in output and "bereit" in output and "Ergebnis: 8" in output
    assert neural_networks.cli("verdoppeln", '{"falsch": 1}') == 1
