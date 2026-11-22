"""Aktienkursprognose: schätzt mit einem LSTM die Schlusskurse der nächsten Handelstage.

Die Gewichte (kurs_lstm.pt) erzeugt train.py. Zeigt zusätzlich ein Diagramm im Chat (NetworkResult mit Bild).
"""
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from Agents.neural_networks import NetworkResult, NeuralNetwork

from .daten import schlusskurse_lesen
from .model import KursLSTM, renditen


class Aktienkursprognose(NeuralNetwork):
    name = "aktienkurs_vorhersagen"
    label = "Aktienkurs vorhersagen"
    description = (
        "Schätzt mit einem LSTM-Netz die Schlusskurse der nächsten Handelstage aus einer CSV-Datei mit historischen "
        "Kursen im Workspace (Spalten Datum und Schlusskurs bzw. Close). Liefert eine Tabelle und ein Diagramm. "
        "Nur eine statistische Schätzung, keine Anlageberatung."
    )
    required_files = ["kurs_lstm.pt"]

    def load(self) -> None:
        gespeichert = torch.load(self.file("kurs_lstm.pt"), map_location="cpu", weights_only=True)
        self.fenster = gespeichert["fenster"]
        self.model = KursLSTM()
        self.model.load_state_dict(gespeichert["state_dict"])
        self.model.eval()

    def predict(self, csv_datei: str, tage: int = 5) -> NetworkResult:
        """Schätzt die nächsten Schlusskurse.

        Args:
            csv_datei: Pfad der CSV-Datei im Workspace, z. B. "Dateien/aktie.csv".
            tage: Anzahl der Handelstage, die geschätzt werden (1 bis 30).
        """
        tage = max(1, min(int(tage), 30))
        daten, kurse = schlusskurse_lesen(self.workspace_file(csv_datei))
        if len(kurse) <= self.fenster:
            raise ValueError(f"Es werden mindestens {self.fenster + 1} Kurse gebraucht, die Datei hat {len(kurse)}.")

        folge = list(renditen(kurse)[-self.fenster:])
        kurs, prognose = kurse[-1], []
        with torch.no_grad():
            for _ in range(tage):  # Tag für Tag: Schätzung anhängen und weiterrechnen
                x = torch.tensor(folge[-self.fenster:], dtype=torch.float32).view(1, -1, 1)
                rendite = self.model(x).item()
                folge.append(rendite)
                kurs *= math.exp(rendite / 100)
                prognose.append(kurs)

        zeilen = [f"Letzter Schlusskurs ({daten[-1]}): {kurse[-1]:.2f}"]
        zeilen += [f"Handelstag +{i}: {p:.2f} ({p / kurse[-1] - 1:+.1%})" for i, p in enumerate(prognose, 1)]
        zeilen.append("Hinweis: Statistische Schätzung eines neuronalen Netzes, keine Anlageberatung.")
        bild = self.new_image_path()
        diagramm_zeichnen(kurse[-90:], prognose, bild)
        return NetworkResult("\n".join(zeilen), bild)


def diagramm_zeichnen(verlauf: list[float], prognose: list[float], ziel: Path) -> None:
    """Einfaches Liniendiagramm mit Pillow: Verlauf blau, Prognose orange."""
    breite, hoehe, rand = 900, 420, 60
    bild = Image.new("RGB", (breite, hoehe), "white")
    zeichnen = ImageDraw.Draw(bild)
    alle = verlauf + prognose
    tief, hoch = min(alle), max(alle)
    spanne = (hoch - tief) or 1.0

    def punkt(i: int, wert: float) -> tuple[float, float]:
        x = rand + i * (breite - 2 * rand) / max(len(alle) - 1, 1)
        return x, hoehe - rand - (wert - tief) / spanne * (hoehe - 2 * rand)

    zeichnen.rectangle([rand, rand, breite - rand, hoehe - rand], outline="#cccccc")
    zeichnen.line([punkt(i, w) for i, w in enumerate(verlauf)], fill="#1f6fb2", width=3)
    zeichnen.line([punkt(i, w) for i, w in enumerate([verlauf[-1]] + prognose, len(verlauf) - 1)], fill="#e8710a", width=3)
    zeichnen.text((rand, 20), "Schlusskurs: Verlauf (blau) und Prognose (orange)", fill="black")
    zeichnen.text((8, rand - 6), f"{hoch:.2f}", fill="#555555")
    zeichnen.text((8, hoehe - rand - 6), f"{tief:.2f}", fill="#555555")
    bild.save(ziel)
