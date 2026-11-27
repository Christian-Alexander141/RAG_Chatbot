"""Aktienkursprognose aus Tageskerzen: LSTM nach dem Projekt "Stock-Market-Prediction-Model" (Harshit Kushwaha,
MIT-Lizenz), in PyTorch nachgebaut (siehe model.py).

Die Gewichte (kerzen_lstm.pt) erzeugt train.py. Zeigt zusätzlich ein Diagramm im Chat (NetworkResult mit Bild).
"""
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from Agents.neural_networks import NetworkResult, NeuralNetwork

from .daten import OhneKerzen, kerzen_lesen
from .model import KerzenLSTM, eingabe


class KerzenPrognose(NeuralNetwork):
    name = "aktienkurs_ohlcv_vorhersagen"
    label = "Aktienkurs vorhersagen (Kerzen-LSTM)"
    description = (
        "Schätzt mit einem LSTM-Netz die Schlusskurse der nächsten Handelstage aus einer CSV-Datei mit Tageskerzen "
        "im Workspace: Spalten Datum, Eröffnung (Open), Hoch (High), Tief (Low), Schluss (Close) und Volumen "
        "(Volume), z. B. ein Export von Yahoo Finance. Hat die Datei nur Schlusskurse, stattdessen "
        "aktienkurs_vorhersagen nutzen. Liefert eine Tabelle und ein Diagramm. Nur eine statistische Schätzung, "
        "keine Anlageberatung."
    )
    required_files = ["kerzen_lstm.pt"]

    def load(self) -> None:
        gespeichert = torch.load(self.file("kerzen_lstm.pt"), map_location="cpu", weights_only=True)
        self.fenster, self.horizont = gespeichert["fenster"], gespeichert["horizont"]
        self.quelle = gespeichert.get("quelle", "")
        self.normen = {k: gespeichert[k].numpy() for k in ("eingabe_mittel", "eingabe_std", "ziel_mittel", "ziel_std")}
        self.model = KerzenLSTM()
        self.model.load_state_dict(gespeichert["state_dict"])
        self.model.eval()

    def predict(self, csv_datei: str, tage: int = 5) -> NetworkResult:
        """Schätzt die nächsten Schlusskurse.

        Args:
            csv_datei: Pfad der CSV-Datei im Workspace, z. B. "Uploads/aapl.csv".
            tage: Anzahl der Handelstage, die geschätzt werden (1 bis 30).
        """
        tage = max(1, min(int(tage), self.horizont))
        try:
            daten, kerzen = kerzen_lesen(self.workspace_file(csv_datei))
        except OhneKerzen as exc:
            raise ValueError(f"{exc} Für Dateien nur mit Schlusskursen gibt es aktienkurs_vorhersagen.") from exc
        if len(kerzen) < self.fenster:
            raise ValueError(f"Es werden mindestens {self.fenster} Handelstage gebraucht, die Datei hat {len(kerzen)}.")

        n = self.normen
        x = (eingabe(kerzen[-self.fenster:]) - n["eingabe_mittel"]) / n["eingabe_std"]
        with torch.no_grad():
            ausgabe = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))[0].numpy()
        abstand = ausgabe * n["ziel_std"] + n["ziel_mittel"]  # Prozent (log) zum letzten Schlusskurs
        letzter = float(kerzen[-1, 3])
        prognose = [letzter * math.exp(a / 100) for a in abstand[:tage]]

        zeilen = [f"Letzter Schlusskurs ({daten[-1][:10]}): {letzter:.2f}"]
        zeilen += [f"Handelstag +{i}: {p:.2f} ({p / letzter - 1:+.1%})" for i, p in enumerate(prognose, 1)]
        if self.quelle:
            zeilen.append(f"Trainiert mit: {self.quelle}.")
        zeilen.append("Hinweis: Statistische Schätzung eines neuronalen Netzes, keine Anlageberatung.")
        bild = self.new_image_path()
        diagramm_zeichnen(list(kerzen[-90:, 3]), prognose, bild)
        return NetworkResult("\n".join(zeilen), bild)


def diagramm_zeichnen(verlauf: list[float], prognose: list[float], ziel: Path) -> None:
    """Liniendiagramm mit Pillow: Schlusskurse blau, Prognose orange."""
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
    zeichnen.text((rand, 20), "Kerzen-LSTM: Schlusskurse (blau) und Prognose (orange)", fill="black")
    zeichnen.text((8, rand - 6), f"{hoch:.2f}", fill="#555555")
    zeichnen.text((8, hoehe - rand - 6), f"{tief:.2f}", fill="#555555")
    bild.save(ziel)
