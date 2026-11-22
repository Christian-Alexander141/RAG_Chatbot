"""Handschrifterkennung: erkennt eine handgeschriebene Ziffer (0 bis 9) auf einem Bild.

Die Gewichte (ziffern_cnn.pt) erzeugt train.py. Fehlen sie, bietet der Chatbot das Netz nicht an.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from Agents.neural_networks import NeuralNetwork

from .model import MEAN, STD, ZiffernCNN


class Handschrifterkennung(NeuralNetwork):
    name = "handschrift_erkennen"
    label = "Handschrift erkennen"
    description = (
        "Erkennt eine einzelne handgeschriebene Ziffer (0 bis 9) auf einem Bild im Workspace, z. B. auf einem an "
        "die Nachricht angehängten Foto. Liefert die erkannte Ziffer und wie sicher sich das Netz ist."
    )
    required_files = ["ziffern_cnn.pt"]

    def load(self) -> None:
        self.model = ZiffernCNN()
        self.model.load_state_dict(torch.load(self.file("ziffern_cnn.pt"), map_location="cpu", weights_only=True))
        self.model.eval()  # Dropout aus; das Netz ist so klein, dass die CPU reicht

    def predict(self, bild: str) -> str:
        """Erkennt die Ziffer auf dem Bild.

        Args:
            bild: Pfad des Bildes im Workspace, z. B. "Uploads/ziffer.png".
        """
        x = bild_vorbereiten(self.workspace_file(bild))
        with torch.no_grad():
            wahrscheinlichkeiten = torch.softmax(self.model(x), dim=1)[0]
        werte, ziffern = torch.topk(wahrscheinlichkeiten, 3)
        weitere = ", ".join(f"{z} ({w:.0%})" for w, z in zip(werte[1:].tolist(), ziffern[1:].tolist()))
        return f"Erkannte Ziffer: {ziffern[0].item()} (Sicherheit {werte[0].item():.0%}). Weitere Kandidaten: {weitere}."


def bild_vorbereiten(pfad: Path) -> torch.Tensor:
    """Bringt ein Foto in die Form der MNIST-Trainingsbilder: 28 x 28 Pixel, helle Ziffer auf schwarzem Grund,
    Ziffer etwa 20 x 20 Pixel groß und mit dem Schwerpunkt in der Bildmitte."""
    graustufen = ImageOps.exif_transpose(Image.open(pfad)).convert("L")
    a = np.asarray(graustufen, dtype=np.float32) / 255.0
    if a.mean() > 0.5:  # dunkle Schrift auf hellem Papier: umdrehen
        a = 1.0 - a
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    a[a < 0.35] = 0.0  # Papierstruktur und Schatten entfernen

    zeilen, spalten = np.nonzero(a)
    if len(zeilen) == 0:
        raise ValueError("Auf dem Bild ist keine Schrift zu erkennen.")
    a = a[zeilen.min():zeilen.max() + 1, spalten.min():spalten.max() + 1]

    hoehe, breite = a.shape
    faktor = 20 / max(hoehe, breite)
    klein = Image.fromarray((a * 255).astype(np.uint8)).resize(
        (max(1, round(breite * faktor)), max(1, round(hoehe * faktor))), Image.LANCZOS
    )
    k = np.asarray(klein, dtype=np.float32) / 255.0
    feld = np.zeros((28, 28), dtype=np.float32)
    oben, links = (28 - k.shape[0]) // 2, (28 - k.shape[1]) // 2
    feld[oben:oben + k.shape[0], links:links + k.shape[1]] = k

    # Schwerpunkt in die Mitte schieben, wie bei MNIST
    gesamt = feld.sum()
    y = (feld.sum(axis=1) * np.arange(28)).sum() / gesamt
    x = (feld.sum(axis=0) * np.arange(28)).sum() / gesamt
    feld = np.roll(feld, (round(13.5 - y), round(13.5 - x)), axis=(0, 1))

    return torch.from_numpy((feld - MEAN) / STD).view(1, 1, 28, 28)
