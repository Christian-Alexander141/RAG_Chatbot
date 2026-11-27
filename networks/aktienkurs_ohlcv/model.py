"""Architektur und Aufbereitung der Daten. Nutzen beide: train.py (Training) und network.py (Einsatz im Chatbot).

Vorlage ist das LSTM aus "Stock-Market-Prediction-Model" von Harshit Kushwaha (MIT-Lizenz, auch veröffentlicht als
hrirkslab/Stock-Market-Prediction-Using-ML), dort mit TensorFlow/Keras. Nachgebaut in PyTorch, mit drei Änderungen:

1. Eingabe nur Eröffnung, Hoch, Tief, Schluss und Volumen. Die vier Kennzahlen des Originals (EPS, Umsatz,
   Eigenkapitalrendite, KGV) waren für alle Tage derselbe heutige Wert, enthielten also keine Information über den
   jeweiligen Tag, dafür aber Wissen aus der Zukunft.
2. Kurse relativ zum letzten Schlusskurs des Fensters statt in Dollar. Das Original skalierte auf den Kursbereich
   der Trainingsaktie und konnte deshalb nur Kurse in diesem Bereich schätzen.
3. Die letzte Schicht schätzt die nächsten HORIZONT Handelstage auf einmal. Das Original schätzte einen Tag und
   wollte für die Zukunft Fenster aus 60 erfundenen Tagen bilden, hatte aber nur 30, und lieferte deshalb Nullen.
"""
import numpy as np
import torch
from torch import nn

FENSTER = 60  # wie im Original: so viele Handelstage sieht das Netz
HORIZONT = 30  # so viele folgende Handelstage schätzt es
MERKMALE = 5  # Eröffnung, Hoch, Tief, Schluss, Volumen


def eingabe(fenster: np.ndarray) -> np.ndarray:
    """Ein Fenster (FENSTER, 5) Kerzen als Eingabe für das Netz.

    Kurse: logarithmischer Abstand zum letzten Schlusskurs in Prozent. Volumen: Logarithmus des Verhältnisses
    zum mittleren Volumen im Fenster. So sehen eine Aktie für 5 € und eine für 500 € gleich aus.
    """
    kurse = np.log(fenster[:, :4] / fenster[-1, 3]) * 100
    volumen = np.log((fenster[:, 4] + 1) / (fenster[:, 4].mean() + 1))
    return np.column_stack([kurse, volumen])


def ziel(schluss_danach: np.ndarray, letzter_schluss: float) -> np.ndarray:
    """Was das Netz lernen soll: Abstand der nächsten HORIZONT Schlusskurse zum letzten bekannten, in Prozent (log)."""
    return np.log(schluss_danach / letzter_schluss) * 100


class KerzenLSTM(nn.Module):
    """Wie im Original: LSTM(50) -> LSTM(50) -> Dense(25) -> Dense, die Dense-Schichten ohne Aktivierung.

    Statt eines Werts liefert die letzte Schicht HORIZONT Werte (Tag +1 bis +HORIZONT).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=MERKMALE, hidden_size=50, num_layers=2, batch_first=True)
        self.dicht = nn.Linear(50, 25)
        self.ausgabe = nn.Linear(25, HORIZONT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Batch, FENSTER, MERKMALE), normiert. Rückgabe: (Batch, HORIZONT), normiert."""
        folge, _ = self.lstm(x)
        return self.ausgabe(self.dicht(folge[:, -1]))
