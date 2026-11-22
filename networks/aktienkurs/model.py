"""Architektur des Netzes. Nutzen beide: train.py (Training) und network.py (Einsatz im Chatbot)."""
import numpy as np
import torch
from torch import nn

FENSTER = 30  # so viele Tagesrenditen sieht das Netz, um die nächste zu schätzen


def renditen(kurse) -> np.ndarray:
    """Tägliche logarithmische Renditen in Prozent. Anders als Kurse sind sie für alle Aktien ähnlich groß,
    deshalb kann ein Netz, das mit einer Aktie trainiert wurde, auch auf andere angewendet werden."""
    return np.diff(np.log(np.asarray(kurse, dtype=np.float64))) * 100


class KursLSTM(nn.Module):
    """LSTM: liest FENSTER Tagesrenditen und schätzt die Rendite des nächsten Tages."""

    def __init__(self, hidden: int = 64, schichten: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=schichten, batch_first=True, dropout=0.1)
        self.ausgabe = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Batch, FENSTER, 1) Renditen in Prozent. Rückgabe: (Batch, 1) nächste Rendite in Prozent."""
        folge, _ = self.lstm(x)
        return self.ausgabe(folge[:, -1])
