"""Architektur des Netzes. Nutzen beide: train.py (Training) und network.py (Einsatz im Chatbot).

Nur PyTorch, keine Imports aus dem Chatbot. So läuft train.py auch ohne den Rest des Projekts.
"""
import torch
from torch import nn

# Mittelwert und Standardabweichung der MNIST-Pixel. Beim Einsatz genauso normalisieren wie beim Training.
MEAN = 0.1307
STD = 0.3081


class ZiffernCNN(nn.Module):
    """Kleines CNN für 28 x 28 Pixel große Graustufenbilder, 10 Klassen (Ziffern 0 bis 9)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Batch, 1, 28, 28), normalisiert mit MEAN und STD. Rückgabe: (Batch, 10) Logits."""
        return self.classifier(self.features(x))
