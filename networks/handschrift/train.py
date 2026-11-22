"""Trainiert das Ziffern-CNN auf MNIST und speichert die Gewichte als ziffern_cnn.pt in diesem Ordner.

Aufruf aus dem Projektordner, mit aktivierter .venv:
    python networks/handschrift/train.py              (3 Epochen, etwa 1 Minute auf GPU, 3-5 Minuten auf CPU)
    python networks/handschrift/train.py --epochen 5

MNIST (etwa 12 MB) wird beim ersten Mal nach networks/handschrift/daten/ heruntergeladen.
"""
import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MEAN, STD, ZiffernCNN  # model.py liegt im selben Ordner

ORDNER = Path(__file__).resolve().parent
DATEN = ORDNER / "daten"
GEWICHTE = ORDNER / "ziffern_cnn.pt"
QUELLEN = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]
DATEIEN = {
    "train_bilder": "train-images-idx3-ubyte.gz",
    "train_ziffern": "train-labels-idx1-ubyte.gz",
    "test_bilder": "t10k-images-idx3-ubyte.gz",
    "test_ziffern": "t10k-labels-idx1-ubyte.gz",
}


def herunterladen(datei: str) -> Path:
    ziel = DATEN / datei
    if ziel.exists():
        return ziel
    DATEN.mkdir(parents=True, exist_ok=True)
    for quelle in QUELLEN:
        try:
            print(f"Lade {quelle}{datei}")
            urllib.request.urlretrieve(quelle + datei, ziel)
            return ziel
        except OSError as exc:
            print(f"  fehlgeschlagen: {exc}")
    raise SystemExit(f"{datei} konnte nicht geladen werden. Datei von Hand nach {DATEN} legen.")


def idx_lesen(pfad: Path) -> np.ndarray:
    """Liest das IDX-Format von MNIST: Kopf mit Anzahl und Größe, danach die Bytes."""
    with gzip.open(pfad, "rb") as f:
        _, _, dimensionen = struct.unpack(">HBB", f.read(4))  # 2 Nullbytes, Datentyp, Anzahl Dimensionen
        form = struct.unpack(">" + "I" * dimensionen, f.read(4 * dimensionen))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(form)


def als_tensoren(bilder: np.ndarray, ziffern: np.ndarray) -> TensorDataset:
    x = (torch.from_numpy(bilder.copy()).float() / 255.0 - MEAN) / STD
    return TensorDataset(x.unsqueeze(1), torch.from_numpy(ziffern.copy()).long())


def genauigkeit(model: nn.Module, daten: DataLoader, device: str) -> float:
    model.eval()
    richtig = 0
    with torch.no_grad():
        for x, y in daten:
            richtig += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
    return richtig / len(daten.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochen", type=int, default=3)
    args = parser.parse_args()

    daten = {name: idx_lesen(herunterladen(datei)) for name, datei in DATEIEN.items()}
    train = DataLoader(als_tensoren(daten["train_bilder"], daten["train_ziffern"]), batch_size=128, shuffle=True)
    test = DataLoader(als_tensoren(daten["test_bilder"], daten["test_ziffern"]), batch_size=1000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZiffernCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Training auf {device}, {len(train.dataset)} Bilder, {args.epochen} Epochen")

    for epoche in range(1, args.epochen + 1):
        model.train()
        for x, y in train:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        print(f"Epoche {epoche}: Verlust {loss.item():.4f}, Genauigkeit Testdaten {genauigkeit(model, test, device):.2%}")

    # Nur die Gewichte speichern (state_dict). network.py baut das Netz aus model.py und lädt sie hinein.
    torch.save(model.cpu().state_dict(), GEWICHTE)
    print(f"Gespeichert: {GEWICHTE}")


if __name__ == "__main__":
    main()
