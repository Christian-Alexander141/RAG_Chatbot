"""Trainiert das Kurs-LSTM und speichert es als kurs_lstm.pt in diesem Ordner.

Aufruf aus dem Projektordner, mit aktivierter .venv:
    python networks/aktienkurs/train.py kurse.csv [weitere.csv ...] [--epochen 30]
    python networks/aktienkurs/train.py --demo      (künstliche Kurse, nur um den Ablauf auszuprobieren)

Die CSV-Dateien brauchen eine Spalte mit Schlusskursen (Close, Schlusskurs, ...), z. B. den Export der
historischen Kurse von Yahoo Finance oder einer Börsenseite. Mehrere Jahre Tageskurse sind sinnvoll.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from daten import schlusskurse_lesen  # daten.py und model.py liegen im selben Ordner
from model import FENSTER, KursLSTM, renditen

ORDNER = Path(__file__).resolve().parent
GEWICHTE = ORDNER / "kurs_lstm.pt"


def demo_kurse(anzahl: int = 8, tage: int = 1500) -> list[np.ndarray]:
    """Künstliche Kurse mit leichtem Trend und Gedächtnis in den Renditen, damit das Netz etwas lernen kann.
    Echte Kurse sind weit weniger vorhersagbar."""
    zufall = np.random.default_rng(0)
    reihen = []
    for _ in range(anzahl):
        r = np.zeros(tage)
        for t in range(1, tage):
            r[t] = 0.03 + 0.3 * r[t - 1] + 0.2 * np.sin(t / 15) + zufall.normal(0, 1.2)
        reihen.append(100 * np.exp(np.cumsum(r) / 100))
    return reihen


def fenster_bilden(kurse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = renditen(kurse)
    x = np.stack([r[i:i + FENSTER] for i in range(len(r) - FENSTER)])
    return x, r[FENSTER:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", type=Path, help="CSV-Dateien mit historischen Kursen")
    parser.add_argument("--demo", action="store_true", help="mit künstlichen Kursen trainieren")
    parser.add_argument("--epochen", type=int, default=30)
    args = parser.parse_args()
    if not args.csv and not args.demo:
        parser.error("CSV-Dateien angeben oder --demo")

    reihen = demo_kurse() if args.demo else [np.array(schlusskurse_lesen(p)[1]) for p in args.csv]
    # Je Reihe die letzten 15 % zum Prüfen zurückhalten (zeitlich danach, wie im echten Einsatz)
    train_x, train_y, test_x, test_y = [], [], [], []
    for kurse in reihen:
        if len(kurse) < FENSTER + 20:
            raise SystemExit(f"Zu wenige Kurse ({len(kurse)}), gebraucht werden mindestens {FENSTER + 20}.")
        x, y = fenster_bilden(kurse)
        grenze = int(len(x) * 0.85)
        train_x.append(x[:grenze]); train_y.append(y[:grenze])
        test_x.append(x[grenze:]); test_y.append(y[grenze:])

    def tensor(teile: list[np.ndarray]) -> torch.Tensor:
        return torch.tensor(np.concatenate(teile), dtype=torch.float32)

    tx, ty = tensor(train_x).unsqueeze(-1), tensor(train_y).unsqueeze(-1)
    vx, vy = tensor(test_x).unsqueeze(-1), tensor(test_y).unsqueeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KursLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print(f"Training auf {device}: {len(tx)} Fenster, Prüfung: {len(vx)} Fenster")

    # Vergleichswert: "morgen ändert sich nichts" (Rendite 0). Ist das Netz nicht besser, hat es nichts gelernt.
    naiv = loss_fn(torch.zeros_like(vy), vy).item()
    for epoche in range(1, args.epochen + 1):
        model.train()
        reihenfolge = torch.randperm(len(tx))
        for start in range(0, len(tx), 256):
            i = reihenfolge[start:start + 256]
            optimizer.zero_grad()
            loss = loss_fn(model(tx[i].to(device)), ty[i].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pruefung = loss_fn(model(vx.to(device)), vy.to(device)).item()
        if epoche == 1 or epoche % 5 == 0 or epoche == args.epochen:
            print(f"Epoche {epoche}: Fehler Prüfdaten {pruefung:.3f} (ohne Netz, Rendite 0: {naiv:.3f})")

    torch.save({"state_dict": model.cpu().state_dict(), "fenster": FENSTER}, GEWICHTE)
    print(f"Gespeichert: {GEWICHTE}")


if __name__ == "__main__":
    main()
