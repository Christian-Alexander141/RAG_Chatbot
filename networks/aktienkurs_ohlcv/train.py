"""Trainiert das Kerzen-LSTM und speichert es als kerzen_lstm.pt in diesem Ordner.

Aufruf aus dem Projektordner, mit aktivierter .venv:
    python networks/aktienkurs_ohlcv/train.py kurse.csv [weitere.csv ...] [--epochen 20]
    python networks/aktienkurs_ohlcv/train.py --demo      (künstliche Kurse, nur um den Ablauf auszuprobieren)

Die CSV-Dateien brauchen Tageskerzen: Eröffnung, Hoch, Tief, Schluss und Volumen (z. B. der Export historischer
Kurse von Yahoo Finance). Die Daten des Originalprojekts (Apple 2010 bis 2024) gehen auch, siehe docs/eigene-netze.md.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from daten import kerzen_lesen  # daten.py und model.py liegen im selben Ordner
from model import FENSTER, HORIZONT, KerzenLSTM, eingabe, ziel

ORDNER = Path(__file__).resolve().parent
GEWICHTE = ORDNER / "kerzen_lstm.pt"


def demo_kerzen(anzahl: int = 6, tage: int = 1500) -> list[np.ndarray]:
    """Künstliche Kerzen mit Trend und Gedächtnis in den Renditen, damit das Netz etwas lernen kann.
    Echte Kurse sind weit weniger vorhersagbar."""
    zufall = np.random.default_rng(0)
    reihen = []
    for _ in range(anzahl):
        r = np.zeros(tage)
        for t in range(1, tage):
            r[t] = 0.03 + 0.3 * r[t - 1] + 0.3 * np.sin(t / 20) + zufall.normal(0, 1.0)
        schluss = 100 * np.exp(np.cumsum(r) / 100)
        eroeffnung = np.r_[schluss[0], schluss[:-1]] * np.exp(zufall.normal(0, 0.3, tage) / 100)
        spanne = np.abs(zufall.normal(0, 0.8, (2, tage))) / 100
        hoch = np.maximum(eroeffnung, schluss) * np.exp(spanne[0])
        tief = np.minimum(eroeffnung, schluss) * np.exp(-spanne[1])
        volumen = 1e6 * np.exp(zufall.normal(0, 0.3, tage) + np.abs(r) / 2)
        reihen.append(np.column_stack([eroeffnung, hoch, tief, schluss, volumen]))
    return reihen


def beispiele(kerzen: np.ndarray, von: int, bis: int) -> tuple[np.ndarray, np.ndarray]:
    """Eingaben und Ziele für alle Tage t mit von <= t < bis (t = letzter Tag des Fensters)."""
    x, y = [], []
    for t in range(max(von, FENSTER - 1), min(bis, len(kerzen) - HORIZONT)):
        x.append(eingabe(kerzen[t - FENSTER + 1:t + 1]))
        y.append(ziel(kerzen[t + 1:t + 1 + HORIZONT, 3], kerzen[t, 3]))
    return np.array(x).reshape(-1, FENSTER, 5), np.array(y).reshape(-1, HORIZONT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", type=Path, help="CSV-Dateien mit Tageskerzen")
    parser.add_argument("--demo", action="store_true", help="mit künstlichen Kursen trainieren")
    parser.add_argument("--epochen", type=int, default=20)
    args = parser.parse_args()
    if not args.csv and not args.demo:
        parser.error("CSV-Dateien angeben oder --demo")

    reihen = demo_kerzen() if args.demo else [kerzen_lesen(p)[1] for p in args.csv]
    quelle = "künstliche Demo-Kurse" if args.demo else ", ".join(p.name for p in args.csv)
    # Je Reihe die letzten 15 % zum Prüfen zurückhalten (zeitlich danach, wie im echten Einsatz). Dazwischen
    # HORIZONT Tage Abstand, sonst kämen die Kurse, die geprüft werden, schon als Trainingsziel vor.
    teile = {"train": ([], []), "test": ([], [])}
    for kerzen in reihen:
        if len(kerzen) < FENSTER + 3 * HORIZONT + 20:
            raise SystemExit(f"Zu wenige Tage ({len(kerzen)}), gebraucht werden mindestens {FENSTER + 3 * HORIZONT + 20}.")
        grenze = int(len(kerzen) * 0.85)
        for name, (von, bis) in {"train": (0, grenze - HORIZONT), "test": (grenze, len(kerzen))}.items():
            x, y = beispiele(kerzen, von, bis)
            teile[name][0].append(x)
            teile[name][1].append(y)
    tx, ty, vx, vy = (np.concatenate(teile[n][i]) for n in ("train", "test") for i in (0, 1))

    # Normieren mit Werten aus den Trainingsdaten (werden mitgespeichert, network.py braucht sie)
    eingabe_mittel, eingabe_std = tx.mean(axis=(0, 1)), tx.std(axis=(0, 1)) + 1e-6
    ziel_mittel, ziel_std = ty.mean(axis=0), ty.std(axis=0) + 1e-6

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, y_train = tensor((tx - eingabe_mittel) / eingabe_std), tensor((ty - ziel_mittel) / ziel_std)
    x_test = tensor((vx - eingabe_mittel) / eingabe_std).to(device)
    model = KerzenLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print(f"Training auf {device}: {len(tx)} Fenster aus {quelle}, Prüfung: {len(vx)} Fenster")

    def fehler() -> np.ndarray:
        """Mittlerer absoluter Fehler auf den Prüfdaten je Tag, in Prozent des letzten Kurses."""
        model.eval()
        with torch.no_grad():
            schaetzung = model(x_test).cpu().numpy() * ziel_std + ziel_mittel
        return np.abs(schaetzung - vy).mean(axis=0)

    naiv = np.abs(vy).mean(axis=0)  # "der Kurs bleibt, wie er ist"
    for epoche in range(1, args.epochen + 1):
        model.train()
        reihenfolge = torch.randperm(len(x_train))
        for start in range(0, len(x_train), 64):
            i = reihenfolge[start:start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[i].to(device)), y_train[i].to(device))
            loss.backward()
            optimizer.step()
        if epoche == 1 or epoche % 5 == 0 or epoche == args.epochen:
            f = fehler()
            print(f"Epoche {epoche}: Fehler Prüfdaten Tag +1 {f[0]:.2f} %, Tag +5 {f[4]:.2f} %, "
                  f"Tag +{HORIZONT} {f[-1]:.2f} % (Kurs bleibt gleich: {naiv[0]:.2f} / {naiv[4]:.2f} / {naiv[-1]:.2f} %)")

    torch.save({
        "state_dict": model.cpu().state_dict(), "fenster": FENSTER, "horizont": HORIZONT, "quelle": quelle,
        "eingabe_mittel": tensor(eingabe_mittel), "eingabe_std": tensor(eingabe_std),
        "ziel_mittel": tensor(ziel_mittel), "ziel_std": tensor(ziel_std),
    }, GEWICHTE)
    print(f"Gespeichert: {GEWICHTE}")


if __name__ == "__main__":
    main()
