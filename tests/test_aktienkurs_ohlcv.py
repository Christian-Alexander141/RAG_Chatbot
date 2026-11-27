"""Beispielnetz networks/aktienkurs_ohlcv (Kerzen-LSTM nach "Stock-Market-Prediction-Model"): Einlesen der Kerzen,
Aufbereitung für das Netz und Aufruf als Tool. Mit zufälligen Gewichten, die echten erzeugt train.py."""
import importlib.util
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from Agents import neural_networks
from rag_chatpipline import config

BEISPIEL = Path(__file__).resolve().parents[1] / "networks" / "aktienkurs_ohlcv"


def _modul(name: str):
    """daten.py, model.py oder train.py direkt aus dem Beispielordner laden (dort importieren sie sich gegenseitig)."""
    vorher = set(sys.modules)
    sys.path.insert(0, str(BEISPIEL))
    try:
        spec = importlib.util.spec_from_file_location(f"ohlcv_{name}", BEISPIEL / f"{name}.py")
        modul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modul)
        return modul
    finally:
        sys.path.remove(str(BEISPIEL))
        for neu in {"daten", "model"} & (set(sys.modules) - vorher):  # train.py importiert sie ohne Paketnamen
            del sys.modules[neu]


def yahoo_csv(tage: int = 120) -> str:
    zeilen = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    for t in range(tage):
        schluss = 100 + t * 0.5 + 3 * math.sin(t / 5)
        zeilen.append(f"2024-{1 + t // 28:02d}-{1 + t % 28:02d},{schluss - 0.4},{schluss + 1},{schluss - 1},"
                      f"{schluss},{schluss * 0.9},{1_000_000 + t * 1000}")
    return "\n".join(zeilen)


@pytest.fixture
def kerzen_netz(workspace):
    """Der Beispielordner ohne trainierte Gewichte, dazu Gewichte eines frischen (untrainierten) Netzes."""
    ziel = config.NETWORKS_DIR / "aktienkurs_ohlcv"
    ziel.mkdir(parents=True)
    for datei in ("daten.py", "model.py", "network.py"):
        shutil.copy(BEISPIEL / datei, ziel / datei)
    model = _modul("model")
    torch.manual_seed(0)
    netz = model.KerzenLSTM()
    torch.save({"state_dict": netz.state_dict(), "fenster": model.FENSTER, "horizont": model.HORIZONT,
                "quelle": "Testdaten", "eingabe_mittel": torch.zeros(5), "eingabe_std": torch.ones(5),
                "ziel_mittel": torch.zeros(model.HORIZONT), "ziel_std": torch.ones(model.HORIZONT)},
               ziel / "kerzen_lstm.pt")
    return next(n for n in neural_networks.networks() if n.name == "aktienkurs_ohlcv_vorhersagen")


def test_reads_yahoo_and_german_exports_oldest_first(tmp_path):
    daten = _modul("daten")
    yahoo = tmp_path / "yahoo.csv"
    yahoo.write_text(yahoo_csv(3))
    tage, kerzen = daten.kerzen_lesen(yahoo)
    assert kerzen.shape == (3, 5) and tage[0] == "2024-01-01"
    assert kerzen[0].tolist() == pytest.approx([99.6, 101, 99, 100, 1_000_000])  # Close, nicht Adj Close

    deutsch = tmp_path / "boerse.csv"
    deutsch.write_text("Datum;Eröffnung;Hoch;Tief;Schlusskurs;Volumen\n"
                       "03.01.2025;1.010,50;1.020,00;1.000,00;1.015,25;12.000\n"
                       "02.01.2025;1.000,00;1.012,00;995,00;1.010,00;10.000\n")
    tage, kerzen = daten.kerzen_lesen(deutsch)
    assert tage == ["02.01.2025", "03.01.2025"]
    assert kerzen[-1].tolist() == pytest.approx([1010.5, 1020, 1000, 1015.25, 12000])

    nur_schluss = tmp_path / "schluss.csv"
    nur_schluss.write_text("Date,Close\n2024-01-01,100\n")
    with pytest.raises(daten.OhneKerzen, match="Eröffnung"):
        daten.kerzen_lesen(nur_schluss)


def test_inputs_are_relative_so_price_level_does_not_matter():
    model = _modul("model")
    zufall = np.random.default_rng(1)
    kerzen = np.column_stack([100 + zufall.normal(0, 1, (model.FENSTER, 4)), zufall.uniform(1e5, 2e5, model.FENSTER)])
    teuer = kerzen * [100, 100, 100, 100, 1]  # gleiche Aktie, Kurse mal 100
    assert np.allclose(model.eingabe(kerzen), model.eingabe(teuer))
    assert model.eingabe(kerzen)[-1, 3] == pytest.approx(0)  # letzter Schlusskurs ist der Bezugspunkt


def test_training_targets_are_the_following_closes():
    train = _modul("train")
    kerzen = train.demo_kerzen(anzahl=1, tage=200)[0]
    x, y = train.beispiele(kerzen, 0, 200)
    t = train.FENSTER - 1  # erstes Beispiel: Fenster endet an Tag t
    assert x.shape[1:] == (train.FENSTER, 5) and y.shape[1] == train.HORIZONT
    assert y[0, 0] == pytest.approx(np.log(kerzen[t + 1, 3] / kerzen[t, 3]) * 100)
    assert y[0, -1] == pytest.approx(np.log(kerzen[t + train.HORIZONT, 3] / kerzen[t, 3]) * 100)
    assert len(x) == 200 - train.FENSTER - train.HORIZONT + 1


def test_tool_returns_table_and_chart(kerzen_netz):
    (config.WORKSPACE_DIR / "Uploads" / "aktie.csv").write_text(yahoo_csv())
    text, artifact = neural_networks.run(kerzen_netz, {"csv_datei": "Uploads/aktie.csv", "tage": 99})
    assert text.count("Handelstag +") == 30  # höchstens 30 Tage
    assert "Testdaten" in text and "keine Anlageberatung" in text
    assert (config.IMAGES_DIR / artifact["name"]).stat().st_size > 1000


def test_tool_points_to_the_other_network_for_close_only_files(kerzen_netz):
    (config.WORKSPACE_DIR / "Uploads" / "schluss.csv").write_text("Date,Close\n" + "\n".join(
        f"2024-01-{d:02d},{100 + d}" for d in range(1, 28)))
    text, _ = neural_networks.run(kerzen_netz, {"csv_datei": "Uploads/schluss.csv"})
    assert "aktienkurs_vorhersagen" in text

    (config.WORKSPACE_DIR / "Uploads" / "kurz.csv").write_text(yahoo_csv(20))
    text, _ = neural_networks.run(kerzen_netz, {"csv_datei": "Uploads/kurz.csv"})
    assert "mindestens 60" in text
