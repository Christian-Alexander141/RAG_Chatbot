"""Tageskerzen (Eröffnung, Hoch, Tief, Schluss, Volumen) aus CSV-Dateien lesen. Nutzen train.py und network.py.

Versteht die üblichen Exporte, z. B. von Yahoo Finance (Date, Open, High, Low, Close, Volume, Komma-getrennt), die
Datei data/stock_data.csv des Originalprojekts und deutsche Börsenseiten (Datum; Eröffnung; Hoch; Tief; Schluss;
Volumen, Semikolon-getrennt, Dezimalkomma). Neueste Kurse dürfen oben oder unten stehen.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

# Je Merkmal mögliche Spaltennamen (klein geschrieben), in dieser Reihenfolge stehen sie im Ergebnis
SPALTEN = {
    "Eröffnung": ("open", "eröffnung", "eroeffnung", "erster", "start"),
    "Hoch": ("high", "hoch", "tageshoch"),
    "Tief": ("low", "tief", "tagestief"),
    "Schluss": ("close", "schluss", "schlusskurs", "zuletzt", "letzter"),
    "Volumen": ("volume", "volumen", "stücke", "stuecke"),
}
DATUM_SPALTEN = ("date", "datum", "zeit", "time")
DATUM_FORMATE = ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%m/%d/%Y", "%Y/%m/%d")


class OhneKerzen(ValueError):
    """Der Datei fehlen Eröffnung, Hoch, Tief oder Volumen (z. B. nur Schlusskurse)."""


def _spalte(kopf: list[str], namen: tuple[str, ...]) -> int:
    for name in namen:  # erst exakte Treffer ("close", nicht "adj close"), dann Teilwörter ("Schluss in EUR")
        if name in kopf:
            return kopf.index(name)
    for name in namen:
        for i, spalte in enumerate(kopf):
            if name in spalte and "adj" not in spalte:
                return i
    return -1


def _zahl(text: str, dezimalkomma: bool) -> float:
    text = text.strip().replace(" ", "")
    if dezimalkomma:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def _datum(text: str):
    for fmt in DATUM_FORMATE:
        try:
            return datetime.strptime(text.strip()[:10], fmt)
        except ValueError:
            pass
    return None


def kerzen_lesen(pfad: Path) -> tuple[list[str], np.ndarray]:
    """Gibt (Datumsangaben, Kerzen) zurück: Kerzen hat die Form (Tage, 5) mit Eröffnung, Hoch, Tief, Schluss,
    Volumen, ältester Tag zuerst. Fehlt eine dieser Spalten, gibt es OhneKerzen."""
    zeilen = pfad.read_text(encoding="utf-8-sig").splitlines()
    if not zeilen:
        raise ValueError(f"{pfad.name} ist leer.")
    trenner = ";" if zeilen[0].count(";") > zeilen[0].count(",") else ","
    leser = csv.reader(zeilen, delimiter=trenner)
    kopf = [k.strip().lower() for k in next(leser)]
    spalten = {merkmal: _spalte(kopf, namen) for merkmal, namen in SPALTEN.items()}
    fehlt = [merkmal for merkmal, i in spalten.items() if i < 0]
    if fehlt:
        raise OhneKerzen(f"{pfad.name}: Spalten fehlen: {', '.join(fehlt)} (gebraucht: Eröffnung/Open, Hoch/High, "
                         "Tief/Low, Schluss/Close und Volumen/Volume).")
    datum_spalte = _spalte(kopf, DATUM_SPALTEN)

    daten, kerzen = [], []
    for zeile in leser:
        try:
            werte = [_zahl(zeile[i], dezimalkomma=trenner == ";") for i in spalten.values()]
        except (IndexError, ValueError):
            continue  # leere Zeilen oder "null" an Feiertagen
        if min(werte[:4]) > 0 and werte[4] >= 0:
            kerzen.append(werte)
            daten.append(zeile[datum_spalte].strip() if datum_spalte >= 0 else str(len(kerzen)))

    erstes, letztes = (_datum(daten[0]), _datum(daten[-1])) if daten else (None, None)
    if erstes and letztes and erstes > letztes:  # neueste oben: umdrehen
        daten.reverse()
        kerzen.reverse()
    return daten, np.array(kerzen, dtype=np.float64).reshape(-1, 5)
