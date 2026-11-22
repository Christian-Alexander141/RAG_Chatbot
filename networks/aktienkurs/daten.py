"""Schlusskurse aus CSV-Dateien lesen. Nutzen beide: train.py und network.py.

Versteht die üblichen Exporte, z. B. von Yahoo Finance (Date, Close, Komma-getrennt) oder deutschen Börsenseiten
(Datum; Schlusskurs, Semikolon-getrennt, Dezimalkomma). Neueste Kurse dürfen oben oder unten stehen.
"""
import csv
from datetime import datetime
from pathlib import Path

KURS_SPALTEN = ("close", "schluss", "schlusskurs", "zuletzt", "kurs", "price", "adj close")
DATUM_SPALTEN = ("date", "datum", "zeit", "time")
DATUM_FORMATE = ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%m/%d/%Y", "%Y/%m/%d")


def _spalte(kopf: list[str], namen: tuple[str, ...]) -> int:
    for name in namen:  # erst exakte Treffer, dann Teilwörter ("Schlusskurs in EUR")
        if name in kopf:
            return kopf.index(name)
    for name in namen:
        for i, spalte in enumerate(kopf):
            if name in spalte:
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


def schlusskurse_lesen(pfad: Path) -> tuple[list[str], list[float]]:
    """Gibt (Datumsangaben, Schlusskurse) zurück, ältester Kurs zuerst."""
    zeilen = pfad.read_text(encoding="utf-8-sig").splitlines()
    if not zeilen:
        raise ValueError(f"{pfad.name} ist leer.")
    trenner = ";" if zeilen[0].count(";") > zeilen[0].count(",") else ","
    leser = csv.reader(zeilen, delimiter=trenner)
    kopf = [k.strip().lower() for k in next(leser)]
    kurs_spalte, datum_spalte = _spalte(kopf, KURS_SPALTEN), _spalte(kopf, DATUM_SPALTEN)
    if kurs_spalte < 0:
        raise ValueError(f"{pfad.name}: keine Spalte mit Schlusskursen gefunden (erwartet z. B. Close oder Schlusskurs).")

    daten, kurse = [], []
    for zeile in leser:
        try:
            kurs = _zahl(zeile[kurs_spalte], dezimalkomma=trenner == ";")
        except (IndexError, ValueError):
            continue  # leere Zeilen oder "null" an Feiertagen
        if kurs > 0:
            kurse.append(kurs)
            daten.append(zeile[datum_spalte].strip() if datum_spalte >= 0 else str(len(kurse)))

    erstes, letztes = (_datum(daten[0]), _datum(daten[-1])) if daten else (None, None)
    if erstes and letztes and erstes > letztes:  # neueste oben: umdrehen
        daten.reverse()
        kurse.reverse()
    return daten, kurse
