"""Deterministische Dokumentsuche: "Gib mir die Reisekostenrichtlinie" liefert direkt das Dokument.

Kleine Modelle finden ein gewünschtes Dokument oft nicht zuverlässig. Deshalb wird eine Dokumentanfrage an festen
Regeln erkannt und der Dateiname ohne Modell mit den Dokumenten der Wissensbasis abgeglichen. Fragen zum Inhalt
("Was steht in der Reisekostenrichtlinie zu Hotels?") gehen wie bisher an die Suche und den Werkzeug-Agenten.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

# Wer ein Dokument haben will, sagt es mit einem dieser Wörter ...
_FETCH = re.compile(
    r"\b(zeig\w*|gib|gebt|öffne\w*|oeffne\w*|schick\w*|sende\w*|hol\w*|such\w*|find\w*|brauche|bräuchte|"
    r"hätte gern|herunterladen|runterladen|download\w*|link|wo (ist|liegt|liegen|sind|finde|gibt))\b",
    re.IGNORECASE,
)
# ... wer etwas über den Inhalt wissen will, mit einem dieser.
_CONTENT = re.compile(
    r"\b(was steht|steht (da|darin|drin|dort)|inhalt\w*|zusammenfass\w*|erklär\w*|wie viel|wieviel|wie hoch|"
    r"wie lange|wann|warum|wieso|darf|dürfen|muss|müssen|laut|gemäß)\b",
    re.IGNORECASE,
)
MIN_SCORE = 0.6


def _normalize(text: str) -> str:
    text = text.lower()
    for umlaut, replacement in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, replacement)
    return text


def _name_tokens(name: str) -> list[str]:
    stem = Path(name).stem
    stem = re.sub(r"([a-zäöü])([A-ZÄÖÜ])", r"\1 \2", stem)  # CamelCase trennen
    return [t for t in re.findall(r"[a-z0-9]+", _normalize(stem)) if len(t) >= 3 or t.isdigit()]


def _token_found(token: str, words: list[str]) -> bool:
    if any(token in w or SequenceMatcher(None, token, w).ratio() >= 0.85 for w in words):
        return True  # Wort enthält den Namensteil, oder kleiner Tippfehler
    # Zusammengesetzte Wörter: Die Wörter der Nachricht müssen den Namensteil weitgehend abdecken.
    # "Reisekosten-Richtlinie" deckt "reisekostenrichtlinie" ab, "Richtlinie" allein deckt "sicherheitsrichtlinie" nicht ab.
    covered = set()
    for w in words:
        start = token.find(w) if len(w) >= 4 else -1
        if start >= 0:
            covered.update(range(start, start + len(w)))
    return len(covered) >= 0.8 * len(token)


def score(message: str, name: str) -> float:
    """Wie gut passt der Dateiname zur Nachricht (0 bis 1, Zahlen im Namen geben einen kleinen Bonus)?"""
    words = re.findall(r"[a-z0-9]+", _normalize(message))
    tokens = _name_tokens(name)
    # Zahlen wie Jahreszahlen fehlen oft in der Frage. Sie entscheiden nur bei Gleichstand.
    numbers = [t for t in tokens if t.isdigit()]
    tokens = [t for t in tokens if not t.isdigit()] or numbers
    if not tokens or not words:
        return 0.0
    bonus = 0.01 * sum(n in words for n in numbers)
    if "".join(tokens) in "".join(words):  # ganzer Name, auch ohne Trennzeichen
        return 1.0 + bonus
    return sum(_token_found(t, words) for t in tokens) / len(tokens) + bonus


def is_fetch_request(message: str) -> bool:
    return bool(_FETCH.search(message)) and not _CONTENT.search(message)


def find_documents(message: str, sources: list[str]) -> list[str]:
    """Die am besten passenden Dokumente zu einer Dokumentanfrage, sonst eine leere Liste."""
    if not is_fetch_request(message):
        return []
    scored = [(score(message, name), name) for name in sources]
    best = max((s for s, _ in scored), default=0.0)
    if best < MIN_SCORE:
        return []
    return sorted(name for s, name in scored if s == best)


def file_url(name: str) -> str:
    return "/files/" + quote(name)


def answer(names: list[str]) -> str:
    links = "\n".join(f"- [{name}]({file_url(name)})" for name in names)
    if len(names) == 1:
        return f"Hier ist das gesuchte Dokument:\n\n{links}"
    return f"Dazu passen mehrere Dokumente:\n\n{links}"
