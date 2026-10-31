"""Okapi BM25 für die hybride Suche: bewertet Textabschnitte nach exakten Begriffen.

score(D, Q) = Σ IDF(t) · f(t, D) · (k1 + 1) / (f(t, D) + k1 · (1 − b + b · |D| / avgdl))

Ergänzt die Embedding-Suche dort, wo sie schwach ist: Firmennamen, Abkürzungen, Norm- und Artikelnummern.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Hashable

K1 = 1.5
B = 0.75

# Häufige deutsche und englische Füllwörter tragen nichts zur Relevanz bei.
STOPWORDS = set("""
aber alle als also am an auch auf aus bei bin bis bitte da damit dann das dass dein dem den der des die dies
diese dieser dir doch du ein eine einem einen einer es für gibt hat hatte habe ich ihr im in ist ja kann
kannst man mein mich mir mit muss nach nicht noch nur oder ob sich sie sind so über um und uns unser unter
vom von vor war was welche welcher welches wer wie wir wird wo zu zum zur
a an and are as at be by for from how i in is it of on or the this to was what when where which who with
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, documents: dict[Hashable, str]):
        self.ids = list(documents)
        self.term_counts = [Counter(tokenize(text)) for text in documents.values()]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        document_frequency = Counter(term for counts in self.term_counts for term in counts)
        n = len(self.ids)
        # IDF mit +1, damit Begriffe in mehr als der Hälfte der Abschnitte nicht negativ zählen
        self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in document_frequency.items()}

    def search(self, query: str, k: int) -> list[tuple[Hashable, float]]:
        """Die k besten Abschnitte als (id, score), nur solche mit mindestens einem Treffer."""
        terms = tokenize(query)
        scores = []
        for doc_id, counts, length in zip(self.ids, self.term_counts, self.lengths):
            score = 0.0
            for term in terms:
                f = counts.get(term)
                if f:
                    norm = K1 * (1 - B + B * length / self.avgdl)
                    score += self.idf[term] * f * (K1 + 1) / (f + norm)
            if score > 0:
                scores.append((doc_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:k]
