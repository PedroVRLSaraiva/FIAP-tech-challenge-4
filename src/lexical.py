"""Busca lexical (BM25) e estatísticas de textos repetidos, sobre os mesmos documentos do índice vetorial."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src import config

# Palavras que não discriminam reviews: artigos, preposições, palavras de pergunta e do domínio.
_STOP = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sobre ao aos ate e ou mas que se
eh foi sao ser ter tem ha sua seu suas seus meu minha me mim eu voce voces ele ela eles elas isso isto esse essa
este esta qual quais quem como quando onde porque por que mais muito muita muitos muitas ja so tambem
quais cliente clientes avaliacao avaliacoes review reviews relatam relata relatado relatados dizem diz
existem existe houve podem pode ficaram foram aparecem aparece aspecto aspectos
""".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def fold(text: str) -> str:
    """Minúsculas e sem acentos."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Chave para detectar textos idênticos: ignora caixa, acento, pontuação e espaços."""
    return _NON_ALNUM_RE.sub(" ", fold(text)).strip()


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN_RE.findall(fold(text)):
        if t in _STOP:
            continue
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") else t)  # plural simples: "entregas" -> "entrega"
    return out


class LexicalIndex:
    """BM25 sobre os textos e contagem de textos idênticos na base inteira."""

    def __init__(self, docs: pd.DataFrame):
        self.docs = docs.reset_index(drop=True)
        self.ids = self.docs["id"].to_numpy()
        texts = self.docs["text"].tolist()
        self._norm = [normalize_text(t) for t in texts]
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        groups: dict[str, list[int]] = {}
        for i, key in enumerate(self._norm):
            groups.setdefault(key, []).append(i)
        self._groups = {k: np.array(v) for k, v in groups.items() if len(v) > 1}

    @classmethod
    def from_parquet(cls, path: Path | None = None) -> LexicalIndex:
        return cls(pd.read_parquet(path or config.PROCESSED_PARQUET))

    def search(self, query: str, n: int, mask: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Top-n (id, score BM25). `mask` restringe a busca (filtros de metadados)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(str(self.ids[i]), float(scores[i])) for i in top if scores[i] > 0]

    def duplicate_info(self, text: str) -> dict:
        """Quantas avaliações da base têm exatamente este texto e com que notas."""
        pos = self._groups.get(normalize_text(text))
        if pos is None:
            return {"count": 1, "notas": {}}
        notas = Counter(int(x) for x in self.docs["review_score"].to_numpy()[pos])
        return {"count": int(len(pos)), "notas": dict(sorted(notas.items()))}
