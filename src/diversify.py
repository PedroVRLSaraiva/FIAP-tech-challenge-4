"""Diversidade e utilidade na seleção das evidências: agrupar textos idênticos e MMR."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.lexical import normalize_text


def group_duplicates(cands: list[dict], stats=None) -> list[dict]:
    """Colapsa candidatos com o mesmo texto em um só, que fica com a posição do melhor.

    Cada grupo recebe `dup_count` (quantas avaliações da base têm esse texto, se `stats` for
    informado; senão, quantas apareceram entre os candidatos) e `dup_notas` (distribuição de notas).
    Assim os k slots cobrem pontos de vista diferentes e a frequência vira informação, e não repetição.
    """
    groups: dict[str, dict] = {}
    for c in cands:  # os candidatos já vêm ordenados por relevância
        key = normalize_text(c["text"])
        if key in groups:
            groups[key]["_pool"].append(c)
        else:
            groups[key] = {"first": c, "_pool": [c]}

    out = []
    for g in groups.values():
        c = dict(g["first"])
        pool = g["_pool"]
        info = stats.duplicate_info(c["text"]) if stats is not None else {"count": 1, "notas": {}}
        if info["count"] > 1:
            c["dup_count"], c["dup_notas"] = info["count"], info["notas"]
        else:
            notas = Counter(int(p.get("metadata", {}).get("review_score", 0)) for p in pool)
            c["dup_count"], c["dup_notas"] = len(pool), dict(sorted(notas.items()))
        out.append(c)
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return np.zeros_like(x) if span < 1e-12 else (x - x.min()) / span


def mmr(cands: list[dict], k: int, lam: float, relevance: list[float] | None = None) -> list[dict]:
    """Maximal Marginal Relevance: a cada passo escolhe a review que maximiza

        lam * relevância(review) - (1 - lam) * (maior similaridade com as já escolhidas)

    A relevância é normalizada para [0, 1] no conjunto de candidatos (min-max) e a similaridade é o
    cosseno entre os embeddings. `lam` = 1 reproduz a ordem por relevância.
    """
    if not cands:
        return []
    k = min(k, len(cands))
    rel = _minmax(np.asarray(relevance if relevance is not None else [c["score"] for c in cands], dtype=float))
    vecs = np.vstack([np.asarray(c["embedding"], dtype=float) for c in cands])
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    chosen = [int(np.argmax(rel))]
    max_sim = vecs @ vecs[chosen[0]]
    while len(chosen) < k:
        scores = lam * rel - (1 - lam) * max_sim
        scores[chosen] = -np.inf
        nxt = int(np.argmax(scores))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, vecs @ vecs[nxt])
    return [cands[i] for i in chosen]
