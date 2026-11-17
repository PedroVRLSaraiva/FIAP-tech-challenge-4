"""Agrupamento por tema (KMeans) para sumarizar perguntas agregadas.

Medimos a silhueta para k de 3 a 6 em perguntas agregadas e ela ficou praticamente constante (~0,04):
o conjunto recuperado é homogêneo demais para a silhueta escolher k. Por isso k é função do tamanho do
conjunto (ver `choose_k`), e o conjunto precisa ser grande (`config.THEME_POOL`) para conter variedade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.lexical import normalize_text, tokenize

MIN_POOL = 12  # abaixo disso não há massa para agrupar


@dataclass
class Theme:
    label: str
    size: int
    share: float                      # fração do conjunto recuperado
    keywords: list[str]
    members: list[int] = field(default_factory=list)          # posições em `cands`
    representatives: list[int] = field(default_factory=list)  # posições em `cands`, mais centrais primeiro


def _keywords(texts_by_cluster: list[list[str]], top: int) -> list[list[str]]:
    docs = [" ".join(t) for t in texts_by_cluster]
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, preprocessor=None)
    try:
        m = vec.fit_transform(docs)
    except ValueError:  # vocabulário vazio
        return [[] for _ in docs]
    terms = np.array(vec.get_feature_names_out())
    out = []
    for row in range(m.shape[0]):
        w = m[row].toarray().ravel()
        order = np.argsort(-w)[: top + 5]  # folga: números soltos são descartados abaixo
        out.append([str(terms[i]) for i in order if w[i] > 0 and not str(terms[i]).isdigit()][:top])
    return out


def choose_k(n: int) -> int:
    """~40 avaliações por tema, entre 3 e `config.THEME_K_MAX` temas."""
    return max(3, min(config.THEME_K_MAX, n // 40))


def cluster_pool(cands: list[dict], reps_per_theme: int = 3, k: int | None = None) -> list[Theme]:
    """Agrupa os candidatos (com `embedding`) em temas e escolhe representantes distintos de cada um.

    Temas ordenados do maior para o menor. Devolve [] se houver poucos candidatos.
    """
    n = len(cands)
    if n < MIN_POOL:
        return []
    vecs = np.vstack([np.asarray(c["embedding"], dtype=float) for c in cands])
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    best_k = k or choose_k(n)
    best = KMeans(n_clusters=best_k, n_init=10, random_state=config.SEED).fit(vecs)

    labels = best.labels_
    texts_by_cluster = [[cands[i]["text"] for i in np.where(labels == c)[0]] for c in range(best_k)]
    kws = _keywords(texts_by_cluster, config.THEME_KEYWORDS)

    themes = []
    for c in range(best_k):
        idx = np.where(labels == c)[0]
        centroid = best.cluster_centers_[c]
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        order = idx[np.argsort(-(vecs[idx] @ centroid))]  # mais central primeiro
        reps, seen = [], set()
        for i in order:
            key = normalize_text(cands[i]["text"])
            if key in seen:
                continue
            seen.add(key)
            reps.append(int(i))
            if len(reps) == reps_per_theme:
                break
        themes.append(Theme("", len(idx), len(idx) / n, kws[c], [int(i) for i in idx], reps))

    themes.sort(key=lambda t: -t.size)
    for i, t in enumerate(themes):
        t.label = f"Tema {chr(ord('A') + i)}"
    return themes
