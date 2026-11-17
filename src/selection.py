"""Seleção das evidências: corte por score, agrupar idênticos, reranking, MMR e temas."""
from __future__ import annotations

import numpy as np

from src import config
from src.diversify import group_duplicates, mmr
from src.options import SearchOptions
from src.themes import cluster_pool


def select_evidences(
    pool: list[dict],
    question: str,
    options: SearchOptions,
    top_k: int,
    *,
    reranker=None,
    stats=None,
) -> tuple[list[dict], dict]:
    """Escolhe as `top_k` evidências a partir do conjunto de candidatos.

    Ordem das etapas (cada uma só roda se a opção estiver ligada):
    corte por SCORE_MIN -> agrupar textos idênticos -> reranking -> MMR -> numerar [1..k].
    Sem nenhuma opção, é o comportamento original: as k mais parecidas com a pergunta.
    """
    top_k = max(1, min(top_k, config.MAX_TOP_K))
    info: dict = {"candidatos": len(pool)}
    pool = [c for c in pool if c["score"] >= config.SCORE_MIN]
    info["apos_score_min"] = len(pool)

    if options.dedup:
        pool = group_duplicates(pool, stats)
        info["apos_agrupar_identicos"] = len(pool)

    relevance = None
    if options.rerank and reranker is not None and pool:
        head = pool[: config.RERANK_POOL]
        scores = reranker.score(question, [c["text"] for c in head])
        order = np.argsort(-scores)
        pool = [{**head[i], "rerank": float(scores[i])} for i in order]
        relevance = [c["rerank"] for c in pool]
        info["reranker"] = reranker.model_name
    elif options.hybrid and pool and all("rrf" in c for c in pool):
        relevance = [c["rrf"] for c in pool]

    if options.mmr and len(pool) > 1 and all("embedding" in c for c in pool):
        chosen = mmr(pool, top_k, options.mmr_lambda, relevance)
        info["mmr_lambda"] = options.mmr_lambda
    else:
        chosen = pool[:top_k]
    return [{**c, "n": i} for i, c in enumerate(chosen, start=1)], info


def select_by_theme(pool: list[dict], top_k: int, stats=None) -> tuple[list[dict], list[dict], dict] | None:
    """Agrupa o conjunto recuperado em temas (KMeans) e escolhe representantes distintos de cada um.

    Retorna (evidências numeradas, temas, info) ou None se houver poucas avaliações para agrupar.
    """
    pool = [c for c in pool if c["score"] >= config.SCORE_MIN and "embedding" in c]
    themes = cluster_pool(pool, reps_per_theme=4)
    if not themes:
        return None
    per_theme = max(1, round(top_k / len(themes)))

    evidences, themes_info, n = [], [], 1
    for t in themes:
        reps = [pool[i] for i in t.representatives[:per_theme]]
        reps = group_duplicates(reps, stats)  # anota a contagem de textos idênticos
        ns = []
        for ev in reps:
            evidences.append({**ev, "n": n, "tema": t.label})
            ns.append(n)
            n += 1
        themes_info.append({
            "label": t.label, "size": t.size, "share": round(t.share, 3),
            "keywords": t.keywords, "evidence_ns": ns,
        })
    return evidences, themes_info, {"pool_temas": len(pool), "n_temas": len(themes)}
