"""Recuperação top-N (vetorial e híbrida) e seleção dos top-k finais."""
from __future__ import annotations

import numpy as np

from src import config, vector_store


def retrieve(
    collection,
    query_embedding: list[float],
    top_n: int | None = None,
    where: dict | None = None,
    with_embeddings: bool = False,
) -> list[dict]:
    """Candidatos vetoriais ordenados por score decrescente, já com o pré-filtro `where` aplicado."""
    cands = vector_store.query(
        collection, query_embedding, top_n or config.TOP_N, where=where, with_embeddings=with_embeddings
    )
    return sorted(cands, key=lambda c: c["score"], reverse=True)


def _cosine(qvec, emb) -> float:
    q, e = np.asarray(qvec, dtype=float), np.asarray(emb, dtype=float)
    return float(q @ e / max(np.linalg.norm(q) * np.linalg.norm(e), 1e-12))


def hybrid_fuse(
    vector_cands: list[dict],
    lexical_hits: list[tuple[str, float]],
    collection,
    query_embedding: list[float],
    rrf_k: int | None = None,
) -> list[dict]:
    """Busca híbrida: funde o ranking vetorial com o lexical (BM25) por Reciprocal Rank Fusion.

    rrf(d) = Σ 1 / (k + posição de d em cada lista), com k = `config.RRF_K` (60).
    Candidatos que só o BM25 trouxe recebem o cosseno com a pergunta (buscado no Chroma), então
    `score` continua significando similaridade vetorial em todos os casos.
    """
    k = rrf_k or config.RRF_K
    by_id = {c["id"]: dict(c) for c in vector_cands}
    vec_rank = {c["id"]: r for r, c in enumerate(vector_cands, start=1)}
    lex_rank = {id_: r for r, (id_, _) in enumerate(lexical_hits, start=1)}

    missing = [id_ for id_ in lex_rank if id_ not in by_id]
    for doc in vector_store.get_by_ids(collection, missing):
        doc["score"] = _cosine(query_embedding, doc["embedding"])
        by_id[doc["id"]] = doc

    fused = []
    for id_, c in by_id.items():
        rv, rl = vec_rank.get(id_), lex_rank.get(id_)
        c["rrf"] = (1 / (k + rv) if rv else 0.0) + (1 / (k + rl) if rl else 0.0)
        c["origem"] = "ambos" if (rv and rl) else ("vetor" if rv else "bm25")
        fused.append(c)
    return sorted(fused, key=lambda c: c["rrf"], reverse=True)


def select(candidates: list[dict], top_k: int) -> list[dict]:
    """Corte por SCORE_MIN e redução para top_k, já numerando [1]..[k]."""
    top_k = max(1, min(top_k, config.MAX_TOP_K))
    kept = [c for c in candidates if c["score"] >= config.SCORE_MIN][:top_k]
    return [{**c, "n": i} for i, c in enumerate(kept, start=1)]
