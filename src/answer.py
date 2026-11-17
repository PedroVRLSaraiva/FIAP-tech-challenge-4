"""Orquestra o fluxo RAG e valida as citações da resposta."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from src import (
    config, context, embeddings, filter_extraction, guardrails, prompts, query, retrieval, selection,
    vector_store,
)
from src.filters import Filters
from src.llm import LLMError
from src.options import SearchOptions

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")
# Intervalo "[1]–[4]" ou "[1]-[4]": o modelo às vezes abrevia assim.
_RANGE_RE = re.compile(r"\[(\d+)\]\s*[–—-]\s*\[(\d+)\]")
# Não separa a frase da citação que vem logo depois do ponto ("... atraso. [2][5]").
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\[\d)|\n+")


def _expand_ranges(text: str) -> str:
    """Troca "[1]–[3]" por "[1][2][3]" para que cada índice seja validado e marcado como citado."""

    def repl(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b or b - a > 50:
            return m.group(0)
        return "".join(f"[{i}]" for i in range(a, b + 1))

    return _RANGE_RE.sub(repl, text)


@dataclass
class AnswerResult:
    question: str
    answer: str
    status: str  # ok | sem_evidencia | fora_de_escopo | erro
    evidences: list[dict] = field(default_factory=list)
    nearest: list[dict] = field(default_factory=list)  # não conclusivas, quando status != ok
    debug: dict = field(default_factory=dict)
    filters: list[str] = field(default_factory=list)  # recorte aplicado, em texto legível
    themes: list[dict] = field(default_factory=list)  # temas (KMeans), quando a opção está ligada


def validate_citations(answer: str, k: int) -> dict:
    """Remove citações com índice fora de {1..k} e mede a qualidade das citações."""
    answer = _expand_ranges(answer)
    found = [int(n) for n in _CITE_RE.findall(answer)]
    valid = [n for n in found if 1 <= n <= k]
    invalid = [n for n in found if not 1 <= n <= k]

    cleaned = _CITE_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= k else "", answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    uncited = [
        s.strip()
        for s in _SENT_SPLIT_RE.split(cleaned)
        if len(s.strip()) >= 40
        and not _CITE_RE.search(s)
        and not s.strip().endswith(":")  # frase de abertura da lista, não afirma nada
    ]
    return {
        "answer": cleaned,
        "cited": sorted(set(valid)),
        "invalid": invalid,
        "valid_rate": (len(valid) / len(found)) if found else 0.0,
        "uncited_sentences": uncited,
    }


def _public(ev: dict) -> dict:
    meta = ev.get("metadata", {})
    out = {
        "n": ev.get("n"),
        "id": ev["id"],
        "text": ev["text"],
        "score": round(ev["score"], 4),
        "nota": meta.get("review_score"),
        "data": meta.get("data"),
        "categoria": meta.get("categoria"),
        "atraso_dias": meta.get("atraso_dias"),
        "entrega": context.entrega_label(meta),
        "dup_count": ev.get("dup_count", 1),
        "dup_notas": ev.get("dup_notas") or {},
    }
    for k in ("tema", "origem", "rerank", "rrf"):
        if k in ev:
            out[k] = round(ev[k], 4) if isinstance(ev[k], float) else ev[k]
    return out


def _refusal(question, message, nearest, debug, filters=()):
    return AnswerResult(question, message, "sem_evidencia", nearest=nearest, debug=debug, filters=list(filters))


def answer(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    *,
    llm=None,
    collection=None,
    embed_fn=None,
    options: SearchOptions | None = None,
    lexical=None,
    reranker=None,
) -> AnswerResult:
    """Fluxo completo. `llm`, `collection`, `embed_fn`, `lexical` (BM25) e `reranker` são injetáveis."""
    options = options or SearchOptions()
    timings: dict[str, float] = {}
    debug: dict = {"timings_s": timings, "params": {"top_k": top_k}, "opcoes": options.describe()}

    question = query.normalize_question(question)
    if not question:
        return AnswerResult("", "Digite uma pergunta.", "erro", debug=debug)

    if llm is None:
        from src.llm import OpenAIClient

        try:
            llm = OpenAIClient()
        except LLMError as e:
            return AnswerResult(question, str(e), "erro", debug=debug)
    collection = collection if collection is not None else vector_store.get_collection()
    embed_fn = embed_fn or embeddings.embed_query

    # 1) Filtros: manuais (interface) têm prioridade sobre os extraídos da pergunta pelo LLM
    t = time.perf_counter()
    manual, auto, auto_status = options.filters, Filters(), "desligado"
    if options.auto_filters:
        auto, auto_status = filter_extraction.extract_filters(question, llm)
    effective = manual.merge_over(auto)
    timings["filtros"] = round(time.perf_counter() - t, 3)
    where = effective.to_where()
    filters_txt = effective.describe()
    debug["query"] = question
    debug["filtros"] = {
        "manuais": manual.to_dict(), "automaticos": auto.to_dict(), "extracao": auto_status,
        "efetivos": effective.to_dict(), "where": where,
    }

    # 2) Embedding da pergunta + busca vetorial (com o filtro dentro da consulta)
    t = time.perf_counter()
    qvec = embed_fn(question)
    pool_n = config.THEME_POOL if options.themes else config.TOP_N
    candidates = retrieval.retrieve(collection, qvec, pool_n, where=where, with_embeddings=options.needs_embeddings)
    timings["retrieval"] = round(time.perf_counter() - t, 3)
    debug["scores_top_candidates"] = [round(c["score"], 4) for c in candidates[:15]]

    # 3) Gate determinístico, sobre os candidatos vetoriais, antes de gastar chamada de geração
    if not candidates and not effective.is_empty():
        debug["gate"] = {"passed": False, "reason": "recorte_vazio", "max_score": 0.0}
        return _refusal(question, config.MSG_SEM_RECORTE, [], debug, filters_txt)
    gate = guardrails.evaluate_gate(candidates)
    debug["gate"] = {
        "passed": gate.passed,
        "reason": gate.reason,
        "max_score": round(gate.max_score, 4),
        "n_above_score_min": gate.n_above,
        "T_MAX": config.T_MAX,
        "SCORE_MIN": config.SCORE_MIN,
        "MIN_DOCS": config.MIN_DOCS,
    }
    if not gate.passed:
        nearest = [_public({**c, "n": i}) for i, c in enumerate(candidates[:3], start=1)]
        return _refusal(question, config.MSG_SEM_EVIDENCIA, nearest, debug, filters_txt)

    # 4) Busca híbrida: funde o ranking vetorial com o BM25 (RRF)
    pool = candidates
    if options.hybrid and lexical is not None:
        t = time.perf_counter()
        mask = None if effective.is_empty() else effective.mask(lexical.docs)
        hits = lexical.search(question, config.BM25_TOP_N, mask)
        pool = retrieval.hybrid_fuse(candidates, hits, collection, qvec)
        timings["hibrida"] = round(time.perf_counter() - t, 3)
        debug["hibrida"] = {
            "bm25": len(hits), "somente_bm25": sum(1 for c in pool if c["origem"] == "bm25"),
            "em_ambos": sum(1 for c in pool if c["origem"] == "ambos"),
        }

    # 5) Seleção das evidências (temas, ou agrupar idênticos + rerank + MMR) e contexto
    t = time.perf_counter()
    themes_info: list[dict] = []
    themed = selection.select_by_theme(pool, top_k, stats=lexical) if options.themes else None
    if themed is not None:
        evidences, themes_info, sel_info = themed
        ctx = context.build_context(evidences, themes_info, pool_size=sel_info["pool_temas"])
    else:
        use_reranker = options.rerank
        if use_reranker and reranker is None:
            from src import rerank

            reranker = rerank.get_reranker() if rerank.is_available() else None
        evidences, sel_info = selection.select_evidences(
            pool, question, options, top_k, reranker=reranker, stats=lexical
        )
        ctx = context.build_context(evidences)
    timings["selecao"] = round(time.perf_counter() - t, 3)
    debug["selecao"] = sel_info
    system = prompts.build_system(themes=bool(themes_info))
    user_prompt = prompts.build_user(question, ctx, filters_txt)
    debug["prompt"] = {"system": system, "user": user_prompt}

    # 6) Geração
    t = time.perf_counter()
    try:
        raw = llm.generate(system, user_prompt)
    except LLMError as e:
        return AnswerResult(
            question, str(e), "erro", [_public(e_) for e_ in evidences], debug=debug,
            filters=filters_txt, themes=themes_info,
        )
    timings["generation"] = round(time.perf_counter() - t, 3)
    debug["raw_answer"] = raw

    status = guardrails.parse_llm_status(raw)
    if status == "sem_evidencia":
        nearest = [_public(e_) for e_ in evidences[:3]]
        return AnswerResult(question, config.MSG_SEM_EVIDENCIA, status, nearest=nearest, debug=debug, filters=filters_txt)
    if status == "fora_de_escopo":
        return AnswerResult(question, config.MSG_FORA_DE_ESCOPO, status, debug=debug, filters=filters_txt)

    # 7) Validação das citações
    check = validate_citations(raw, k=len(evidences))
    debug["citations"] = {k: v for k, v in check.items() if k != "answer"}
    if check["invalid"]:
        log.warning("Citações inválidas removidas: %s", check["invalid"])

    public = []
    for ev in evidences:
        p = _public(ev)
        p["cited"] = ev["n"] in check["cited"]
        public.append(p)
    return AnswerResult(
        question, check["answer"], "ok", evidences=public, debug=debug, filters=filters_txt, themes=themes_info
    )
