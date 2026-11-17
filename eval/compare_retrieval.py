"""Compara as técnicas de seleção das evidências, sem chamar o LLM de geração.

Uso: python -m eval.compare_retrieval [--k 10]

Para cada pergunta respondível de eval/perguntas.csv e cada configuração, escolhe as k evidências e mede:
- textos_distintos: quantos textos diferentes entre as k (repetição = valor baixo);
- redundancia: similaridade (cosseno) média entre pares de evidências escolhidas (menor = mais variado);
- cobertura_temas: fração dos 6 temas (KMeans fixo sobre os 100 vizinhos mais próximos) que as evidências cobrem;
- cosseno_medio: similaridade média com a pergunta (proxy de relevância: não deve cair muito);
- conformidade_filtro (bloco B): fração das evidências que satisfazem o filtro esperado.

Usa os filtros esperados (coluna filtro_esperado) como filtros manuais, para ser determinístico e sem LLM.
Grava eval/retrieval_comparison.csv, eval/retrieval_comparison.md e eval/themes_demo.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src import config, embeddings, query, rerank, retrieval, selection, vector_store
from src.filters import Filters
from src.lexical import LexicalIndex, normalize_text
from src.options import SearchOptions
from src.themes import cluster_pool

HERE = Path(__file__).parent

CONFIGS: dict[str, SearchOptions] = {
    "baseline (vetorial pura)": SearchOptions(),
    "+ agrupar idênticos": SearchOptions(dedup=True),
    "+ MMR": SearchOptions(mmr=True),
    "+ agrupar idênticos + MMR": SearchOptions(dedup=True, mmr=True),
    "híbrida (BM25+RRF)": SearchOptions(hybrid=True),
    "híbrida + agrupar + MMR": SearchOptions(hybrid=True, dedup=True, mmr=True),
    "rerank": SearchOptions(rerank=True),
    "rerank + agrupar": SearchOptions(rerank=True, dedup=True),
    "rerank + agrupar + MMR": SearchOptions(rerank=True, dedup=True, mmr=True),
    "tudo (híbrida+rerank+agrupar+MMR)": SearchOptions(hybrid=True, rerank=True, dedup=True, mmr=True),
}
LAMBDAS = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
N_THEMES = 6


def load_questions() -> list[dict]:
    rows = list(csv.DictReader((HERE / "perguntas.csv").open(encoding="utf-8")))
    out = []
    for r in rows:
        if r["esperado"] != "responde":
            continue
        raw = json.loads(r["filtro_esperado"]) if r.get("filtro_esperado") else {}
        out.append({**r, "filtro": Filters.from_dict(raw)})
    return out


def retrieve_selected(col, lex, reranker, q, qvec, options, k, filters: Filters):
    where = filters.to_where()
    cands = retrieval.retrieve(col, qvec, config.TOP_N, where=where, with_embeddings=True)
    pool = cands
    if options.hybrid:
        mask = None if filters.is_empty() else filters.mask(lex.docs)
        pool = retrieval.hybrid_fuse(cands, lex.search(q, config.BM25_TOP_N, mask), col, qvec)
    ev, _ = selection.select_evidences(pool, q, options, k, reranker=reranker, stats=lex)
    return ev, cands


def metrics(ev: list[dict], km: KMeans | None) -> dict:
    if not ev:
        return {"textos_distintos": 0, "redundancia": float("nan"), "cobertura_temas": 0.0, "cosseno_medio": float("nan")}
    vecs = np.vstack([np.asarray(e["embedding"], dtype=float) for e in ev])
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    pairs = [float(vecs[i] @ vecs[j]) for i, j in combinations(range(len(ev)), 2)]
    cover = len(set(km.predict(vecs))) / km.n_clusters if km is not None else float("nan")
    return {
        "textos_distintos": len({normalize_text(e["text"]) for e in ev}),
        "redundancia": float(np.mean(pairs)) if pairs else float("nan"),
        "cobertura_temas": cover,
        "cosseno_medio": float(np.mean([e["score"] for e in ev])),
    }


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] == r[key]]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    args = ap.parse_args()

    col = vector_store.get_collection()
    print("Carregando índice lexical (BM25)...")
    lex = LexicalIndex.from_parquet()
    print("Carregando reranker...")
    reranker = rerank.get_reranker() if rerank.is_available() else None
    questions = load_questions()

    # embedding da pergunta e agrupamento de referência (KMeans fixo), uma vez por pergunta
    prep = {}
    for r in questions:
        q = query.normalize_question(r["pergunta"])
        qvec = embeddings.embed_query(q)
        cands = retrieval.retrieve(col, qvec, 100, with_embeddings=True)
        km = KMeans(n_clusters=N_THEMES, n_init=10, random_state=config.SEED).fit(
            np.vstack([c["embedding"] for c in cands]).astype(float)
        )
        prep[r["id"]] = (q, qvec, km)

    # ---------------- Bloco A: técnicas de seleção (sem filtros) ----------------
    all_rows: list[dict] = []
    summary: dict[str, dict] = {}

    def run(label: str, options: SearchOptions) -> None:
        rows, t0 = [], time.perf_counter()
        for r in questions:
            q, qvec, km = prep[r["id"]]
            ev, _ = retrieve_selected(col, lex, reranker, q, qvec, options, args.k, Filters())
            m = metrics(ev, km)
            rows.append({"bloco": "A", "config": label, "id": r["id"], "tipo": r["tipo"], **m})
        all_rows.extend(rows)
        summary[label] = {
            **{k: mean(rows, k) for k in ("textos_distintos", "redundancia", "cobertura_temas", "cosseno_medio")},
            "latencia_ms": (time.perf_counter() - t0) / len(rows) * 1000,
        }
        s = summary[label]
        print(f"{label:<38} distintos={s['textos_distintos']:.2f}/{args.k}  redund={s['redundancia']:.3f}  "
              f"temas={s['cobertura_temas']:.2f}  cos={s['cosseno_medio']:.3f}  {s['latencia_ms']:.0f}ms/pergunta")

    print(f"\n== Bloco A: {len(questions)} perguntas, k={args.k}, sem filtros ==")
    for label, options in CONFIGS.items():
        run(label, options)

    print("\n== Varredura de λ (agrupar idênticos + MMR) ==")
    lam_summary: dict[float, dict] = {}
    for lam in LAMBDAS:
        rows = []
        for r in questions:
            q, qvec, km = prep[r["id"]]
            ev, _ = retrieve_selected(col, lex, reranker, q, qvec, SearchOptions(dedup=True, mmr=True, mmr_lambda=lam),
                                      args.k, Filters())
            rows.append(metrics(ev, km))
        lam_summary[lam] = {k: mean(rows, k) for k in ("textos_distintos", "redundancia", "cobertura_temas", "cosseno_medio")}
        s = lam_summary[lam]
        print(f"λ={lam:<4} distintos={s['textos_distintos']:.2f}  redund={s['redundancia']:.3f}  "
              f"temas={s['cobertura_temas']:.2f}  cos={s['cosseno_medio']:.3f}")

    # ---------------- Bloco B: filtros ----------------
    print("\n== Bloco B: filtros (perguntas com filtro esperado) ==")
    pos = {id_: i for i, id_ in enumerate(lex.ids)}
    filt_rows = []
    for r in questions:
        if r["filtro"].is_empty():
            continue
        q, qvec, km = prep[r["id"]]
        mask = r["filtro"].mask(lex.docs)
        for label, options in (("baseline (vetorial pura)", SearchOptions()), ("agrupar + MMR", SearchOptions(dedup=True, mmr=True))):
            for com in (False, True):
                ev, cands = retrieve_selected(col, lex, reranker, q, qvec, options, args.k,
                                              r["filtro"] if com else Filters())
                conf = float(np.mean([mask[pos[e["id"]]] for e in ev])) if ev else float("nan")
                m = metrics(ev, km)
                filt_rows.append({"bloco": "B", "config": label, "filtro": "com filtro" if com else "sem filtro",
                                  "id": r["id"], "pergunta": r["pergunta"], "n_candidatos": len(cands),
                                  "conformidade_filtro": conf, "n_evidencias": len(ev), **m})
    for label in ("baseline (vetorial pura)", "agrupar + MMR"):
        for fl in ("sem filtro", "com filtro"):
            rows = [x for x in filt_rows if x["config"] == label and x["filtro"] == fl]
            print(f"{label:<26} {fl:<11} conformidade={mean(rows, 'conformidade_filtro'):.2f}  "
                  f"distintos={mean(rows, 'textos_distintos'):.2f}  cos={mean(rows, 'cosseno_medio'):.3f}")

    # ---------------- temas (demonstração) ----------------
    demo = {}
    for r in questions:
        if r["id"] not in ("1", "4", "7"):
            continue
        q, qvec, _ = prep[r["id"]]
        cands = [c for c in retrieval.retrieve(col, qvec, config.THEME_POOL, with_embeddings=True)
                 if c["score"] >= config.SCORE_MIN]
        themes = cluster_pool(cands, reps_per_theme=2)
        demo[q] = {"avaliacoes_no_conjunto": len(cands), "temas": [
            {"tema": t.label, "tamanho": t.size, "share": round(t.share, 3), "palavras": t.keywords,
             "exemplos": [cands[i]["text"][:110] for i in t.representatives]} for t in themes]}
    (HERE / "themes_demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2))

    # ---------------- saídas ----------------
    fields = sorted({k for r in all_rows + filt_rows for k in r})
    with (HERE / "retrieval_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows + filt_rows)

    md = [f"# Comparação de recuperação (k={args.k}, {len(questions)} perguntas, sem LLM de geração)\n",
          "## A. Técnicas de seleção (sem filtros)\n",
          "| Configuração | Textos distintos | Redundância ↓ | Cobertura de temas | Cosseno médio | ms/pergunta |",
          "|---|---|---|---|---|---|"]
    for label, s in summary.items():
        md.append(f"| {label} | {s['textos_distintos']:.2f} / {args.k} | {s['redundancia']:.3f} | "
                  f"{s['cobertura_temas']:.2f} | {s['cosseno_medio']:.3f} | {s['latencia_ms']:.0f} |")
    md += ["", "## Varredura de λ (agrupar idênticos + MMR)\n",
           "| λ | Textos distintos | Redundância ↓ | Cobertura de temas | Cosseno médio |", "|---|---|---|---|---|"]
    for lam, s in lam_summary.items():
        md.append(f"| {lam} | {s['textos_distintos']:.2f} | {s['redundancia']:.3f} | {s['cobertura_temas']:.2f} | {s['cosseno_medio']:.3f} |")
    md += ["", "## B. Filtros (perguntas com filtro esperado)\n",
           "| Configuração | Filtro | Conformidade com o filtro | Textos distintos | Cosseno médio |", "|---|---|---|---|---|"]
    for label in ("baseline (vetorial pura)", "agrupar + MMR"):
        for fl in ("sem filtro", "com filtro"):
            rows = [x for x in filt_rows if x["config"] == label and x["filtro"] == fl]
            md.append(f"| {label} | {fl} | {mean(rows, 'conformidade_filtro'):.2f} | "
                      f"{mean(rows, 'textos_distintos'):.2f} | {mean(rows, 'cosseno_medio'):.3f} |")
    (HERE / "retrieval_comparison.md").write_text("\n".join(md) + "\n")
    print("\nGravado: eval/retrieval_comparison.csv, eval/retrieval_comparison.md, eval/themes_demo.json")


if __name__ == "__main__":
    main()
