"""Roda o pipeline em lote sobre eval/perguntas.csv e grava um CSV de resultados.

Uso:
    python -m eval.run_eval                        # configuração recomendada -> eval/results.csv
    python -m eval.run_eval --preset baseline      # busca vetorial pura (comportamento anterior)
    python -m eval.run_eval --preset temas --ids 1,4,21 --out eval/results_temas.csv
Depois, preencha as colunas manuais (precisao_at_k, fidelidade, acerto_recusa, observacoes).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from src import config
from src.answer import answer
from src.filters import Filters
from src.lexical import LexicalIndex, normalize_text
from src.llm import OpenAIClient
from src.options import SearchOptions

HERE = Path(__file__).parent

PRESETS = {
    "baseline": lambda: SearchOptions(),
    "recomendado": lambda: SearchOptions.recommended(),
    "temas": lambda: SearchOptions.recommended(themes=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    ap.add_argument("--preset", choices=PRESETS, default="recomendado")
    ap.add_argument("--ids", default="", help="lista de ids separados por vírgula (padrão: todas)")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    args = ap.parse_args()

    options = PRESETS[args.preset]()
    rows = list(csv.DictReader((HERE / "perguntas.csv").open(encoding="utf-8")))
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]

    llm = OpenAIClient()  # um único cliente para toda a execução
    lexical = LexicalIndex.from_parquet() if (options.dedup or options.hybrid) else None
    print(f"Opções: {', '.join(options.describe())} | top_k={args.top_k} | {len(rows)} perguntas")

    fields = [
        "id", "tipo", "esperado", "pergunta", "opcoes", "status", "acertou_recusa_auto", "max_score",
        "n_evidencias", "n_textos_distintos", "n_temas", "filtros_aplicados", "filtro_esperado",
        "filtro_extraido_ok", "citacoes_validas", "citacoes_invalidas", "frases_sem_citacao",
        "latencia_total_s", "resposta", "evidencias_json",
        "precisao_at_k", "fidelidade", "acerto_recusa", "observacoes",  # preencher à mão
    ]
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            res = answer(r["pergunta"], top_k=args.top_k, llm=llm, options=options, lexical=lexical)
            cit = res.debug.get("citations", {})
            refused = res.status in ("sem_evidencia", "fora_de_escopo")
            expected = Filters.from_dict(json.loads(r["filtro_esperado"])) if r.get("filtro_esperado") else Filters()
            fdebug = res.debug.get("filtros", {})
            extraction_ok = ""
            if options.auto_filters and fdebug:
                extraction_ok = int(Filters.from_dict(fdebug.get("automaticos")) == expected)
            w.writerow({
                "id": r["id"], "tipo": r["tipo"], "esperado": r["esperado"], "pergunta": r["pergunta"],
                "opcoes": "+".join(options.describe()),
                "status": res.status,
                "acertou_recusa_auto": int(refused == (r["esperado"] == "recusa")),
                "max_score": res.debug.get("gate", {}).get("max_score"),
                "n_evidencias": len(res.evidences),
                "n_textos_distintos": len({normalize_text(e["text"]) for e in res.evidences}),
                "n_temas": len(res.themes),
                "filtros_aplicados": "; ".join(res.filters),
                "filtro_esperado": r.get("filtro_esperado", ""),
                "filtro_extraido_ok": extraction_ok,
                "citacoes_validas": len(cit.get("cited", [])),
                "citacoes_invalidas": len(cit.get("invalid", [])),
                "frases_sem_citacao": len(cit.get("uncited_sentences", [])),
                "latencia_total_s": round(sum(res.debug.get("timings_s", {}).values()), 2),
                "resposta": res.answer,
                "evidencias_json": json.dumps(res.evidences, ensure_ascii=False),
            })
            f.flush()
            filt = f"  [{'; '.join(res.filters)}]" if res.filters else ""
            print(f"[{r['id']:>2}] {res.status:<15} esperado={r['esperado']:<9} {r['pergunta'][:52]}{filt}")
    print(f"\nResultados em {out_path}")


if __name__ == "__main__":
    main()
