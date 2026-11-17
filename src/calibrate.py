"""Calibração dos limiares do gate (T_MAX, SCORE_MIN, MIN_DOCS).

Uso: python -m src.calibrate
Roda perguntas DENTRO e FORA do escopo pelo mesmo caminho do pipeline (embedding da pergunta +
recuperação). Não chama o LLM de geração: só a API de embeddings. Mostra max_score e a quantidade
de docs acima de cada limiar candidato, e sugere T_MAX / SCORE_MIN.
Grave os valores escolhidos em src/config.py.
"""
from __future__ import annotations

import numpy as np

from src import embeddings, query, retrieval, vector_store

IN_SCOPE = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas de entrega?",
    "Quais aspectos da experiência de compra são mais elogiados?",
    "Quais padrões aparecem nas avaliações de clientes insatisfeitos?",
    "Há reclamações sobre o prazo de entrega?",
    "Os clientes reclamam da qualidade dos produtos?",
    "Houve problemas com produto diferente do anunciado?",
    "O que os clientes dizem sobre o atendimento da loja?",
    "Produtos chegaram quebrados ou danificados?",
    "Os clientes elogiam a rapidez da entrega?",
    "Há reclamações sobre embalagem?",
    "Clientes receberam o produto errado?",
    "Quais problemas de entrega aparecem com mais frequência?",
    "O que agrada os clientes nos produtos comprados?",
    "Clientes ficaram satisfeitos com o custo-benefício?",
]
OUT_OF_SCOPE = [
    "Qual a capital da França?",
    "Como funciona um motor de carro?",
    "Quem ganhou a Copa do Mundo de 2018?",
    "Qual é a receita de bolo de chocolate?",
    "Explique a teoria da relatividade.",
    "Quanto é 15 vezes 12?",
    "Quem escreveu Dom Casmurro?",
    "Qual a previsão do tempo para amanhã?",
    "Como instalar Python no Windows?",
    "Qual o melhor time de futebol do Brasil?",
    "Quantos planetas existem no sistema solar?",
    "Como declarar imposto de renda?",
    "O que é fotossíntese?",
    "Qual a cotação do dólar hoje?",
    "Quem foi Santos Dumont?",
]


THRESHOLDS = np.arange(0.35, 0.71, 0.05)


def profile(questions: list[str], collection) -> list[dict]:
    rows = []
    for q in questions:
        cands = retrieval.retrieve(collection, embeddings.embed_query(query.normalize_question(q)))
        scores = np.array([c["score"] for c in cands])
        rows.append({
            "q": q,
            "max": float(scores.max()),
            **{f"n>={t:.2f}": int((scores >= t).sum()) for t in THRESHOLDS},
        })
    return rows


def main() -> None:
    col = vector_store.get_collection()
    inn, out = profile(IN_SCOPE, col), profile(OUT_OF_SCOPE, col)

    print(f"{'ESCOPO':<6} {'max_score':>9}  pergunta")
    for tag, rows in (("DENTRO", inn), ("FORA", out)):
        for r in rows:
            print(f"{tag:<6} {r['max']:>9.4f}  {r['q']}")

    in_max = np.array([r["max"] for r in inn])
    out_max = np.array([r["max"] for r in out])
    print(f"\nDentro: min={in_max.min():.4f} média={in_max.mean():.4f}")
    print(f"Fora:   max={out_max.max():.4f} média={out_max.mean():.4f}")
    if in_max.min() > out_max.max():
        t = (in_max.min() + out_max.max()) / 2
        print(f"Separação limpa. Sugestão: T_MAX ≈ {t:.3f}")
    else:
        print("Os conjuntos se sobrepõem: escolha T_MAX priorizando recusar o fora de escopo "
              "e revise as perguntas dentro do escopo que ficaram abaixo.")
    print("\nPara SCORE_MIN, olhe as colunas n>=limiar nas perguntas DENTRO: "
          "escolha um valor que ainda deixe >= MIN_DOCS reviews.")
    for r in inn:
        print(r["q"][:50].ljust(52), " ".join(f"{k[2:]}:{v:>2}" for k, v in r.items() if k.startswith("n>=")))


if __name__ == "__main__":
    main()
