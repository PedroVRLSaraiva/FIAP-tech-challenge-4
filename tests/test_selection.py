import numpy as np
import pandas as pd
import pytest

from src import data_prep
from src.diversify import group_duplicates, mmr
from src.lexical import LexicalIndex, normalize_text, tokenize
from src.options import SearchOptions
from src.retrieval import hybrid_fuse
from src.selection import select_by_theme, select_evidences
from src.themes import cluster_pool


def cand(id_, text, score, vec, nota=1):
    return {"id": id_, "text": text, "score": score, "metadata": {"review_score": nota}, "embedding": np.asarray(vec, float)}


# ---- normalização e BM25 ------------------------------------------------------
def test_normalize_text_ignores_case_accent_and_punctuation():
    assert normalize_text("Muita demora na ENTREGA!!") == normalize_text("muita demora na entrega.")
    assert normalize_text("Não chegou") == normalize_text("nao chegou")


def test_tokenize_drops_stopwords_and_plural():
    assert tokenize("Quais são as reclamações sobre entregas?") == ["reclamaco", "entrega"] or \
        "entrega" in tokenize("Quais são as reclamações sobre entregas?")
    assert "os" not in tokenize("os clientes")


def _docs():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "text": ["Muita demora na entrega", "muita demora na entrega.", "Produto chegou quebrado",
                 "Atendimento horrível, ninguém responde", "Adorei a embalagem"],
        "review_score": [1, 5, 1, 1, 5],
    })


def test_lexical_search_and_mask():
    lex = LexicalIndex(_docs())
    hits = lex.search("demora entrega", 3)
    assert {h[0] for h in hits[:2]} == {"a", "b"}
    masked = lex.search("demora entrega", 3, mask=np.array([False, True, True, True, True]))
    assert masked[0][0] == "b"
    assert lex.search("palavrainexistente", 3) == []


def test_duplicate_info_counts_whole_base_with_notes():
    lex = LexicalIndex(_docs())
    info = lex.duplicate_info("MUITA demora na entrega!")
    assert info["count"] == 2 and info["notas"] == {1: 1, 5: 1}
    assert lex.duplicate_info("Produto chegou quebrado")["count"] == 1


# ---- agrupar idênticos ---------------------------------------------------------
def test_group_duplicates_collapses_and_counts_from_base():
    lex = LexicalIndex(_docs())
    pool = [cand("a", "Muita demora na entrega", 0.7, [1, 0]), cand("b", "muita demora na entrega.", 0.69, [1, 0]),
            cand("c", "Produto chegou quebrado", 0.6, [0, 1])]
    out = group_duplicates(pool, lex)
    assert [c["id"] for c in out] == ["a", "c"]  # fica o melhor de cada grupo, na ordem original
    assert out[0]["dup_count"] == 2 and out[0]["dup_notas"] == {1: 1, 5: 1}
    assert out[1]["dup_count"] == 1


def test_group_duplicates_without_stats_counts_the_pool():
    pool = [cand("a", "Boa", 0.7, [1, 0], nota=5), cand("b", "boa!", 0.6, [1, 0], nota=4)]
    out = group_duplicates(pool)
    assert len(out) == 1 and out[0]["dup_count"] == 2 and out[0]["dup_notas"] == {4: 1, 5: 1}


# ---- MMR -----------------------------------------------------------------------
def test_mmr_prefers_diverse_over_near_copy():
    pool = [cand("1", "A", 0.90, [1.0, 0.0]), cand("2", "A2", 0.89, [0.999, 0.04]), cand("3", "B", 0.80, [0.0, 1.0])]
    assert [c["id"] for c in mmr(pool, 2, lam=0.5)] == ["1", "3"]
    # lam = 1 reproduz a ordem por relevância
    assert [c["id"] for c in mmr(pool, 2, lam=1.0)] == ["1", "2"]


def test_mmr_handles_small_pool_and_empty():
    assert mmr([], 3, 0.5) == []
    assert len(mmr([cand("1", "A", 0.9, [1, 0])], 5, 0.5)) == 1


# ---- seleção completa ----------------------------------------------------------
def test_select_evidences_baseline_equals_top_k_by_score():
    pool = [cand(str(i), f"t{i}", 0.9 - i * 0.01, np.eye(10)[i]) for i in range(6)]
    ev, info = select_evidences(pool, "q", SearchOptions(), 3)
    assert [e["id"] for e in ev] == ["0", "1", "2"] and [e["n"] for e in ev] == [1, 2, 3]


def test_select_evidences_dedup_frees_slots():
    pool = [cand("1", "igual", 0.9, [1, 0, 0]), cand("2", "IGUAL!", 0.89, [1, 0, 0]),
            cand("3", "outro", 0.8, [0, 1, 0]), cand("4", "terceiro", 0.7, [0, 0, 1])]
    ev, _ = select_evidences(pool, "q", SearchOptions(dedup=True), 3)
    assert [e["text"] for e in ev] == ["igual", "outro", "terceiro"]
    assert ev[0]["dup_count"] == 2


class FakeReranker:
    model_name = "fake"

    def score(self, query, texts):
        return np.array([1.0 if "atendimento" in t else 0.0 for t in texts])


def test_select_evidences_rerank_reorders():
    pool = [cand("1", "entrega faltando", 0.9, [1, 0]), cand("2", "atendimento péssimo", 0.8, [0, 1])]
    ev, info = select_evidences(pool, "atendimento", SearchOptions(rerank=True), 2, reranker=FakeReranker())
    assert [e["id"] for e in ev] == ["2", "1"] and info["reranker"] == "fake"


# ---- híbrida (RRF) ---------------------------------------------------------------
class FakeStore:
    def get(self, ids, include):
        return {"ids": list(ids), "documents": [f"doc {i}" for i in ids],
                "metadatas": [{"review_score": 1} for _ in ids], "embeddings": [[1.0, 0.0] for _ in ids]}


def test_hybrid_fuse_rrf_and_bm25_only_gets_cosine():
    vec = [cand("v1", "x", 0.9, [1, 0]), cand("both", "y", 0.8, [1, 0]), cand("v3", "z", 0.7, [1, 0])]
    fused = hybrid_fuse(vec, [("both", 12.0), ("lex_only", 9.0)], FakeStore(), [1.0, 0.0], rrf_k=60)
    by = {c["id"]: c for c in fused}
    assert by["both"]["origem"] == "ambos" and by["lex_only"]["origem"] == "bm25" and by["v1"]["origem"] == "vetor"
    assert fused[0]["id"] == "both"  # aparece nas duas listas: maior RRF
    assert by["both"]["rrf"] == pytest.approx(1 / 62 + 1 / 61)
    assert by["lex_only"]["score"] == pytest.approx(1.0)  # cosseno calculado para quem só o BM25 trouxe


# ---- temas ---------------------------------------------------------------------
def _themed_pool():
    rng = np.random.default_rng(0)
    pool = []
    for t, (base, words) in enumerate([(np.array([1, 0, 0.0]), "atraso demora prazo"),
                                       (np.array([0, 1, 0.0]), "quebrado danificado embalagem"),
                                       (np.array([0, 0, 1.0]), "atendimento resposta loja")]):
        for i in range(8 + t * 4):
            pool.append(cand(f"{t}-{i}", f"{words} {i}", 0.7, base + rng.normal(scale=0.03, size=3)))
    return pool


def test_cluster_pool_finds_three_themes_with_sizes_and_keywords():
    themes = cluster_pool(_themed_pool(), reps_per_theme=2)
    assert [t.size for t in themes] == [16, 12, 8] and themes[0].label == "Tema A"
    assert abs(sum(t.share for t in themes) - 1.0) < 1e-9
    assert all(len(t.representatives) == 2 for t in themes)
    assert set(themes[0].keywords) & {"atendimento", "resposta", "loja"}   # o maior (16)
    assert set(themes[2].keywords) & {"atraso", "demora", "prazo"}         # o menor (8)
    assert not any(k.isdigit() for t in themes for k in t.keywords)


def test_cluster_pool_too_small_returns_empty():
    assert cluster_pool(_themed_pool()[:5]) == []


def test_select_by_theme_numbers_representatives_across_themes():
    ev, themes, info = select_by_theme(_themed_pool(), top_k=6)
    assert [e["n"] for e in ev] == list(range(1, len(ev) + 1))
    assert {e["tema"] for e in ev} == {"Tema A", "Tema B", "Tema C"}
    assert sorted(n for t in themes for n in t["evidence_ns"]) == [e["n"] for e in ev]
    assert select_by_theme(_themed_pool()[:5], 6) is None


# ---- enriquecimento dos dados do pedido -------------------------------------------
def test_enrich_with_orders_derives_metadata():
    docs = pd.DataFrame({"id": ["r1", "r2", "r3"], "order_id": ["o1", "o2", "o3"], "text": ["a"] * 3,
                         "review_score": [1, 5, 3], "data": ["2018-01-10"] * 3})
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"], "order_status": ["delivered", "delivered", "shipped"],
        "order_purchase_timestamp": ["2018-01-02 10:00:00", "2017-12-05 08:00:00", "2018-02-01 00:00:00"],
        "order_delivered_customer_date": ["2018-01-20 10:00:00", "2017-12-10 08:00:00", None],
        "order_estimated_delivery_date": ["2018-01-15 10:00:00", "2017-12-20 08:00:00", "2018-02-20 00:00:00"],
    })
    items = pd.DataFrame({"order_id": ["o1", "o1", "o1", "o2", "o2"], "order_item_id": [1, 2, 3, 1, 2],
                          "product_id": ["pA", "pB", "pB", "pC", "pD"]})
    products = pd.DataFrame({"product_id": ["pA", "pB", "pC", "pD"],
                             "product_category_name": ["telefonia", "pc_gamer", "beleza_saude", "esporte_lazer"]})
    tr = pd.DataFrame({"product_category_name": ["telefonia", "beleza_saude", "esporte_lazer"],
                       "product_category_name_english": ["telephony", "health_beauty", "sports_leisure"]})
    out = data_prep.enrich_with_orders(docs, orders, items, products, tr).set_index("id")
    assert out.loc["r1", "ano_mes"] == 201801 and out.loc["r2", "ano_mes"] == 201712
    assert out.loc["r1", "atraso_dias"] == 5.0 and bool(out.loc["r1", "entregue_atrasado"]) is True
    assert out.loc["r2", "atraso_dias"] == -10.0 and bool(out.loc["r2", "entregue_atrasado"]) is False
    assert pd.isna(out.loc["r3", "atraso_dias"]) and pd.isna(out.loc["r3", "entregue_atrasado"])
    # categoria: a mais frequente entre os itens (pc_gamer x2 sem tradução vence telefonia x1); empate: o primeiro item
    assert out.loc["r1", "categoria"] == "pc_gamer"
    assert out.loc["r2", "categoria"] == "health_beauty"
    assert pd.isna(out.loc["r3", "categoria"])


def test_to_metadata_omits_missing_values_with_primitive_types():
    df = pd.DataFrame({"review_score": [1, 5], "data": ["2018-01-01"] * 2, "order_status": ["delivered", "shipped"],
                       "categoria": ["telephony", None], "ano_mes": pd.array([201801, None], dtype="Int64"),
                       "data_compra_ts": pd.array([1514764800, None], dtype="Int64"),
                       "atraso_dias": [5.0, np.nan], "entregue_atrasado": [True, None]})
    m = data_prep.to_metadata(df)
    assert m[0] == {"review_score": 1, "data": "2018-01-01", "order_status": "delivered", "categoria": "telephony",
                    "ano_mes": 201801, "data_compra_ts": 1514764800, "atraso_dias": 5.0, "entregue_atrasado": True}
    assert m[1] == {"review_score": 5, "data": "2018-01-01", "order_status": "shipped"}
    assert type(m[0]["ano_mes"]) is int and type(m[0]["entregue_atrasado"]) is bool
