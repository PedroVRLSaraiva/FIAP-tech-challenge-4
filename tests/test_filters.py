import json

import chromadb
import numpy as np
import pandas as pd
import pytest

from src import config, filter_extraction
from src.filter_extraction import extract_filters, needs_extraction
from src.filters import Filters

CATS = {"electronics": "eletronicos", "telephony": "telefonia", "bed_bath_table": "cama_mesa_banho"}


# ---- where do Chroma ----------------------------------------------------------
def test_empty_filters_have_no_where():
    f = Filters()
    assert f.is_empty() and f.to_where() is None and f.describe() == []


def test_single_condition_is_not_wrapped_in_and():
    assert Filters(nota_max=2).to_where() == {"review_score": {"$lte": 2}}


def test_multiple_conditions_use_and():
    w = Filters(nota_max=2, categorias=("telephony",), periodo_inicio=201801, entregue_atrasado=True).to_where()
    assert set(w) == {"$and"} and len(w["$and"]) == 4
    assert {"entregue_atrasado": {"$eq": True}} in w["$and"]
    assert {"categoria": {"$in": ["telephony"]}} in w["$and"]


def test_false_boolean_counts_as_a_filter():
    f = Filters(entregue_atrasado=False)
    assert not f.is_empty() and f.to_where() == {"entregue_atrasado": {"$eq": False}}


def test_merge_manual_has_priority_over_auto():
    manual = Filters(nota_max=1, categorias=("telephony",))
    auto = Filters(nota_max=2, periodo_inicio=201701, periodo_fim=201712)
    m = manual.merge_over(auto)
    assert m.nota_max == 1 and m.categorias == ("telephony",)
    assert (m.periodo_inicio, m.periodo_fim) == (201701, 201712)


def test_describe_is_readable():
    d = Filters(nota_max=2, categorias=("bed_bath_table",), periodo_inicio=201801, periodo_fim=201803,
                entregue_atrasado=True).describe()
    assert d == ["nota ≤ 2", "categoria: bed bath table", "compra de jan/2018 a mar/2018", "entregue com atraso"]


# ---- validação ----------------------------------------------------------------
def test_from_dict_drops_invalid_fields():
    f = Filters.from_dict(
        {"nota_min": 9, "nota_max": "2", "categorias": ["electronics", "inventada"],
         "periodo_inicio": 201713, "periodo_fim": 201712, "entregue_atrasado": "sim"},
        valid_categories=set(CATS),
    )
    assert f.nota_min is None and f.nota_max == 2
    assert f.categorias == ("electronics",)
    assert f.periodo_inicio is None and f.periodo_fim == 201712
    assert f.entregue_atrasado is None


def test_from_dict_rejects_inverted_ranges_and_garbage():
    f = Filters.from_dict({"nota_min": 4, "nota_max": 2, "periodo_inicio": 201806, "periodo_fim": 201801})
    assert f.is_empty()
    assert Filters.from_dict(None).is_empty() and Filters.from_dict("x").is_empty()


# ---- pandas x Chroma: mesma semântica -----------------------------------------
def _frame():
    return pd.DataFrame({
        "id": list("abcdef"),
        "review_score": [1, 2, 5, 4, 1, 3],
        "categoria": ["telephony", "telephony", "electronics", None, "electronics", "telephony"],
        "ano_mes": pd.array([201801, 201802, 201712, 201805, 201803, None], dtype="Int64"),
        "entregue_atrasado": [True, False, None, True, True, False],
    })


CASES = [
    Filters(nota_max=2),
    Filters(nota_min=4),
    Filters(categorias=("telephony",)),
    Filters(periodo_inicio=201801, periodo_fim=201803),
    Filters(entregue_atrasado=True),
    Filters(entregue_atrasado=False),
    Filters(nota_max=2, categorias=("electronics", "telephony"), periodo_inicio=201801, entregue_atrasado=True),
]


@pytest.mark.parametrize("f", CASES)
def test_mask_matches_chroma_where(f):
    df = _frame()
    col = chromadb.EphemeralClient().create_collection(f"t{abs(hash(str(f)))}", metadata={"hnsw:space": "cosine"})
    metas = []
    for r in df.itertuples():
        m = {"review_score": int(r.review_score)}
        if r.categoria is not None:
            m["categoria"] = r.categoria
        if not pd.isna(r.ano_mes):
            m["ano_mes"] = int(r.ano_mes)
        if not pd.isna(r.entregue_atrasado):
            m["entregue_atrasado"] = bool(r.entregue_atrasado)
        metas.append(m)
    col.add(ids=df["id"].tolist(), embeddings=np.eye(6).tolist(), metadatas=metas)
    where = f.to_where()
    chroma_ids = set(col.get(where=where)["ids"])
    mask_ids = set(df["id"][f.mask(df)])
    assert chroma_ids == mask_ids


# ---- extração automática -----------------------------------------------------
@pytest.mark.parametrize("q,expected", [
    ("Quais os principais problemas em avaliações com nota baixa?", True),
    ("O que os clientes dizem sobre eletrônicos em 2017?", True),
    ("Pedidos entregues com atraso reclamam do quê?", True),
    ("Quais padrões aparecem nas avaliações de clientes insatisfeitos?", True),
    ("Os clientes reclamam de falta de itens no pedido?", False),
    ("O que os clientes relatam sobre problemas relacionados à entrega?", False),
    ("Qual a capital da França?", False),
])
def test_needs_extraction(q, expected):
    assert needs_extraction(q) is expected


class JSONLLM:
    model = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, system, user, max_tokens=None):
        self.calls += 1
        return self.reply


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(filter_extraction, "_CACHE_FILE", tmp_path / "filters.json")


def test_extract_skips_llm_without_cue(tmp_cache):
    llm = JSONLLM("{}")
    f, status = extract_filters("Os clientes reclamam de falta de itens no pedido?", llm, CATS)
    assert f.is_empty() and status == "sem_indicio" and llm.calls == 0


def test_extract_parses_validates_and_caches(tmp_cache):
    reply = 'Claro! {"nota_min": null, "nota_max": 2, "categorias": ["telephony", "xyz"], ' \
            '"periodo_inicio": 201801, "periodo_fim": 201806, "entregue_atrasado": null}'
    llm = JSONLLM(reply)
    q = "Principais problemas de telefonia com nota baixa no primeiro semestre de 2018?"
    f, status = extract_filters(q, llm, CATS)
    assert status == "ok" and llm.calls == 1
    assert f == Filters(nota_max=2, categorias=("telephony",), periodo_inicio=201801, periodo_fim=201806)
    f2, status2 = extract_filters(q, llm, CATS)
    assert status2 == "cached" and llm.calls == 1 and f2 == f


def test_extract_invalid_json_or_llm_error_is_safe(tmp_cache):
    f, status = extract_filters("avaliações com nota baixa", JSONLLM("não sei"), CATS)
    assert f.is_empty() and status == "falha"

    class Boom:
        model = "fake"

        def generate(self, *a, **k):
            raise RuntimeError("fora do ar")

    f, status = extract_filters("avaliações com nota baixa", Boom(), CATS)
    assert f.is_empty() and status == "falha"
