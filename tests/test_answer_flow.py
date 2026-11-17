import numpy as np
import pytest

from src import config, guardrails
from src.answer import answer, validate_citations
from src.context import build_context
from src.retrieval import select


# ---- validador de citações ---------------------------------------------------
def test_validate_citations_removes_out_of_range():
    r = validate_citations("Atrasos frequentes [1][2]. Produto errado [9].", k=3)
    assert r["invalid"] == [9]
    assert "[9]" not in r["answer"]
    assert r["cited"] == [1, 2]
    assert r["valid_rate"] == pytest.approx(2 / 3)


def test_validate_citations_expands_ranges():
    r = validate_citations("Há atrasos [1]–[3] e demora [5]-[6].", k=6)
    assert r["cited"] == [1, 2, 3, 5, 6]
    assert r["invalid"] == []
    assert "[2]" in r["answer"]


def test_validate_citations_range_out_of_bounds_is_cleaned():
    r = validate_citations("Atrasos [2]–[5].", k=3)
    assert r["cited"] == [2, 3]
    assert r["invalid"] == [4, 5]


def test_citation_after_period_is_not_flagged_as_uncited():
    r = validate_citations(
        "- Atraso na entrega do produto, que impediu clientes de avaliar a compra. [2][5]", k=6
    )
    assert r["uncited_sentences"] == []


def test_intro_line_ending_with_colon_is_not_flagged():
    r = validate_citations("Nas avaliações recuperadas, os principais problemas relatados são:\n- Atraso [1].", k=2)
    assert r["uncited_sentences"] == []


def test_validate_citations_flags_uncited_sentences():
    r = validate_citations("Os clientes reclamam muito do prazo de entrega nas avaliações.", k=2)
    assert r["uncited_sentences"]
    assert r["valid_rate"] == 0.0


# ---- guardrails ----------------------------------------------------------------
def _cands(scores):
    return [{"id": str(i), "text": f"t{i}", "metadata": {}, "score": s} for i, s in enumerate(scores)]


def test_gate_no_relevant_results():
    g = guardrails.evaluate_gate(_cands([config.T_MAX - 0.05, config.T_MAX - 0.1]))
    assert not g.passed and g.reason == "nenhum_resultado_relevante"


def test_gate_insufficient_evidence():
    g = guardrails.evaluate_gate(_cands([config.T_MAX + 0.05] + [config.SCORE_MIN - 0.1] * 5))
    assert not g.passed and g.reason == "evidencia_insuficiente"


def test_gate_passes():
    g = guardrails.evaluate_gate(_cands([config.T_MAX + 0.05] * config.MIN_DOCS))
    assert g.passed


def test_gate_empty():
    assert not guardrails.evaluate_gate([]).passed


def test_parse_llm_status():
    assert guardrails.parse_llm_status("SEM_EVIDENCIA") == "sem_evidencia"
    assert guardrails.parse_llm_status(" sem_evidencia. ") == "sem_evidencia"
    assert guardrails.parse_llm_status("FORA_DE_ESCOPO") == "fora_de_escopo"
    assert guardrails.parse_llm_status("Há atrasos [1].") == "ok"
    # a palavra no meio de uma resposta real não deve virar recusa
    assert guardrails.parse_llm_status("Não há SEM_EVIDENCIA aqui, mas sim atrasos [1].") == "ok"


# ---- seleção e contexto --------------------------------------------------------
def test_select_numbers_and_cuts():
    cands = sorted(_cands([0.9, 0.85, 0.8, config.SCORE_MIN - 0.2]), key=lambda c: -c["score"])
    sel = select(cands, top_k=2)
    assert [c["n"] for c in sel] == [1, 2]
    assert len(select(cands, top_k=10)) == 3  # o de baixo score é cortado


def test_build_context_format():
    ctx = build_context([{"n": 1, "text": 'Chegou "quebrado"', "metadata": {"review_score": 1, "data": "2018-03-01"}}])
    assert ctx.startswith("[1] (nota: 1/5 | data: 2018-03-01)")
    assert '"quebrado"' not in ctx


# ---- fluxo completo com dublês -------------------------------------------------
class FakeCollection:
    """Coleção falsa: devolve `scores` em ordem; respeita `where` apenas devolvendo vazio se `empty_with_where`."""

    def __init__(self, scores, empty_with_where=False):
        self._scores = scores
        self.empty_with_where = empty_with_where
        self.last_where = None

    def count(self):
        return len(self._scores)

    def query(self, query_embeddings, n_results, include, where=None):
        self.last_where = where
        n = 0 if (where and self.empty_with_where) else min(n_results, len(self._scores))
        out = {
            "ids": [[f"r{i}" for i in range(n)]],
            "documents": [[f"review {i} sobre entrega atrasada" for i in range(n)]],
            "metadatas": [[{"review_score": 1, "data": "2018-01-01", "categoria": "telephony", "atraso_dias": 6.0}
                           for _ in range(n)]],
            "distances": [[1 - s for s in self._scores[:n]]],
        }
        if "embeddings" in include:
            rng = np.random.default_rng(0)
            out["embeddings"] = [[rng.normal(size=8) for _ in range(n)]]
        return out


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system, user, max_tokens=None):
        self.calls += 1
        return self.reply


HI = [config.T_MAX + 0.15, config.T_MAX + 0.12, config.T_MAX + 0.1, config.T_MAX + 0.08]


def _run(scores, reply):
    llm = FakeLLM(reply)
    res = answer("Quais problemas de entrega?", top_k=5, llm=llm, collection=FakeCollection(scores), embed_fn=lambda q: [0.0])
    return res, llm


def test_flow_ok_with_citations():
    res, _ = _run(HI, "Há atrasos frequentes [1][2] e demora [3]. Algo [99].")
    assert res.status == "ok"
    assert "[99]" not in res.answer
    assert [e["n"] for e in res.evidences if e["cited"]] == [1, 2, 3]
    assert res.debug["query"] == "Quais problemas de entrega?"


def test_flow_gate_blocks_llm_generation():
    low = config.T_MAX - 0.1
    res, llm = _run([low, low - 0.05, low - 0.1], "não deveria ser chamado")
    assert res.status == "sem_evidencia"
    assert res.answer == config.MSG_SEM_EVIDENCIA
    assert llm.calls == 0  # o gate recusa antes de chamar o LLM
    assert res.nearest


def test_flow_llm_refusal_tokens():
    res, _ = _run(HI[:3], "SEM_EVIDENCIA")
    assert res.status == "sem_evidencia"
    res, _ = _run(HI[:3], "FORA_DE_ESCOPO")
    assert res.status == "fora_de_escopo"
    assert res.answer == config.MSG_FORA_DE_ESCOPO


def test_flow_empty_question():
    res = answer("   ", llm=FakeLLM("x"), collection=FakeCollection([0.9]), embed_fn=lambda q: [0.0])
    assert res.status == "erro"


# ---- opções: filtros, recorte vazio, MMR, contexto enriquecido -------------------
from src.filters import Filters  # noqa: E402
from src.options import SearchOptions  # noqa: E402


def test_manual_filter_goes_to_chroma_where_and_is_described():
    col = FakeCollection(HI)
    opts = SearchOptions(filters=Filters(nota_max=2, entregue_atrasado=True))
    res = answer("Quais problemas?", top_k=5, llm=FakeLLM("Atrasos [1][2]."), collection=col,
                 embed_fn=lambda q: [0.0], options=opts)
    assert res.status == "ok"
    assert col.last_where == {"$and": [{"review_score": {"$lte": 2}}, {"entregue_atrasado": {"$eq": True}}]}
    assert res.filters == ["nota ≤ 2", "entregue com atraso"]


def test_filters_with_no_candidates_report_empty_slice_without_calling_llm():
    llm = FakeLLM("não deveria ser chamado")
    res = answer("Quais problemas?", llm=llm, collection=FakeCollection(HI, empty_with_where=True),
                 embed_fn=lambda q: [0.0], options=SearchOptions(filters=Filters(nota_max=1)))
    assert res.status == "sem_evidencia" and res.answer == config.MSG_SEM_RECORTE
    assert llm.calls == 0 and res.debug["gate"]["reason"] == "recorte_vazio"


def test_auto_filters_use_llm_json_and_manual_wins():
    class LLM(FakeLLM):
        model = "fake"

        def generate(self, system, user, max_tokens=None):
            self.calls += 1
            if "<reviews>" in user:
                return "Resposta [1]."
            return '{"nota_min": null, "nota_max": 2, "categorias": [], "periodo_inicio": null, ' \
                   '"periodo_fim": null, "entregue_atrasado": null}'

    col = FakeCollection(HI)
    opts = SearchOptions(auto_filters=True, filters=Filters(nota_max=1))
    res = answer("Problemas em avaliações com nota baixa?", llm=LLM(""), collection=col,
                 embed_fn=lambda q: [0.0], options=opts)
    assert res.debug["filtros"]["automaticos"] == {"nota_max": 2}
    assert res.debug["filtros"]["efetivos"] == {"nota_max": 1}  # o manual vence
    assert col.last_where == {"review_score": {"$lte": 1}}


def test_mmr_and_dedup_path_runs_and_enriches_context():
    llm = FakeLLM("Atrasos [1][2].")
    opts = SearchOptions(dedup=True, mmr=True)
    res = answer("Quais problemas de entrega?", top_k=3, llm=llm, collection=FakeCollection(HI),
                 embed_fn=lambda q: [0.0], options=opts)
    assert res.status == "ok" and len(res.evidences) == 3
    assert res.debug["selecao"]["mmr_lambda"] == opts.mmr_lambda
    user_prompt = res.debug["prompt"]["user"]
    assert "categoria: telephony" in user_prompt and "entregue 6 dias após o prazo" in user_prompt
    assert res.evidences[0]["categoria"] == "telephony" and res.evidences[0]["entrega"]


def test_context_shows_duplicate_count_and_themes():
    ev = [{"n": 1, "text": "Muita demora na entrega", "metadata": {"review_score": 1, "data": "2018-01-01"},
           "dup_count": 9, "dup_notas": {1: 8, 5: 1}},
          {"n": 2, "text": "Produto chegou quebrado", "metadata": {"review_score": 1, "data": "2018-01-02"}}]
    ctx = build_context(ev)
    assert "texto idêntico em 9 avaliações da base (notas: 1★×8, 5★×1)" in ctx
    themed = build_context(ev, [{"label": "Tema A", "size": 40, "share": 0.4, "keywords": ["entrega", "atraso"],
                                 "evidence_ns": [1]},
                                {"label": "Tema B", "size": 10, "share": 0.1, "keywords": [], "evidence_ns": [2]}],
                           pool_size=100)
    assert "## Tema A: 40 de 100 avaliações recuperadas (40%). Palavras frequentes: entrega, atraso" in themed
    assert themed.index("## Tema A") < themed.index("[1]") < themed.index("## Tema B") < themed.index("[2]")
