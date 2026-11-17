"""Site de busca em linguagem natural sobre as avaliações da Olist (RAG)."""
from __future__ import annotations

import csv
import html
import io
import re

import streamlit as st

from src import config
from src.answer import AnswerResult, answer
from src.filters import Filters, fmt_ym, pretty_category
from src.llm import LLMError, OpenAIClient
from src.options import SearchOptions

st.set_page_config(page_title="Voice of Customer", page_icon="💬", layout="wide")

EXAMPLES = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados?",
    "Quais padrões aparecem nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que há problemas recorrentes com o prazo de entrega?",
    "Quais problemas aparecem nos pedidos entregues com atraso?",
    "Qual a capital da França?",
]

CSS = """
<style>
.ev-card {border:1px solid rgba(128,128,128,.35); border-radius:8px; padding:.7rem .9rem; margin-bottom:.6rem;}
.ev-card.cited {border-left:4px solid #2e7d32;}
.ev-head {font-size:.82rem; opacity:.8; margin-bottom:.3rem;}
.ev-n {font-weight:700;}
.ev-text {white-space:pre-wrap;}
.badge {display:inline-block; padding:0 .45rem; border-radius:4px; font-size:.75rem; margin-left:.4rem;
        background:rgba(46,125,50,.18);}
.badge.muted {background:rgba(128,128,128,.2);}
.badge.dup {background:rgba(25,118,210,.18);}
.chip {display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.8rem; margin:0 .35rem .35rem 0;
       background:rgba(25,118,210,.15);}
.theme {margin-bottom:.7rem;}
.theme-bar {height:8px; border-radius:4px; background:rgba(25,118,210,.75);}
.theme-track {height:8px; border-radius:4px; background:rgba(128,128,128,.2); width:100%;}
.theme-kw {font-size:.8rem; opacity:.75;}
</style>
"""


# --- Recursos pesados carregados uma única vez ------------------------------
@st.cache_resource(show_spinner="Carregando índice...")
def load_resources():
    from src import vector_store

    if not config.CHROMA_DIR.exists():
        return None, "Índice não encontrado. Rode `python build_index.py` antes de abrir o site."
    try:
        collection = vector_store.get_collection()
    except Exception:
        return None, "Índice não encontrado. Rode `python build_index.py` antes de abrir o site."
    return collection, None


@st.cache_resource
def load_llm():
    try:
        return OpenAIClient(), None
    except LLMError as e:
        return None, str(e)


@st.cache_resource(show_spinner="Carregando índice de busca lexical (BM25)...")
def load_lexical():
    """BM25, contagem de textos idênticos e metadados. Sem o parquet, essas opções ficam indisponíveis."""
    from src.lexical import LexicalIndex

    try:
        return LexicalIndex.from_parquet()
    except Exception:
        return None


@st.cache_resource(show_spinner="Carregando o modelo de reranking (na primeira vez baixa ~470 MB)...")
def load_reranker():
    from src import rerank

    return rerank.get_reranker() if rerank.is_available() else None


# --- Renderização ------------------------------------------------------------
def stars(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return "★" * n + "☆" * (5 - n)


def render_answer_html(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\[(\d+)\]", r'<a href="#ev-\1">[\1]</a>', safe)


def render_evidence(ev: dict, muted: bool = False) -> str:
    cited = ev.get("cited")
    cls = "ev-card cited" if cited else "ev-card"
    badges = ""
    if muted:
        badges += '<span class="badge muted">não conclusiva</span>'
    elif cited:
        badges += '<span class="badge">citada na resposta</span>'
    if ev.get("dup_count", 1) > 1:
        notas = ", ".join(f"{k}★×{v}" for k, v in sorted((ev.get("dup_notas") or {}).items()))
        badges += (f'<span class="badge dup">texto idêntico em {ev["dup_count"]} avaliações'
                   f'{" (" + html.escape(notas) + ")" if notas else ""}</span>')
    meta = []
    if ev.get("categoria"):
        meta.append(html.escape(pretty_category(ev["categoria"])))
    if ev.get("entrega"):
        meta.append(html.escape(ev["entrega"]))
    if ev.get("tema"):
        meta.append(html.escape(ev["tema"]))
    meta_txt = f' · {" · ".join(meta)}' if meta else ""
    return (
        f'<div class="{cls}" id="ev-{ev["n"]}">'
        f'<div class="ev-head"><span class="ev-n">[{ev["n"]}]</span> '
        f'{stars(ev.get("nota"))} · {html.escape(str(ev.get("data")))}{meta_txt} · '
        f'score {ev["score"]:.3f}{badges}</div>'
        f'<div class="ev-text">{html.escape(ev["text"])}</div></div>'
    )


def render_themes(themes: list[dict], pool: int | None) -> str:
    out = []
    for t in themes:
        refs = " ".join(f'<a href="#ev-{n}">[{n}]</a>' for n in t["evidence_ns"])
        out.append(
            f'<div class="theme"><b>{html.escape(t["label"])}</b> · {t["size"]} de {pool} avaliações recuperadas '
            f'({t["share"]:.0%}) · exemplos {refs}'
            f'<div class="theme-track"><div class="theme-bar" style="width:{t["share"] * 100:.0f}%"></div></div>'
            f'<div class="theme-kw">palavras frequentes: {html.escape(", ".join(t["keywords"]) or "-")}</div></div>'
        )
    return "".join(out)


def evidences_csv(evidences: list[dict]) -> bytes:
    cols = ["n", "id", "text", "nota", "data", "categoria", "entrega", "atraso_dias", "score", "cited",
            "tema", "dup_count"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["n", "review_id", "texto", "nota", "data", "categoria", "entrega", "atraso_dias", "score",
                "citada", "tema", "textos_identicos_na_base"])
    for e in evidences:
        w.writerow([e.get(c, "") if e.get(c) is not None else "" for c in cols])
    return buf.getvalue().encode("utf-8-sig")  # BOM: o Excel abre os acentos corretamente


def show_result(res: AnswerResult) -> None:
    if res.filters:
        st.markdown(
            "".join(f'<span class="chip">{html.escape(f)}</span>' for f in res.filters)
            + '<span class="theme-kw"> recorte aplicado à busca</span>',
            unsafe_allow_html=True,
        )

    if res.status == "ok":
        st.markdown("### Resposta")
        st.markdown(render_answer_html(res.answer), unsafe_allow_html=True)
        st.caption(f"Baseada em {len(res.evidences)} avaliações recuperadas (amostra da base).")
        unc = res.debug.get("citations", {}).get("uncited_sentences", [])
        if unc:
            st.warning("Algumas afirmações da resposta não têm citação. Confira com as evidências.")
    elif res.status == "sem_evidencia":
        st.warning(res.answer)
    elif res.status == "fora_de_escopo":
        st.info(res.answer)
    else:
        st.error(res.answer)

    if res.themes:
        st.markdown("### Temas")
        st.markdown(render_themes(res.themes, res.debug.get("selecao", {}).get("pool_temas")), unsafe_allow_html=True)

    if res.evidences:
        head, btn = st.columns([4, 1])
        head.markdown(f"### Evidências ({len(res.evidences)})")
        btn.download_button("Exportar CSV", evidences_csv(res.evidences), "evidencias.csv", "text/csv",
                            use_container_width=True)
        st.markdown("".join(render_evidence(e) for e in res.evidences), unsafe_allow_html=True)
    elif res.nearest:
        with st.expander("Avaliações mais próximas encontradas (não conclusivas)"):
            st.markdown("".join(render_evidence(e, muted=True) for e in res.nearest), unsafe_allow_html=True)

    with st.expander("Como a resposta foi gerada"):
        d = res.debug
        st.markdown(f"**Pergunta (usada na busca):** {d.get('query', res.question)}")
        st.markdown(f"**Opções de busca:** {', '.join(d.get('opcoes', []))}")
        st.markdown("**Filtros**")
        st.json(d.get("filtros", {}), expanded=False)
        st.markdown("**Gate de evidência**")
        st.json(d.get("gate", {}), expanded=False)
        st.markdown("**Scores dos melhores candidatos**")
        st.write(d.get("scores_top_candidates", []))
        if "hibrida" in d:
            st.markdown("**Busca híbrida (BM25 + vetorial)**")
            st.json(d["hibrida"], expanded=False)
        if "selecao" in d:
            st.markdown("**Seleção das evidências**")
            st.json(d["selecao"], expanded=False)
        st.markdown("**Latência por etapa (s)**")
        st.json(d.get("timings_s", {}), expanded=False)
        if "citations" in d:
            st.markdown("**Validação de citações**")
            st.json(d["citations"], expanded=False)
        if "prompt" in d:
            st.markdown("**Prompt enviado ao LLM**")
            st.code(d["prompt"]["system"] + "\n\n" + d["prompt"]["user"], language="text")


# --- Página ------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.title("Voice of Customer Intelligence")
st.caption(
    "Pergunte em linguagem natural sobre as avaliações de clientes da Olist. "
    "As respostas usam somente as avaliações recuperadas e citam as evidências."
)

lexical = load_lexical()
REC = config.RECOMMENDED_OPTIONS
MONTHS = [y * 100 + m for y in (2016, 2017, 2018) for m in range(1, 13) if 201609 <= y * 100 + m <= 201810]

with st.sidebar:
    st.header("Parâmetros")
    top_k = st.slider("Avaliações no contexto (top_k)", 3, config.MAX_TOP_K, config.DEFAULT_TOP_K)
    st.caption("Para perguntas agregadas (ex.: 'principais problemas'), valores maiores dão uma amostra mais ampla.")

    st.header("Seleção das evidências")
    dedup = st.checkbox("Agrupar textos idênticos", REC["dedup"], disabled=lexical is None,
                        help="Colapsa cópias do mesmo texto e informa em quantas avaliações da base ele aparece.")
    mmr = st.checkbox("Diversificar (MMR)", REC["mmr"], help="Evita evidências parecidas entre si.")
    mmr_lambda = st.slider("λ: relevância × diversidade", 0.0, 1.0, config.MMR_LAMBDA, 0.05, disabled=not mmr)
    rerank_on = st.checkbox("Reordenar com cross-encoder", REC["rerank"],
                            help="Lê pergunta e avaliação juntas e reordena por utilidade (~0,1 a 0,4 s).")
    hybrid = st.checkbox("Busca híbrida (BM25 + vetorial)", REC["hybrid"], disabled=lexical is None,
                         help="Funde a busca por palavras (BM25) com a semântica por Reciprocal Rank Fusion.")
    themes_on = st.checkbox("Resumir por temas (KMeans)", REC["themes"],
                            help="Agrupa as avaliações recuperadas em temas e informa o tamanho de cada um.")

    st.header("Filtros")
    auto_filters = st.checkbox("Extrair filtros da pergunta", REC["auto_filters"],
                               help="Usa o LLM para entender 'nota baixa', 'em 2017', 'entregue com atraso'... "
                                    "Só consulta o LLM quando a pergunta sugere um filtro.")
    nota = st.slider("Nota", 1, 5, (1, 5))
    cats = st.multiselect(
        "Categoria", sorted(lexical.docs["categoria"].dropna().unique()) if lexical is not None else [],
        format_func=pretty_category, disabled=lexical is None,
    )
    p1, p2 = st.columns(2)
    p_ini = p1.selectbox("Compra de", [None] + MONTHS, format_func=lambda v: "-" if v is None else fmt_ym(v))
    p_fim = p2.selectbox("até", [None] + MONTHS, format_func=lambda v: "-" if v is None else fmt_ym(v))
    entrega = st.selectbox("Entrega", ["Qualquer", "Entregue com atraso", "Entregue no prazo"])
    st.caption("Filtros manuais têm prioridade sobre os extraídos da pergunta.")

    st.divider()
    st.header("Histórico")
    history = st.session_state.setdefault("history", [])
    history_slot = st.empty()  # redesenhado depois da busca, para incluir a pergunta atual


def render_history() -> None:
    with history_slot.container():
        for h in reversed(history[-10:]):
            st.caption(f"• {h}")


render_history()

options = SearchOptions(
    hybrid=hybrid and lexical is not None, dedup=dedup and lexical is not None, mmr=mmr, mmr_lambda=mmr_lambda,
    rerank=rerank_on, themes=themes_on, auto_filters=auto_filters,
    filters=Filters(
        nota_min=nota[0] if nota[0] > 1 else None,
        nota_max=nota[1] if nota[1] < 5 else None,
        categorias=tuple(cats),
        periodo_inicio=p_ini, periodo_fim=p_fim,
        entregue_atrasado={"Qualquer": None, "Entregue com atraso": True, "Entregue no prazo": False}[entrega],
    ),
)

collection, err = load_resources()
if err:
    st.error(err)
    st.stop()
llm, llm_err = load_llm()
if llm_err:
    st.error(llm_err)
    st.stop()

st.session_state.setdefault("question", "")
st.markdown("**Exemplos**")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state["question"] = ex
        st.session_state["run"] = True

with st.form("search"):
    question = st.text_input(
        "Sua pergunta", key="question", max_chars=config.MAX_QUESTION_CHARS,
        placeholder="Ex.: O que os clientes reclamam sobre a entrega?",
    )
    submitted = st.form_submit_button("Buscar")

if submitted or st.session_state.pop("run", False):
    q = (st.session_state["question"] or "").strip()
    if not q:
        st.warning("Digite uma pergunta.")
    else:
        cache = st.session_state.setdefault("cache", {})
        key = (q, top_k, options.key())
        if key not in cache:
            with st.spinner("Buscando avaliações e gerando a resposta..."):
                try:
                    res = answer(
                        q, top_k=top_k, llm=llm, collection=collection, options=options, lexical=lexical,
                        reranker=load_reranker() if options.rerank else None,
                    )
                except Exception:
                    res = AnswerResult(q, "Falha inesperada ao processar a pergunta. Tente novamente.", "erro")
            if res.status != "erro":
                cache[key] = res
        else:
            res = cache[key]
        st.session_state["last"] = res
        if q not in history:
            history.append(q)
            render_history()

# O último resultado é redesenhado a cada execução: o botão de download recarrega a página.
if "last" in st.session_state:
    show_result(st.session_state["last"])
