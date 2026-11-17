"""Extração automática de filtros a partir da pergunta (LLM com saída JSON validada).

Custo controlado: o LLM só é chamado quando a pergunta contém um indício de filtro (nota, período,
categoria, atraso...). Perguntas sem indício não gastam requisição. O resultado é validado contra o
esquema (`Filters.from_dict`) e guardado em disco, então a mesma pergunta não é extraída duas vezes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from functools import lru_cache

import pandas as pd

from src import config
from src.filters import Filters

log = logging.getLogger(__name__)

_CACHE_FILE = config.CACHE_DIR / "filters.json"
PROMPT_VERSION = "v1"

_CUES = (
    # nota / sentimento
    r"notas?\b|estrelas?\b|insatisf|satisf|negativ|positiv|baixas?\b|altas?\b|ruim\b|bem avaliad|mal avaliad",
    # entrega
    r"atras|no prazo|adiantad|dentro do prazo",
    # período
    r"20(?:16|17|18)\b|janeiro|fevereiro|marco\b|abril|maio\b|junho|julho|agosto|setembro|outubro|novembro"
    r"|dezembro|semestre|trimestre|mes\b|ano\b|periodo",
    # categoria / tipo de produto
    r"categoria|cama\b|mesa\b|banho|beleza|saude|informatica|computador|esporte|lazer|movel|moveis|decorac"
    r"|utilidades|domestic|relogio|presente|telefon|celular|automotiv|carros?\b|brinquedo|ferramenta|jardim"
    r"|perfum|bebes?\b|eletronic|eletrodomestic|papelaria|bolsa|acessorio|pet shop|livro|malas?\b|construc"
    r"|alimento|bebida|musica|instrumento|fotografia|roupa|calcado|audio\b|games?\b|flores?\b",
)
# \b no início: evita casar "alta" dentro de "falta", por exemplo
_CUE_RE = re.compile(r"\b(?:" + "|".join(_CUES) + ")")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def needs_extraction(question: str) -> bool:
    """Indício barato de que a pergunta restringe nota, período, categoria ou entrega."""
    return bool(_CUE_RE.search(_fold(question)))


@lru_cache(maxsize=1)
def load_category_names() -> dict[str, str]:
    """{categoria em inglês (valor do metadado): nome em português}. Vazio se o CSV não existir."""
    try:
        tr = pd.read_csv(config.CATEGORY_TRANSLATION_CSV)
    except OSError:
        return {}
    return dict(zip(tr["product_category_name_english"], tr["product_category_name"]))


def _system_prompt(categories: dict[str, str]) -> str:
    cat_lines = "\n".join(f"- {en} ({pt})" if pt != en else f"- {en}" for en, pt in sorted(categories.items()))
    return f"""Você extrai FILTROS DE METADADOS de perguntas sobre avaliações de clientes de um e-commerce brasileiro \
(compras de setembro/2016 a outubro/2018). Responda SOMENTE com um objeto JSON, sem texto extra, com estas chaves:

{{"nota_min": int|null, "nota_max": int|null, "categorias": [string], "periodo_inicio": int|null, \
"periodo_fim": int|null, "entregue_atrasado": bool|null}}

Regras:
- nota (1 a 5): "nota baixa/ruim/negativa" e "clientes insatisfeitos" => nota_max=2. \
"nota alta/boa/positiva" e "clientes satisfeitos" => nota_min=4. "nota 1" => nota_min=1 e nota_max=1.
- periodo_inicio e periodo_fim no formato AAAAMM, referentes à data da COMPRA. "em 2017" => 201701 e 201712. \
"primeiro semestre de 2018" => 201801 e 201806. "em dezembro de 2017" => 201712 e 201712.
- entregue_atrasado: true SOMENTE se a pergunta restringe a pedidos entregues com atraso; false SOMENTE se restringe a \
pedidos entregues no prazo. NÃO preencha só porque a pergunta fala de "prazo de entrega" ou "atraso" como assunto.
- categorias: use SOMENTE nomes desta lista (o texto antes dos parênteses). Use [] se a pergunta não citar categoria \
ou tipo de produto da lista.
- Na dúvida, use null ou []. Nunca invente filtros que a pergunta não pede.

Categorias válidas:
{cat_lines}

Exemplos:
Pergunta: Quais os principais problemas em avaliações com nota baixa?
{{"nota_min": null, "nota_max": 2, "categorias": [], "periodo_inicio": null, "periodo_fim": null, "entregue_atrasado": null}}
Pergunta: O que os clientes elogiam em eletrônicos em 2017?
{{"nota_min": null, "nota_max": null, "categorias": ["electronics"], "periodo_inicio": 201701, "periodo_fim": 201712, "entregue_atrasado": null}}
Pergunta: Há reclamações sobre o prazo de entrega?
{{"nota_min": null, "nota_max": null, "categorias": [], "periodo_inicio": null, "periodo_fim": null, "entregue_atrasado": null}}"""


def _cache_key(question: str, model: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{question}".encode()).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
    except OSError:
        pass


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_filters(
    question: str, llm, categories: dict[str, str] | None = None
) -> tuple[Filters, str]:
    """Retorna (filtros, status). Status: "sem_indicio" | "ok" | "cached" | "falha".

    Falhas (LLM fora do ar, JSON inválido) devolvem filtros vazios: a busca segue sem filtro.
    """
    if not needs_extraction(question):
        return Filters(), "sem_indicio"

    categories = categories if categories is not None else load_category_names()
    model = getattr(llm, "model", None)
    key = _cache_key(question, model) if model else None
    if key:
        hit = _load_cache().get(key)
        if hit is not None:
            return Filters.from_dict(hit, set(categories) or None), "cached"

    try:
        raw = llm.generate(_system_prompt(categories), question, max_tokens=200)
    except Exception as e:
        log.warning("Extração de filtros falhou (%s); seguindo sem filtro.", e)
        return Filters(), "falha"
    data = _parse_json(raw)
    if data is None:
        log.warning("Extração de filtros devolveu JSON inválido; seguindo sem filtro.")
        return Filters(), "falha"

    filters = Filters.from_dict(data, set(categories) or None)
    if key:
        cache = _load_cache()
        cache[key] = filters.to_dict()
        _save_cache(cache)
    return filters, "ok"
