"""Montagem do contexto numerado [1..k] enviado ao LLM."""
from __future__ import annotations

from src import config


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def entrega_label(meta: dict) -> str | None:
    d = meta.get("atraso_dias")
    if d is None:
        return None
    if d > 0.5:
        return f"entregue {d:.0f} dias após o prazo"
    if d < -0.5:
        return f"entregue {-d:.0f} dias antes do prazo"
    return "entregue no prazo"


def _dup_label(ev: dict) -> str | None:
    n = ev.get("dup_count", 1)
    if n <= 1:
        return None
    notas = ev.get("dup_notas") or {}
    dist = ", ".join(f"{k}★×{v}" for k, v in sorted(notas.items()))
    return f"texto idêntico em {n} avaliações da base" + (f" (notas: {dist})" if dist else "")


def _line(ev: dict) -> str:
    meta = ev.get("metadata", {})
    text = _truncate(ev["text"], config.MAX_REVIEW_CHARS_IN_CONTEXT)
    text = text.replace('"', "'")  # aspas escapadas para o texto da review não "fechar" o formato
    parts = [f'nota: {meta.get("review_score", "?")}/5', f'data: {meta.get("data", "?")}']
    if meta.get("categoria"):
        parts.append(f'categoria: {meta["categoria"]}')
    for extra in (entrega_label(meta), _dup_label(ev)):
        if extra:
            parts.append(extra)
    return f'[{ev["n"]}] ({" | ".join(parts)}) "{text}"'


def build_context(evidences: list[dict], themes: list[dict] | None = None, pool_size: int | None = None) -> str:
    """Contexto numerado. Com `themes`, agrupa as evidências sob um cabeçalho por tema."""
    if not themes:
        return "\n".join(_line(ev) for ev in evidences)

    by_n = {ev["n"]: ev for ev in evidences}
    blocks = []
    for t in themes:
        kws = ", ".join(t["keywords"]) if t["keywords"] else "sem palavras frequentes"
        header = (f'## {t["label"]}: {t["size"]} de {pool_size} avaliações recuperadas '
                  f'({t["share"]:.0%}). Palavras frequentes: {kws}')
        blocks.append("\n".join([header] + [_line(by_n[n]) for n in t["evidence_ns"]]))
    return "\n\n".join(blocks)
