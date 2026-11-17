"""Regras de falta de evidência, aplicadas antes e depois do LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src import config

_TOKEN_RE = re.compile(r"^\W*(SEM_EVIDENCIA|FORA_DE_ESCOPO)\W*$", re.IGNORECASE)


@dataclass
class GateResult:
    passed: bool
    reason: str | None
    max_score: float
    n_above: int


def evaluate_gate(candidates: list[dict]) -> GateResult:
    """Decide se há evidência suficiente para chamar o LLM.

    - nenhum resultado relevante: max_score < T_MAX
    - evidência insuficiente: menos de MIN_DOCS reviews com score >= SCORE_MIN
    """
    scores = [c["score"] for c in candidates]
    max_score = max(scores) if scores else 0.0
    n_above = sum(1 for s in scores if s >= config.SCORE_MIN)
    if not scores or max_score < config.T_MAX:
        return GateResult(False, "nenhum_resultado_relevante", max_score, n_above)
    if n_above < config.MIN_DOCS:
        return GateResult(False, "evidencia_insuficiente", max_score, n_above)
    return GateResult(True, None, max_score, n_above)


def parse_llm_status(answer: str) -> str:
    """Converte a saída do LLM em status: ok | sem_evidencia | fora_de_escopo."""
    m = _TOKEN_RE.match((answer or "").strip())
    if not m:
        return "ok"
    return "fora_de_escopo" if m.group(1).upper() == "FORA_DE_ESCOPO" else "sem_evidencia"
