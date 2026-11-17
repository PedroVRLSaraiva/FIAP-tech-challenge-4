"""Processamento da pergunta: normalização.

A reescrita da query via LLM (query expansion) foi retirada do escopo obrigatório e ficou como
item opcional do segundo momento (ver Documentação/Plano de Execução.md, seção 6).
"""
from __future__ import annotations

import re

from src import config

_WS_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    q = _WS_RE.sub(" ", (question or "")).strip()
    return q[: config.MAX_QUESTION_CHARS]
