"""Reranking com cross-encoder multilíngue (carregado só quando a opção é ligada)."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src import config

log = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class Reranker:
    """Pontua pares (pergunta, review) lendo os dois textos juntos.

    O bi-encoder do índice compara vetores calculados separadamente; o cross-encoder enxerga a
    pergunta e a review ao mesmo tempo e reordena por utilidade, mais do que por similaridade.
    """

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        log.info("Carregando reranker %s", self.model_name)
        self._model = CrossEncoder(self.model_name, max_length=config.RERANK_MAX_LENGTH)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return np.asarray(self._model.predict([(query, t) for t in texts]), dtype=float)


@lru_cache(maxsize=1)
def get_reranker(model_name: str | None = None) -> Reranker:
    return Reranker(model_name)
