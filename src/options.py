"""Opções de busca: cada técnica de seleção das evidências pode ser ligada ou desligada."""
from __future__ import annotations

from dataclasses import dataclass, field

from src import config
from src.filters import Filters


@dataclass(frozen=True)
class SearchOptions:
    """Por padrão tudo desligado: reproduz a busca vetorial pura (linha de base da avaliação)."""

    hybrid: bool = False        # BM25 + vetorial com Reciprocal Rank Fusion
    dedup: bool = False         # agrupa textos idênticos e informa a contagem
    mmr: bool = False           # diversidade entre as reviews escolhidas
    mmr_lambda: float = config.MMR_LAMBDA
    rerank: bool = False        # cross-encoder reordena os candidatos
    themes: bool = False        # agrupa em temas (KMeans) para perguntas agregadas
    auto_filters: bool = False  # extrai filtros da pergunta com o LLM
    filters: Filters = field(default_factory=Filters)  # filtros manuais (prioridade sobre os automáticos)

    @classmethod
    def recommended(cls, **overrides) -> SearchOptions:
        """Configuração padrão do site, definida pela comparação em eval/compare_retrieval.py."""
        base = dict(config.RECOMMENDED_OPTIONS)
        base.update(overrides)
        return cls(**base)

    @property
    def needs_embeddings(self) -> bool:
        return self.mmr or self.themes

    def key(self) -> tuple:
        return (self.hybrid, self.dedup, self.mmr, round(self.mmr_lambda, 3), self.rerank, self.themes,
                self.auto_filters, repr(self.filters))

    def describe(self) -> list[str]:
        on = [n for n, v in (("híbrida", self.hybrid), ("dedup", self.dedup), ("MMR", self.mmr),
                             ("rerank", self.rerank), ("temas", self.themes),
                             ("filtros automáticos", self.auto_filters)) if v]
        return on or ["vetorial pura"]
