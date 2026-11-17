"""Filtros por metadados (nota, categoria, período da compra, entrega com atraso).

Um mesmo objeto `Filters` gera:
- o `where` do Chroma, aplicado na consulta ao vetor (pré-filtro, e não depois do top-k);
- a máscara equivalente em pandas, usada pela busca lexical (BM25);
- a descrição em português exibida na interface.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

_MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
ANO_MIN, ANO_MAX = 2016, 2018  # período coberto pela base


def _valid_ym(v: int | None) -> bool:
    return v is not None and ANO_MIN <= v // 100 <= ANO_MAX and 1 <= v % 100 <= 12


def fmt_ym(v: int) -> str:
    return f"{_MESES[v % 100 - 1]}/{v // 100}"


def pretty_category(c: str) -> str:
    return c.replace("_", " ")


@dataclass(frozen=True)
class Filters:
    nota_min: int | None = None
    nota_max: int | None = None
    categorias: tuple[str, ...] = ()
    periodo_inicio: int | None = None  # AAAAMM da compra
    periodo_fim: int | None = None
    entregue_atrasado: bool | None = None

    # ---- estado -------------------------------------------------------------
    def is_empty(self) -> bool:
        return all(not getattr(self, f.name) and getattr(self, f.name) is not False for f in fields(self))

    def to_dict(self) -> dict:
        """Só os campos preenchidos (serializável em JSON)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None or v == ():
                continue
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    def merge_over(self, fallback: Filters) -> Filters:
        """Os campos preenchidos aqui têm prioridade; os vazios vêm de `fallback`."""
        vals = {}
        for f in fields(self):
            mine, other = getattr(self, f.name), getattr(fallback, f.name)
            vals[f.name] = mine if (mine is not None and mine != ()) else other
        return Filters(**vals)

    # ---- Chroma -------------------------------------------------------------
    def to_where(self) -> dict | None:
        conds: list[dict] = []
        if self.nota_min is not None:
            conds.append({"review_score": {"$gte": self.nota_min}})
        if self.nota_max is not None:
            conds.append({"review_score": {"$lte": self.nota_max}})
        if self.categorias:
            conds.append({"categoria": {"$in": list(self.categorias)}})
        if self.periodo_inicio is not None:
            conds.append({"ano_mes": {"$gte": self.periodo_inicio}})
        if self.periodo_fim is not None:
            conds.append({"ano_mes": {"$lte": self.periodo_fim}})
        if self.entregue_atrasado is not None:
            conds.append({"entregue_atrasado": {"$eq": self.entregue_atrasado}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    # ---- pandas (busca lexical) --------------------------------------------
    def mask(self, df) -> np.ndarray:
        """Máscara booleana equivalente a `to_where`; metadado ausente nunca satisfaz um filtro."""
        m = np.ones(len(df), dtype=bool)
        if self.nota_min is not None:
            m &= (df["review_score"] >= self.nota_min).to_numpy()
        if self.nota_max is not None:
            m &= (df["review_score"] <= self.nota_max).to_numpy()
        if self.categorias:
            m &= df["categoria"].isin(self.categorias).to_numpy()
        if self.periodo_inicio is not None:
            m &= (df["ano_mes"] >= self.periodo_inicio).fillna(False).to_numpy(dtype=bool)
        if self.periodo_fim is not None:
            m &= (df["ano_mes"] <= self.periodo_fim).fillna(False).to_numpy(dtype=bool)
        if self.entregue_atrasado is not None:
            col = df["entregue_atrasado"]
            m &= (col == self.entregue_atrasado).fillna(False).to_numpy(dtype=bool)
        return m

    # ---- interface ----------------------------------------------------------
    def describe(self) -> list[str]:
        out = []
        if self.nota_min is not None and self.nota_max is not None:
            out.append(f"nota {self.nota_min}" if self.nota_min == self.nota_max
                       else f"nota de {self.nota_min} a {self.nota_max}")
        elif self.nota_min is not None:
            out.append(f"nota ≥ {self.nota_min}")
        elif self.nota_max is not None:
            out.append(f"nota ≤ {self.nota_max}")
        if self.categorias:
            out.append("categoria: " + ", ".join(pretty_category(c) for c in self.categorias))
        if self.periodo_inicio is not None and self.periodo_fim is not None:
            out.append(f"compra em {fmt_ym(self.periodo_inicio)}" if self.periodo_inicio == self.periodo_fim
                       else f"compra de {fmt_ym(self.periodo_inicio)} a {fmt_ym(self.periodo_fim)}")
        elif self.periodo_inicio is not None:
            out.append(f"compra a partir de {fmt_ym(self.periodo_inicio)}")
        elif self.periodo_fim is not None:
            out.append(f"compra até {fmt_ym(self.periodo_fim)}")
        if self.entregue_atrasado is True:
            out.append("entregue com atraso")
        elif self.entregue_atrasado is False:
            out.append("entregue no prazo")
        return out

    # ---- construção validada ------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict | None, valid_categories: set[str] | None = None) -> Filters:
        """Valida e normaliza um dicionário (ex.: saída do LLM). Campo inválido é descartado."""
        raw = raw if isinstance(raw, dict) else {}

        def as_int(v):
            if isinstance(v, bool) or v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        nmin, nmax = as_int(raw.get("nota_min")), as_int(raw.get("nota_max"))
        nmin = nmin if nmin is not None and 1 <= nmin <= 5 else None
        nmax = nmax if nmax is not None and 1 <= nmax <= 5 else None
        if nmin is not None and nmax is not None and nmin > nmax:
            nmin = nmax = None

        cats = raw.get("categorias") or []
        cats = tuple(dict.fromkeys(
            c for c in (cats if isinstance(cats, (list, tuple)) else [cats])
            if isinstance(c, str) and (valid_categories is None or c in valid_categories)
        ))

        pi, pf = as_int(raw.get("periodo_inicio")), as_int(raw.get("periodo_fim"))
        pi = pi if _valid_ym(pi) else None
        pf = pf if _valid_ym(pf) else None
        if pi is not None and pf is not None and pi > pf:
            pi = pf = None

        ea = raw.get("entregue_atrasado")
        ea = ea if isinstance(ea, bool) else None
        return cls(nmin, nmax, cats, pi, pf, ea)
