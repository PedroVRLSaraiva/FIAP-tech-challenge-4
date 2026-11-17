"""Carga, limpeza e montagem dos documentos a partir das reviews da Olist."""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normaliza espaços/quebras, troca URLs por token e remove caracteres de controle.

    Não remove acentos, stopwords nem altera a caixa: embeddings de transformers
    dependem do texto natural.
    """
    if text is None:
        return ""
    text = str(text)
    text = _URL_RE.sub(config.URL_TOKEN, text)
    # remove caracteres de controle (mantém \n e \t para virarem espaço depois)
    text = "".join(
        ch for ch in text
        if ch in "\n\t\r" or unicodedata.category(ch)[0] != "C"
    )
    text = text.replace("�", " ")  # lixo de encoding
    return _WS_RE.sub(" ", text).strip()


def is_too_short(text: str) -> bool:
    return len(text) < config.MIN_CHARS or len(text.split()) < config.MIN_WORDS


def build_documents(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Transforma o CSV bruto de reviews em documentos prontos para indexar.

    Retorna (DataFrame de documentos, contagens por etapa para o log/README).
    """
    stats: dict[str, int] = {"total": len(raw)}

    df = raw.copy()
    df["review_comment_message"] = df["review_comment_message"].fillna("").astype(str)
    df["review_comment_title"] = df["review_comment_title"].fillna("").astype(str)

    # 1) Seleção: só reviews com mensagem não vazia
    df = df[df["review_comment_message"].str.strip() != ""]
    stats["com_mensagem"] = len(df)

    # 2) Título + mensagem
    title = df["review_comment_title"].map(clean_text)
    message = df["review_comment_message"].map(clean_text)
    df["text"] = [
        (f"{t}. {m}" if t and not t.endswith((".", "!", "?")) else f"{t} {m}".strip())
        if t else m
        for t, m in zip(title, message)
    ]
    df["text"] = df["text"].map(clean_text)

    # 3) Textos curtos
    short = df["text"].map(is_too_short)
    stats["descartadas_curtas"] = int(short.sum())
    df = df[~short]

    # 4) Duplicatas exatas por (order_id, texto)
    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "text"])
    stats["descartadas_duplicadas_pedido"] = before - len(df)

    # 5) review_id único (o dataset tem review_id repetidos)
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    stats["descartadas_review_id_repetido"] = before - len(df)

    df["data"] = pd.to_datetime(df["review_creation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["data"] = df["data"].fillna("desconhecida")
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce").fillna(0).astype(int)

    out = df.rename(columns={"review_id": "id"})[["id", "order_id", "text", "review_score", "data"]]
    out = out.reset_index(drop=True)
    stats["documentos_finais"] = len(out)

    for k, v in stats.items():
        log.info("%-34s %d", k, v)
    return out, stats


ENRICH_COLUMNS = ["order_status", "categoria", "ano_mes", "data_compra_ts", "atraso_dias", "entregue_atrasado"]


def _category_per_order(items: pd.DataFrame, products: pd.DataFrame, translation: pd.DataFrame) -> pd.Series:
    """Uma categoria por pedido: a mais frequente entre os itens (empate: o primeiro item).

    Usa a tradução em inglês; sem tradução, mantém o nome original em português.
    """
    tr = dict(zip(translation["product_category_name"], translation["product_category_name_english"]))
    it = items[["order_id", "order_item_id", "product_id"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    it = it.dropna(subset=["product_category_name"]).sort_values(["order_id", "order_item_id"])
    it["categoria"] = it["product_category_name"].map(lambda c: tr.get(c, c))

    def pick(g: pd.Series) -> str:
        counts = g.value_counts()
        top = counts[counts == counts.max()].index
        return next(c for c in g if c in top)  # primeira ocorrência entre as empatadas

    return it.groupby("order_id")["categoria"].agg(pick)


def enrich_with_orders(
    docs: pd.DataFrame,
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    """Junta reviews e pedido (`order_id`) e deriva os metadados usados nos filtros.

    - order_status
    - categoria: uma por pedido (ver `_category_per_order`)
    - ano_mes (int AAAAMM) e data_compra_ts (int, epoch em segundos), da data da compra
    - atraso_dias = data de entrega - data estimada, em dias (vazio se não entregue)
    - entregue_atrasado = atraso_dias > 0 (vazio se não entregue)
    Valores ausentes ficam como NaN no DataFrame e são omitidos dos metadados do Chroma.
    """
    o = orders[
        ["order_id", "order_status", "order_purchase_timestamp",
         "order_delivered_customer_date", "order_estimated_delivery_date"]
    ].copy()
    purchase = pd.to_datetime(o["order_purchase_timestamp"], errors="coerce")
    delivered = pd.to_datetime(o["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(o["order_estimated_delivery_date"], errors="coerce")
    o["ano_mes"] = (purchase.dt.year * 100 + purchase.dt.month).astype("Int64")
    o["data_compra_ts"] = (purchase.astype("int64") // 10**9).where(purchase.notna()).astype("Int64")
    o["atraso_dias"] = ((delivered - estimated).dt.total_seconds() / 86400).round(1)
    o["entregue_atrasado"] = (o["atraso_dias"] > 0).where(o["atraso_dias"].notna())
    o = o[["order_id", "order_status", "ano_mes", "data_compra_ts", "atraso_dias", "entregue_atrasado"]]

    cat = _category_per_order(items, products, translation).rename("categoria").reset_index()
    out = docs.merge(o, on="order_id", how="left").merge(cat, on="order_id", how="left")
    log.info("com pedido: %d | com categoria: %d | com entrega registrada: %d",
             out["order_status"].notna().sum(), out["categoria"].notna().sum(), out["atraso_dias"].notna().sum())
    return out


def to_metadata(docs: pd.DataFrame) -> list[dict]:
    """Metadados primitivos para o Chroma. Valores ausentes são omitidos (o Chroma não aceita None)."""
    cols = [c for c in ["review_score", "data", *ENRICH_COLUMNS] if c in docs.columns]
    out = []
    for row in docs[cols].itertuples(index=False):
        meta = {}
        for col, val in zip(cols, row):
            if val is None or (not isinstance(val, str) and pd.isna(val)):
                continue
            if col in ("review_score", "ano_mes", "data_compra_ts"):
                meta[col] = int(val)
            elif col == "atraso_dias":
                meta[col] = float(val)
            elif col == "entregue_atrasado":
                meta[col] = bool(val)
            else:
                meta[col] = str(val)
        out.append(meta)
    return out


def load_orders_tables() -> tuple[pd.DataFrame, ...] | None:
    paths = [config.ORDERS_CSV, config.ORDER_ITEMS_CSV, config.PRODUCTS_CSV, config.CATEGORY_TRANSLATION_CSV]
    if not all(p.exists() for p in paths):
        return None
    return tuple(pd.read_csv(p) for p in paths)


def load_raw_reviews(path: Path | None = None) -> pd.DataFrame:
    path = path or config.RAW_REVIEWS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe o dataset do Kaggle (olistbr/brazilian-ecommerce) para data/."
        )
    return pd.read_csv(path)


def prepare_and_save(path: Path | None = None) -> pd.DataFrame:
    docs, _ = build_documents(load_raw_reviews(path))
    tables = load_orders_tables()
    if tables is None:
        log.warning("CSVs de pedidos não encontrados: índice sem metadados de pedido (rode python download_data.py).")
    else:
        docs = enrich_with_orders(docs, *tables)
    config.PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(config.PROCESSED_PARQUET, index=False)
    log.info("Salvo em %s", config.PROCESSED_PARQUET)
    return docs
