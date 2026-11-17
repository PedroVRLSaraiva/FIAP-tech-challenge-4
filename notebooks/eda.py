"""Exploratória das reviews (rode com `python notebooks/eda.py`). Registre os números no README."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import data_prep

raw = data_prep.load_raw_reviews()
print("Linhas:", len(raw))
print("review_id duplicados:", raw["review_id"].duplicated().sum())
print("order_id com >1 review:", (raw["order_id"].value_counts() > 1).sum())
print("\nDistribuição de notas (%):\n", (raw["review_score"].value_counts(normalize=True).sort_index() * 100).round(1))
print("\n% nulos título:", round(raw["review_comment_title"].isna().mean() * 100, 1))
print("% nulos mensagem:", round(raw["review_comment_message"].isna().mean() * 100, 1))

msg = raw["review_comment_message"].dropna().map(data_prep.clean_text)
n_chars = msg.str.len()
print("\nComprimento (caracteres):", n_chars.describe(percentiles=[.5, .95, .99]).round(1).to_dict())
print("Comprimento (palavras) p95:", msg.str.split().str.len().quantile(.95))

short = msg[msg.map(data_prep.is_too_short)]
print(f"\nTextos curtos (<{data_prep.config.MIN_CHARS} chars ou <{data_prep.config.MIN_WORDS} palavras):", len(short))
print("Exemplos descartados:", short.sample(min(20, len(short)), random_state=42).tolist())

docs, stats = data_prep.build_documents(raw)
print("\nFunil:", stats)
