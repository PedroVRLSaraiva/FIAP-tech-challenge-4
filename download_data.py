"""Baixa os CSVs da Olist usados no projeto (via kagglehub) para data/.

Uso: python download_data.py

- olist_order_reviews_dataset.csv: base de conhecimento (obrigatório)
- olist_orders_dataset.csv, olist_order_items_dataset.csv, olist_products_dataset.csv,
  product_category_name_translation.csv: metadados e filtros (status, datas, atraso, categoria)
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import kagglehub

from src import config

DATASET = "olistbr/brazilian-ecommerce"
FILES = [
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]


def download(file: str) -> Path:
    dest = config.DATA_DIR / file
    path = Path(kagglehub.dataset_download(DATASET, path=file))
    # dependendo da versão do kagglehub, o arquivo único chega compactado em .zip
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z, z.open(file) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy(path, dest)
    return dest


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        print(f"Salvo em {download(f)}")


if __name__ == "__main__":
    main()
