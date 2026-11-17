"""Pipeline offline: CSV de reviews -> documentos limpos -> embeddings -> ChromaDB.

Uso:
    python build_index.py            # reaproveita cache e faz upsert (idempotente)
    python build_index.py --reindex  # recria a collection reaproveitando o cache de embeddings (sem custo de API)
    python build_index.py --rebuild  # recria tudo, inclusive o cache de embeddings (gera embeddings de novo)
"""
from __future__ import annotations

import argparse
import logging
import time

from src import config, data_prep, embeddings, vector_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="recria a collection e o cache de embeddings")
    parser.add_argument("--reindex", action="store_true", help="recria a collection e mantém o cache de embeddings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    t0 = time.time()

    if args.rebuild:
        embeddings.clear_cache()

    docs = data_prep.prepare_and_save()
    ids = docs["id"].tolist()
    texts = docs["text"].tolist()

    vecs = embeddings.embed_with_cache(ids, texts)

    collection = (
        vector_store.reset_collection()
        if (args.rebuild or args.reindex)
        else vector_store.get_or_create_collection()
    )
    metadatas = data_prep.to_metadata(docs)
    vector_store.upsert_documents(collection, ids, texts, vecs, metadatas)

    count = collection.count()
    print(f"Índice pronto: {count} documentos em {time.time() - t0:.1f}s")
    if count != len(ids):
        raise SystemExit(f"Contagem inconsistente: índice={count}, documentos={len(ids)}")


if __name__ == "__main__":
    main()
