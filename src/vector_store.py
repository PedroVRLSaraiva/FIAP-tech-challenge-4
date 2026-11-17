"""ChromaDB: criação da collection, upsert em lotes e consulta."""
from __future__ import annotations

import chromadb
import numpy as np
from chromadb.config import Settings

from src import config

UPSERT_BATCH = 5000


def get_client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def get_collection(client=None):
    client = client or get_client()
    return client.get_collection(config.COLLECTION_NAME)


def reset_collection(client=None):
    client = client or get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_or_create_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def upsert_documents(collection, ids, texts, embeddings, metadatas) -> None:
    for start in range(0, len(ids), UPSERT_BATCH):
        end = start + UPSERT_BATCH
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in embeddings[start:end]],
            metadatas=metadatas[start:end],
        )


def query(
    collection,
    query_embedding: list[float],
    n_results: int,
    where: dict | None = None,
    with_embeddings: bool = False,
) -> list[dict]:
    """Retorna candidatos como dicts com id, text, metadata e score (= 1 - distância cosseno).

    `where` é um pré-filtro de metadados aplicado dentro da busca vetorial (não depois do top-k).
    Com `with_embeddings`, cada candidato traz também o vetor (`embedding`), usado por MMR e temas.
    """
    n = min(n_results, max(collection.count(), 1))
    include = ["documents", "metadatas", "distances"] + (["embeddings"] if with_embeddings else [])
    kwargs = {"query_embeddings": [query_embedding], "n_results": n, "include": include}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    embs = res["embeddings"][0] if with_embeddings else [None] * len(res["ids"][0])
    out = []
    for id_, doc, meta, dist, emb in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], embs
    ):
        item = {"id": id_, "text": doc, "metadata": meta or {}, "score": 1.0 - float(dist)}
        if emb is not None:
            item["embedding"] = np.asarray(emb, dtype=np.float32)
        out.append(item)
    return out


def get_by_ids(collection, ids: list[str]) -> list[dict]:
    """Busca documentos pelo id, com texto, metadados e vetor."""
    if not ids:
        return []
    res = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    return [
        {"id": i, "text": d, "metadata": m or {}, "embedding": np.asarray(e, dtype=np.float32)}
        for i, d, m, e in zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"])
    ]
