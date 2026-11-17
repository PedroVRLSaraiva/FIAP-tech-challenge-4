"""Embeddings via API da OpenAI, geração em lote e cache em disco."""
from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache

import numpy as np

from src import config

log = logging.getLogger(__name__)

def _is_quota_error(e: Exception) -> bool:
    """429 por falta de créditos/cota (não adianta tentar de novo)."""
    body = getattr(e, "body", None) or {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    codes = {getattr(e, "code", None), err.get("code"), err.get("type")}
    return bool(codes & {"insufficient_quota", "credit_balance_exhausted"})


_SAVE_EVERY = 2048  # grava o cache a cada N textos novos, para permitir retomada


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada. Defina a chave no arquivo .env.")
    return OpenAI(api_key=config.OPENAI_API_KEY, timeout=config.LLM_TIMEOUT_S, max_retries=0)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    import openai

    inputs = [t[: config.EMBEDDING_MAX_CHARS] or " " for t in texts]
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = _client().embeddings.create(model=config.EMBEDDING_MODEL, input=inputs)
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        except openai.RateLimitError as e:
            if _is_quota_error(e):
                raise RuntimeError(
                    "A conta da OpenAI está sem créditos (insufficient_quota). "
                    "Adicione créditos em platform.openai.com/settings/organization/billing."
                ) from e
            last = e
            time.sleep(2 ** attempt)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError("A API de embeddings da OpenAI não respondeu após várias tentativas.") from last


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def embed_query(text: str) -> list[float]:
    return _normalize(np.asarray(_embed_batch([text]), dtype=np.float32))[0].tolist()


def embed_passages(texts: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        out.extend(_embed_batch(texts[start:start + config.EMBEDDING_BATCH_SIZE]))
    return _normalize(np.asarray(out, dtype=np.float32))


# --- cache em disco (um arquivo por modelo, para não misturar dimensões) -----
def _cache_paths():
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", config.EMBEDDING_MODEL)
    return (
        config.CACHE_DIR / f"embeddings_{slug}.npy",
        config.CACHE_DIR / f"embeddings_{slug}_ids.json",
    )


def clear_cache() -> None:
    for p in _cache_paths():
        p.unlink(missing_ok=True)


def embed_with_cache(ids: list[str], texts: list[str]) -> np.ndarray:
    """Gera embeddings reaproveitando o cache em disco (por id) e salva o resultado."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vec_path, ids_path = _cache_paths()
    cached: dict[str, np.ndarray] = {}
    if vec_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text())
        cached_vecs = np.load(vec_path)
        if len(cached_ids) == len(cached_vecs):
            cached = dict(zip(cached_ids, cached_vecs))
            log.info("Cache: %d embeddings encontrados", len(cached))

    missing = [i for i, id_ in enumerate(ids) if id_ not in cached]
    log.info("Gerando %d embeddings novos (de %d) com %s", len(missing), len(ids), config.EMBEDDING_MODEL)
    for start in range(0, len(missing), _SAVE_EVERY):
        idx = missing[start:start + _SAVE_EVERY]
        vecs = embed_passages([texts[i] for i in idx])
        for i, v in zip(idx, vecs):
            cached[ids[i]] = v
        _save_cache(cached)
        log.info("  %d/%d", min(start + _SAVE_EVERY, len(missing)), len(missing))

    return np.vstack([cached[i] for i in ids])


def _save_cache(cached: dict[str, np.ndarray]) -> None:
    vec_path, ids_path = _cache_paths()
    keys = list(cached.keys())
    np.save(vec_path, np.vstack([cached[k] for k in keys]))
    ids_path.write_text(json.dumps(keys))
