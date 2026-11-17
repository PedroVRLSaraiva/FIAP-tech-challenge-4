"""Cliente de LLM (OpenAI) atrás de uma interface mínima."""
from __future__ import annotations

import logging
import re
import time
from typing import Protocol

from src import config
from src.embeddings import _is_quota_error

log = logging.getLogger(__name__)


def _is_daily_limit(e: Exception) -> bool:
    msg = str(getattr(e, "message", "") or e)
    return "per day" in msg or "(RPD)" in msg


def _retry_hint(e: Exception) -> str | None:
    m = re.search(r"try again in ([0-9hms.]+)", str(getattr(e, "message", "") or e))
    return m.group(1) if m else None


class LLMError(RuntimeError):
    """Falha de LLM com mensagem segura para exibir na UI (sem segredos)."""


class LLMClient(Protocol):
    def generate(self, system: str, user: str, max_tokens: int | None = None) -> str: ...


class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI

        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise LLMError("OPENAI_API_KEY não configurada. Defina a chave no arquivo .env.")
        self.model = model or config.LLM_MODEL
        self._client = OpenAI(api_key=key, timeout=config.LLM_TIMEOUT_S, max_retries=0)
        # Parâmetros que o modelo recusar são removidos nas tentativas seguintes.
        self._unsupported: set[str] = set()

    def generate(self, system: str, user: str, max_tokens: int | None = None) -> str:
        import openai

        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        last_exc: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRIES):
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": config.LLM_TEMPERATURE,
                "max_completion_tokens": max_tokens,
            }
            for p in self._unsupported:
                params.pop(p, None)
            try:
                resp = self._client.chat.completions.create(**params)
                return (resp.choices[0].message.content or "").strip()
            except openai.BadRequestError as e:
                # ex.: modelo que não aceita `temperature` ou `max_completion_tokens`
                param = getattr(e, "param", None)
                if param in params and param not in ("model", "messages"):
                    log.warning("Parâmetro '%s' não suportado pelo modelo; removendo.", param)
                    self._unsupported.add(param)
                    last_exc = e
                    continue
                raise LLMError(f"Requisição rejeitada pela API: {e.message}") from e
            except openai.RateLimitError as e:
                if _is_quota_error(e):
                    raise LLMError(
                        "A conta da OpenAI está sem créditos (insufficient_quota). "
                        "Adicione créditos para continuar."
                    ) from e
                if _is_daily_limit(e):
                    raise LLMError(
                        "Limite diário de requisições do modelo atingido (RPD). "
                        + (f"A API sugere tentar de novo em {_retry_hint(e)}." if _retry_hint(e) else "Tente mais tarde.")
                    ) from e
                last_exc = e
                time.sleep(2 ** attempt)
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                last_exc = e
                time.sleep(2 ** attempt)
            except openai.AuthenticationError as e:
                raise LLMError("Chave da OpenAI inválida ou sem permissão.") from e
            except openai.NotFoundError as e:
                raise LLMError(f"Modelo '{self.model}' não encontrado para esta chave.") from e
            except openai.APIError as e:
                raise LLMError(f"Erro da API da OpenAI: {e.message}") from e
        raise LLMError("A API da OpenAI não respondeu após várias tentativas. Tente novamente.") from last_exc
