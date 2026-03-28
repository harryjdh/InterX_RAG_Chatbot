import logging

import openai
from openai import AsyncOpenAI
from typing import List

from .circuit_breaker import CircuitBreaker
from .config import config
from .retry import retry

logger = logging.getLogger(__name__)

_NON_RETRYABLE = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)


class EmbeddingClient:
    def __init__(self, redis=None):
        self._client = AsyncOpenAI(
            api_key=config.EMBEDDING_API_KEY.get_secret_value(),
            base_url=config.EMBEDDING_BASE_URL,
            timeout=config.EMBEDDING_TIMEOUT,
        )
        self._cb = CircuitBreaker(name="embedding", redis=redis)

    async def embed(self, text: str) -> List[float]:
        return await self._cb.call(self._embed_raw, text)

    async def _embed_raw(self, text: str) -> List[float]:
        response = await retry(
            self._client.embeddings.create,
            model=config.EMBEDDING_MODEL_NAME,
            input=text,
            non_retryable=_NON_RETRYABLE,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self._cb.call(self._embed_batch_raw, texts)

    async def _embed_batch_raw(self, texts: List[str]) -> List[List[float]]:
        response = await retry(
            self._client.embeddings.create,
            model=config.EMBEDDING_MODEL_NAME,
            input=texts,
            non_retryable=_NON_RETRYABLE,
        )
        # API may return items out of order; sort by index
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def ping(self) -> bool:
        """Embedding API 연결 확인 (임베딩 생성 없이 /models 엔드포인트 호출)."""
        return await self._cb.call(self._ping_raw)

    async def _ping_raw(self) -> bool:
        await self._client.models.list()
        return True

    async def close(self):
        await self._client.close()
