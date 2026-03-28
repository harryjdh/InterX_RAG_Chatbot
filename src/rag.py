import asyncio
import logging
import time
from typing import AsyncIterator, List, Optional, Tuple

from .circuit_breaker import CircuitBreakerOpen
from .config import config
from .embeddings import EmbeddingClient
from .llm import LLMClient
from .metrics import (
    rag_context_chars,
    rag_embedding_duration,
    rag_llm_duration,
    rag_llm_ttfb,
    rag_retrieval_duration,
    rag_retrieval_score,
)
from .vectordb import VectorDB

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "당신은 주어진 문맥을 바탕으로 사용자의 질문에 답변하는 한국어 QA 챗봇입니다.\n"
    "반드시 제공된 문맥 정보를 기반으로 답변하세요.\n"
    "문맥에 답변이 없는 경우, '주어진 정보에서 해당 내용을 찾을 수 없습니다.'라고 답변하세요.\n"
    "답변은 간결하고 명확하게 작성하세요."
)


class RetrievalError(Exception):
    """임베딩 또는 벡터 검색 실패 시 발생합니다."""


class LLMError(Exception):
    """LLM 호출 실패 시 발생합니다."""


def _build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(
        f"[문맥 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return (
        f"다음 문맥들을 참고하여 질문에 답변해주세요.\n\n"
        f"{context_block}\n\n"
        f"질문: {query}\n"
        f"답변:"
    )


class NaiveRAG:
    def __init__(self, vectordb: VectorDB, embedder: EmbeddingClient, llm: LLMClient, redis=None):
        self.vectordb = vectordb
        self.embedder = embedder
        self.llm = llm
        self._redis = redis

    @classmethod
    async def create(cls) -> "NaiveRAG":
        vectordb = await VectorDB.create()

        redis = None
        if config.REDIS_URL:
            from redis.asyncio import from_url as redis_from_url

            redis = await redis_from_url(config.REDIS_URL, decode_responses=True)
            logger.info("Redis 연결 완료: %s (분산 CB/Rate Limiter 활성화)", config.REDIS_URL)

        embedder = EmbeddingClient(redis=redis)
        llm = LLMClient(redis=redis)
        return cls(vectordb, embedder, llm, redis=redis)

    async def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, Optional[str], float]]:
        """Returns list of (content, title, similarity)."""
        top_k = top_k or config.TOP_K
        try:
            t0 = time.perf_counter()
            query_embedding = await self.embedder.embed(query)
            t1 = time.perf_counter()
            embed_secs = t1 - t0
            rag_embedding_duration.observe(embed_secs)  # 임베딩 완료 즉시 기록 (DB 실패와 무관)

            results = await self.vectordb.search(query_embedding, top_k=top_k)
            t2 = time.perf_counter()
            search_secs = t2 - t1
            rag_retrieval_duration.observe(search_secs)

            logger.info(
                "retrieve | embed=%.3fs search=%.3fs total=%.3fs top_k=%d",
                embed_secs, search_secs, t2 - t0, top_k,
            )
            for rank, (_, title, score) in enumerate(results, 1):
                logger.debug("retrieve | rank=%d score=%.4f title=%s", rank, score, title)
                rag_retrieval_score.labels(rank=str(rank)).observe(score)
            return results
        except CircuitBreakerOpen:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise RetrievalError(str(e)) from e

    async def answer(self, query: str) -> str:
        """Non-streaming: returns full answer string."""
        t0 = time.perf_counter()

        results = await self.retrieve(query)
        contexts = [content[:config.MAX_CONTEXT_CHARS] for content, _, _ in results]
        total_chars = sum(len(c) for c in contexts)
        rag_context_chars.observe(total_chars)
        prompt = _build_prompt(query, contexts)

        t1 = time.perf_counter()
        try:
            answer = await self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        except CircuitBreakerOpen:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise LLMError(str(e)) from e
        t2 = time.perf_counter()

        llm_secs = t2 - t1
        rag_llm_duration.observe(llm_secs)
        logger.info(
            "answer | retrieve=%.3fs llm=%.3fs total=%.3fs",
            t1 - t0, llm_secs, t2 - t0,
        )
        return answer

    async def answer_stream(self, query: str) -> AsyncIterator[str]:
        """Streaming: yields answer chunks as they arrive."""
        t0 = time.perf_counter()

        results = await self.retrieve(query)
        contexts = [content[:config.MAX_CONTEXT_CHARS] for content, _, _ in results]
        total_chars = sum(len(c) for c in contexts)
        rag_context_chars.observe(total_chars)
        prompt = _build_prompt(query, contexts)

        t1 = time.perf_counter()
        logger.info("answer_stream | retrieve=%.3fs", t1 - t0)

        try:
            first_chunk = True
            async for chunk in self.llm.generate_stream(prompt, system_prompt=SYSTEM_PROMPT):
                if first_chunk:
                    ttfb = time.perf_counter() - t0
                    rag_llm_ttfb.observe(ttfb)
                    logger.info(
                        "answer_stream | time_to_first_chunk=%.3fs", ttfb
                    )
                    first_chunk = False
                yield chunk
        except CircuitBreakerOpen:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise LLMError(str(e)) from e

    async def close(self):
        await asyncio.gather(
            self.vectordb.close(),
            self.embedder.close(),
            self.llm.close(),
        )
        if self._redis:
            await self._redis.aclose()
