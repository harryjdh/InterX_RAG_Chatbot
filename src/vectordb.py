import logging

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from typing import List, Tuple, Optional

from .config import config

logger = logging.getLogger(__name__)

_ACQUIRE_TIMEOUT = 5.0
_COMMAND_TIMEOUT = 10.0


class VectorDB:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def create(cls) -> "VectorDB":
        # register_vector는 pool init 시 PostgreSQL에서 vector 타입을 조회합니다.
        # extension이 없으면 "unknown type: public.vector"로 pool 생성 자체가 실패하므로,
        # pool 생성 전에 codec 없는 임시 커넥션으로 extension을 먼저 보장합니다.
        _conn_args = dict(
            host=config.POSTGRES_HOST,
            port=int(config.POSTGRES_PORT),
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD.get_secret_value(),
        )
        _tmp = await asyncpg.connect(**_conn_args)
        try:
            await _tmp.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await _tmp.close()

        async def _init(conn):
            await register_vector(conn)

        pool = await asyncpg.create_pool(
            **_conn_args,
            min_size=config.DB_POOL_MIN_SIZE,
            max_size=config.DB_POOL_MAX_SIZE,
            command_timeout=_COMMAND_TIMEOUT,
            init=_init,
        )
        return cls(pool)

    async def ensure_schema(self) -> None:
        """vector 확장과 documents 테이블이 없으면 생성합니다 (멱등).

        alembic 마이그레이션 미실행 환경(docker compose up 직후 등)에서도
        API가 즉시 동작할 수 있도록 보장합니다.
        alembic이 이미 실행된 환경에서는 IF NOT EXISTS로 인해 아무 변화가 없습니다.
        """
        async with self._pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id          SERIAL PRIMARY KEY,
                    content     TEXT NOT NULL,
                    title       TEXT,
                    embedding   vector({config.EMBEDDING_DIM}),
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        logger.info("스키마 확인 완료 (EMBEDDING_DIM=%d)", config.EMBEDDING_DIM)

    async def insert_batch(self, documents: List[dict]):
        """
        documents: List of {"content": str, "embedding": List[float], "title": str (optional)}
        """
        async with self._pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO documents (content, title, embedding) VALUES ($1, $2, $3)",
                    [
                        (doc["content"], doc.get("title"), np.array(doc["embedding"]))
                        for doc in documents
                    ],
                )

    async def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[str, Optional[str], float]]:
        """
        Returns list of (content, title, similarity) tuples, ordered by similarity descending.
        Uses cosine similarity: 1 - cosine_distance.
        """
        async with self._pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            rows = await conn.fetch(
                """
                SELECT content, title, 1 - (embedding <=> $1) AS similarity
                FROM documents
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                np.array(query_embedding),
                top_k,
            )
        return [(row["content"], row["title"], float(row["similarity"])) for row in rows]

    async def count(self) -> int:
        async with self._pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM documents")

    async def ping(self) -> bool:
        async with self._pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            await conn.fetchval("SELECT 1")
        return True

    async def close(self):
        if self._pool:
            await self._pool.close()
