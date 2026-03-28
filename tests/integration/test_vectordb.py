"""
VectorDB 통합 테스트.

실제 PostgreSQL + pgvector 인스턴스에 연결하여 end-to-end 경로를 검증합니다.
Alembic 마이그레이션이 먼저 적용되어 있어야 합니다 (alembic upgrade head).
"""
import math

import pytest
import pytest_asyncio

from src.config import config

pytestmark = pytest.mark.integration

_DIM = config.EMBEDDING_DIM


def _fake_embedding(seed: int = 0) -> list[float]:
    """재현 가능한 단위 벡터 생성 (실제 모델 호출 없음)."""
    vec = [float((i + seed + 1) % 100) for i in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest_asyncio.fixture(autouse=True)
async def clean_table(db):
    """각 테스트 실행 전 documents 테이블 초기화."""
    async with db._pool.acquire() as conn:
        await conn.execute("DELETE FROM documents")
    yield


async def test_ping(db):
    assert await db.ping() is True


async def test_initial_count_is_zero(db):
    assert await db.count() == 0


async def test_insert_and_count(db):
    docs = [
        {"content": f"문서 {i}", "title": f"제목 {i}", "embedding": _fake_embedding(i)}
        for i in range(3)
    ]
    await db.insert_batch(docs)
    assert await db.count() == 3


async def test_search_returns_top_k(db):
    docs = [
        {"content": f"문서 {i}", "title": f"제목 {i}", "embedding": _fake_embedding(i)}
        for i in range(5)
    ]
    await db.insert_batch(docs)

    results = await db.search(_fake_embedding(0), top_k=3)

    assert len(results) == 3
    for content, title, score in results:
        assert isinstance(content, str)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0 + 1e-6


async def test_search_similarity_descending(db):
    """검색 결과가 유사도 내림차순으로 정렬되어야 합니다."""
    docs = [
        {"content": f"문서 {i}", "title": None, "embedding": _fake_embedding(i)}
        for i in range(5)
    ]
    await db.insert_batch(docs)

    results = await db.search(_fake_embedding(2), top_k=5)
    scores = [score for _, _, score in results]
    assert scores == sorted(scores, reverse=True)


async def test_insert_without_title(db):
    docs = [{"content": "제목 없는 문서", "embedding": _fake_embedding(99)}]
    await db.insert_batch(docs)

    results = await db.search(_fake_embedding(99), top_k=1)
    assert len(results) == 1
    content, title, _ = results[0]
    assert content == "제목 없는 문서"
    assert title is None
