"""
통합 테스트 픽스처.

실제 PostgreSQL 연결이 필요합니다.
환경 변수: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""
import pytest

from src.vectordb import VectorDB


@pytest.fixture(scope="module")
async def db():
    """모듈 범위 VectorDB 커넥션 풀 — 테스트 완료 후 자동 정리."""
    vectordb = await VectorDB.create()
    yield vectordb
    await vectordb.close()
