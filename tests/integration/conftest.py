"""
통합 테스트 픽스처.

실제 PostgreSQL 연결이 필요합니다.
환경 변수: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""
import pytest_asyncio

from src.vectordb import VectorDB


@pytest_asyncio.fixture
async def db():
    """테스트마다 독립적인 VectorDB 커넥션 풀 생성 및 정리."""
    vectordb = await VectorDB.create()
    yield vectordb
    await vectordb.close()
