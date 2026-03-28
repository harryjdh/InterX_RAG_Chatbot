import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from main import app


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.answer = AsyncMock(return_value="테스트 답변입니다.")
    rag.vectordb.ping = AsyncMock()
    rag.embedder.ping = AsyncMock()
    rag.llm.ping = AsyncMock()

    async def _stream(query: str):
        yield "테스트 "
        yield "답변입니다."

    rag.answer_stream = _stream
    return rag


@pytest_asyncio.fixture
async def client(mock_rag):
    app.state.rag = mock_rag
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    if hasattr(app.state, "rag"):
        del app.state.rag
