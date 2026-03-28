"""FastAPI 엔드포인트 통합 테스트."""
from unittest.mock import AsyncMock


async def test_chat_success(client):
    response = await client.post("/chat", json={"query": "서울의 인구는?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "테스트 답변입니다."
    assert "X-Request-ID" in response.headers


async def test_chat_empty_query(client):
    response = await client.post("/chat", json={"query": ""})
    assert response.status_code == 422


async def test_chat_blank_query(client):
    response = await client.post("/chat", json={"query": "   "})
    assert response.status_code == 422


async def test_chat_missing_query(client):
    response = await client.post("/chat", json={})
    assert response.status_code == 422


async def test_chat_stream_success(client):
    response = await client.post("/chat/stream", json={"query": "서울의 인구는?"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: delta" in response.text
    assert "event: done" in response.text
    assert "[DONE]" in response.text


async def test_health_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


async def test_health_ready_all_ok(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["db"] == "ok"
    assert data["embedding"] == "ok"
    assert data["llm"] == "ok"


async def test_health_ready_degraded_when_embedding_fails(client, mock_rag):
    mock_rag.embedder.ping = AsyncMock(side_effect=Exception("연결 실패"))
    response = await client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["embedding"] == "error"
    assert data["db"] == "ok"
    assert data["llm"] == "ok"


async def test_service_unavailable_without_rag(client):
    del client.app.state.rag
    response = await client.post("/chat", json={"query": "질문"})
    assert response.status_code == 503
