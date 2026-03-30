"""FastAPI 엔드포인트 통합 테스트."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
from main import app
from src.circuit_breaker import CircuitBreakerOpen
from src.rag import LLMError, RetrievalError


async def test_chat_success(client):
    response = await client.post("/v1/chat", json={"query": "서울의 인구는?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "테스트 답변입니다."
    assert "X-Request-ID" in response.headers


async def test_chat_empty_query(client):
    response = await client.post("/v1/chat", json={"query": ""})
    assert response.status_code == 422


async def test_chat_blank_query(client):
    response = await client.post("/v1/chat", json={"query": "   "})
    assert response.status_code == 422


async def test_chat_missing_query(client):
    response = await client.post("/v1/chat", json={})
    assert response.status_code == 422


async def test_chat_stream_success(client):
    response = await client.post("/v1/chat/stream", json={"query": "서울의 인구는?"})
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
    del app.state.rag
    response = await client.post("/v1/chat", json={"query": "질문"})
    assert response.status_code == 503


# --- 에러 HTTP 상태코드 ---

async def test_chat_retrieval_error_returns_502(client, mock_rag):
    mock_rag.answer = AsyncMock(side_effect=RetrievalError("임베딩 API 오류"))
    response = await client.post("/v1/chat", json={"query": "질문"})
    assert response.status_code == 502
    assert "문서 검색" in response.json()["detail"]


async def test_chat_llm_error_returns_502(client, mock_rag):
    mock_rag.answer = AsyncMock(side_effect=LLMError("LLM 호출 실패"))
    response = await client.post("/v1/chat", json={"query": "질문"})
    assert response.status_code == 502
    assert "응답 생성" in response.json()["detail"]


async def test_chat_circuit_breaker_open_returns_503(client, mock_rag):
    mock_rag.answer = AsyncMock(side_effect=CircuitBreakerOpen("서킷 열림"))
    response = await client.post("/v1/chat", json={"query": "질문"})
    assert response.status_code == 503


async def test_chat_timeout_returns_504(client, mock_rag):
    mock_rag.answer = AsyncMock(side_effect=asyncio.TimeoutError())
    response = await client.post("/v1/chat", json={"query": "질문"})
    assert response.status_code == 504


# --- SSE 스트리밍 에러 이벤트 ---

async def test_stream_retrieval_error_yields_error_event(client, mock_rag):
    async def _fail(query: str):
        raise RetrievalError("벡터 검색 실패")
        yield  # async generator로 인식시킴

    mock_rag.answer_stream = _fail
    response = await client.post("/v1/chat/stream", json={"query": "질문"})
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "retrieval" in response.text
    assert "event: done" in response.text  # finally 블록이 항상 done 전송


async def test_stream_llm_error_yields_error_event(client, mock_rag):
    async def _fail(query: str):
        raise LLMError("LLM 스트리밍 실패")
        yield

    mock_rag.answer_stream = _fail
    response = await client.post("/v1/chat/stream", json={"query": "질문"})
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "llm" in response.text
    assert "event: done" in response.text


# --- lifespan 환경변수 검증 ---

async def test_lifespan_raises_on_missing_llm_url(monkeypatch):
    monkeypatch.setattr(main.config, "LLM_BASE_URL", "")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        async with main.lifespan(MagicMock()):
            pass


async def test_lifespan_raises_on_missing_embedding_url(monkeypatch):
    monkeypatch.setattr(main.config, "LLM_BASE_URL", "http://llm-endpoint/v1")
    monkeypatch.setattr(main.config, "EMBEDDING_BASE_URL", "")
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        async with main.lifespan(MagicMock()):
            pass
