import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.circuit_breaker import CircuitBreakerOpen
from src.config import config
from src.rag import LLMError, NaiveRAG, RetrievalError

# --- Logging Setup ---

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """ELK / Datadog / CloudWatch 등 로그 집계 시스템을 위한 JSON 포매터."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
_handler.addFilter(_RequestIdFilter())
logging.root.setLevel(config.LOG_LEVEL.upper())
logging.root.addHandler(_handler)

logger = logging.getLogger(__name__)


class StreamErrorType(str, Enum):
    RETRIEVAL = "retrieval"
    LLM = "llm"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.rag = await NaiveRAG.create()
    except Exception:
        logger.critical("애플리케이션 시작 실패: RAG 초기화 중 오류 발생", exc_info=True)
        raise
    yield
    await app.state.rag.close()


limiter = Limiter(
    key_func=get_remote_address,
    headers_enabled=True,
    storage_uri=config.REDIS_URL or "memory://",
)

app = FastAPI(
    title="RAG Chatbot API",
    description="KorQuAD 기반 한국어 RAG 챗봇 API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.DOCS_ENABLED else None,
    redoc_url="/redoc" if config.DOCS_ENABLED else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)


# --- Middleware ---

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


# --- Global Exception Handlers ---

@app.exception_handler(CircuitBreakerOpen)
async def circuit_breaker_open_handler(request: Request, exc: CircuitBreakerOpen):
    logger.warning("서킷 열림 [%s %s]: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError):
    logger.error("검색 오류 [%s %s]: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(status_code=502, content={"detail": f"검색 오류: {exc}"})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLM 오류 [%s %s]: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(status_code=502, content={"detail": f"LLM 오류: {exc}"})


@app.exception_handler(asyncio.TimeoutError)
async def timeout_error_handler(request: Request, exc: asyncio.TimeoutError):
    logger.warning("요청 타임아웃 [%s %s]", request.method, request.url.path)
    return JSONResponse(status_code=504, content={"detail": "요청 시간이 초과되었습니다."})


# --- Dependencies ---

async def get_rag(request: Request) -> NaiveRAG:
    if not hasattr(request.app.state, "rag"):
        raise HTTPException(status_code=503, detail="서비스가 아직 준비되지 않았습니다.")
    return request.app.state.rag


# --- Models ---

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=config.MAX_QUERY_LEN)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query는 공백만으로 이루어질 수 없습니다.")
        return v.strip()


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    db: str
    embedding: str
    llm: str


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="챗봇 질의 (non-streaming)")
@limiter.limit(config.RATE_LIMIT)
async def chat(request: Request, body: ChatRequest, rag: NaiveRAG = Depends(get_rag)):
    """
    질문을 입력받아 RAG를 통해 생성된 답변을 반환합니다.
    전체 답변이 완성된 후 한 번에 반환됩니다.
    """
    async with asyncio.timeout(config.REQUEST_TIMEOUT):
        answer = await rag.answer(body.query)
    return ChatResponse(answer=answer)


@app.post("/chat/stream", summary="챗봇 질의 (streaming SSE)")
@limiter.limit(config.RATE_LIMIT)
async def chat_stream(request: Request, body: ChatRequest, rag: NaiveRAG = Depends(get_rag)):
    """
    질문을 입력받아 RAG를 통해 생성된 답변을 Server-Sent Events로 스트리밍합니다.

    각 이벤트 형식:
    - `event: delta` / `data: {"delta": "..."}` — 답변 청크
    - `event: error` / `data: {"error": "...", "type": "retrieval|llm|timeout|unknown"}` — 오류 발생
    - `event: done` / `data: [DONE]` — 스트림 종료
    """
    async def event_generator():
        try:
            async with asyncio.timeout(config.REQUEST_TIMEOUT):
                async for chunk in rag.answer_stream(body.query):
                    data = json.dumps({"delta": chunk}, ensure_ascii=False)
                    yield f"event: delta\ndata: {data}\n\n"
        except asyncio.TimeoutError:
            error = json.dumps({"error": "요청 시간이 초과되었습니다.", "type": StreamErrorType.TIMEOUT}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"
        except RetrievalError as e:
            error = json.dumps({"error": str(e), "type": StreamErrorType.RETRIEVAL}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"
        except LLMError as e:
            error = json.dumps({"error": str(e), "type": StreamErrorType.LLM}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"
        except CircuitBreakerOpen as e:
            error = json.dumps({"error": str(e), "type": StreamErrorType.CIRCUIT_OPEN}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"
        except Exception as e:
            logger.exception("스트리밍 중 예상치 못한 오류 발생")
            error = json.dumps({"error": str(e), "type": StreamErrorType.UNKNOWN}, ensure_ascii=False)
            yield f"event: error\ndata: {error}\n\n"
        finally:
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/health", response_model=HealthResponse, summary="헬스 체크")
async def health(rag: NaiveRAG = Depends(get_rag)):
    try:
        async with asyncio.timeout(config.HEALTH_DB_TIMEOUT):
            await rag.vectordb.ping()
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="데이터베이스 응답 없음")
    return HealthResponse(status="healthy")


@app.get("/health/ready", response_model=ReadyResponse, summary="외부 서비스 연결 확인")
async def health_ready(rag: NaiveRAG = Depends(get_rag)):
    """DB, 임베딩 API, LLM API 연결 상태를 모두 병렬로 확인합니다."""

    async def check_db():
        async with asyncio.timeout(config.HEALTH_DB_TIMEOUT):
            await rag.vectordb.ping()

    async def check_embedding():
        async with asyncio.timeout(config.HEALTH_EMBEDDING_TIMEOUT):
            await rag.embedder.ping()

    async def check_llm():
        async with asyncio.timeout(config.HEALTH_LLM_TIMEOUT):
            await rag.llm.ping()

    checks = await asyncio.gather(
        check_db(), check_embedding(), check_llm(), return_exceptions=True
    )

    names = ["db", "embedding", "llm"]
    results = {}
    for name, result in zip(names, checks):
        if isinstance(result, Exception):
            logger.warning("헬스체크 %s 연결 실패: %s", name, result, exc_info=result)
            results[name] = "error"
        else:
            results[name] = "ok"

    healthy = all(v == "ok" for v in results.values())
    status = "healthy" if healthy else "degraded"
    response = ReadyResponse(status=status, **results)
    if not healthy:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response
