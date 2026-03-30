import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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
    for var, val in [
        ("LLM_BASE_URL", config.LLM_BASE_URL),
        ("EMBEDDING_BASE_URL", config.EMBEDDING_BASE_URL),
    ]:
        if not val:
            logger.critical(
                "필수 환경변수 %s가 설정되지 않았습니다. .env 파일을 확인하세요.", var
            )
            raise ValueError(f"{var}이(가) 설정되지 않았습니다. .env 파일을 확인하세요.")
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

# 워커 프로세스당 활성 SSE 스트림 수 (asyncio 단일 스레드 내 원자적 접근)
_active_streams: int = 0

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    # allow_origins=["*"] 와 allow_credentials=True 의 동시 허용을 차단합니다.
    # Starlette은 wildcard origin 요청에 요청 Origin을 그대로 에코하므로,
    # 두 옵션이 함께 설정되면 임의 도메인의 credentialed 요청이 허용됩니다.
    allow_credentials=config.CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _verify_metrics_token(request: Request):
    """METRICS_TOKEN 설정 시 Bearer 토큰으로 /metrics 엔드포인트를 보호합니다."""
    if config.METRICS_TOKEN is None:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {config.METRICS_TOKEN.get_secret_value()}":
        raise HTTPException(status_code=401, detail="Unauthorized")


Instrumentator().instrument(app).expose(
    app,
    dependencies=[Depends(_verify_metrics_token)],
)


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
    return JSONResponse(status_code=502, content={"detail": "문서 검색 중 오류가 발생했습니다."})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLM 오류 [%s %s]: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(status_code=502, content={"detail": "응답 생성 중 오류가 발생했습니다."})


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

v1_router = APIRouter(prefix="/v1")


@v1_router.post("/chat", response_model=ChatResponse, summary="챗봇 질의 (non-streaming)")
@limiter.limit(config.RATE_LIMIT)
async def chat(request: Request, response: Response, body: ChatRequest, rag: NaiveRAG = Depends(get_rag)):
    """
    질문을 입력받아 RAG를 통해 생성된 답변을 반환합니다.
    전체 답변이 완성된 후 한 번에 반환됩니다.
    """
    async with asyncio.timeout(config.REQUEST_TIMEOUT):
        answer = await rag.answer(body.query)
    return ChatResponse(answer=answer)


@v1_router.post("/chat/stream", summary="챗봇 질의 (streaming SSE)")
@limiter.limit(config.RATE_LIMIT)
async def chat_stream(request: Request, response: Response, body: ChatRequest, rag: NaiveRAG = Depends(get_rag)):
    """
    질문을 입력받아 RAG를 통해 생성된 답변을 Server-Sent Events로 스트리밍합니다.

    각 이벤트 형식:
    - `event: delta` / `data: {"delta": "..."}` — 답변 청크
    - `event: error` / `data: {"error": "...", "type": "retrieval|llm|timeout|unknown"}` — 오류 발생
    - `event: done` / `data: [DONE]` — 스트림 종료
    """
    # 전역 동시 SSE 연결 수 상한 초과 시 조기 거절.
    # 제너레이터 진입 전에 확인하여 클라이언트에 즉시 503을 반환합니다.
    if _active_streams >= config.MAX_CONCURRENT_STREAMS:
        raise HTTPException(
            status_code=503,
            detail=f"동시 스트리밍 연결이 최대치({config.MAX_CONCURRENT_STREAMS})에 도달했습니다. 잠시 후 재시도하세요.",
        )

    async def event_generator():
        # 미들웨어가 request_id_var를 리셋한 후에도 스트리밍 구간 로그에 request_id가
        # 올바르게 기록되도록 제너레이터 진입 시점에 컨텍스트를 재설정합니다.
        _token = request_id_var.set(request.state.request_id)

        # retrieve() + LLM TTFB 구간(첫 번째 청크 도착 전)에서도 연결 해제를 감지하기 위해
        # 별도 태스크에서 0.5초 주기로 연결 상태를 폴링합니다.
        # _done_event: 스트리밍 종료 신호 — 정상 완료 후 watcher가 태스크를 취소하는
        # race condition을 방지하기 위해 done 이벤트 yield 직전에 설정합니다.
        _done_event = asyncio.Event()
        _current_task = asyncio.current_task()

        async def _disconnect_watcher():
            while not _done_event.is_set():
                if await request.is_disconnected():
                    # _done_event가 await 중 설정되었을 수 있으므로 이중 확인
                    if not _done_event.is_set():
                        logger.info(
                            "연결 해제 조기 감지 (retrieve/TTFB 구간) [%s]",
                            request.state.request_id,
                        )
                        _current_task.cancel()
                    return
                await asyncio.sleep(0.5)

        _watcher = asyncio.create_task(_disconnect_watcher())
        # try 바깥에서 generator를 생성해야 outer finally에서 aclose()로 즉시 정리 가능합니다.
        stream = rag.answer_stream(body.query)
        global _active_streams
        _active_streams += 1
        try:
            try:
                async with asyncio.timeout(config.REQUEST_TIMEOUT):
                    async for chunk in stream:
                        if await request.is_disconnected():
                            logger.info("클라이언트 연결 해제 감지 — 스트리밍 중단 [%s]", request.state.request_id)
                            return
                        data = json.dumps({"delta": chunk}, ensure_ascii=False)
                        yield f"event: delta\ndata: {data}\n\n"
            except asyncio.CancelledError:
                # uvicorn 또는 watcher가 연결을 강제 취소한 경우 — CancelledError를 소비하여
                # finally 블록의 yield가 재진입 CancelledError를 일으키지 않도록 합니다.
                logger.info("스트리밍 태스크 취소됨 (CancelledError) [%s]", request.state.request_id)
                return
            except asyncio.TimeoutError:
                error = json.dumps({"error": "요청 시간이 초과되었습니다.", "type": StreamErrorType.TIMEOUT}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"
            except RetrievalError as e:
                logger.error("스트리밍 중 검색 오류: %s", e, exc_info=e)
                error = json.dumps({"error": "문서 검색 중 오류가 발생했습니다.", "type": StreamErrorType.RETRIEVAL}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"
            except LLMError as e:
                logger.error("스트리밍 중 LLM 오류: %s", e, exc_info=e)
                error = json.dumps({"error": "응답 생성 중 오류가 발생했습니다.", "type": StreamErrorType.LLM}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"
            except CircuitBreakerOpen as e:
                error = json.dumps({"error": str(e), "type": StreamErrorType.CIRCUIT_OPEN}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"
            except Exception:
                logger.exception("스트리밍 중 예상치 못한 오류 발생")
                error = json.dumps({"error": "예상치 못한 오류가 발생했습니다.", "type": StreamErrorType.UNKNOWN}, ensure_ascii=False)
                yield f"event: error\ndata: {error}\n\n"
            finally:
                # _done_event를 yield 전에 설정하여 race condition 방지:
                # 정상 완료 직후 HTTP 연결이 닫혀도 watcher가 태스크를 취소하지 않습니다.
                _done_event.set()
                yield "event: done\ndata: [DONE]\n\n"
        finally:
            # watcher 정리: cancel 전 _done_event가 이미 설정되어 있으므로
            # watcher는 다음 이벤트 루프 틱에서 안전하게 종료됩니다.
            _watcher.cancel()
            try:
                await _watcher
            except asyncio.CancelledError:
                pass
            # disconnect / cancel / timeout / 정상 종료 어떤 경우에도
            # LLM API 스트리밍 연결을 즉시 닫아 upstream 리소스를 회수합니다.
            await stream.aclose()
            request_id_var.reset(_token)
            _active_streams -= 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


app.include_router(v1_router)


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
