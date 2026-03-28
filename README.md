# RAG 기반 한국어 QA 챗봇

KorQuAD v1.0 데이터셋과 pgvector(PostgreSQL)를 사용한 NaiveRAG 챗봇 시스템입니다.

---

## 프로젝트 구조

```
interX_rag_chatbot/
├── main.py                         # FastAPI 애플리케이션
├── cli.py                          # CLI 인터페이스
├── Dockerfile                      # API 서버 이미지 (프로덕션)
├── Dockerfile.build                # VectorDB 구축 이미지
├── Dockerfile.build.dockerignore   # 빌드 이미지 전용 ignore
├── .dockerignore                   # API 이미지 전용 ignore
├── docker-compose.yml              # PostgreSQL + API 서버 구성
├── alembic.ini                     # Alembic 마이그레이션 설정
├── pytest.ini                      # pytest 설정
├── requirements.txt                # 프로덕션 의존성 (pinned)
├── requirements.in                 # 프로덕션 직접 의존성 (pip-compile 입력)
├── requirements-build.txt          # 빌드 전용 의존성 (pinned)
├── requirements-build.in           # 빌드 직접 의존성 (pip-compile 입력)
├── requirements-dev.txt            # 개발/테스트 의존성
├── Makefile                        # 의존성 관리, Docker, 마이그레이션, 테스트 명령 모음
├── .env                            # API 키 및 환경변수 (gitignore 대상)
├── .env.example                    # 환경변수 템플릿
├── alembic/
│   ├── env.py                      # Alembic 실행 환경 (asyncio + SQLAlchemy)
│   ├── script.py.mako              # 마이그레이션 파일 템플릿
│   └── versions/
│       └── 001_initial_schema.py   # 초기 스키마 (vector 익스텐션 + documents 테이블)
├── scripts/
│   ├── __init__.py
│   └── build_vectordb.py           # 데이터 로딩 및 임베딩 스크립트
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixtures (mock_rag, AsyncClient)
│   ├── test_api.py                 # FastAPI 엔드포인트 통합 테스트
│   └── test_retry.py               # retry 모듈 단위 테스트
└── src/
    ├── config.py                   # 환경변수 설정 (Pydantic Settings)
    ├── vectordb.py                 # pgvector 연동 (asyncpg)
    ├── embeddings.py               # Embedding API 클라이언트
    ├── llm.py                      # LLM API 클라이언트 (스트리밍 지원)
    ├── rag.py                      # NaiveRAG 파이프라인
    └── retry.py                    # 지수 백오프 + jitter 재시도
```

---

## 실행 방법 (Docker 권장)

### 1. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에서 API 키 및 접속 정보 수정
```

### 2. VectorDB 구축 (최초 1회)

Alembic 마이그레이션 실행 후 임베딩을 생성하여 PostgreSQL에 저장합니다.

```bash
docker compose --profile build up build-vectordb

# 또는 Makefile 사용
make build-vectordb
```

### 3. API 서버 실행

```bash
docker compose up -d

# 또는 Makefile 사용
make up
```

PostgreSQL + API 서버가 함께 시작됩니다. 서버는 `http://localhost:8000`에서 접근 가능합니다.

### 4. CLI 인터페이스 실행

```bash
# Docker 환경 (권장)
docker exec -it rag_api python cli.py

# 로컬 환경
python cli.py
```

> `quit`, `exit`, `q` 또는 `Ctrl+C`로 종료

---

## 로컬 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. PostgreSQL 시작

```bash
docker compose up postgres -d
```

### 3. DB 마이그레이션

```bash
make migrate
# 또는
alembic upgrade head
```

### 4. VectorDB 구축

```bash
# 전체 데이터 구축
python -m scripts.build_vectordb --limit 0

# 일부만 구축 (빠른 테스트용)
python -m scripts.build_vectordb --limit 1000

# 기존 데이터 있을 때 확인 없이 추가
python -m scripts.build_vectordb --limit 0 --skip-confirm
```

### 5. API 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## DB 마이그레이션 (Alembic)

스키마 변경 이력을 Alembic으로 관리합니다. `build_vectordb.py` 실행 시 자동으로 `alembic upgrade head`가 선행 실행됩니다.

```bash
# 마이그레이션 적용
make migrate

# 마이그레이션 이력 조회
make migrate-history

# 새 마이그레이션 파일 생성
alembic revision -m "add_new_column"

# 특정 리비전으로 롤백
alembic downgrade -1
```

### 마이그레이션 구조

| 리비전 | 설명 |
|--------|------|
| `001` | `vector` 익스텐션 활성화 + `documents` 테이블 생성 |

> `alembic/env.py`는 `src.config`에서 DB 접속 정보를 읽어 SQLAlchemy asyncio 엔진으로 마이그레이션을 실행합니다. 별도 URL 하드코딩 없이 `.env` 설정만으로 동작합니다.

---

## 테스트

### 개발 의존성 설치

```bash
pip install -r requirements-dev.txt
```

### 테스트 실행

```bash
# 전체 테스트
make test

# 상세 출력
make test-v

# 또는 직접 실행
pytest
pytest -v
```

### 테스트 구성

| 파일 | 대상 | 테스트 수 |
|------|------|----------|
| `tests/test_api.py` | FastAPI 엔드포인트 통합 테스트 | 9개 |
| `tests/test_retry.py` | `src.retry` 지수 백오프 단위 테스트 | 6개 |

**`test_api.py` 주요 케이스:**
- `/chat` 정상 응답, 빈 쿼리 422, 공백 쿼리 422, 필드 누락 422
- `/chat/stream` SSE 스트리밍 응답 (event: delta / done 포함 확인)
- `/health` DB 연결 확인
- `/health/ready` 전체 서비스 정상 / Embedding 장애 시 503 degraded
- RAG 미초기화 상태에서 503 반환

**`test_retry.py` 주요 케이스:**
- 첫 시도 성공, 1회 실패 후 재시도 성공
- 재시도 소진 시 원본 예외 raise (traceback 보존)
- `non_retryable` 예외 즉시 raise
- `Retry-After` 헤더 대기 시간 준수
- `max_delay` 상한 적용 검증

---

## 의존성 관리 (pip-tools)

```bash
# pip-tools 설치
pip install pip-tools

# lock file 재생성 (의존성 추가/변경 시)
make lock        # requirements.txt 재생성
make lock-build  # requirements-build.txt 재생성
make lock-dev    # requirements-dev.txt 재생성

# 환경 동기화
make sync        # 프로덕션
make sync-dev    # 개발
```

`.in` 파일에 직접 의존성을 명시하고 `pip-compile`로 전체 의존성 트리를 고정합니다.

| 파일 | 용도 |
|------|------|
| `requirements.in` / `.txt` | API 서버 프로덕션 실행 |
| `requirements-build.in` / `.txt` | VectorDB 구축 (datasets, alembic 포함) |
| `requirements-dev.txt` | 로컬 개발 및 테스트 (pytest, httpx 포함) |

---

## API 호출 흐름

```
클라이언트 요청
    │
    ▼
[Middleware] X-Request-ID 생성 / 주입 → 모든 로그에 request_id 포함
    │
    ▼
[Pydantic] ChatRequest 검증 (min_length=1, max_length, 공백 쿼리 차단)
    │
    ▼
[1] 질문 임베딩
    EmbeddingClient.embed(query)
    → Embedding API 호출 (타임아웃: EMBEDDING_TIMEOUT, 재시도: 최대 3회)
    │
    ▼
[2] 유사 문맥 검색
    VectorDB.search(query_embedding, top_k=TOP_K)
    → pgvector 코사인 유사도 기준 상위 K개 문맥 반환
    │
    ▼
[3] 프롬프트 구성
    _build_prompt(query, contexts)
    → 검색된 문맥 + 질문을 LLM 입력 형식으로 조합
    │
    ▼
[4] LLM 답변 생성
    LLMClient.generate() / generate_stream()
    → 답변 생성 (non-streaming / SSE streaming)
    → <think> 블록 자동 제거 (Qwen3 thinking mode 대응)
    │
    ▼
[Global Exception Handler] RetrievalError → 502 / LLMError → 502 / TimeoutError → 504
    │
    ▼
[Middleware] 응답 헤더에 X-Request-ID 추가
    │
    ▼
클라이언트 응답
```

---

## API 엔드포인트

### 헬스 체크

```bash
# DB 연결 확인 (경량)
curl http://localhost:8000/health

# 전체 서비스 연결 확인 (DB + Embedding API + LLM API 병렬 체크)
curl http://localhost:8000/health/ready
```

응답 예시 (`/health/ready`):
```json
{"status": "healthy", "db": "ok", "embedding": "ok", "llm": "ok"}
```

서비스 장애 시 HTTP 503 반환:
```json
{"status": "degraded", "db": "ok", "embedding": "error", "llm": "ok"}
```

### 챗봇 질의 (non-streaming)

```bash
# Linux / Git Bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "대한민국의 수도는 어디인가요?"}'

# PowerShell
python -c "import requests; r = requests.post('http://localhost:8000/chat', json={'query': '대한민국의 수도는 어디인가요?'}); print(r.json()['answer'])"
```

응답:
```json
{"answer": "대한민국의 수도는 서울입니다."}
```

### 챗봇 질의 (streaming SSE)

```bash
# Linux / Git Bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "대한민국의 수도는 어디인가요?"}' \
  --no-buffer
```

SSE 이벤트 형식:
```
event: delta
data: {"delta": "대한민국의"}

event: delta
data: {"delta": " 수도는 서울입니다."}

event: done
data: [DONE]
```

오류 발생 시:
```
event: error
data: {"error": "...", "type": "retrieval|llm|timeout|unknown"}

event: done
data: [DONE]
```

클라이언트 측 이벤트 처리 (JavaScript):
```javascript
const source = new EventSource('/chat/stream');
source.addEventListener('delta', e => process(JSON.parse(e.data).delta));
source.addEventListener('error', e => handleError(JSON.parse(e.data)));
source.addEventListener('done',  () => source.close());
```

### Prometheus 메트릭

```bash
curl http://localhost:8000/metrics
```

엔드포인트별 요청 수, 레이턴시 히스토그램, 진행 중 요청 수를 Prometheus 형식으로 노출합니다.

### API 문서 (Swagger UI)

기본 비활성화. 개발 환경에서 활성화하려면 `.env`에 `DOCS_ENABLED=True` 설정:

```
http://localhost:8000/docs
```

---

## 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `LLM_API_KEY` | LLM API 키 | token-abc123 |
| `LLM_BASE_URL` | LLM API 엔드포인트 | - |
| `LLM_MODEL_NAME` | LLM 모델명 | Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 |
| `LLM_TIMEOUT` | LLM API 타임아웃 (초) | 120.0 |
| `EMBEDDING_API_KEY` | Embedding API 키 | token-abc123 |
| `EMBEDDING_BASE_URL` | Embedding API 엔드포인트 | - |
| `EMBEDDING_MODEL_NAME` | Embedding 모델명 | Qwen/Qwen3-Embedding-4B |
| `EMBEDDING_DIM` | 임베딩 벡터 차원 | 2560 |
| `EMBEDDING_TIMEOUT` | Embedding API 타임아웃 (초) | 30.0 |
| `POSTGRES_HOST` | PostgreSQL 호스트 | localhost |
| `POSTGRES_PORT` | PostgreSQL 포트 | 5432 |
| `POSTGRES_DB` | 데이터베이스명 | ragdb |
| `POSTGRES_USER` | 사용자명 | raguser |
| `POSTGRES_PASSWORD` | 비밀번호 | ragpassword |
| `TOP_K` | 검색할 문맥 수 | 5 |
| `MAX_QUERY_LEN` | 최대 쿼리 길이 | 500 |
| `REQUEST_TIMEOUT` | 전체 요청 타임아웃 (초) | 180.0 |
| `DB_POOL_MIN_SIZE` | DB 커넥션 풀 최소 크기 | 2 |
| `DB_POOL_MAX_SIZE` | DB 커넥션 풀 최대 크기 | 10 |
| `HEALTH_DB_TIMEOUT` | DB 헬스체크 타임아웃 (초) | 5.0 |
| `HEALTH_EMBEDDING_TIMEOUT` | Embedding 헬스체크 타임아웃 (초) | 30.0 |
| `HEALTH_LLM_TIMEOUT` | LLM 헬스체크 타임아웃 (초) | 120.0 |
| `LOG_LEVEL` | 로그 레벨 (DEBUG/INFO/WARNING/ERROR) | INFO |
| `DOCS_ENABLED` | Swagger UI 활성화 여부 | False |
| `WORKERS` | uvicorn 워커 수 (Docker) | 4 |

---

## 아키텍처 개선 사항

### API 호출 흐름

| 항목 | 내용 |
|------|------|
| 완전 비동기 처리 | asyncpg, AsyncOpenAI, asyncio.gather 사용 |
| 요청 추적 | ContextVar 기반 X-Request-ID 미들웨어, 전 로그에 request_id 포함 |
| 전역 예외 처리 | RetrievalError→502, LLMError→502, TimeoutError→504, 로깅 포함 |
| 타임아웃 일관성 | 전 엔드포인트 asyncio.timeout() 통일 |
| 입력 검증 | Pydantic Field + field_validator로 공백/길이 검증 |
| 헬스체크 병렬화 | asyncio.gather로 DB·Embedding·LLM 동시 확인 |
| 헬스체크 경량화 | /models 엔드포인트로 ping (토큰 생성 없음) |
| 서비스 저하 감지 | /health/ready 장애 시 HTTP 503 반환 |
| SSE 표준 준수 | event: delta/error/done 타입 필드 |
| 재시도 전략 | 지수 백오프 + jitter + max_delay 상한 + Retry-After 헤더 준수 |
| startup 안전성 | lifespan 실패 시 critical 로깅, get_rag() 미준비 시 503 |
| 워커 안전성 | 스키마 초기화를 build_vectordb.py로 분리 (멀티워커 DDL race 제거) |
| 관측성 | Prometheus /metrics 엔드포인트 노출 |

### 패키징

| 항목 | 내용 |
|------|------|
| 멀티스테이지 빌드 | 컴파일 도구 분리, 런타임 이미지 최소화 |
| 보안 | non-root 유저, COPY --chown, SecretStr 크리덴셜 마스킹 |
| 시그널 처리 | exec uvicorn으로 PID 1 교체, SIGTERM 정상 전달 |
| Graceful Shutdown | stop_grace_period 35s + --timeout-graceful-shutdown 30 |
| 재시작 정책 | restart: unless-stopped |
| 의존성 분리 | 프로덕션/빌드/개발 requirements 분리, .in 파일 + pip-compile 워크플로우 |
| 네트워크 격리 | rag-network 전용 bridge, postgres 외부 포트 비노출 |
| 리소스 제한 | CPU/메모리 limits + reservations |
| 환경변수 | Pydantic Settings, .env.example 문서화 |
| Python 재현성 | 패치 버전 고정 (python:3.11.12-slim) |
| Docker 최적화 | PYTHONUNBUFFERED + PYTHONDONTWRITEBYTECODE, 레이어 최소화 |
| Makefile | lock/sync/migrate/test/up/build-vectordb 등 표준 명령 제공 |

### 마이그레이션 및 테스트

| 항목 | 내용 |
|------|------|
| 스키마 버전 관리 | Alembic + SQLAlchemy asyncio 엔진으로 마이그레이션 실행 |
| 마이그레이션 자동화 | build_vectordb.py 실행 시 alembic upgrade head 선행 실행 |
| 롤백 지원 | downgrade() 구현으로 이전 스키마 버전으로 복구 가능 |
| 엔드포인트 통합 테스트 | pytest + httpx AsyncClient + NaiveRAG mock으로 HTTP 레이어 검증 |
| 단위 테스트 | retry 로직(백오프·non_retryable·Retry-After·max_delay) 독립 검증 |
| 테스트 격리 | 실제 DB/API 호출 없이 mock으로 완전 격리 |
| asyncio 테스트 | pytest-asyncio (asyncio_mode=auto)로 async 테스트 자동 인식 |
