from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    LLM_API_KEY: SecretStr = SecretStr("token-abc123")
    LLM_BASE_URL: str = "http://211.47.56.81:17972/v1"
    LLM_MODEL_NAME: str = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

    # Embedding
    EMBEDDING_API_KEY: SecretStr = SecretStr("token-abc123")
    EMBEDDING_BASE_URL: str = "http://211.47.56.71:15653/v1"
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-4B"
    EMBEDDING_DIM: int = 2560

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ragdb"
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: SecretStr = SecretStr("ragpassword")

    # RAG
    TOP_K: int = 5

    # DB 커넥션 풀 (workers 수에 맞게 조정: WORKERS × DB_POOL_MAX_SIZE ≤ DB 최대 연결 수)
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10

    # 요청 처리
    MAX_QUERY_LEN: int = 500
    REQUEST_TIMEOUT: float = 180.0

    # 로깅
    LOG_LEVEL: str = "INFO"

    # OpenAPI 문서 (프로덕션에서는 False 권장)
    DOCS_ENABLED: bool = False

    # Redis (분산 Rate Limiter / 서킷 브레이커 공유 상태)
    # 미설정 시 인메모리 폴백 사용 (단일 워커 개발 환경)
    REDIS_URL: Optional[str] = None

    # Rate Limiting: IP당 요청 상한 (SlowAPI limits 형식: "N/period")
    RATE_LIMIT: str = "20/minute"

    # LLM 비용 상한: 문맥당 최대 글자 수 (초과 시 잘라냄)
    MAX_CONTEXT_CHARS: int = 2000

    # API 클라이언트 타임아웃
    LLM_TIMEOUT: float = 120.0
    EMBEDDING_TIMEOUT: float = 30.0

    # 헬스체크 타임아웃
    HEALTH_DB_TIMEOUT: float = 5.0
    HEALTH_EMBEDDING_TIMEOUT: float = 30.0
    HEALTH_LLM_TIMEOUT: float = 120.0


config = Config()
