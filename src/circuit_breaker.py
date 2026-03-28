"""
3-state 서킷 브레이커 (Closed → Open → Half-Open).

- CLOSED   : 정상 동작. 연속 실패가 failure_threshold에 도달하면 OPEN 전환.
- OPEN     : 모든 요청을 즉시 차단. recovery_timeout 경과 후 HALF_OPEN 전환.
- HALF_OPEN: 탐침 요청 허용. success_threshold 연속 성공 시 CLOSED, 실패 시 OPEN 재전환.

동작 모드
---------
- redis=None  : 단일 프로세스 인메모리 모드 (기본값, 로컬 개발 / 단일 워커 배포).
- redis=Redis : 분산 모드. 모든 uvicorn 워커가 Redis를 통해 상태를 공유합니다.
                REDIS_URL 환경 변수 설정 시 NaiveRAG.create() 에서 자동으로 활성화됩니다.

주의: 분산 모드에서는 time.time() (wall clock)을 사용합니다.
인메모리 모드에서도 일관성을 위해 time.time()을 사용합니다.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """서킷이 열려 있어 요청이 차단되었습니다."""


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        name: str = "circuit",
        redis=None,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._name = name
        self._redis = redis

        # 인메모리 폴백 상태 (redis=None 일 때 사용)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    def _key(self, field: str) -> str:
        return f"cb:{self._name}:{field}"

    async def call(self, func: Callable, *args, **kwargs):
        """서킷 브레이커를 통해 비동기 함수를 호출합니다."""
        await self._check_state()
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception as exc:
            await self._record_failure(exc)
            raise

    # --- 상태 읽기/쓰기 ---

    async def _get_state(self) -> tuple:
        """(state, failure_count, success_count, opened_at) 반환."""
        if self._redis is None:
            return self._state, self._failure_count, self._success_count, self._opened_at

        pipe = self._redis.pipeline()
        pipe.get(self._key("state"))
        pipe.get(self._key("failures"))
        pipe.get(self._key("successes"))
        pipe.get(self._key("opened_at"))
        results = await pipe.execute()

        raw_state = results[0]
        state = CircuitState(raw_state if raw_state else CircuitState.CLOSED.value)
        failures = int(results[1] or 0)
        successes = int(results[2] or 0)
        opened_at = float(results[3] or 0.0)
        return state, failures, successes, opened_at

    async def _set_state(
        self,
        state: CircuitState,
        failure_count: int,
        success_count: int,
        opened_at: float,
    ) -> None:
        if self._redis is None:
            self._state = state
            self._failure_count = failure_count
            self._success_count = success_count
            self._opened_at = opened_at
        else:
            pipe = self._redis.pipeline()
            pipe.set(self._key("state"), state.value)
            pipe.set(self._key("failures"), failure_count)
            pipe.set(self._key("successes"), success_count)
            pipe.set(self._key("opened_at"), opened_at)
            await pipe.execute()

        # Prometheus 게이지 업데이트 (임포트 실패 시 무시)
        try:
            from .metrics import _CB_STATE_MAP, rag_circuit_state

            rag_circuit_state.labels(name=self._name).set(
                _CB_STATE_MAP.get(state.value, 0)
            )
        except Exception:
            pass

    # --- 상태 전이 ---

    async def _check_state(self) -> None:
        async with self._lock:
            state, failures, successes, opened_at = await self._get_state()
            if state == CircuitState.OPEN:
                elapsed = time.time() - opened_at
                if elapsed >= self._recovery_timeout:
                    await self._set_state(CircuitState.HALF_OPEN, 0, 0, opened_at)
                    logger.info(
                        "서킷 브레이커 [%s] HALF_OPEN 전환 (%.1fs 경과)",
                        self._name,
                        elapsed,
                    )
                else:
                    remaining = self._recovery_timeout - elapsed
                    raise CircuitBreakerOpen(
                        f"서킷 열림 [{self._name}]: {remaining:.0f}s 후 재시도 가능"
                    )

    async def _record_success(self) -> None:
        async with self._lock:
            state, failures, successes, opened_at = await self._get_state()
            if state == CircuitState.HALF_OPEN:
                successes += 1
                if successes >= self._success_threshold:
                    await self._set_state(CircuitState.CLOSED, 0, 0, 0.0)
                    logger.info(
                        "서킷 브레이커 [%s] CLOSED 전환 (복구 완료)", self._name
                    )
                else:
                    await self._set_state(state, failures, successes, opened_at)
            elif state == CircuitState.CLOSED:
                # 성공 시 연속 실패 카운터 리셋
                await self._set_state(state, 0, successes, opened_at)

    async def _record_failure(self, exc: Exception) -> None:
        async with self._lock:
            state, failures, successes, opened_at = await self._get_state()
            # OPEN 상태는 _check_state에서 이미 차단 — 경쟁 상태 방어
            if state == CircuitState.OPEN:
                return

            failures += 1
            now = time.time()

            if state == CircuitState.HALF_OPEN:
                await self._set_state(CircuitState.OPEN, failures, 0, now)
                logger.warning(
                    "서킷 브레이커 [%s] OPEN 재전환 (HALF_OPEN 중 실패): %s",
                    self._name,
                    exc,
                )
            elif (
                state == CircuitState.CLOSED
                and failures >= self._failure_threshold
            ):
                await self._set_state(CircuitState.OPEN, failures, 0, now)
                logger.warning(
                    "서킷 브레이커 [%s] OPEN 전환 (연속 실패 %d회): %s",
                    self._name,
                    failures,
                    exc,
                )
            else:
                await self._set_state(state, failures, successes, opened_at)
