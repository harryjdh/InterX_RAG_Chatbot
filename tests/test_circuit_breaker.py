"""Circuit Breaker 단위 테스트 (인메모리 모드)."""
import pytest
from unittest.mock import AsyncMock

from src.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState


# --- CLOSED 상태 ---

async def test_closed_allows_calls():
    cb = CircuitBreaker(failure_threshold=3)
    ok = AsyncMock(return_value="result")
    assert await cb.call(ok) == "result"
    ok.assert_called_once()


async def test_success_resets_failure_count():
    """연속 실패 후 성공하면 failure_count가 0으로 초기화된다."""
    cb = CircuitBreaker(failure_threshold=5)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb._failure_count == 3

    await cb.call(AsyncMock(return_value="ok"))
    assert cb._failure_count == 0


# --- CLOSED → OPEN 전환 ---

async def test_trips_open_after_threshold_failures():
    """연속 실패가 failure_threshold에 도달하면 OPEN으로 전환된다."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=999)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb._state == CircuitState.OPEN


async def test_open_blocks_all_calls_and_does_not_invoke_function():
    """OPEN 상태에서 모든 요청이 즉시 차단되고 실제 함수는 호출되지 않는다."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=999)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)

    ok = AsyncMock(return_value="ok")
    with pytest.raises(CircuitBreakerOpen):
        await cb.call(ok)
    ok.assert_not_called()


# --- OPEN → HALF_OPEN 전환 ---

async def test_transitions_to_half_open_after_recovery_timeout():
    """recovery_timeout 경과 후 OPEN → HALF_OPEN 전환, 탐침 요청이 통과된다."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb._state == CircuitState.OPEN

    # recovery_timeout=0.0 → 다음 호출 즉시 HALF_OPEN 전환
    result = await cb.call(AsyncMock(return_value="probe_ok"))
    assert result == "probe_ok"


# --- HALF_OPEN → OPEN 재전환 ---

async def test_half_open_failure_reopens_circuit():
    """HALF_OPEN 중 탐침 호출 실패 시 다시 OPEN 상태로 돌아간다."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)

    with pytest.raises(ValueError):
        await cb.call(AsyncMock(side_effect=ValueError("probe fail")))

    assert cb._state == CircuitState.OPEN


# --- HALF_OPEN → CLOSED 복구 ---

async def test_half_open_success_threshold_closes_circuit():
    """HALF_OPEN에서 success_threshold 연속 성공 시 CLOSED로 완전 복구된다."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0, success_threshold=2)
    fail = AsyncMock(side_effect=ValueError("fail"))
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)

    ok = AsyncMock(return_value="ok")
    await cb.call(ok)  # 1번째 성공 — HALF_OPEN 유지 (successes=1 < threshold=2)
    assert cb._state == CircuitState.HALF_OPEN

    await cb.call(ok)  # 2번째 성공 — CLOSED 전환
    assert cb._state == CircuitState.CLOSED

    # CLOSED 복구 후 정상 동작 확인
    for _ in range(5):
        assert await cb.call(ok) == "ok"
