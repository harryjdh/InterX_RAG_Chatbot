"""src.retry 모듈 단위 테스트."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.retry import retry


async def test_success_on_first_attempt():
    fn = AsyncMock(return_value="ok")
    result = await retry(fn, retries=3, delay=0)
    assert result == "ok"
    fn.assert_called_once()


async def test_success_after_one_failure():
    fn = AsyncMock(side_effect=[ValueError("일시 오류"), "ok"])
    result = await retry(fn, retries=3, delay=0)
    assert result == "ok"
    assert fn.call_count == 2


async def test_raises_after_all_retries_exhausted():
    fn = AsyncMock(side_effect=ValueError("항상 실패"))
    with pytest.raises(ValueError, match="항상 실패"):
        await retry(fn, retries=3, delay=0)
    assert fn.call_count == 3


async def test_non_retryable_exception_raises_immediately():
    fn = AsyncMock(side_effect=KeyError("재시도 불가"))
    with pytest.raises(KeyError):
        await retry(fn, retries=3, delay=0, non_retryable=(KeyError,))
    fn.assert_called_once()


async def test_retry_after_header_respected():
    """Retry-After 헤더가 있으면 해당 값을 대기 시간으로 사용한다."""
    exc = ValueError("rate limited")
    response_mock = MagicMock()
    response_mock.headers = {"Retry-After": "0"}
    exc.response = response_mock

    fn = AsyncMock(side_effect=[exc, "ok"])
    result = await retry(fn, retries=3, delay=0)
    assert result == "ok"
    assert fn.call_count == 2


async def test_max_delay_cap_applied():
    """지수 백오프가 max_delay를 초과하지 않는다."""
    call_count = 0

    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("fail")
        return "ok"

    result = await retry(fn, retries=3, delay=0, backoff=100.0, max_delay=0.001)
    assert result == "ok"
    assert call_count == 3
