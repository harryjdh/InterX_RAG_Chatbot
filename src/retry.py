import asyncio
import logging
import random
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


async def retry(
    func: Callable,
    *args,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    non_retryable: Tuple[Type[Exception], ...] = (),
    **kwargs,
):
    """지수 백오프 + jitter로 비동기 함수를 재시도합니다.

    Args:
        non_retryable: 재시도하지 않을 예외 타입 (인증 실패 등).
        max_delay: 재시도 대기 시간 상한 (초).
    """
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except non_retryable:
            raise
        except asyncio.TimeoutError:
            # 외부 asyncio.timeout() 스코프에서 발생한 타임아웃 — 재시도해도 즉시 재발하므로 전파
            raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            # Retry-After 헤더가 있으면 서버가 지정한 대기 시간을 우선 사용
            retry_after = _parse_retry_after(e)
            if retry_after is not None:
                wait = retry_after
            else:
                base_wait = min(delay * (backoff ** attempt), max_delay)
                wait = base_wait * (0.5 + random.random() * 0.5)
            logger.warning(
                "재시도 %d/%d (%.1fs 후): %s", attempt + 1, retries, wait, e
            )
            await asyncio.sleep(wait)


def _parse_retry_after(exc: Exception) -> float | None:
    """예외의 response 헤더에서 Retry-After 값을 추출합니다."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
