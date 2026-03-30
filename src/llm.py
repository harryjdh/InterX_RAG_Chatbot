import asyncio
import logging
import random
import re

import openai
from openai import AsyncOpenAI
from typing import AsyncIterator, List, Optional

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from .config import config
from .retry import retry

logger = logging.getLogger(__name__)

# 스트림 도중 네트워크 단절 시 재연결 최대 횟수.
# 아직 클라이언트로 청크를 전송하지 않은 경우에만 재시도합니다.
_STREAM_RETRIES = 2

_NON_RETRYABLE = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
)


class LLMClient:
    def __init__(self, redis=None):
        self._client = AsyncOpenAI(
            api_key=config.LLM_API_KEY.get_secret_value(),
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
        )
        self._cb = CircuitBreaker(name="llm", redis=redis)

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str], history: Optional[List[dict]]
    ) -> List[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt, history)
        return await self._cb.call(self._generate_raw, messages)

    async def _generate_raw(self, messages: List[dict]) -> str:
        response = await retry(
            self._client.chat.completions.create,
            model=config.LLM_MODEL_NAME,
            messages=messages,
            stream=False,
            non_retryable=_NON_RETRYABLE,
        )
        content = response.choices[0].message.content or ""
        return self._strip_thinking(content)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """
        Yields text chunks, skipping <think>...</think> blocks that Qwen3 models
        may emit before the actual answer.

        스트림 생성과 반복을 하나의 CB 단위로 처리합니다.
        - 스트림 생성 실패 / mid-stream 단절 모두 CB 실패로 기록됩니다.
        - 아직 클라이언트에 청크를 전송하지 않은 경우에만 재시도합니다.
        """
        messages = self._build_messages(prompt, system_prompt, history)

        for attempt in range(_STREAM_RETRIES + 1):
            # 매 시도 전 CB 상태 확인 (OPEN이면 CircuitBreakerOpen 발생)
            await self._cb.check_state()

            chunks_yielded = 0
            stream = None  # 재시도 시 이전 스트림을 추적하여 명시적으로 닫기 위해 초기화
            try:
                # _create_stream은 CB를 통하지 않음 — 성공/실패를 아래에서 명시적으로 기록
                stream = await self._create_stream(messages)
                async for chunk in self._iter_stream(stream):
                    chunks_yielded += 1
                    yield chunk
                # 스트림 전체 완료 시에만 성공 기록
                await self._cb.record_success()
                return
            except CircuitBreakerOpen:
                raise
            except Exception as exc:
                # 스트림 생성 실패와 mid-stream 단절 모두 CB 실패로 기록
                await self._cb.record_failure(exc)
                if chunks_yielded > 0 or attempt >= _STREAM_RETRIES:
                    raise
                # 재시도 전 이전 HTTP 스트림 연결을 명시적으로 닫아 커넥션 풀 누수 방지.
                # chunks_yielded == 0 경로에서만 도달하므로 클라이언트에는 아직 아무것도 전송되지 않은 상태.
                if stream is not None:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass
                wait = min(1.0 * (2.0 ** attempt), 10.0) * (0.5 + random.random() * 0.5)
                logger.warning(
                    "스트리밍 재연결 시도 %d/%d (%.1fs 후): %s",
                    attempt + 1,
                    _STREAM_RETRIES,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)

    async def _create_stream(self, messages: List[dict]):
        return await retry(
            self._client.chat.completions.create,
            model=config.LLM_MODEL_NAME,
            messages=messages,
            stream=True,
            non_retryable=_NON_RETRYABLE,
        )

    async def _iter_stream(self, stream) -> AsyncIterator[str]:
        buffer = ""
        in_thinking = False

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            buffer += delta

            # Process buffer: skip <think>...</think> content
            while True:
                if in_thinking:
                    end_idx = buffer.find("</think>")
                    if end_idx != -1:
                        # Discard everything up to and including </think>
                        buffer = buffer[end_idx + len("</think>"):]
                        in_thinking = False
                    else:
                        # Still inside thinking block, discard buffered content
                        # Keep last few chars in case tag spans chunk boundary
                        if len(buffer) > len("</think>"):
                            buffer = buffer[-len("</think>"):]
                        break
                else:
                    start_idx = buffer.find("<think>")
                    if start_idx != -1:
                        # Yield content that appears before the thinking tag
                        if start_idx > 0:
                            yield buffer[:start_idx]
                        buffer = buffer[start_idx + len("<think>"):]
                        in_thinking = True
                    else:
                        # No thinking tag found; yield safely, keeping a small
                        # tail in case "<think>" is split across chunks
                        tail = len("<think>") - 1
                        if len(buffer) > tail:
                            yield buffer[:-tail]
                            buffer = buffer[-tail:]
                        break

        # Flush remaining buffer after stream ends
        if buffer and not in_thinking:
            yield buffer

    async def ping(self) -> bool:
        """LLM API 연결 확인 (토큰 생성 없이 /models 엔드포인트 호출).

        헬스 체크는 CB를 통과하지 않습니다 — 반복적인 헬스 체크가 CB를 트립시키는 것을 방지합니다.
        """
        return await self._ping_raw()

    async def _ping_raw(self) -> bool:
        await self._client.models.list()
        return True

    async def close(self):
        await self._client.close()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from non-streaming output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
