"""LLMClient 단위 테스트 — 외부 API 호출 없이 순수 로직만 검증."""
from unittest.mock import MagicMock

from src.llm import LLMClient


# --- _strip_thinking (staticmethod) ---

def test_strip_thinking_removes_single_block():
    result = LLMClient._strip_thinking("<think>내부 추론</think>실제 답변")
    assert result == "실제 답변"


def test_strip_thinking_preserves_normal_text():
    result = LLMClient._strip_thinking("일반 텍스트입니다.")
    assert result == "일반 텍스트입니다."


def test_strip_thinking_removes_multiple_blocks():
    result = LLMClient._strip_thinking("<think>1차 추론</think>중간<think>2차 추론</think>끝")
    assert result == "중간끝"
    assert "추론" not in result


def test_strip_thinking_empty_input():
    assert LLMClient._strip_thinking("") == ""


def test_strip_thinking_multiline_block():
    result = LLMClient._strip_thinking("<think>\n여러 줄\n추론\n</think>답변")
    assert result == "답변"
    assert "추론" not in result


# --- _build_messages ---

def test_build_messages_with_system_prompt():
    client = object.__new__(LLMClient)
    msgs = client._build_messages("질문", "시스템 지시", None)
    assert msgs[0] == {"role": "system", "content": "시스템 지시"}
    assert msgs[-1] == {"role": "user", "content": "질문"}


def test_build_messages_without_system_prompt():
    client = object.__new__(LLMClient)
    msgs = client._build_messages("질문", None, None)
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "질문"}


def test_build_messages_with_history():
    client = object.__new__(LLMClient)
    history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]
    msgs = client._build_messages("새 질문", None, history)
    assert msgs[0] == {"role": "user", "content": "이전 질문"}
    assert msgs[1] == {"role": "assistant", "content": "이전 답변"}
    assert msgs[2] == {"role": "user", "content": "새 질문"}


# --- _iter_stream ---

def _make_chunk(content):
    """청크 모의 객체 생성."""
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


async def test_iter_stream_normal_text():
    """<think> 태그 없는 일반 텍스트는 그대로 출력된다."""
    async def mock_stream():
        for c in ["Hello", " World"]:
            yield _make_chunk(c)

    client = object.__new__(LLMClient)
    result = []
    async for text in client._iter_stream(mock_stream()):
        result.append(text)
    assert "".join(result) == "Hello World"


async def test_iter_stream_skips_think_block():
    """<think>...</think> 전체가 제거된다."""
    async def mock_stream():
        for c in ["<think>reasoning step</think>", "answer"]:
            yield _make_chunk(c)

    client = object.__new__(LLMClient)
    result = []
    async for text in client._iter_stream(mock_stream()):
        result.append(text)
    combined = "".join(result)
    assert combined == "answer"
    assert "reasoning" not in combined


async def test_iter_stream_think_tag_split_across_chunks():
    """<think> 태그가 청크 경계에 걸쳐 분할되어도 정상 처리된다."""
    async def mock_stream():
        for c in ["prefix <thi", "nk>thought</think> suffix"]:
            yield _make_chunk(c)

    client = object.__new__(LLMClient)
    result = []
    async for text in client._iter_stream(mock_stream()):
        result.append(text)
    combined = "".join(result)
    assert "<think>" not in combined
    assert "thought" not in combined
    assert "prefix" in combined
    assert "suffix" in combined


async def test_iter_stream_none_delta_is_skipped():
    """delta.content가 None인 청크는 무시된다."""
    async def mock_stream():
        yield _make_chunk(None)   # None 청크
        yield _make_chunk("ok")

    client = object.__new__(LLMClient)
    result = []
    async for text in client._iter_stream(mock_stream()):
        result.append(text)
    assert "ok" in "".join(result)
