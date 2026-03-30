"""NaiveRAG 파이프라인 단위 테스트 — 외부 API 호출 없이 순수 로직만 검증."""
from src.rag import _build_prompt


def test_build_prompt_contains_query_and_contexts():
    prompt = _build_prompt("수도는?", ["서울에 관한 문맥", "한국에 관한 문맥"])
    assert "수도는?" in prompt
    assert "서울에 관한 문맥" in prompt
    assert "한국에 관한 문맥" in prompt
    assert "답변:" in prompt


def test_build_prompt_numbers_each_context():
    """문맥이 [문맥 N] 형식으로 번호가 매겨진다."""
    prompt = _build_prompt("q", ["ctx1", "ctx2", "ctx3"])
    assert "[문맥 1]" in prompt
    assert "[문맥 2]" in prompt
    assert "[문맥 3]" in prompt


def test_build_prompt_single_context():
    prompt = _build_prompt("q", ["only context"])
    assert "[문맥 1]" in prompt
    assert "[문맥 2]" not in prompt


def test_build_prompt_empty_contexts():
    """문맥이 없어도 프롬프트가 생성되고 질문과 답변 마커가 포함된다."""
    prompt = _build_prompt("질문", [])
    assert "질문" in prompt
    assert "답변:" in prompt
    assert "[문맥" not in prompt
