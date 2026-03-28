"""
커스텀 RAG 관측성 메트릭 (Prometheus).

prometheus-fastapi-instrumentator 가 제공하는 HTTP 레벨 메트릭 외에
RAG 파이프라인 전용 지표를 여기서 정의합니다.
"""
from prometheus_client import Gauge, Histogram

# 임베딩 API 호출 지연 (초)
rag_embedding_duration = Histogram(
    "rag_embedding_duration_seconds",
    "Time to generate query embedding",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# 벡터 유사도 검색 지연 (초)
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Time to perform vector similarity search",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# 검색 결과 유사도 점수 (rank 레이블: "1"~"5")
rag_retrieval_score = Histogram(
    "rag_retrieval_score",
    "Cosine similarity score of retrieved documents",
    ["rank"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# LLM 전체 응답 생성 지연, non-streaming (초)
rag_llm_duration = Histogram(
    "rag_llm_duration_seconds",
    "Time for full LLM response generation (non-streaming)",
    buckets=[1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# LLM 스트리밍 첫 청크 도착 시간 (초)
rag_llm_ttfb = Histogram(
    "rag_llm_ttfb_seconds",
    "Time to first token from LLM streaming response",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# 프롬프트에 삽입된 문맥 총 글자 수
rag_context_chars = Histogram(
    "rag_context_chars_total",
    "Total characters of context inserted into LLM prompt",
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000],
)

# 서킷 브레이커 상태 게이지 (0=closed, 1=open, 2=half_open)
rag_circuit_state = Gauge(
    "rag_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["name"],
)

_CB_STATE_MAP = {"closed": 0, "open": 1, "half_open": 2}
