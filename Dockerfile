# 1단계: 빌드 (gcc 등 컴파일 도구 포함)
FROM python:3.11.12-slim AS builder

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# 2단계: 실행 (컴파일된 패키지만 복사)
FROM python:3.11.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser . .

# COPY 후 root 권한으로 실행 권한 부여 (USER 전환 전)
RUN chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

# curl 없이 Python 표준 라이브러리만으로 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# tini가 PID 1을 담당: 좀비 프로세스 회수 + SIGTERM을 uvicorn master에 안전하게 전달
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/entrypoint.sh"]
