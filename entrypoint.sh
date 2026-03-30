#!/bin/sh
# exec로 sh를 uvicorn으로 대체하여 tini의 직접 자식이 uvicorn이 되도록 합니다.
# tini (PID 1) → uvicorn master → uvicorn workers
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS:-4}" \
    --timeout-graceful-shutdown 30
