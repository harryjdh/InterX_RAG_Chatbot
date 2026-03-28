# 의존성 관리
# pip-tools 설치: pip install pip-tools

.PHONY: lock lock-build sync sync-build lock-dev sync-dev

## requirements.txt (프로덕션 lock file) 재생성
lock:
	pip-compile requirements.in -o requirements.txt --strip-extras

## requirements-build.txt (빌드 lock file) 재생성
lock-build:
	pip-compile requirements-build.in -o requirements-build.txt --strip-extras

## requirements-dev.txt (개발 lock file) 재생성
lock-dev:
	pip-compile requirements-dev.in -o requirements-dev.txt --strip-extras

## lock file 전체 재생성
lock-all: lock lock-build lock-dev

## 현재 환경을 lock file과 동기화
sync:
	pip-sync requirements.txt

sync-build:
	pip-sync requirements-build.txt

sync-dev:
	pip-sync requirements-dev.txt

# Docker
.PHONY: up down build build-vectordb logs

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

build-vectordb:
	docker compose --profile build run --rm build-vectordb

logs:
	docker compose logs -f api

# DB 마이그레이션
.PHONY: migrate migrate-history

## Alembic 마이그레이션 적용 (upgrade head)
migrate:
	alembic upgrade head

## 마이그레이션 이력 조회
migrate-history:
	alembic history --verbose

# 테스트
.PHONY: test test-v test-integration test-all

## 유닛 테스트 실행
test:
	pytest -m "not integration"

## 유닛 테스트 (상세 출력)
test-v:
	pytest -v -m "not integration"

## 통합 테스트 (실제 PostgreSQL 필요: POSTGRES_HOST, POSTGRES_DB 등 설정 필요)
test-integration:
	pytest -v -m integration

## 유닛 + 통합 테스트 전체 실행
test-all:
	pytest -v
