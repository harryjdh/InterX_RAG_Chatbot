import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = None


def _get_url() -> str:
    from src.config import config as app_config

    return (
        f"postgresql+asyncpg://"
        f"{app_config.POSTGRES_USER}"
        f":{app_config.POSTGRES_PASSWORD.get_secret_value()}"
        f"@{app_config.POSTGRES_HOST}"
        f":{app_config.POSTGRES_PORT}"
        f"/{app_config.POSTGRES_DB}"
    )


def run_migrations_offline() -> None:
    """오프라인 모드: DB 연결 없이 SQL 스크립트 생성."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """온라인 모드: 실제 DB에 마이그레이션 적용."""
    engine = create_async_engine(_get_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
