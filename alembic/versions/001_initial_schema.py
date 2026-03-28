"""initial schema: documents table with pgvector

Revision ID: 001
Revises:
Create Date: 2026-03-28

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # vector 익스텐션은 DB 레벨 설치이므로 IF NOT EXISTS 유지 (중복 설치 무해)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # IF NOT EXISTS 제거: 테이블이 이미 존재하면 Alembic이 명시적으로 오류를 발생시켜
    # 스키마 불일치를 조기에 감지합니다.
    # 기존 DB(ensure_schema()로 생성된)에서 마이그레이션을 처음 적용할 경우:
    #   alembic stamp head  # 현재 상태를 마이그레이션 완료로 표시
    op.execute(
        """
        CREATE TABLE documents (
            id          SERIAL PRIMARY KEY,
            content     TEXT NOT NULL,
            title       TEXT,
            embedding   vector(2560),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
