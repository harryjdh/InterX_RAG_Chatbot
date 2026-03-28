"""
KorQuAD 데이터셋에서 문맥을 추출하여 pgvector에 저장하는 스크립트.

사용법:
    # 초기 구축 (1,000개)
    python -m scripts.build_vectordb --limit 1000

    # 전체 데이터 구축
    python -m scripts.build_vectordb --limit 0

    # 배치 크기 조정
    python -m scripts.build_vectordb --limit 5000 --batch-size 64
"""
import argparse
import asyncio

from datasets import load_dataset
from tqdm import tqdm

from src.embeddings import EmbeddingClient
from src.vectordb import VectorDB


def _run_migrations() -> None:
    """Alembic 마이그레이션을 실행합니다 (alembic upgrade head)."""
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command

    print("DB 스키마 마이그레이션 실행 중 (alembic upgrade head)...")
    cfg = AlembicConfig("alembic.ini")
    alembic_command.upgrade(cfg, "head")
    print("마이그레이션 완료.")


def extract_unique_contexts(dataset) -> list[dict]:
    """데이터셋에서 중복 없는 문맥 목록을 추출합니다."""
    seen: set[str] = set()
    contexts: list[dict] = []

    for item in dataset:
        content = item["context"].strip()
        if content in seen:
            continue
        seen.add(content)
        contexts.append({"content": content, "title": item.get("title", "")})

    return contexts


async def main():
    parser = argparse.ArgumentParser(description="KorQuAD → pgvector 임베딩 구축")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="처리할 문맥 수 (0 = 전체, 기본값: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="임베딩 API 배치 크기 (기본값: 32)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="데이터셋 split (기본값: train)",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="기존 데이터 확인 없이 바로 추가",
    )
    args = parser.parse_args()

    # 1. 데이터셋 로드
    print(f"KorQuAD 데이터셋 로딩 중 (split={args.split})...")
    dataset = load_dataset("KorQuAD/squad_kor_v1", split=args.split)
    print(f"전체 레코드 수: {len(dataset)}")

    # 2. 고유 문맥 추출
    print("고유 문맥 추출 중...")
    contexts = extract_unique_contexts(dataset)
    print(f"고유 문맥 수: {len(contexts)}")

    if args.limit > 0:
        contexts = contexts[: args.limit]
        print(f"처리 대상: {len(contexts)}개 (--limit {args.limit})")

    # 3. VectorDB 초기화 (스키마는 _run_migrations()에서 alembic이 처리)
    print("VectorDB 연결 중...")
    db = await VectorDB.create()
    embedder = EmbeddingClient()

    try:
        existing = await db.count()
        if existing > 0 and not args.skip_confirm:
            print(f"\n주의: 기존에 {existing}개 문서가 저장되어 있습니다.")
            answer = input("계속 추가하시겠습니까? (y/n): ").strip().lower()
            if answer != "y":
                print("종료합니다.")
                return

        # 4. 배치 임베딩 및 저장
        print(f"\n임베딩 생성 및 저장 시작 (배치 크기: {args.batch_size})...")
        batch_size = args.batch_size
        failed = 0

        for i in tqdm(range(0, len(contexts), batch_size), desc="배치 처리"):
            batch = contexts[i : i + batch_size]
            texts = [doc["content"] for doc in batch]

            try:
                embeddings = await embedder.embed_batch(texts)
                docs_to_insert = [
                    {
                        "content": doc["content"],
                        "title": doc["title"],
                        "embedding": emb,
                    }
                    for doc, emb in zip(batch, embeddings)
                ]
                await db.insert_batch(docs_to_insert)

            except Exception as e:
                tqdm.write(f"배치 {i // batch_size + 1} 실패: {e} — 개별 처리 시도...")
                for doc in batch:
                    try:
                        emb = await embedder.embed(doc["content"])
                        await db.insert_batch(
                            [
                                {
                                    "content": doc["content"],
                                    "title": doc["title"],
                                    "embedding": emb,
                                }
                            ]
                        )
                    except Exception as e2:
                        tqdm.write(f"  문서 건너뜀: {e2}")
                        failed += 1

        final_count = await db.count()
        print(f"\n완료! 총 {final_count}개 문서 저장됨 (실패: {failed}개)")

        print("HNSW 인덱스 생성 중...")
        await db.create_index()
    finally:
        await embedder.close()
        await db.close()


if __name__ == "__main__":
    _run_migrations()
    asyncio.run(main())
