"""
사전에 생성된 parquet 파일을 읽어 pgvector DB에 적재합니다.
임베딩 API 호출 없이 동작하므로 평가자 환경에서도 빠르게 실행됩니다.

사용법:
    python -m scripts.build_vectordb
    python -m scripts.build_vectordb --input data/embeddings.parquet

사전 조건:
    scripts/embed_and_export.py 를 먼저 실행하여 data/embeddings.parquet 생성
"""
import argparse
import asyncio

import pandas as pd
from tqdm import tqdm

from src.vectordb import VectorDB


async def main():
    parser = argparse.ArgumentParser(description="parquet → pgvector 적재")
    parser.add_argument(
        "--input",
        type=str,
        default="data/embeddings.parquet",
        help="입력 parquet 파일 경로 (기본값: data/embeddings.parquet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="DB 삽입 배치 크기 (기본값: 256)",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="기존 데이터 확인 없이 바로 추가",
    )
    args = parser.parse_args()

    print(f"parquet 로딩 중: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"총 {len(df)}개 문서 로드 완료")

    print("VectorDB 연결 중...")
    db = await VectorDB.create()
    await db.ensure_schema()

    try:
        existing = await db.count()
        if existing > 0 and not args.skip_confirm:
            print(f"\n주의: 기존에 {existing}개 문서가 저장되어 있습니다.")
            answer = input("계속 추가하시겠습니까? (y/n): ").strip().lower()
            if answer != "y":
                print("종료합니다.")
                return

        print(f"\nDB 적재 시작 (배치 크기: {args.batch_size})...")
        records = df.to_dict("records")
        failed = 0

        for i in tqdm(range(0, len(records), args.batch_size), desc="적재 중"):
            batch = records[i : i + args.batch_size]
            docs = [
                {
                    "content": row["content"],
                    "title": row.get("title", ""),
                    "embedding": list(row["embedding"]),
                }
                for row in batch
            ]
            try:
                await db.insert_batch(docs)
            except Exception as e:
                tqdm.write(f"배치 {i // args.batch_size + 1} 실패: {e}")
                failed += len(batch)

        final_count = await db.count()
        print(f"\n완료! 총 {final_count}개 문서 저장됨 (실패: {failed}개)")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
