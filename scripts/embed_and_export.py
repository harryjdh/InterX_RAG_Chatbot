"""
KorQuAD 데이터셋에서 문맥을 추출하여 임베딩을 생성하고 parquet 파일로 저장합니다.
평가자 환경에서 임베딩 API 없이 VectorDB를 구축할 수 있도록 사전에 1회 실행합니다.

사용법:
    # 전체 데이터 임베딩 (기본값)
    python -m scripts.embed_and_export

    # 일부만 임베딩
    python -m scripts.embed_and_export --limit 1000

결과물:
    data/embeddings.parquet  (columns: content, title, embedding)
"""
import argparse
import asyncio
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.embeddings import EmbeddingClient


def extract_unique_contexts(dataset) -> list[dict]:
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
    parser = argparse.ArgumentParser(description="KorQuAD 임베딩 생성 및 parquet 저장")
    parser.add_argument("--limit", type=int, default=0, help="처리할 문맥 수 (0 = 전체)")
    parser.add_argument("--batch-size", type=int, default=32, help="임베딩 API 배치 크기")
    parser.add_argument("--split", type=str, default="train", help="데이터셋 split")
    parser.add_argument("--output", type=str, default="data/embeddings.parquet", help="출력 파일 경로")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"KorQuAD 데이터셋 로딩 중 (split={args.split})...")
    dataset = load_dataset("KorQuAD/squad_kor_v1", split=args.split)
    print(f"전체 레코드 수: {len(dataset)}")

    print("고유 문맥 추출 중...")
    contexts = extract_unique_contexts(dataset)
    print(f"고유 문맥 수: {len(contexts)}")

    if args.limit > 0:
        contexts = contexts[: args.limit]
        print(f"처리 대상: {len(contexts)}개 (--limit {args.limit})")

    embedder = EmbeddingClient()
    records = []
    failed = 0

    print(f"\n임베딩 생성 중 (배치 크기: {args.batch_size})...")
    try:
        for i in tqdm(range(0, len(contexts), args.batch_size), desc="배치 처리"):
            batch = contexts[i : i + args.batch_size]
            texts = [doc["content"] for doc in batch]
            try:
                embeddings = await embedder.embed_batch(texts)
                for doc, emb in zip(batch, embeddings):
                    records.append({
                        "content": doc["content"],
                        "title": doc["title"],
                        "embedding": emb,
                    })
            except Exception as e:
                tqdm.write(f"배치 {i // args.batch_size + 1} 실패: {e} — 개별 처리 시도...")
                for doc in batch:
                    try:
                        emb = await embedder.embed(doc["content"])
                        records.append({
                            "content": doc["content"],
                            "title": doc["title"],
                            "embedding": emb,
                        })
                    except Exception as e2:
                        tqdm.write(f"  문서 건너뜀: {e2}")
                        failed += 1
    finally:
        await embedder.close()

    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)
    print(f"\n완료! {len(records)}개 저장 → {args.output} (실패: {failed}개)")


if __name__ == "__main__":
    asyncio.run(main())
