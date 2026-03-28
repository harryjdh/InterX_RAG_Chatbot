"""
커맨드 라인 인터페이스 — 질문을 입력하면 스트리밍으로 답변을 출력합니다.

사용법:
    python cli.py
"""
import asyncio

from src.rag import NaiveRAG


async def main():
    rag = await NaiveRAG.create()
    print("=" * 60)
    print(" RAG 챗봇 (KorQuAD 기반 한국어 QA)")
    print(" 종료하려면 'quit', 'exit', 또는 Ctrl+C")
    print("=" * 60)

    try:
        while True:
            query = input("\n질문: ").encode("utf-8", errors="ignore").decode("utf-8").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("챗봇을 종료합니다.")
                break

            print("답변: ", end="", flush=True)
            async for chunk in rag.answer_stream(query):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print("\n챗봇을 종료합니다.")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
