"""
Headless end-to-end demo: ingest a sample document and ask it questions.
Useful for CI, quick smoke-testing, or environments without a browser.

Usage:
    python cli_demo.py
    python cli_demo.py path/to/your.pdf "Your question here"
"""
import sys
from src.pipeline import RAGPipeline
from src.config import DEFAULT_CONFIG


DEMO_QUESTIONS = [
    "What is the refund policy?",
    "How long is the warranty period?",
    "Can I get a refund after the warranty expires?",
    "What is not covered under the warranty?",
]


def main():
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_policy.md"
    questions = [sys.argv[2]] if len(sys.argv) > 2 else DEMO_QUESTIONS

    pipeline = RAGPipeline(DEFAULT_CONFIG)

    print(f"Ingesting: {doc_path}")
    stats = pipeline.ingest([doc_path])
    print(f"  -> embedder: {stats['_embedder']}, total chunks indexed: {stats['_total_chunks']}\n")

    for q in questions:
        result = pipeline.chat(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['source']} (page {s['page']}, score {s['score']}): {s['preview'][:80]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
