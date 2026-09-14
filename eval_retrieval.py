"""
Lightweight retrieval evaluation: given a set of (question, expected_keyword)
pairs, checks whether the retrieved top-k chunks actually contain the
keyword that would let an LLM answer correctly. This is the kind of check
you'd wire into CI before shipping changes to chunking/embedding config.

Usage: python eval_retrieval.py
"""
from src.pipeline import RAGPipeline
from src.config import DEFAULT_CONFIG

EVAL_SET = [
    ("What is the warranty length?", "24-month"),
    ("How many days do I have for a full refund?", "30 days"),
    ("Is there expedited shipping?", "Expedited shipping"),
    ("Are gift cards refundable?", "non-refundable"),
    ("What are the support hours?", "9am-6pm"),
]


def main():
    pipeline = RAGPipeline(DEFAULT_CONFIG)
    pipeline.ingest(["data/sample_policy.md"])

    hits = 0
    for question, keyword in EVAL_SET:
        retrieved = pipeline.retriever.retrieve(question)
        found = any(keyword.lower() in c.text.lower() for c, _ in retrieved)
        hits += found
        print(f"[{'HIT' if found else 'MISS'}] {question!r} -> expected keyword {keyword!r}")

    print(f"\nRetrieval hit-rate: {hits}/{len(EVAL_SET)} ({100*hits/len(EVAL_SET):.0f}%)")


if __name__ == "__main__":
    main()
