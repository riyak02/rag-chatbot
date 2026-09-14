"""
Debug/inspection script: shows the raw intermediate output of every RAG stage
(parsing, chunking, embedding, vector storage, retrieval) instead of just the
summary stats the Streamlit UI shows.

Usage:
    python inspect_pipeline.py
    python inspect_pipeline.py path/to/your.pdf "your question"
"""
import sys
import numpy as np

from src.pipeline import RAGPipeline
from src.config import DEFAULT_CONFIG

np.set_printoptions(precision=4, suppress=True)


def main():
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_policy.md"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the refund policy?"

    pipeline = RAGPipeline(DEFAULT_CONFIG)

    print("=" * 80)
    print(f"STAGE 1-2: Upload + Parse + Chunk  ({doc_path})")
    print("=" * 80)
    stats = pipeline.ingest([doc_path])
    print(stats)

    print("\n" + "=" * 80)
    print("STAGE 3: Chunks produced")
    print("=" * 80)
    for i, chunk in enumerate(pipeline.store.chunks):
        print(f"[{i}] source={chunk.source} page={chunk.page_number} "
              f"len={len(chunk.text)}")
        print(f"    text preview: {chunk.text[:120]!r}")

    print("\n" + "=" * 80)
    print(f"STAGE 4: Embedding Generation  (embedder={pipeline.embedder.name})")
    print("=" * 80)
    vectors = pipeline.store.all_vectors()
    print(f"embedding matrix shape: {vectors.shape}  (n_chunks x embedding_dim)")
    print(f"vector for chunk[0] (first 10 dims): {vectors[0][:10]}")
    print(f"L2 norm of chunk[0] vector: {np.linalg.norm(vectors[0]):.4f}  "
          f"(should be ~1.0 since vectors are normalized for cosine similarity)")

    print("\n" + "=" * 80)
    print("STAGE 5: Vector Storage")
    print("=" * 80)
    backend = "FAISS IndexFlatIP" if hasattr(pipeline.store, "_index") else "NumPy matrix"
    print(f"backend: {backend}")
    print(f"total vectors stored: {len(pipeline.store.chunks)}")
    print(f"dimension: {pipeline.store.dim}")

    print("\n" + "=" * 80)
    print(f"STAGE 6: Retrieval  (query: {question!r})")
    print("=" * 80)
    results = pipeline.retriever.retrieve(question)
    for chunk, score in results:
        print(f"  score={score:.4f}  source={chunk.source} page={chunk.page_number}")
        print(f"    {chunk.text[:150]!r}")

    print("\n" + "=" * 80)
    print("STAGE 7: Response Generation")
    print("=" * 80)
    result = pipeline.chat(question, use_history=False)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
