"""
Ties together every stage:
  Upload -> Parse -> Chunk -> Embed -> Store -> Retrieve -> Generate

Also owns conversation memory (for multi-turn chat) and index persistence
(so a user doesn't have to re-ingest documents every session).
"""
from typing import List, Dict, Tuple
from pathlib import Path

from .config import Config, DEFAULT_CONFIG
from .loaders import load_document
from .chunker import chunk_pages, Chunk
from .embedder import build_embedder, embedder_from_state, BaseEmbedder
from .vectorstore import VectorStore
from .retriever import Retriever
from .llm import build_generator


class RAGPipeline:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.embedder: BaseEmbedder = None
        self.store: VectorStore = None
        self.retriever: Retriever = None
        self.generator = build_generator(self.cfg.llm)
        self.history: List[Dict[str, str]] = []
        self._all_chunk_texts: List[str] = []  # kept to refit TF-IDF if more docs are added

    # ---------------- Ingestion ----------------
    def ingest(self, filepaths: List[str]) -> Dict:
        """Parse, chunk, embed, and index one or more documents."""
        all_chunks: List[Chunk] = []
        stats = {}
        for path in filepaths:
            pages = load_document(path)
            chunks = chunk_pages(pages, self.cfg.chunk)
            all_chunks.extend(chunks)
            stats[Path(path).name] = {"pages": len(pages), "chunks": len(chunks)}

        if not all_chunks:
            raise ValueError("No extractable text found in the provided document(s).")

        self._all_chunk_texts.extend(c.text for c in all_chunks)

        if self.embedder is None:
            self.embedder = build_embedder(self.cfg.embedding)

        # TF-IDF needs a fit() over the corpus; sentence-transformers ignores it.
        self.embedder.fit(self._all_chunk_texts)

        vectors = self.embedder.embed([c.text for c in all_chunks])

        if self.store is None:
            self.store = VectorStore(dim=vectors.shape[1])
        elif self.embedder.name == "tfidf":
            # TF-IDF vocabulary can grow with new docs -> must rebuild the whole index
            # since old vectors were computed against a smaller vocabulary.
            old_chunks = self.store.chunks
            self.store = VectorStore(dim=vectors.shape[1])
            if old_chunks:
                old_vecs = self.embedder.embed([c.text for c in old_chunks])
                self.store.add(old_vecs, old_chunks)

        self.store.add(vectors, all_chunks)
        self.retriever = Retriever(self.store, self.embedder, self.cfg.retrieval)

        stats["_embedder"] = self.embedder.name
        stats["_total_chunks"] = len(self.store.chunks)
        return stats

    # ---------------- Chat ----------------
    def chat(self, query: str, use_history: bool = True) -> Dict:
        if self.retriever is None:
            return {"answer": "Please upload at least one document first.", "sources": []}

        results = self.retriever.retrieve(query)
        history = self.history if use_history else []
        answer = self.generator.generate(query, results, history)

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        # keep memory bounded
        self.history = self.history[-12:]

        sources = [
            {"source": c.source, "page": c.page_number, "score": round(score, 3), "preview": c.text[:200]}
            for c, score in results
        ]
        return {"answer": answer, "sources": sources}

    def reset_history(self):
        self.history = []

    # ---------------- Persistence ----------------
    def save(self, path: str = None):
        path = path or self.cfg.persist_dir
        Path(path).mkdir(parents=True, exist_ok=True)
        self.store.save(path)
        import json
        with open(Path(path) / "embedder_state.json", "w") as f:
            json.dump(self.embedder.to_state(), f)

    @classmethod
    def load(cls, path: str = None, cfg: Config = None) -> "RAGPipeline":
        cfg = cfg or DEFAULT_CONFIG
        path = path or cfg.persist_dir
        import json
        pipeline = cls(cfg)
        with open(Path(path) / "embedder_state.json") as f:
            state = json.load(f)
        pipeline.embedder = embedder_from_state(state)
        pipeline.store = VectorStore.load(path)
        pipeline.retriever = Retriever(pipeline.store, pipeline.embedder, cfg.retrieval)
        pipeline._all_chunk_texts = [c.text for c in pipeline.store.chunks]
        return pipeline
