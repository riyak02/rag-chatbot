"""
Retrieval stage.

Two-phase retrieval:
  1. Pull `fetch_k` nearest neighbours by cosine similarity (recall pass).
  2. Re-rank down to `top_k` using Maximal Marginal Relevance (MMR), which
     trades a little raw similarity for diversity -- this stops the answer
     being built from 5 near-duplicate chunks that all say the same thing,
     a common failure mode in naive top-k RAG.
"""
from typing import List, Tuple
import numpy as np

from .chunker import Chunk
from .vectorstore import VectorStore
from .embedder import BaseEmbedder
from .config import RetrievalConfig


def _mmr(query_vec: np.ndarray, candidates: List[Tuple[Chunk, float, np.ndarray]],
         k: int, lambda_mult: float) -> List[Tuple[Chunk, float]]:
    if not candidates:
        return []
    selected: List[int] = []
    remaining = list(range(len(candidates)))
    vecs = np.array([c[2] for c in candidates])

    while remaining and len(selected) < k:
        if not selected:
            # first pick: highest relevance
            best = max(remaining, key=lambda i: candidates[i][1])
        else:
            sel_vecs = vecs[selected]
            best, best_score = None, -1e9
            for i in remaining:
                relevance = candidates[i][1]
                diversity_penalty = float(np.max(sel_vecs @ vecs[i])) if len(selected) else 0.0
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                if mmr_score > best_score:
                    best_score, best = mmr_score, i
        selected.append(best)
        remaining.remove(best)

    return [(candidates[i][0], candidates[i][1]) for i in selected]


class Retriever:
    def __init__(self, store: VectorStore, embedder: BaseEmbedder, cfg: RetrievalConfig):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        q_vec = self.embedder.embed([query])[0]
        results = self.store.search(q_vec, self.cfg.fetch_k)
        results = [(c, s) for c, s in results if s >= self.cfg.score_threshold]

        if not self.cfg.use_mmr:
            return results[: self.cfg.top_k]

        # need per-candidate vectors for MMR; re-embed candidate texts
        # (cheap: only fetch_k items, not the whole corpus)
        cand_texts = [c.text for c, _ in results]
        if not cand_texts:
            return []
        cand_vecs = self.embedder.embed(cand_texts)
        candidates = [(results[i][0], results[i][1], cand_vecs[i]) for i in range(len(results))]
        return _mmr(q_vec, candidates, self.cfg.top_k, self.cfg.mmr_lambda)
