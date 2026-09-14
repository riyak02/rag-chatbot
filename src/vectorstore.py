"""
Vector Storage stage.

Uses FAISS (IndexFlatIP over L2-normalized vectors == cosine similarity)
when available for speed at scale; falls back to a plain NumPy matrix +
matrix-multiply cosine search otherwise (perfectly fine up to tens of
thousands of chunks, and zero extra dependencies).
"""
import json
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .chunker import Chunk

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: List[Chunk] = []
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        assert vectors.shape[0] == len(chunks)
        vectors = vectors.astype(np.float32)
        if _HAS_FAISS:
            self._index.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors]) if self._matrix.size else vectors
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        k = min(k, len(self.chunks))
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        if _HAS_FAISS:
            scores, ids = self._index.search(query_vec, k)
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
        else:
            sims = (self._matrix @ query_vec.T).ravel()
            top_idx = np.argsort(-sims)[:k]
            return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def all_vectors(self) -> np.ndarray:
        if _HAS_FAISS:
            return self._index.reconstruct_n(0, self._index.ntotal)
        return self._matrix

    # ---- persistence ----
    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        vecs = self.all_vectors()
        np.save(Path(path) / "vectors.npy", vecs)
        with open(Path(path) / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "backend": "faiss" if _HAS_FAISS else "numpy"}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(Path(path) / "meta.json") as f:
            meta = json.load(f)
        store = cls(meta["dim"])
        with open(Path(path) / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        vecs = np.load(Path(path) / "vectors.npy")
        store.add(vecs, chunks)
        return store
