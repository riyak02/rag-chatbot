"""
Embedding Generation stage.

Design goal: never hard-fail. If sentence-transformers (and its model
weights, which require a one-time download) is unavailable -- e.g. no
internet access, air-gapped environment, or the package isn't installed --
the system transparently falls back to a TF-IDF vectorizer, which is pure
scikit-learn math and needs no external downloads at all. Everything above
this module (vector store, retriever) only ever sees "give me a matrix of
vectors for these texts", so the swap is invisible to the rest of the
pipeline.
"""
from typing import List, Optional
import numpy as np

from .config import EmbeddingConfig


class BaseEmbedder:
    dim: int

    def fit(self, corpus: List[str]) -> None:
        """Optional: called once with the full corpus (TF-IDF needs this)."""

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def to_state(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> "BaseEmbedder":
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # may raise ImportError
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def to_state(self) -> dict:
        return {"type": "sentence-transformers", "model_name": self.model_name}

    @classmethod
    def from_state(cls, state: dict) -> "SentenceTransformerEmbedder":
        return cls(state["model_name"])


class TfidfEmbedder(BaseEmbedder):
    """Fully offline fallback: TF-IDF + L2 normalization -> cosine similarity works
    the same way it would with dense embeddings for the downstream vector store."""
    name = "tfidf"

    def __init__(self, max_features: int = 20000, vocabulary: Optional[dict] = None, idf: Optional[np.ndarray] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        self._fitted = False
        if vocabulary is not None and idf is not None:
            self.vectorizer.vocabulary_ = vocabulary
            self.vectorizer.idf_ = idf
            self.vectorizer._tfidf._idf_diag = None  # rebuilt lazily by sklearn on first use
            import scipy.sparse as sp
            self.vectorizer._tfidf._idf_diag = sp.diags(idf)
            self._fitted = True
            self.dim = len(vocabulary)

    def fit(self, corpus: List[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True
        self.dim = len(self.vectorizer.vocabulary_)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() on a corpus before embedding.")
        mat = self.vectorizer.transform(texts)
        arr = mat.toarray().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def to_state(self) -> dict:
        return {
            "type": "tfidf",
            "max_features": self.max_features,
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
            "idf": self.vectorizer.idf_.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "TfidfEmbedder":
        import numpy as np
        return cls(state["max_features"], state["vocabulary"], np.array(state["idf"]))


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    """Try the higher-quality embedder first, fall back gracefully."""
    try:
        return SentenceTransformerEmbedder(cfg.st_model_name)
    except Exception:
        return TfidfEmbedder(max_features=cfg.tfidf_max_features)


def embedder_from_state(state: dict) -> BaseEmbedder:
    if state["type"] == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder.from_state(state)
        except Exception:
            raise RuntimeError(
                "Index was built with sentence-transformers but the model isn't "
                "available in this environment. Re-ingest documents to rebuild "
                "with the TF-IDF fallback."
            )
    return TfidfEmbedder.from_state(state)
