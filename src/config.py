"""
Central configuration for the RAG chatbot.
Every tunable knob in the pipeline lives here so the whole system can be
re-tuned without touching business logic.
"""
from dataclasses import dataclass, field
import os


@dataclass
class ChunkConfig:
    chunk_size: int = 800          # characters per chunk (approx ~150-200 tokens)
    chunk_overlap: int = 120       # sliding window overlap to preserve context across chunks
    min_chunk_size: int = 50       # drop tiny leftover chunks


@dataclass
class RetrievalConfig:
    top_k: int = 5                 # chunks returned to the generator
    fetch_k: int = 20              # candidates considered before re-ranking / MMR
    use_mmr: bool = True           # Maximal Marginal Relevance -> diversity, less redundancy
    mmr_lambda: float = 0.5        # 1.0 = pure relevance, 0.0 = pure diversity
    score_threshold: float = 0.0   # minimum cosine similarity to keep a chunk


@dataclass
class EmbeddingConfig:
    # Preferred model if sentence-transformers + internet/model cache are available.
    st_model_name: str = "all-MiniLM-L6-v2"
    # Fallback (fully offline, no downloads): TF-IDF vectorization.
    fallback: str = "tfidf"
    tfidf_max_features: int = 20000


@dataclass
class LLMConfig:
    provider: str = "anthropic"                     # "anthropic" | "extractive"
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 1000
    temperature: float = 0.2
    system_prompt: str = (
        "You are a helpful assistant answering questions using ONLY the "
        "provided context excerpts from the user's documents. "
        "If the answer is not contained in the context, say you don't know "
        "rather than guessing. Always cite the source chunk numbers you used, "
        "like [Source 2]."
    )


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    persist_dir: str = os.path.join(os.path.dirname(__file__), "..", "storage")


DEFAULT_CONFIG = Config()
