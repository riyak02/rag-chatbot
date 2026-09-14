import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAGPipeline
from src.config import DEFAULT_CONFIG
from src.chunker import chunk_pages, ChunkConfig
from src.loaders import Page, load_document

SAMPLE_DOC = str(Path(__file__).resolve().parents[1] / "data" / "sample_policy.md")


def test_loader_reads_markdown():
    pages = load_document(SAMPLE_DOC)
    assert len(pages) == 1
    assert "Warranty" in pages[0].text


def test_chunker_respects_overlap_and_min_size():
    page = Page(text="A" * 50 + ". " + "B" * 2000, source="x.txt", page_number=1)
    cfg = ChunkConfig(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
    chunks = chunk_pages([page], cfg)
    assert len(chunks) > 1
    assert all(len(c.text) <= 600 for c in chunks)  # size + overlap slack


def test_end_to_end_ingest_and_chat():
    pipeline = RAGPipeline(DEFAULT_CONFIG)
    stats = pipeline.ingest([SAMPLE_DOC])
    assert stats["_total_chunks"] > 0

    result = pipeline.chat("How long is the warranty?")
    assert result["answer"]
    assert len(result["sources"]) > 0
    # the correct fact ("24-month") should surface somewhere in top sources
    assert any("24-month" in s["preview"] or "24 months" in s["preview"] for s in result["sources"])


def test_save_and_load_roundtrip(tmp_path):
    pipeline = RAGPipeline(DEFAULT_CONFIG)
    pipeline.ingest([SAMPLE_DOC])
    pipeline.save(str(tmp_path))

    reloaded = RAGPipeline.load(str(tmp_path), DEFAULT_CONFIG)
    result = reloaded.chat("What is the refund policy?")
    assert result["answer"]
    assert len(reloaded.store.chunks) == len(pipeline.store.chunks)


def test_multiple_documents_grow_index():
    pipeline = RAGPipeline(DEFAULT_CONFIG)
    pipeline.ingest([SAMPLE_DOC])
    n1 = len(pipeline.store.chunks)
    pipeline.ingest([SAMPLE_DOC])  # ingest again -> should double
    n2 = len(pipeline.store.chunks)
    assert n2 == 2 * n1
