"""
Chunking stage.

Uses a recursive splitter: try to break on paragraph boundaries first,
then sentences, then hard character windows -- always with overlap so
context isn't lost at chunk edges. This mirrors the standard
"RecursiveCharacterTextSplitter" pattern used in production RAG systems.
"""
import re
from dataclasses import dataclass
from typing import List

from .loaders import Page
from .config import ChunkConfig

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


def _split_text(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # Hard split as last resort.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_seps = separators[0], separators[1:]
    parts = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest_seps))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append((tail + " " + cur).strip())
    return out


def chunk_pages(pages: List[Page], cfg: ChunkConfig) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for page in pages:
        clean = re.sub(r"[ \t]+", " ", page.text).strip()
        raw_pieces = _split_text(clean, cfg.chunk_size, SEPARATORS)
        raw_pieces = _add_overlap(raw_pieces, cfg.chunk_overlap)
        for piece in raw_pieces:
            piece = piece.strip()
            if len(piece) < cfg.min_chunk_size:
                continue
            chunks.append(Chunk(
                id=f"{page.source}::p{page.page_number}::c{idx}",
                text=piece,
                source=page.source,
                page_number=page.page_number,
                chunk_index=idx,
            ))
            idx += 1
    return chunks
