"""
ingestion/chunker.py
====================
Semantic text chunking utilities.
Splits text into overlapping chunks suitable for embedding.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """Represents a single text chunk with metadata."""
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    page_num: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on sentence boundaries while preserving punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_words(text: str) -> int:
    """Count approximate word tokens."""
    return len(text.split())


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    page_num: Optional[int] = None,
    section: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.

    Strategy:
    1. Split by paragraphs first (natural boundaries)
    2. If paragraph is too large, split by sentences
    3. Apply word-count-based sliding window with overlap

    Args:
        text: Raw text content
        chunk_size: Max words per chunk
        chunk_overlap: Words to overlap between consecutive chunks
        page_num: Source page number
        section: Source section/heading
        metadata: Additional metadata

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}
    text = text.strip()

    # Split by paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Flatten paragraphs into word tokens with char position tracking
    words: list[tuple[str, int]] = []  # (word, char_position)

    char_offset = 0
    for para in paragraphs:
        para_words = para.split()
        for word in para_words:
            pos = text.find(word, char_offset)
            if pos == -1:
                pos = char_offset
            words.append((word, pos))
            char_offset = pos + len(word)

    if not words:
        return []

    chunks: list[TextChunk] = []
    chunk_index = 0
    i = 0

    while i < len(words):
        # Take chunk_size words
        end = min(i + chunk_size, len(words))
        chunk_words = words[i:end]

        chunk_text_content = " ".join(w[0] for w in chunk_words)
        char_start = chunk_words[0][1]
        char_end = chunk_words[-1][1] + len(chunk_words[-1][0])

        chunk = TextChunk(
            content=chunk_text_content,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            page_num=page_num,
            section=section,
            metadata=meta.copy(),
        )
        chunks.append(chunk)
        chunk_index += 1

        # Advance with overlap
        i += chunk_size - chunk_overlap

    logger.debug(f"Chunked text into {len(chunks)} chunks.")
    return chunks


def chunk_document_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    base_metadata: Optional[dict] = None,
) -> list[TextChunk]:
    """
    Chunk a multi-page document.

    Args:
        pages: List of dicts with keys: text, page_num, section (optional)
        chunk_size: Max words per chunk
        chunk_overlap: Overlap words
        base_metadata: Metadata applied to all chunks

    Returns:
        All chunks across all pages, with global chunk_index
    """
    all_chunks: list[TextChunk] = []
    global_index = 0

    for page in pages:
        text = page.get("text", "")
        page_num = page.get("page_num")
        section = page.get("section", "")

        page_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_num=page_num,
            section=section,
            metadata=base_metadata,
        )

        for chunk in page_chunks:
            chunk.chunk_index = global_index
            all_chunks.append(chunk)
            global_index += 1

    logger.info(f"Total chunks across document: {len(all_chunks)}")
    return all_chunks
