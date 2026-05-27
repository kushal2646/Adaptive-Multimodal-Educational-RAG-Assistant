"""
core/context_compressor.py
===========================
STEP 9 — Metadata Filtering / Context Compression

Performs:
  - Duplicate removal (exact + near-duplicate)
  - Irrelevant chunk filtering below quality threshold
  - Token-aware context compression
  - Final chunk ordering for optimal LLM reading
"""

import re
from utils.logger import get_logger

logger = get_logger(__name__)

# Approximate token ratio (words to tokens)
WORDS_PER_TOKEN = 0.75


def _word_count(text: str) -> int:
    return len(text.split())


def _approx_tokens(text: str) -> int:
    return int(_word_count(text) / WORDS_PER_TOKEN)


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    """
    Check if two texts are near-duplicates using word overlap (Jaccard similarity).
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return False
    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union)
    return jaccard >= threshold


def _normalize_whitespace(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def compress_context(
    chunks: list[dict],
    max_tokens: int = 6000,
    min_quality_score: float = 0.0,
) -> list[dict]:
    """
    Compress and clean the retrieved context.

    Steps:
      1. Filter by minimum quality score
      2. Remove exact duplicates
      3. Remove near-duplicates
      4. Trim to max token budget
      5. Normalize whitespace

    Args:
        chunks: Re-ranked chunk dicts
        max_tokens: Maximum total tokens allowed in context
        min_quality_score: Minimum quality score threshold

    Returns:
        Compressed list of chunk dicts
    """
    from config.settings import MAX_CONTEXT_TOKENS
    max_tokens = max_tokens or MAX_CONTEXT_TOKENS

    if not chunks:
        return []

    # Step 1: Filter by quality score
    filtered = [c for c in chunks if c.get("final_score", 1.0) >= min_quality_score]
    if not filtered:
        filtered = chunks  # Don't filter out everything

    # Step 2: Remove exact duplicates
    seen_contents: set[str] = set()
    deduped = []
    for chunk in filtered:
        content_key = chunk.get("content", "").strip().lower()[:200]
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            deduped.append(chunk)

    # Step 3: Remove near-duplicates
    final_chunks = []
    for chunk in deduped:
        content = chunk.get("content", "")
        is_dup = any(
            _is_near_duplicate(content, fc.get("content", ""))
            for fc in final_chunks
        )
        if not is_dup:
            final_chunks.append(chunk)

    # Step 4: Trim to token budget
    total_tokens = 0
    budget_chunks = []
    for chunk in final_chunks:
        content = chunk.get("content", "")
        chunk_tokens = _approx_tokens(content)
        if total_tokens + chunk_tokens <= max_tokens:
            # Clean whitespace
            chunk = chunk.copy()
            chunk["content"] = _normalize_whitespace(content)
            budget_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Try to fit a truncated version
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:  # Worth including
                words = content.split()
                target_words = int(remaining_tokens * WORDS_PER_TOKEN)
                truncated = " ".join(words[:target_words]) + "..."
                chunk = chunk.copy()
                chunk["content"] = _normalize_whitespace(truncated)
                chunk["truncated"] = True
                budget_chunks.append(chunk)
            break

    logger.info(
        f"Context compression: {len(chunks)} → {len(budget_chunks)} chunks "
        f"(~{total_tokens} tokens)"
    )
    return budget_chunks


def format_context_block(chunks: list[dict]) -> str:
    """
    Format compressed chunks into a single context string for the LLM prompt.
    """
    if not chunks:
        return "No relevant context found in the knowledge base."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("filename", "")
        subject = chunk.get("subject", "")
        page = chunk.get("page_num")

        header_parts = [f"[Source {i}]"]
        if source:
            header_parts.append(f"📄 {source}")
        if subject:
            header_parts.append(f"| {subject}")
        if page:
            header_parts.append(f"| Page {page}")

        header = " ".join(header_parts)
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)
