"""
core/hybrid_retriever.py
========================
STEPS 5/6 — Hybrid Retrieval Engine

Combines:
  1. Semantic search via pgvector (cosine similarity)
  2. BM25 keyword search (rank_bm25)
  3. Metadata filtering
  4. Score fusion (weighted combination)

Returns merged, deduplicated ranked candidate chunks.
"""

import numpy as np
from typing import Optional

from core.retrieval_router import RetrievalPlan
from database.vector_store import semantic_search, get_all_chunks_text, metadata_filter_search
from utils.logger import get_logger

logger = get_logger(__name__)

# Cached BM25 corpus to avoid rebuilding on every query
_bm25_cache: dict = {"corpus": None, "model": None, "chunk_ids": None}


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def _build_bm25_index():
    """Build BM25 index from all stored chunks."""
    from rank_bm25 import BM25Okapi

    logger.info("Building BM25 index from stored chunks...")
    rows = get_all_chunks_text()

    if not rows:
        logger.warning("No chunks in DB — BM25 index empty.")
        return None, [], []

    corpus = [row["content"] for row in rows]
    chunk_ids = [row["id"] for row in rows]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    _bm25_cache["corpus"] = corpus
    _bm25_cache["model"] = bm25
    _bm25_cache["chunk_ids"] = chunk_ids

    logger.info(f"BM25 index built with {len(corpus)} documents.")
    return bm25, corpus, chunk_ids


def _get_bm25_index():
    """Get or build the BM25 index."""
    if _bm25_cache["model"] is None:
        return _build_bm25_index()
    return _bm25_cache["model"], _bm25_cache["corpus"], _bm25_cache["chunk_ids"]


def invalidate_bm25_cache():
    """Call this after new documents are ingested."""
    _bm25_cache["corpus"] = None
    _bm25_cache["model"] = None
    _bm25_cache["chunk_ids"] = None
    logger.debug("BM25 cache invalidated.")


def bm25_search(
    query: str,
    top_k: int = 8,
) -> list[dict]:
    """
    BM25 keyword search over all stored chunks.

    Returns list of chunk dicts with bm25_score field.
    """
    bm25, corpus, chunk_ids = _get_bm25_index()

    if bm25 is None or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "id": chunk_ids[idx],
                "content": corpus[idx],
                "bm25_score": float(scores[idx]),
                "source": "bm25",
            })

    return results


def retrieve(
    query: str,
    plan: RetrievalPlan,
    rewritten_queries=None,
) -> list[dict]:
    """
    Execute the retrieval plan and return merged, fused candidates.

    Args:
        query: Original query
        plan: RetrievalPlan from the router
        rewritten_queries: RewrittenQuery object (optional, for better retrieval)

    Returns:
        List of candidate chunk dicts with merged scores
    """
    from config.settings import SEMANTIC_WEIGHT, BM25_WEIGHT

    if plan.strategy == "none":
        return []

    semantic_results: list[dict] = []
    bm25_results: list[dict] = []

    # ── Semantic retrieval ───────────────────────────────────────────────────
    if plan.use_semantic:
        model = _get_embedding_model()

        # Use primary rewritten query for embedding if available
        embed_text = query
        if rewritten_queries:
            embed_text = rewritten_queries.primary

        logger.info(f"Running semantic search: '{embed_text[:60]}...'")
        query_embedding = model.encode([embed_text])[0].tolist()

        semantic_results = semantic_search(
            query_embedding=query_embedding,
            top_k=plan.top_k,
            subject=plan.subject_filter or None,
            chapter=plan.chapter_filter or None,
        )
        logger.info(f"Semantic search returned {len(semantic_results)} chunks.")

    # ── BM25 keyword retrieval ───────────────────────────────────────────────
    if plan.use_bm25:
        keyword_text = query
        if rewritten_queries:
            keyword_text = rewritten_queries.keyword

        logger.info(f"Running BM25 search: '{keyword_text[:60]}'")
        bm25_results = bm25_search(query=keyword_text, top_k=plan.top_k)
        logger.info(f"BM25 search returned {len(bm25_results)} chunks.")

    # ── Merge & Fuse scores ──────────────────────────────────────────────────
    merged: dict[int, dict] = {}

    # Normalize BM25 scores to [0, 1]
    if bm25_results:
        max_bm25 = max(r["bm25_score"] for r in bm25_results) or 1.0
        for r in bm25_results:
            chunk_id = r["id"]
            r["normalized_bm25"] = r["bm25_score"] / max_bm25
            r["fused_score"] = BM25_WEIGHT * r["normalized_bm25"]
            merged[chunk_id] = r

    # Add/update semantic scores
    for r in semantic_results:
        chunk_id = r["id"]
        semantic_score = float(r.get("similarity_score", 0))
        r["fused_score"] = SEMANTIC_WEIGHT * semantic_score

        if chunk_id in merged:
            merged[chunk_id]["fused_score"] += SEMANTIC_WEIGHT * semantic_score
            merged[chunk_id]["similarity_score"] = semantic_score
        else:
            r["source"] = "semantic"
            merged[chunk_id] = r

    # Sort by fused score
    candidates = sorted(merged.values(), key=lambda x: x.get("fused_score", 0), reverse=True)
    candidates = candidates[:plan.top_k]

    logger.info(f"Hybrid retrieval: {len(candidates)} fused candidates.")
    return candidates
