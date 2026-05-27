"""
core/reranker.py
================
STEP 8 — Re-Ranking

Re-ranks retrieved candidate chunks based on:
  - Semantic similarity to query
  - Educational relevance signals
  - Content quality heuristics
  - Contextual continuity
"""

import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def _content_quality_score(text: str) -> float:
    """
    Heuristic quality score for educational content.
    Higher = better quality chunk.

    Signals:
      - Length (not too short, not too long)
      - Sentence count
      - Presence of educational keywords
    """
    words = text.split()
    word_count = len(words)

    # Penalize very short or very long chunks
    if word_count < 20:
        length_score = 0.3
    elif word_count < 50:
        length_score = 0.7
    elif word_count <= 300:
        length_score = 1.0
    else:
        length_score = 0.8

    # Reward educational content signals
    edu_keywords = {
        "definition", "example", "theorem", "formula", "algorithm",
        "concept", "principle", "important", "note", "key", "step",
        "because", "therefore", "however", "thus", "hence",
        "summarize", "explain", "analyze",
    }
    text_lower = text.lower()
    keyword_hits = sum(1 for kw in edu_keywords if kw in text_lower)
    keyword_score = min(1.0, keyword_hits / 3)

    return 0.6 * length_score + 0.4 * keyword_score


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    rewritten_query: str = "",
) -> list[dict]:
    """
    Re-rank retrieved candidates using cross-query semantic similarity.

    Args:
        query: Original user query
        candidates: Retrieved chunk dicts
        top_n: Number of top chunks to return after re-ranking
        rewritten_query: Optional expanded query for better comparison

    Returns:
        Top-n re-ranked chunks with final_score field
    """
    from config.settings import RERANK_TOP_N
    top_n = top_n or RERANK_TOP_N

    if not candidates:
        return []

    if len(candidates) <= 1:
        candidates[0]["final_score"] = candidates[0].get("fused_score", 1.0)
        return candidates[:top_n]

    model = _get_embedding_model()

    # Encode query (use rewritten if available)
    query_text = rewritten_query if rewritten_query else query
    texts_to_encode = [query_text] + [c["content"] for c in candidates]
    embeddings = model.encode(texts_to_encode, batch_size=32, show_progress_bar=False)

    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    # Compute cosine similarities
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    chunk_norms = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
    similarities = chunk_norms @ query_norm  # shape: (n_chunks,)

    # Compute final score: semantic similarity + quality + retrieval score
    for i, chunk in enumerate(candidates):
        sim_score = float(similarities[i])
        quality = _content_quality_score(chunk.get("content", ""))
        retrieval_score = chunk.get("fused_score", 0.5)

        # Weighted final score
        final = 0.5 * sim_score + 0.3 * quality + 0.2 * retrieval_score
        chunk["rerank_similarity"] = sim_score
        chunk["quality_score"] = quality
        chunk["final_score"] = final

    # Sort by final score descending
    reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    selected = reranked[:top_n]

    logger.info(
        f"Re-ranking: {len(candidates)} → {len(selected)} chunks. "
        f"Top score: {selected[0]['final_score']:.3f}"
    )
    return selected
