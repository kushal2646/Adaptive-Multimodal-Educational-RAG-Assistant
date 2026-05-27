"""
core/retrieval_router.py
========================
STEP 5 — Retrieval Strategy Selection

Adaptive decision engine that selects the optimal retrieval strategy
based on query intent, content type, and database state.
"""

from dataclasses import dataclass

from core.intent_classifier import QueryIntent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalPlan:
    """The selected retrieval strategy and parameters."""
    strategy: str              # "none" | "semantic" | "keyword" | "hybrid" | "metadata"
    use_semantic: bool
    use_bm25: bool
    use_metadata_filter: bool
    subject_filter: str = ""
    chapter_filter: str = ""
    top_k: int = 8
    reason: str = ""


def select_retrieval_strategy(
    intent: QueryIntent,
    has_documents: bool = True,
    force_strategy: str = "",
) -> RetrievalPlan:
    """
    Adaptively select the optimal retrieval strategy.

    Decision logic:
      - Conversational queries → no retrieval
      - Coding / math with no subject hint → direct LLM
      - Research / conceptual → hybrid retrieval
      - Factual → semantic retrieval
      - Summarization → semantic retrieval
      - Metadata signals present → metadata filter + semantic

    Args:
        intent: Classified query intent
        has_documents: Whether the vector DB has documents
        force_strategy: Override strategy (optional)

    Returns:
        RetrievalPlan with selected strategy and parameters
    """
    from config.settings import TOP_K_RESULTS

    # ── Force override ────────────────────────────────────────────────────────
    if force_strategy and force_strategy != "auto":
        plan = RetrievalPlan(
            strategy=force_strategy,
            use_semantic=force_strategy in ("semantic", "hybrid"),
            use_bm25=force_strategy in ("keyword", "hybrid"),
            use_metadata_filter=force_strategy == "metadata",
            top_k=TOP_K_RESULTS,
            reason=f"User-forced strategy: {force_strategy}",
        )
        logger.info(f"Retrieval strategy forced: {force_strategy}")
        return plan

    # ── No documents in DB → skip retrieval ──────────────────────────────────
    if not has_documents:
        logger.info("No documents in DB — skipping retrieval.")
        return RetrievalPlan(
            strategy="none",
            use_semantic=False,
            use_bm25=False,
            use_metadata_filter=False,
            reason="No documents ingested yet",
        )

    # ── Conversational → no retrieval ────────────────────────────────────────
    if intent.intent == "general_conversational":
        return RetrievalPlan(
            strategy="none",
            use_semantic=False,
            use_bm25=False,
            use_metadata_filter=False,
            reason="Conversational query — no retrieval needed",
        )

    # ── Pure coding without subject context → no retrieval ───────────────────
    if intent.intent == "coding_programming" and not intent.subject_hint:
        return RetrievalPlan(
            strategy="none",
            use_semantic=False,
            use_bm25=False,
            use_metadata_filter=False,
            reason="Standalone coding query — answered directly by LLM",
        )

    # ── Research / Conceptual → full hybrid ──────────────────────────────────
    if intent.intent in ("research_oriented", "conceptual_deep_learning", "comparative_analysis"):
        return RetrievalPlan(
            strategy="hybrid",
            use_semantic=True,
            use_bm25=True,
            use_metadata_filter=bool(intent.subject_hint),
            subject_filter=intent.subject_hint,
            top_k=TOP_K_RESULTS,
            reason="Deep/research query — hybrid semantic + keyword retrieval",
        )

    # ── Factual / Quiz / Summarization → semantic ─────────────────────────────
    if intent.intent in ("factual_educational", "summarization", "quiz_generation"):
        return RetrievalPlan(
            strategy="semantic",
            use_semantic=True,
            use_bm25=False,
            use_metadata_filter=bool(intent.subject_hint),
            subject_filter=intent.subject_hint,
            top_k=TOP_K_RESULTS,
            reason="Factual/summarization — semantic retrieval",
        )

    # ── Mathematical → hybrid (formulas need exact keyword match too) ─────────
    if intent.intent == "mathematical_reasoning":
        return RetrievalPlan(
            strategy="hybrid",
            use_semantic=True,
            use_bm25=True,
            use_metadata_filter=bool(intent.subject_hint),
            subject_filter=intent.subject_hint,
            top_k=TOP_K_RESULTS,
            reason="Mathematical query — hybrid retrieval for formulas",
        )

    # ── Default → hybrid ─────────────────────────────────────────────────────
    return RetrievalPlan(
        strategy="hybrid",
        use_semantic=True,
        use_bm25=True,
        use_metadata_filter=bool(intent.subject_hint),
        subject_filter=intent.subject_hint,
        top_k=TOP_K_RESULTS,
        reason="Default hybrid retrieval strategy",
    )
