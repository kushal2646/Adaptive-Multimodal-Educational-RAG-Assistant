"""
core/answer_validator.py
=========================
STEP 12 — Answer Validation / Citation Check

Validates:
  - Factual grounding (is answer based on context?)
  - Hallucination risk signals
  - Retrieval relevance
  - Educational accuracy signals

Triggers retry if confidence is low.
"""

import re
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_valid: bool
    confidence: float          # 0.0 – 1.0
    needs_retry: bool
    issues: list[str]
    grounding_score: float     # How well grounded in context (0-1)
    hallucination_risk: str    # "low" | "medium" | "high"


# ── HALLUCINATION RISK SIGNALS ────────────────────────────────────────────────

_HIGH_CONFIDENCE_SIGNALS = {
    "according to", "the document states", "as mentioned",
    "based on the context", "the source indicates", "retrieved context",
    "the text explains", "as described",
}

_UNCERTAINTY_SIGNALS = {
    "i believe", "i think", "probably", "i'm not sure",
    "i cannot confirm", "insufficient information",
    "not available in the knowledge base", "i don't have",
}

_HALLUCINATION_SIGNALS = {
    "studies show that", "research has proven", "according to experts",
    "it is widely known", "universally accepted", "scientifically proven",
    "in 20", "% of cases", "according to [",  # fake citations pattern
}


def _check_grounding(answer: str, context_block: str) -> float:
    """
    Estimate how grounded the answer is in the retrieved context.
    Uses word overlap between answer and context.
    """
    if not context_block or "No relevant context" in context_block:
        return 0.5  # No context to judge against

    answer_words = set(answer.lower().split())
    context_words = set(context_block.lower().split())

    # Remove stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or",
        "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "this", "that", "it", "its", "be", "have", "has", "had",
    }
    answer_words -= stop_words
    context_words -= stop_words

    if not answer_words:
        return 0.5

    overlap = answer_words & context_words
    score = len(overlap) / max(len(answer_words), 1)
    return min(1.0, score * 2)  # Scale up (full overlap rarely happens)


def _detect_hallucination_risk(answer: str, has_context: bool) -> str:
    """Detect hallucination risk level."""
    answer_lower = answer.lower()

    high_confidence_hits = sum(1 for s in _HIGH_CONFIDENCE_SIGNALS if s in answer_lower)
    hallucination_hits = sum(1 for s in _HALLUCINATION_SIGNALS if s in answer_lower)
    uncertainty_hits = sum(1 for s in _UNCERTAINTY_SIGNALS if s in answer_lower)

    if not has_context:
        # No retrieval used — higher risk if making specific factual claims
        if hallucination_hits > 0:
            return "high"
        return "medium"

    if hallucination_hits > 1:
        return "high"
    elif hallucination_hits == 1 and high_confidence_hits == 0:
        return "medium"
    elif uncertainty_hits > 0 or high_confidence_hits > 0:
        return "low"
    else:
        return "low"


def validate_answer(
    answer: str,
    query: str,
    context_block: str = "",
    intent: str = "",
    attempt: int = 1,
) -> ValidationResult:
    """
    Validate the generated answer.

    Args:
        answer: Generated LLM response
        query: Original user query
        context_block: Retrieved context used
        intent: Query intent label
        attempt: Current attempt number (for retry logic)

    Returns:
        ValidationResult
    """
    issues = []

    # ── Basic checks ──────────────────────────────────────────────────────────
    if not answer or len(answer.strip()) < 20:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            needs_retry=attempt < 2,
            issues=["Answer is too short or empty"],
            grounding_score=0.0,
            hallucination_risk="high",
        )

    # ── Error response check ──────────────────────────────────────────────────
    if answer.strip().startswith("❌"):
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            needs_retry=False,
            issues=["LLM generation error"],
            grounding_score=0.0,
            hallucination_risk="high",
        )

    has_context = bool(context_block) and "No relevant context" not in context_block

    # ── Grounding score ───────────────────────────────────────────────────────
    grounding_score = _check_grounding(answer, context_block)

    # ── Hallucination risk ────────────────────────────────────────────────────
    hallucination_risk = _detect_hallucination_risk(answer, has_context)

    if hallucination_risk == "high":
        issues.append("Potential hallucination detected (fabricated references or unsupported claims)")

    # ── Relevance check (very basic: does it address the query topic?) ────────
    query_keywords = set(query.lower().split()) - {"what", "how", "why", "explain", "the", "a", "is"}
    answer_lower = answer.lower()
    keyword_coverage = sum(1 for kw in query_keywords if kw in answer_lower)
    relevance_score = min(1.0, keyword_coverage / max(len(query_keywords), 1))

    if relevance_score < 0.2:
        issues.append("Answer may not address the query topic")

    # ── Length check ──────────────────────────────────────────────────────────
    word_count = len(answer.split())
    if word_count < 30 and intent not in ("general_conversational",):
        issues.append("Answer is unusually short for an educational response")

    # ── Compute overall confidence ────────────────────────────────────────────
    confidence = (
        0.4 * grounding_score
        + 0.3 * relevance_score
        + 0.2 * (0.0 if hallucination_risk == "high" else 0.5 if hallucination_risk == "medium" else 1.0)
        + 0.1 * (1.0 if word_count >= 50 else 0.5)
    )

    # ── Determine if retry is needed ──────────────────────────────────────────
    needs_retry = (
        attempt < 2
        and (
            hallucination_risk == "high"
            or (has_context and grounding_score < 0.15)
            or confidence < 0.25
        )
    )

    is_valid = confidence >= 0.2 and hallucination_risk != "high"

    logger.info(
        f"Validation [attempt {attempt}]: confidence={confidence:.2f}, "
        f"grounding={grounding_score:.2f}, risk={hallucination_risk}, "
        f"retry={needs_retry}"
    )

    return ValidationResult(
        is_valid=is_valid,
        confidence=confidence,
        needs_retry=needs_retry,
        issues=issues,
        grounding_score=grounding_score,
        hallucination_risk=hallucination_risk,
    )
