"""
core/intent_classifier.py
==========================
STEP 2 — Query Understanding / Intent Classification

Classifies:
  - Query intent (10 types)
  - User expertise level (beginner / intermediate / advanced)
  - Urgency, ambiguity, and hidden intent flags
"""

import re
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryIntent:
    """Result of intent classification."""
    intent: str                        # primary intent label
    confidence: float                  # 0.0 – 1.0
    expertise: str                     # beginner | intermediate | advanced
    requires_retrieval: bool
    is_multimodal: bool
    is_ambiguous: bool
    is_mathematical: bool
    is_coding: bool
    subject_hint: str = ""             # extracted subject if any
    chapter_hint: str = ""
    secondary_intents: list[str] = field(default_factory=list)


# ── KEYWORD SIGNAL BANKS ───────────────────────────────────────────────────────

_CONVERSATIONAL_SIGNALS = {
    "hello", "hi", "hey", "how are you", "what's up", "thanks", "thank you",
    "bye", "goodbye", "who are you", "what can you do", "help me",
    "good morning", "good evening",
}

_FACTUAL_SIGNALS = {
    "what is", "define", "who is", "when did", "where is",
    "what does", "meaning of", "tell me about", "give me info",
}

_CONCEPTUAL_SIGNALS = {
    "explain", "how does", "why does", "how do", "describe", "elaborate",
    "concept of", "principle of", "theory of", "what happens when",
    "mechanism", "intuition behind",
}

_SUMMARIZATION_SIGNALS = {
    "summarize", "summary", "tldr", "brief", "key points", "main ideas",
    "overview", "recap", "gist", "shorten",
}

_COMPARATIVE_SIGNALS = {
    "compare", "difference between", "vs", "versus", "similarities",
    "contrast", "which is better", "pros and cons", "advantages",
}

_RESEARCH_SIGNALS = {
    "research", "paper", "study", "literature", "journal", "findings",
    "hypothesis", "methodology", "experiment", "cite", "reference",
    "state of the art", "recent advances",
}

_CODING_SIGNALS = {
    "code", "program", "function", "debug", "error", "syntax",
    "implement", "algorithm", "script", "class", "method", "compile",
    "runtime", "python", "java", "c++", "javascript", "sql", "api",
    "library", "framework", "bug", "fix", "output", "input",
}

_MATHEMATICAL_SIGNALS = {
    "calculate", "solve", "equation", "formula", "proof", "theorem",
    "integral", "derivative", "matrix", "vector", "probability",
    "statistics", "compute", "find the value", "simplify", "factor",
}

_QUIZ_SIGNALS = {
    "quiz", "test", "mcq", "multiple choice", "flashcard", "practice",
    "question", "exam", "assignment", "homework", "exercise",
}

_BEGINNER_SIGNALS = {
    "simple", "easy", "basic", "beginner", "explain like", "eli5",
    "what is", "introduction", "fundamentals", "start", "first time",
}

_ADVANCED_SIGNALS = {
    "advanced", "deep dive", "formal", "rigorous", "proof",
    "complexity", "optimization", "architecture", "internals",
    "benchmark", "tradeoff", "state of the art",
}


def _matches(text: str, signals: set) -> bool:
    """Check if any signal keyword appears in text."""
    text_lower = text.lower()
    return any(sig in text_lower for sig in signals)


def _count_matches(text: str, signals: set) -> int:
    text_lower = text.lower()
    return sum(1 for sig in signals if sig in text_lower)


def classify_intent(query: str) -> QueryIntent:
    """
    Classify query intent and extract educational signals.

    Args:
        query: Raw user query string

    Returns:
        QueryIntent dataclass with all classification results
    """
    q = query.strip()
    q_lower = q.lower()

    # ── Conversational check ─────────────────────────────────────────────────
    if _matches(q_lower, _CONVERSATIONAL_SIGNALS) and len(q.split()) < 12:
        return QueryIntent(
            intent="general_conversational",
            confidence=0.92,
            expertise="intermediate",
            requires_retrieval=False,
            is_multimodal=False,
            is_ambiguous=False,
            is_mathematical=False,
            is_coding=False,
        )

    # ── Scoring each intent ──────────────────────────────────────────────────
    scores = {
        "factual_educational":     _count_matches(q_lower, _FACTUAL_SIGNALS),
        "conceptual_deep_learning": _count_matches(q_lower, _CONCEPTUAL_SIGNALS),
        "summarization":           _count_matches(q_lower, _SUMMARIZATION_SIGNALS),
        "comparative_analysis":    _count_matches(q_lower, _COMPARATIVE_SIGNALS),
        "research_oriented":       _count_matches(q_lower, _RESEARCH_SIGNALS),
        "coding_programming":      _count_matches(q_lower, _CODING_SIGNALS),
        "mathematical_reasoning":  _count_matches(q_lower, _MATHEMATICAL_SIGNALS),
        "quiz_generation":         _count_matches(q_lower, _QUIZ_SIGNALS),
    }

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Default to retrieval_required if nothing matches well
    if best_score == 0:
        best_intent = "retrieval_required"
        confidence = 0.6
    else:
        confidence = min(0.95, 0.6 + best_score * 0.1)

    secondary = [k for k, v in scores.items() if v > 0 and k != best_intent]

    # ── Expertise detection ──────────────────────────────────────────────────
    if _matches(q_lower, _ADVANCED_SIGNALS):
        expertise = "advanced"
    elif _matches(q_lower, _BEGINNER_SIGNALS):
        expertise = "beginner"
    else:
        expertise = "intermediate"

    # ── Special flags ────────────────────────────────────────────────────────
    is_coding = best_intent == "coding_programming" or _matches(q_lower, _CODING_SIGNALS)
    is_math = best_intent == "mathematical_reasoning" or _matches(q_lower, _MATHEMATICAL_SIGNALS)
    is_ambiguous = len(q.split()) < 5 or q.endswith("?") and len(q.split()) < 4

    # Retrieval not required for purely conversational or standalone coding
    no_retrieval_intents = {"general_conversational"}
    requires_retrieval = best_intent not in no_retrieval_intents

    # ── Subject extraction (simple heuristic) ────────────────────────────────
    subject_hint = ""
    subjects_map = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "math": "Mathematics",
        "mathematics": "Mathematics",
        "history": "History",
        "geography": "Geography",
        "computer science": "Computer Science",
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "economics": "Economics",
        "english": "English",
        "science": "Science",
    }
    for keyword, label in subjects_map.items():
        if keyword in q_lower:
            subject_hint = label
            break

    logger.debug(
        f"Intent: {best_intent} ({confidence:.2f}), Expertise: {expertise}, "
        f"Retrieval: {requires_retrieval}"
    )

    return QueryIntent(
        intent=best_intent,
        confidence=confidence,
        expertise=expertise,
        requires_retrieval=requires_retrieval,
        is_multimodal=False,
        is_ambiguous=is_ambiguous,
        is_mathematical=is_math,
        is_coding=is_coding,
        subject_hint=subject_hint,
        secondary_intents=secondary[:3],
    )
