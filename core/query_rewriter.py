"""
core/query_rewriter.py
======================
STEP 4 — Query Rewriting / Expansion

Uses Groq LLM to:
  - Expand abbreviations
  - Add missing educational context
  - Generate semantic variants
  - Create BM25-optimized keyword queries
"""

from dataclasses import dataclass

from config.settings import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewrittenQuery:
    """Holds the original + rewritten query variants."""
    original: str
    primary: str          # main rewritten query for embedding
    semantic: str         # semantically expanded for vector search
    keyword: str          # keyword-dense variant for BM25
    hybrid: str           # combined for hybrid search


def _get_groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


_REWRITE_SYSTEM_PROMPT = """You are an expert educational query optimizer for a RAG system.
Your task is to rewrite and expand student/user queries into optimized retrieval queries.

You MUST respond ONLY with a valid JSON object in this EXACT format:
{
  "primary": "Clear, grammatically correct, context-rich version of the query",
  "semantic": "Semantically expanded version with related concepts and synonyms",
  "keyword": "Space-separated keyword-rich version with important terms",
  "hybrid": "A combined query using both semantic meaning and key terms"
}

Rules:
- Expand abbreviations (e.g., OS → Operating System, ML → Machine Learning)
- Add educational context if the query is vague
- Never answer the question, only rewrite it
- Keep all variants under 100 words each
- Preserve the original educational intent
"""


def rewrite_query(
    query: str,
    intent: str = "",
    subject_hint: str = "",
) -> RewrittenQuery:
    """
    Rewrite and expand user query for optimal retrieval.

    Args:
        query: Original user query
        intent: Detected intent label
        subject_hint: Detected subject (optional)

    Returns:
        RewrittenQuery with multiple variants
    """
    if not query.strip():
        return RewrittenQuery(
            original=query,
            primary=query,
            semantic=query,
            keyword=query,
            hybrid=query,
        )

    context_hint = ""
    if subject_hint:
        context_hint = f"\nSubject context: {subject_hint}"
    if intent:
        context_hint += f"\nQuery intent: {intent.replace('_', ' ')}"

    user_prompt = f"""Rewrite this educational query:{context_hint}

Query: "{query}"

Return ONLY the JSON object, no explanation."""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON
        import json
        # Extract JSON if wrapped in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        result = RewrittenQuery(
            original=query,
            primary=data.get("primary", query),
            semantic=data.get("semantic", query),
            keyword=data.get("keyword", query),
            hybrid=data.get("hybrid", query),
        )
        logger.debug(f"Query rewritten: '{query[:50]}...' → '{result.primary[:50]}...'")
        return result

    except Exception as e:
        logger.warning(f"Query rewrite failed ({e}), using original query.")
        return RewrittenQuery(
            original=query,
            primary=query,
            semantic=query,
            keyword=query,
            hybrid=query,
        )
