"""
core/prompt_assembler.py
=========================
STEP 10 — Context Assembly / Prompt Augmentation

Builds the final LLM prompt by combining:
  - User query
  - Rewritten query
  - Retrieved + compressed context
  - Expertise-level instructions
  - Response format requirements
  - Hallucination prevention rules
"""

from core.intent_classifier import QueryIntent
from utils.logger import get_logger

logger = get_logger(__name__)


# ── SYSTEM PROMPTS BY EXPERTISE ───────────────────────────────────────────────

_SYSTEM_BEGINNER = """You are EduRAG, an expert AI tutor specialized in making complex topics easy to understand.

Your student is a BEGINNER. Follow these guidelines STRICTLY:
- Use very simple language, avoid jargon
- Use everyday analogies and real-world examples
- Explain step-by-step, assuming no prior knowledge
- Use bullet points and numbered lists for clarity
- Define technical terms when you first use them
- Be encouraging and supportive in tone

HALLUCINATION PREVENTION:
- Base your answer ONLY on the provided context
- If context is insufficient, say: "I don't have enough information on this in the knowledge base."
- Never invent facts, formulas, or citations
"""

_SYSTEM_INTERMEDIATE = """You are EduRAG, an expert AI tutor and educational research assistant.

Your student has INTERMEDIATE knowledge. Follow these guidelines:
- Use clear technical language with appropriate depth
- Include conceptual explanations with practical examples
- Show connections between concepts
- Use structured formatting (headers, bullets, numbered lists)
- Include relevant formulas or code when appropriate

HALLUCINATION PREVENTION:
- Ground all claims in the provided context
- If context is insufficient, explicitly state limitations
- Do not fabricate research references or statistics
"""

_SYSTEM_ADVANCED = """You are EduRAG, an expert AI tutor for advanced learners.

Your student is ADVANCED. Follow these guidelines:
- Use formal, precise technical language
- Provide rigorous explanations with mathematical depth where appropriate
- Discuss tradeoffs, edge cases, and nuances
- Reference theoretical foundations
- Go beyond surface-level explanations

HALLUCINATION PREVENTION:
- All claims must be grounded in the provided context
- Clearly distinguish between retrieved knowledge and general reasoning
- Do not fabricate papers, benchmarks, or specific numerical claims not in context
"""

_SYSTEM_CODING = """You are EduRAG, an expert programming tutor.

Guidelines:
- Provide clean, well-commented, working code examples
- Explain each step of the code
- Mention time/space complexity where relevant
- Suggest best practices and common pitfalls
- Format all code in proper code blocks with language specifiers

HALLUCINATION PREVENTION:
- Only write code you are confident is correct
- If referencing library APIs, note the version/documentation
- Base factual claims on provided context
"""

_SYSTEM_MATH = """You are EduRAG, an expert mathematics tutor.

Guidelines:
- Show complete step-by-step solutions
- Explain the reasoning behind each step
- Use proper mathematical notation in plain text (e.g., x^2, sqrt(x))
- Provide alternative solution methods when useful
- Verify final answers where possible

HALLUCINATION PREVENTION:
- Do not invent theorems or formulas not in the context
- Always verify that formulas from context are applied correctly
"""


# ── RESPONSE FORMAT TEMPLATE ──────────────────────────────────────────────────

_RESPONSE_FORMAT = """
Respond using this structure:

# [Topic Title]

## 📖 Explanation
[Your main educational explanation]

## 🔑 Key Concepts
- [Key point 1]
- [Key point 2]
- [Key point 3]

## 💡 Example
[Practical example, code, or worked problem]

## 📝 Summary
[Concise 2-3 sentence summary for revision]

## 📚 References
[Mention source document names if context was used, otherwise omit this section]
"""

_RESPONSE_FORMAT_CONVERSATIONAL = """
Respond naturally and helpfully. Keep your response concise and friendly.
"""


def build_system_prompt(intent: QueryIntent) -> str:
    """Select and return the appropriate system prompt."""
    if intent.intent == "general_conversational":
        return (
            "You are EduRAG, a friendly AI educational assistant. "
            "Answer conversationally and helpfully."
        )
    if intent.is_coding:
        return _SYSTEM_CODING
    if intent.is_mathematical:
        return _SYSTEM_MATH

    expertise_prompts = {
        "beginner": _SYSTEM_BEGINNER,
        "intermediate": _SYSTEM_INTERMEDIATE,
        "advanced": _SYSTEM_ADVANCED,
    }
    return expertise_prompts.get(intent.expertise, _SYSTEM_INTERMEDIATE)


def assemble_prompt(
    query: str,
    intent: QueryIntent,
    context_block: str = "",
    rewritten_query: str = "",
    chat_history: list[dict] = None,
) -> list[dict]:
    """
    Assemble the full message list for the LLM.

    Args:
        query: Original user query
        intent: QueryIntent from classifier
        context_block: Formatted context string from compressor
        rewritten_query: Rewritten query (for LLM transparency)
        chat_history: Previous messages [{role, content}]

    Returns:
        List of message dicts for Groq API
    """
    system_prompt = build_system_prompt(intent)

    # Add response format to system prompt (except conversational)
    if intent.intent != "general_conversational":
        system_prompt += _RESPONSE_FORMAT

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Add condensed chat history (last 6 turns)
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    # ── User message with context ─────────────────────────────────────────────
    user_parts = []

    if context_block and "No relevant context" not in context_block:
        user_parts.append(
            "## 📚 Retrieved Educational Context\n\n"
            + context_block
            + "\n\n---"
        )

    if rewritten_query and rewritten_query != query:
        user_parts.append(f"**Optimized query:** {rewritten_query}")

    user_parts.append(f"**Question:** {query}")

    if context_block and "No relevant context" not in context_block:
        user_parts.append(
            "\n⚠️ IMPORTANT: Base your answer on the retrieved context above. "
            "If the context doesn't contain enough information, clearly state that."
        )

    user_message = "\n\n".join(user_parts)
    messages.append({"role": "user", "content": user_message})

    logger.debug(f"Assembled prompt: {len(messages)} messages, context length: {len(context_block)}")
    return messages
