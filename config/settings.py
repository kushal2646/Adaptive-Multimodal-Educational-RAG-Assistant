"""
config/settings.py
==================
Centralized configuration management.
Loads all settings from environment variables (.env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


# ── GROQ LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

# ── DATABASE ──────────────────────────────────────────────────────────────────
NEON_DATABASE_URL: str = os.getenv("NEON_DATABASE_URL", "")

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "8"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))

# ── CHUNKING ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── WHISPER ───────────────────────────────────────────────────────────────────
WHISPER_MODE: str = os.getenv("WHISPER_MODE", "groq")        # "groq" | "local"
WHISPER_LOCAL_MODEL: str = os.getenv("WHISPER_LOCAL_MODEL", "base")

# ── APP ───────────────────────────────────────────────────────────────────────
APP_TITLE: str = os.getenv("APP_TITLE", "EduRAG Assistant")
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def validate_config() -> list[str]:
    """
    Validate that all required config values are set.
    Returns a list of missing variable names.
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not NEON_DATABASE_URL:
        missing.append("NEON_DATABASE_URL")
    return missing


# Intent classification labels
INTENT_LABELS = [
    "general_conversational",
    "factual_educational",
    "conceptual_deep_learning",
    "retrieval_required",
    "multimodal",
    "summarization",
    "comparative_analysis",
    "research_oriented",
    "coding_programming",
    "mathematical_reasoning",
]

# Expertise levels
EXPERTISE_LEVELS = ["beginner", "intermediate", "advanced"]

# Retrieval strategies
RETRIEVAL_STRATEGIES = [
    "semantic",
    "keyword",
    "hybrid",
    "metadata",
    "multimodal",
    "none",
]
