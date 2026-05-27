"""
ingestion/audio_ingester.py
===========================
Audio/Video ingestion pipeline:
  1. Transcribe using Groq Whisper API (default) or local Whisper
  2. Chunk transcript
  3. Embed + store in Neon
"""

import uuid
import os
from pathlib import Path

from config.settings import WHISPER_MODE, WHISPER_LOCAL_MODEL, GROQ_API_KEY
from ingestion.chunker import chunk_text
from database.vector_store import upsert_document, upsert_chunks
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_FORMATS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv"}


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def transcribe_with_groq(audio_path: str) -> str:
    """Transcribe audio using Groq's Whisper API."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    logger.info(f"Transcribing with Groq Whisper: '{audio_path}'")
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text",
        )
    return transcription


def transcribe_with_local_whisper(audio_path: str) -> str:
    """Transcribe audio using local openai-whisper package."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper\n"
            "Or set WHISPER_MODE=groq in your .env file."
        )

    logger.info(f"Transcribing with local Whisper ({WHISPER_LOCAL_MODEL}): '{audio_path}'")
    model = whisper.load_model(WHISPER_LOCAL_MODEL)
    result = model.transcribe(audio_path)
    return result["text"]


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio/video file using the configured Whisper mode.
    Supports: WHISPER_MODE=groq | local
    """
    if WHISPER_MODE == "groq":
        return transcribe_with_groq(audio_path)
    elif WHISPER_MODE == "local":
        return transcribe_with_local_whisper(audio_path)
    else:
        raise ValueError(f"Unknown WHISPER_MODE: {WHISPER_MODE}. Use 'groq' or 'local'.")


def ingest_audio(
    audio_path: str,
    subject: str = "",
    chapter: str = "",
    topic: str = "",
    source_type: str = "audio",  # "audio" or "video"
    progress_callback=None,
) -> dict:
    """
    Full audio/video ingestion pipeline.

    Args:
        audio_path: Path to audio or video file
        subject/chapter/topic: Metadata tags
        source_type: 'audio' or 'video'
        progress_callback: Optional callable(step, total, message)

    Returns:
        Summary dict
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    doc_id = str(uuid.uuid4())
    filename = path.name

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        logger.info(f"[{step}/{total}] {msg}")

    _progress(1, 4, f"Transcribing '{filename}' using {WHISPER_MODE} Whisper...")
    transcript = transcribe_audio(audio_path)

    if not transcript.strip():
        logger.warning("Empty transcript — audio may be silent or unclear.")
        return {"doc_id": doc_id, "filename": filename, "total_chunks": 0}

    _progress(2, 4, f"Chunking transcript ({len(transcript)} chars)...")
    chunks = chunk_text(
        text=transcript,
        metadata={
            "doc_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "subject": subject,
            "chapter": chapter,
            "topic": topic,
        },
    )

    _progress(3, 4, f"Generating embeddings for {len(chunks)} chunks...")
    model = _get_embedding_model()
    texts = [c.content for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    _progress(4, 4, "Storing transcript in Neon pgvector...")
    chunk_dicts = [
        {
            "doc_id": doc_id,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "embedding": emb.tolist(),
            "source_type": source_type,
            "subject": subject,
            "chapter": chapter,
            "topic": topic,
            "page_num": None,
            "filename": filename,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "metadata": c.metadata,
        }
        for c, emb in zip(chunks, embeddings)
    ]

    upsert_document(
        doc_id=doc_id,
        filename=filename,
        source_type=source_type,
        total_chunks=len(chunks),
        subject=subject,
        chapter=chapter,
        topic=topic,
    )
    upsert_chunks(chunk_dicts)

    logger.info(f"{source_type.capitalize()} '{filename}' ingested: {len(chunks)} chunks.")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "transcript_chars": len(transcript),
    }
