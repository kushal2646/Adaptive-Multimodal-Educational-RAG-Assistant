"""
database/vector_store.py
========================
pgvector-based vector store for educational document chunks.
Handles:
  - Schema creation
  - Chunk upsert (embeddings + metadata)
  - Semantic similarity search
  - Metadata-filtered search
  - Listing all stored documents
"""

import json
import numpy as np
from typing import Optional
from datetime import datetime

from database.neon_client import get_cursor
from config.settings import EMBEDDING_DIMENSION, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)

# ── SCHEMA ────────────────────────────────────────────────────────────────────

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS educational_documents (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    filename    TEXT,
    source_type TEXT DEFAULT 'unknown',  -- pdf, image, audio, video, text
    subject     TEXT,
    chapter     TEXT,
    topic       TEXT,
    total_chunks INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'
);
"""

CREATE_CHUNKS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({EMBEDDING_DIMENSION}),
    source_type TEXT DEFAULT 'unknown',
    subject     TEXT,
    chapter     TEXT,
    topic       TEXT,
    page_num    INT,
    filename    TEXT,
    char_start  INT,
    char_end    INT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{{}}'
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

CREATE_DOC_ID_IDX = "CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON document_chunks(doc_id);"
CREATE_SUBJECT_IDX = "CREATE INDEX IF NOT EXISTS chunks_subject_idx ON document_chunks(subject);"


def create_tables() -> None:
    """Initialize the database schema with pgvector."""
    try:
        with get_cursor() as cur:
            cur.execute(CREATE_EXTENSION_SQL)
            cur.execute(CREATE_DOCUMENTS_TABLE_SQL)
            cur.execute(CREATE_CHUNKS_TABLE_SQL)
            cur.execute(CREATE_DOC_ID_IDX)
            cur.execute(CREATE_SUBJECT_IDX)
            try:
                cur.execute(CREATE_INDEX_SQL)
            except Exception:
                pass  # Index creation may fail on small datasets — not critical
        logger.info("Database schema ready.")
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        raise


# ── UPSERT ────────────────────────────────────────────────────────────────────

def upsert_document(
    doc_id: str,
    filename: str,
    source_type: str,
    total_chunks: int,
    subject: str = "",
    chapter: str = "",
    topic: str = "",
    metadata: dict = None,
) -> None:
    """Register a document in the documents table."""
    meta = json.dumps(metadata or {})
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO educational_documents
                (doc_id, filename, source_type, subject, chapter, topic, total_chunks, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (doc_id, filename, source_type, subject, chapter, topic, total_chunks, meta),
        )


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Store document chunks with their embeddings.

    Each chunk dict should have:
        doc_id, chunk_index, content, embedding (list[float]),
        source_type, subject, chapter, topic, page_num,
        filename, char_start, char_end, metadata
    """
    if not chunks:
        return 0

    inserted = 0
    with get_cursor() as cur:
        for chunk in chunks:
            embedding_str = "[" + ",".join(str(v) for v in chunk["embedding"]) + "]"
            cur.execute(
                """
                INSERT INTO document_chunks
                    (doc_id, chunk_index, content, embedding, source_type,
                     subject, chapter, topic, page_num, filename,
                     char_start, char_end, metadata)
                VALUES (%s,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING;
                """,
                (
                    chunk.get("doc_id", ""),
                    chunk.get("chunk_index", 0),
                    chunk.get("content", ""),
                    embedding_str,
                    chunk.get("source_type", "unknown"),
                    chunk.get("subject", ""),
                    chunk.get("chapter", ""),
                    chunk.get("topic", ""),
                    chunk.get("page_num"),
                    chunk.get("filename", ""),
                    chunk.get("char_start"),
                    chunk.get("char_end"),
                    json.dumps(chunk.get("metadata", {})),
                ),
            )
            inserted += 1
    logger.info(f"Upserted {inserted} chunks.")
    return inserted


# ── SEMANTIC SEARCH ───────────────────────────────────────────────────────────

def semantic_search(
    query_embedding: list[float],
    top_k: int = TOP_K_RESULTS,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Cosine similarity search using pgvector.
    Returns top_k most semantically similar chunks above the similarity threshold.
    """
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    filter_params: list = []
    where_parts: list[str] = []

    # Threshold filter (uses embedding param #2)
    where_parts.append(f"1 - (embedding <=> %s::vector) >= {similarity_threshold}")

    if subject:
        where_parts.append("subject ILIKE %s")
        filter_params.append(f"%{subject}%")
    if chapter:
        where_parts.append("chapter ILIKE %s")
        filter_params.append(f"%{chapter}%")

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT
            id, doc_id, chunk_index, content,
            source_type, subject, chapter, topic,
            page_num, filename,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM document_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    # Param order matches SQL placeholders:
    #   SELECT clause  → embedding_str  (similarity_score calc)
    #   WHERE clause   → embedding_str  (threshold filter) + filter_params
    #   ORDER BY       → embedding_str
    #   LIMIT          → top_k
    query_params = (
        [embedding_str]    # SELECT similarity_score
        + [embedding_str]  # WHERE threshold
        + filter_params    # optional subject/chapter filters
        + [embedding_str]  # ORDER BY
        + [top_k]
    )

    with get_cursor() as cur:
        cur.execute(sql, query_params)
        rows = cur.fetchall()

    return [dict(r) for r in rows]


# ── METADATA FILTER SEARCH ────────────────────────────────────────────────────

def metadata_filter_search(
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = TOP_K_RESULTS,
) -> list[dict]:
    """Retrieve chunks by metadata filters (no vector similarity)."""
    conditions = []
    params = []

    if subject:
        conditions.append("subject ILIKE %s")
        params.append(f"%{subject}%")
    if chapter:
        conditions.append("chapter ILIKE %s")
        params.append(f"%{chapter}%")
    if topic:
        conditions.append("topic ILIKE %s")
        params.append(f"%{topic}%")

    where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)

    sql = f"""
        SELECT id, doc_id, chunk_index, content,
               source_type, subject, chapter, topic,
               page_num, filename
        FROM document_chunks
        {where_sql}
        ORDER BY id
        LIMIT %s;
    """

    with get_cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [dict(r) for r in rows]


# ── DOCUMENT LISTING ──────────────────────────────────────────────────────────

def list_documents() -> list[dict]:
    """List all ingested documents."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, filename, source_type, subject, chapter,
                   topic, total_chunks, created_at
            FROM educational_documents
            ORDER BY created_at DESC;
            """
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_all_chunks_text() -> list[str]:
    """Retrieve all chunk text (for BM25 corpus building)."""
    with get_cursor() as cur:
        cur.execute("SELECT id, content FROM document_chunks ORDER BY id;")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> int:
    """Delete all chunks for a document."""
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE doc_id = %s;", (doc_id,)
        )
        deleted = cur.rowcount
        cur.execute(
            "DELETE FROM educational_documents WHERE doc_id = %s;", (doc_id,)
        )
    logger.info(f"Deleted document {doc_id} ({deleted} chunks).")
    return deleted


def get_document_count() -> dict:
    """Get counts of documents and chunks."""
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM educational_documents;")
        doc_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) AS cnt FROM document_chunks;")
        chunk_count = cur.fetchone()["cnt"]
    return {"documents": doc_count, "chunks": chunk_count}
