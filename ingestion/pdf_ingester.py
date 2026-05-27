"""
ingestion/pdf_ingester.py
=========================
PDF ingestion pipeline:
  1. Extract text + structure using PyMuPDF
  2. Chunk text
  3. Generate embeddings
  4. Store in Neon pgvector
"""

import uuid
from pathlib import Path

import fitz  # PyMuPDF

from ingestion.chunker import chunk_document_pages, TextChunk
from database.vector_store import upsert_document, upsert_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_embedding_model():
    """Lazy load the embedding model."""
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF using PyMuPDF.

    Returns:
        List of dicts: {page_num, text, section}
    """
    pages = []
    doc = fitz.open(pdf_path)

    current_section = ""
    for page_num, page in enumerate(doc, start=1):
        # Extract text blocks
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # sort by y, then x

        page_text_parts = []
        for block in blocks:
            if block[6] == 0:  # text block (not image)
                text = block[4].strip()
                if text:
                    # Detect headings heuristically (short, ends without period)
                    if len(text) < 80 and not text.endswith("."):
                        current_section = text
                    page_text_parts.append(text)

        full_page_text = "\n".join(page_text_parts)
        if full_page_text.strip():
            pages.append({
                "page_num": page_num,
                "text": full_page_text,
                "section": current_section,
            })

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from '{pdf_path}'")
    return pages


def ingest_pdf(
    pdf_path: str,
    subject: str = "",
    chapter: str = "",
    topic: str = "",
    progress_callback=None,
) -> dict:
    """
    Full PDF ingestion pipeline.

    Args:
        pdf_path: Path to PDF file
        subject: Subject metadata tag
        chapter: Chapter metadata tag
        topic: Topic metadata tag
        progress_callback: Optional callable(step, total, message) for UI updates

    Returns:
        Summary dict {doc_id, filename, total_chunks}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = str(uuid.uuid4())
    filename = path.name

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        logger.info(f"[{step}/{total}] {msg}")

    _progress(1, 5, f"Extracting pages from '{filename}'...")
    pages = extract_pages_from_pdf(pdf_path)

    _progress(2, 5, "Chunking extracted text...")
    base_metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "subject": subject,
        "chapter": chapter,
        "topic": topic,
        "source_type": "pdf",
    }
    chunks: list[TextChunk] = chunk_document_pages(
        pages=pages,
        base_metadata=base_metadata,
    )

    if not chunks:
        logger.warning("No chunks generated from PDF.")
        return {"doc_id": doc_id, "filename": filename, "total_chunks": 0}

    _progress(3, 5, f"Generating embeddings for {len(chunks)} chunks...")
    model = _get_embedding_model()
    texts = [c.content for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    _progress(4, 5, "Storing chunks in Neon pgvector...")
    chunk_dicts = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_dicts.append({
            "doc_id": doc_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "embedding": embedding.tolist(),
            "source_type": "pdf",
            "subject": subject,
            "chapter": chapter,
            "topic": topic,
            "page_num": chunk.page_num,
            "filename": filename,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "metadata": chunk.metadata,
        })

    upsert_document(
        doc_id=doc_id,
        filename=filename,
        source_type="pdf",
        total_chunks=len(chunks),
        subject=subject,
        chapter=chapter,
        topic=topic,
        metadata={"path": str(path)},
    )
    upsert_chunks(chunk_dicts)

    _progress(5, 5, f"✅ PDF ingested: {len(chunks)} chunks stored.")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "pages": len(pages),
    }
