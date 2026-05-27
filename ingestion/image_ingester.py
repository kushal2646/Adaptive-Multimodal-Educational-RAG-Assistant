"""
ingestion/image_ingester.py
===========================
Image ingestion pipeline using OCR:
  1. Load image (Pillow)
  2. Pre-process for OCR
  3. Extract text with Tesseract
  4. Chunk + embed + store
"""

import uuid
from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np

from ingestion.chunker import chunk_text
from database.vector_store import upsert_document, upsert_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def preprocess_image_for_ocr(image_path: str) -> Image.Image:
    """
    Pre-process image to improve OCR accuracy.
    - Convert to grayscale
    - Apply Otsu thresholding
    - Denoise
    """
    img = cv2.imread(image_path)
    if img is None:
        # Fallback: open with PIL
        return Image.open(image_path).convert("RGB")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return Image.fromarray(denoised)


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using Tesseract OCR.
    Returns raw extracted text.
    """
    processed = preprocess_image_for_ocr(image_path)
    config = r"--oem 3 --psm 6"  # LSTM OCR engine, assume uniform text block
    text = pytesseract.image_to_string(processed, config=config)
    logger.info(f"OCR extracted {len(text)} characters from '{image_path}'")
    return text.strip()


def ingest_image(
    image_path: str,
    subject: str = "",
    chapter: str = "",
    topic: str = "",
    description: str = "",
    progress_callback=None,
) -> dict:
    """
    Full image ingestion pipeline.

    Args:
        image_path: Path to image file
        subject/chapter/topic: Metadata tags
        description: Optional user-provided description of the image
        progress_callback: Optional callable(step, total, message)

    Returns:
        Summary dict
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    doc_id = str(uuid.uuid4())
    filename = path.name

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)
        logger.info(f"[{step}/{total}] {msg}")

    _progress(1, 4, f"Running OCR on '{filename}'...")
    extracted_text = extract_text_from_image(image_path)

    # Combine OCR text with user description if available
    if description:
        extracted_text = f"[Description: {description}]\n\n{extracted_text}"

    if not extracted_text.strip():
        logger.warning("No text extracted from image.")
        return {"doc_id": doc_id, "filename": filename, "total_chunks": 0}

    _progress(2, 4, "Chunking OCR text...")
    chunks = chunk_text(
        text=extracted_text,
        metadata={
            "doc_id": doc_id,
            "filename": filename,
            "source_type": "image",
            "subject": subject,
            "chapter": chapter,
            "topic": topic,
        },
    )

    _progress(3, 4, f"Generating embeddings for {len(chunks)} chunks...")
    model = _get_embedding_model()
    texts = [c.content for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    _progress(4, 4, "Storing in Neon pgvector...")
    chunk_dicts = [
        {
            "doc_id": doc_id,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "embedding": emb.tolist(),
            "source_type": "image",
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
        source_type="image",
        total_chunks=len(chunks),
        subject=subject,
        chapter=chapter,
        topic=topic,
    )
    upsert_chunks(chunk_dicts)

    logger.info(f"Image '{filename}' ingested: {len(chunks)} chunks.")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "extracted_chars": len(extracted_text),
    }
