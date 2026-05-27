# 🎓 Adaptive Multimodal Educational RAG Assistant

A production-grade AI tutoring system powered by **Groq Llama 3.3 / 3.1**, **Neon PostgreSQL + pgvector**, and **Hybrid RAG Retrieval**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **LLM** | Groq Llama 3.3 70B (streaming + fallback) |
| **Vector DB** | Neon PostgreSQL + pgvector |
| **Retrieval** | Hybrid: Semantic (cosine) + BM25 keyword |
| **Multimodal** | PDF, Image (OCR), Audio, Video |
| **Pipeline** | 13-step Adaptive RAG with query rewriting, re-ranking, context compression, answer validation |
| **UI** | Streamlit dark-mode premium interface |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd "Adaptive Multimodal Educational RAG Assistant"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` and fill in:
- `GROQ_API_KEY` — from [console.groq.com/keys](https://console.groq.com/keys)
- `NEON_DATABASE_URL` — from [console.neon.tech](https://console.neon.tech)

### 3. Install Tesseract OCR (for image ingestion)

- **Windows**: Download from [tesseract-ocr.github.io](https://tesseract-ocr.github.io/tessdoc/Installation.html)
  - Add to PATH: `C:\Program Files\Tesseract-OCR`

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Architecture

```
User Query
  │
  ├─ [Step 2] Intent Classifier → 10 intent types + expertise detection
  ├─ [Step 4] Query Rewriter → semantic + keyword + hybrid variants
  ├─ [Step 5] Retrieval Router → adaptive strategy selection
  ├─ [Step 6] Hybrid Retriever → pgvector + BM25 + score fusion
  ├─ [Step 8] Re-Ranker → semantic re-ranking + quality scoring
  ├─ [Step 9] Context Compressor → dedup + token budget
  ├─ [Step 10] Prompt Assembler → expertise-adaptive prompts
  ├─ [Step 11] Groq LLM → Llama 3.3 70B generation
  └─ [Step 12] Answer Validator → grounding + hallucination check
```

---

## 📁 Project Structure

```
app.py                     # Main Streamlit application
requirements.txt
.env.example               # Environment variables template

config/
  settings.py              # Centralized config

core/
  intent_classifier.py     # Query intent + expertise detection
  query_rewriter.py        # LLM-powered query expansion
  retrieval_router.py      # Adaptive strategy selection
  hybrid_retriever.py      # Semantic + BM25 retrieval
  reranker.py              # Re-ranking with quality scoring
  context_compressor.py    # Dedup + token compression
  prompt_assembler.py      # Expertise-adaptive prompt building
  llm_generator.py         # Groq API with streaming + fallback
  answer_validator.py      # Grounding + hallucination detection

database/
  neon_client.py           # PostgreSQL connection pool
  vector_store.py          # pgvector CRUD + search

ingestion/
  chunker.py               # Semantic text chunking
  pdf_ingester.py          # PDF → chunks → embeddings → Neon
  image_ingester.py        # Image OCR → chunks → embeddings
  audio_ingester.py        # Audio/Video → transcript → embeddings

ui/
  components.py            # Reusable Streamlit components
  styles.css               # Premium dark-mode CSS

utils/
  logger.py                # Rich structured logging
```

---

## 🧠 RAG Pipeline Steps

| Step | Module | Description |
|---|---|---|
| 1 | `app.py` | Receive user query |
| 2 | `intent_classifier.py` | Classify intent + expertise |
| 3 | `retrieval_router.py` | Decide if retrieval needed |
| 4 | `query_rewriter.py` | Rewrite + expand query |
| 5 | `retrieval_router.py` | Select retrieval strategy |
| 6 | `hybrid_retriever.py` | Vector + BM25 search |
| 7 | `hybrid_retriever.py` | Merge + fuse candidates |
| 8 | `reranker.py` | Re-rank by semantic similarity |
| 9 | `context_compressor.py` | Compress + deduplicate |
| 10 | `prompt_assembler.py` | Build augmented prompt |
| 11 | `llm_generator.py` | Groq Llama 3.3/3.1 generation |
| 12 | `answer_validator.py` | Validate + retry if needed |
| 13 | `app.py` | Return final response |

---

## 📋 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key |
| `NEON_DATABASE_URL` | ✅ | Neon PostgreSQL connection string |
| `GROQ_MODEL` | Optional | Default: `llama-3.3-70b-versatile` |
| `WHISPER_MODE` | Optional | `groq` (default) or `local` |
| `TOP_K_RESULTS` | Optional | Retrieved chunks (default: 8) |
| `CHUNK_SIZE` | Optional | Words per chunk (default: 512) |

---

## 🔧 Supported File Types

| Type | Formats |
|---|---|
| PDF | `.pdf` |
| Image | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` |
| Video | `.mp4`, `.mkv`, `.webm`, `.mov` |
