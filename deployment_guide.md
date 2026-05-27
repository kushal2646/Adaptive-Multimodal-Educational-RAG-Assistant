# 🚀 Deployment Guide: Live Hosting for EduRAG Assistant

This guide explains how to deploy the **Adaptive Multimodal Educational RAG Assistant** live to the web so others can access it.

---

## 🛠️ Prerequisites Before Going Live

1. **GitHub Repository**: The codebase has been successfully pushed to:
   `https://github.com/kushal2646/Adaptive-Multimodal-Educational-RAG-Assistant.git`
2. **Neon PostgreSQL Database**: Ensure your database is active and contains the schema tables (automatically initialized on first run).
3. **API Keys**: Make sure you have your active `GROQ_API_KEY` ready.

---

## ⚡ Option 1: Streamlit Community Cloud (Recommended & Free)

This is the fastest, easiest, and completely free hosting solution for Streamlit applications.

### Step-by-Step Setup:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your repository details:
   * **Repository**: `kushal2646/Adaptive-Multimodal-Educational-RAG-Assistant`
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. Click **"Advanced settings..."** (next to the deploy button).
5. In the **Secrets** text area, paste your environment variables in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_xxxx..."
   NEON_DATABASE_URL = "postgresql://neondb_owner:xxxx..."
   GROQ_MODEL = "llama-3.3-70b-versatile"
   WHISPER_MODE = "groq"
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   EMBEDDING_DIMENSION = 384
   TOP_K_RESULTS = 8
   SIMILARITY_THRESHOLD = 0.35
   BM25_WEIGHT = 0.3
   SEMANTIC_WEIGHT = 0.7
   MAX_CONTEXT_TOKENS = 6000
   RERANK_TOP_N = 5
   CHUNK_SIZE = 512
   CHUNK_OVERLAP = 64
   APP_TITLE = "EduRAG Assistant"
   LOG_LEVEL = "INFO"
   ```
6. Click **"Save"**, then click **"Deploy!"**.
7. Streamlit will boot up a container, install your `requirements.txt` dependencies, and make your app live at a public URL (e.g., `https://edurag-assistant.streamlit.app`).

---

## 🖼️ Option 2: Hugging Face Spaces (Free & Easy)

Hugging Face Spaces is another free hosting service designed specifically for machine learning and web apps.

### Step-by-Step Setup:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2. Click **"Create new Space"**.
3. Fill in the details:
   * **Space name**: `Adaptive-Multimodal-Educational-RAG-Assistant`
   * **SDK**: `Streamlit`
   * **Space hardware**: `CPU Basic (Free)`
4. Click **"Create Space"**.
5. Go to the **Settings** tab of your new Space:
   * Scroll down to **Variables and secrets**.
   * Click **"New secret"** and add each environment variable individually:
     * Key: `GROQ_API_KEY`, Value: `your_key_here`
     * Key: `NEON_DATABASE_URL`, Value: `your_neon_url_here`
     * Key: `GROQ_MODEL`, Value: `llama-3.3-70b-versatile`
6. Push your repository to Hugging Face or sync it with GitHub:
   * You can set up a GitHub Action to automatically push updates to Hugging Face on commits.
7. Your app will build and be live at `https://huggingface.co/spaces/your-username/Adaptive-Multimodal-Educational-RAG-Assistant`.

---

## 🚄 Option 3: Railway or Render (Paid / Auto-Scale)

For higher reliability, custom domain routing, or production loads, cloud container platforms are excellent.

### Step-by-Step Setup on Render:
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **"New +"** → **"Web Service"**.
3. Connect your GitHub repository.
4. Set the configuration details:
   * **Runtime**: `Python`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Advanced"** to add **Environment Variables**:
   * Add `GROQ_API_KEY` and `NEON_DATABASE_URL`.
6. Click **"Create Web Service"**. Render will deploy it automatically.

---

## 🔏 Special Considerations for Going Live

### 1. Ingestion Capabilities (Audio / Video / OCR)
* **Whisper Mode**: On cloud hosting platforms (like Streamlit Cloud or Hugging Face), make sure `WHISPER_MODE` is set to `groq` in the secrets. Running local Whisper on free tier instances will crash the container due to CPU/RAM limits (local models are heavy!).
* **Tesseract OCR**: If using Image OCR, the hosting platform needs the binary `tesseract` installed.
  * For **Streamlit Cloud**: Create a file named `packages.txt` in the root directory containing `tesseract-ocr` and `libgl1-mesa-glx` to auto-install it.
  * For **Hugging Face / Docker**: Use a Docker space or custom configurations.

### 2. Database Schema
* The Neon PostgreSQL schema is initialized automatically when the live app launches. No database seeding is required.
