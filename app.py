"""
app.py
======
EduRAG Assistant — Main Streamlit Application
Orchestrates all 13 pipeline steps:

  1. User Query
  2. Intent Classification
  3. Retrieval Decision
  4. Query Rewriting
  5. Retrieval Strategy Selection
  6. Hybrid Retrieval (pgvector + BM25)
  7. Candidate Collection
  8. Re-Ranking
  9. Context Compression
  10. Prompt Assembly
  11. LLM Generation (Groq Llama 3)
  12. Answer Validation
  13. Final Response
"""

import os
import sys
import tempfile
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EduRAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
from ui.components import (
    load_css, render_header, render_chat_message, render_source_cards,
    render_pipeline_status, render_stats_cards, render_empty_state,
    render_ingestion_result, render_config_warning,
)
load_css()

# ── Config validation ─────────────────────────────────────────────────────────
from config.settings import validate_config, APP_TITLE
missing_config = validate_config()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "messages": [],           # [{role, content, metadata}]
        "pipeline_steps": [],     # for status panel
        "last_sources": [],       # retrieved chunks for source panel
        "db_initialized": False,
        "doc_stats": {"documents": 0, "chunks": 0},
        "force_strategy": "auto",
        "expertise_override": "auto",
        "show_sources": True,
        "show_pipeline": True,
        "rewrite_enabled": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initialize DB schema (cached — runs once per session)."""
    if missing_config:
        return False
    try:
        from database.vector_store import create_tables, get_document_count
        create_tables()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False


def get_db_stats() -> dict:
    """Fetch current document/chunk counts."""
    try:
        from database.vector_store import get_document_count
        return get_document_count()
    except Exception:
        return {"documents": 0, "chunks": 0}


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(
    query: str,
    chat_history: list[dict],
    force_strategy: str = "auto",
    expertise_override: str = "auto",
    rewrite_enabled: bool = True,
) -> dict:
    """
    Full 13-step RAG pipeline.

    Returns:
        dict with keys: answer, intent, strategy, expertise, confidence,
                        hallucination_risk, sources, pipeline_steps
    """
    pipeline_steps = [
        {"name": "Query Understanding",   "icon": "🔍", "status": "pending"},
        {"name": "Query Rewriting",        "icon": "✏️", "status": "pending"},
        {"name": "Retrieval Strategy",     "icon": "🎯", "status": "pending"},
        {"name": "Hybrid Retrieval",       "icon": "📡", "status": "pending"},
        {"name": "Re-Ranking",             "icon": "📊", "status": "pending"},
        {"name": "Context Compression",    "icon": "🗜️", "status": "pending"},
        {"name": "Prompt Assembly",        "icon": "🔧", "status": "pending"},
        {"name": "LLM Generation",         "icon": "🧠", "status": "pending"},
        {"name": "Answer Validation",      "icon": "✅", "status": "pending"},
    ]

    # ── STEP 2: Intent Classification ────────────────────────────────────────
    pipeline_steps[0]["status"] = "active"
    from core.intent_classifier import classify_intent
    intent = classify_intent(query)

    if expertise_override != "auto":
        intent.expertise = expertise_override

    pipeline_steps[0]["status"] = "done"

    # ── STEP 4: Query Rewriting ───────────────────────────────────────────────
    pipeline_steps[1]["status"] = "active"
    rewritten = None
    if rewrite_enabled and intent.intent != "general_conversational":
        from core.query_rewriter import rewrite_query
        rewritten = rewrite_query(
            query=query,
            intent=intent.intent,
            subject_hint=intent.subject_hint,
        )
    pipeline_steps[1]["status"] = "done" if rewritten else "skipped"

    # ── STEP 3/5: Retrieval Decision + Strategy Selection ────────────────────
    pipeline_steps[2]["status"] = "active"
    db_stats = get_db_stats()
    has_docs = db_stats.get("chunks", 0) > 0

    from core.retrieval_router import select_retrieval_strategy
    plan = select_retrieval_strategy(
        intent=intent,
        has_documents=has_docs,
        force_strategy=force_strategy if force_strategy != "auto" else "",
    )
    pipeline_steps[2]["status"] = "done"

    # ── STEPS 6/7: Hybrid Retrieval ───────────────────────────────────────────
    candidates = []
    if plan.strategy != "none":
        pipeline_steps[3]["status"] = "active"
        from core.hybrid_retriever import retrieve
        candidates = retrieve(
            query=query,
            plan=plan,
            rewritten_queries=rewritten,
        )
        pipeline_steps[3]["status"] = "done"
    else:
        pipeline_steps[3]["status"] = "skipped"

    # ── STEP 8: Re-Ranking ────────────────────────────────────────────────────
    if candidates:
        pipeline_steps[4]["status"] = "active"
        from core.reranker import rerank
        candidates = rerank(
            query=query,
            candidates=candidates,
            rewritten_query=rewritten.primary if rewritten else "",
        )
        pipeline_steps[4]["status"] = "done"
    else:
        pipeline_steps[4]["status"] = "skipped"

    # ── STEP 9: Context Compression ───────────────────────────────────────────
    pipeline_steps[5]["status"] = "active"
    from core.context_compressor import compress_context, format_context_block
    compressed = compress_context(candidates)
    context_block = format_context_block(compressed)
    pipeline_steps[5]["status"] = "done"

    # ── STEP 10: Prompt Assembly ──────────────────────────────────────────────
    pipeline_steps[6]["status"] = "active"
    from core.prompt_assembler import assemble_prompt
    messages = assemble_prompt(
        query=query,
        intent=intent,
        context_block=context_block,
        rewritten_query=rewritten.primary if rewritten else "",
        chat_history=chat_history,
    )
    pipeline_steps[6]["status"] = "done"

    # ── STEP 11: LLM Generation ───────────────────────────────────────────────
    pipeline_steps[7]["status"] = "active"
    from core.llm_generator import generate_with_fallback
    answer = generate_with_fallback(messages=messages, stream=False)
    pipeline_steps[7]["status"] = "done"

    # ── STEP 12: Answer Validation ────────────────────────────────────────────
    pipeline_steps[8]["status"] = "active"
    from core.answer_validator import validate_answer
    validation = validate_answer(
        answer=answer,
        query=query,
        context_block=context_block,
        intent=intent.intent,
        attempt=1,
    )

    # Retry if needed
    if validation.needs_retry:
        from core.query_rewriter import rewrite_query
        from core.hybrid_retriever import retrieve
        from core.reranker import rerank
        from core.context_compressor import compress_context, format_context_block
        from core.prompt_assembler import assemble_prompt

        # Retry with expanded semantic query
        retry_rewritten = rewrite_query(query=query, intent=intent.intent)
        retry_candidates = retrieve(query=query, plan=plan, rewritten_queries=retry_rewritten)
        retry_candidates = rerank(query=query, candidates=retry_candidates)
        retry_compressed = compress_context(retry_candidates)
        retry_context = format_context_block(retry_compressed)

        retry_messages = assemble_prompt(
            query=query,
            intent=intent,
            context_block=retry_context,
            rewritten_query=retry_rewritten.semantic,
            chat_history=chat_history,
        )
        answer = generate_with_fallback(messages=retry_messages, stream=False)
        validation = validate_answer(
            answer=answer,
            query=query,
            context_block=retry_context,
            intent=intent.intent,
            attempt=2,
        )
        compressed = retry_compressed

    pipeline_steps[8]["status"] = "done"

    return {
        "answer": answer,
        "intent": intent.intent,
        "strategy": plan.strategy,
        "expertise": intent.expertise,
        "confidence": validation.confidence,
        "hallucination_risk": validation.hallucination_risk,
        "is_valid": validation.is_valid,
        "issues": validation.issues,
        "sources": compressed,
        "pipeline_steps": pipeline_steps,
        "rewritten_query": rewritten.primary if rewritten else query,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎓 EduRAG Assistant")
    st.markdown("---")

    # ── Config warning ────────────────────────────────────────────────────────
    if missing_config:
        render_config_warning(missing_config)

    # ── DB Stats ──────────────────────────────────────────────────────────────
    if not missing_config:
        db_ok = initialize_database()
        if db_ok:
            stats = get_db_stats()
            st.session_state.doc_stats = stats
            render_stats_cards(stats["documents"], stats["chunks"])

    # ── Document Upload ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">📤 Upload Materials</div></div>',
        unsafe_allow_html=True,
    )

    upload_type = st.selectbox(
        "File type",
        ["PDF", "Image (OCR)", "Audio", "Video"],
        key="upload_type",
    )

    # Metadata inputs
    with st.expander("📌 Metadata Tags (optional)"):
        subject_tag = st.text_input("Subject", placeholder="e.g. Machine Learning", key="tag_subject")
        chapter_tag = st.text_input("Chapter", placeholder="e.g. Chapter 3", key="tag_chapter")
        topic_tag   = st.text_input("Topic",   placeholder="e.g. Neural Networks", key="tag_topic")

    # File uploader
    accept_map = {
        "PDF":          ["pdf"],
        "Image (OCR)":  ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        "Audio":        ["mp3", "wav", "m4a", "ogg", "flac"],
        "Video":        ["mp4", "mkv", "webm", "mov"],
    }
    uploaded_file = st.file_uploader(
        f"Choose {upload_type} file",
        type=accept_map.get(upload_type, []),
        key="file_uploader",
    )

    if uploaded_file and not missing_config:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{uploaded_file.name.split('.')[-1]}",
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {upload_type}..."):
                try:
                    progress_placeholder = st.empty()

                    def progress_cb(step, total, msg):
                        progress_placeholder.progress(step / total, text=msg)

                    if upload_type == "PDF":
                        from ingestion.pdf_ingester import ingest_pdf
                        result = ingest_pdf(
                            tmp_path,
                            subject=subject_tag,
                            chapter=chapter_tag,
                            topic=topic_tag,
                            progress_callback=progress_cb,
                        )
                    elif upload_type == "Image (OCR)":
                        from ingestion.image_ingester import ingest_image
                        result = ingest_image(
                            tmp_path,
                            subject=subject_tag,
                            chapter=chapter_tag,
                            topic=topic_tag,
                            progress_callback=progress_cb,
                        )
                    elif upload_type in ("Audio", "Video"):
                        from ingestion.audio_ingester import ingest_audio
                        result = ingest_audio(
                            tmp_path,
                            subject=subject_tag,
                            chapter=chapter_tag,
                            topic=topic_tag,
                            source_type=upload_type.lower(),
                            progress_callback=progress_cb,
                        )

                    # Invalidate BM25 cache after ingestion
                    from core.hybrid_retriever import invalidate_bm25_cache
                    invalidate_bm25_cache()

                    progress_placeholder.empty()
                    render_ingestion_result(result, upload_type)

                    # Refresh stats
                    st.session_state.doc_stats = get_db_stats()
                    st.rerun()

                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        st.session_state.force_strategy = st.selectbox(
            "Retrieval Strategy",
            ["auto", "hybrid", "semantic", "keyword", "none"],
            key="strategy_select",
        )
        st.session_state.expertise_override = st.selectbox(
            "User Expertise",
            ["auto", "beginner", "intermediate", "advanced"],
            key="expertise_select",
        )
        st.session_state.rewrite_enabled = st.toggle(
            "Query Rewriting", value=True, key="rewrite_toggle"
        )
        st.session_state.show_sources = st.toggle(
            "Show Sources", value=True, key="sources_toggle"
        )
        st.session_state.show_pipeline = st.toggle(
            "Show Pipeline", value=True, key="pipeline_toggle"
        )

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.markdown("---")

    # ── Pipeline Status Panel ─────────────────────────────────────────────────
    if st.session_state.show_pipeline and st.session_state.pipeline_steps:
        render_pipeline_status(st.session_state.pipeline_steps)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════════════════════

render_header()

# ── Two-column layout: Chat + Sources ────────────────────────────────────────
col_chat, col_sources = st.columns([3, 1])

with col_chat:
    # ── Chat History ──────────────────────────────────────────────────────────
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            render_empty_state()
        else:
            for msg in st.session_state.messages:
                render_chat_message(
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata"),
                )

    # ── Chat Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask me anything... (upload documents first for grounded responses)",
        key="chat_input",
    )

    if user_input:
        if missing_config:
            st.error(
                "Please configure your API keys first. "
                "Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` and `NEON_DATABASE_URL`."
            )
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "metadata": None,
            })

            # Build chat history for context
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            # ── Run Pipeline ──────────────────────────────────────────────────
            with st.spinner("🧠 Processing through RAG pipeline..."):
                try:
                    result = run_rag_pipeline(
                        query=user_input,
                        chat_history=chat_history,
                        force_strategy=st.session_state.force_strategy,
                        expertise_override=st.session_state.expertise_override,
                        rewrite_enabled=st.session_state.rewrite_enabled,
                    )

                    # Update session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "intent": result["intent"],
                            "strategy": result["strategy"],
                            "expertise": result["expertise"],
                            "confidence": result["confidence"],
                            "hallucination_risk": result["hallucination_risk"],
                        },
                    })
                    st.session_state.last_sources = result["sources"]
                    st.session_state.pipeline_steps = result["pipeline_steps"]

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ An error occurred: {str(e)}\n\nPlease check your configuration and try again.",
                        "metadata": None,
                    })

            st.rerun()

with col_sources:
    if st.session_state.show_sources:
        if st.session_state.last_sources:
            render_source_cards(st.session_state.last_sources)
        else:
            st.markdown(
                """
                <div style="padding: 1.5rem; text-align: center; color: #6e7681;
                            background: #161b22; border-radius: 12px; border: 1px solid #30363d;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                    <p style="font-size: 0.82rem; line-height: 1.5;">
                        Retrieved sources will appear here after your first retrieval-grounded response.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Quick Questions ───────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        example_questions = [
            "Explain this concept step by step",
            "Summarize the key points",
            "Generate 5 quiz questions",
            "What are the advantages and disadvantages?",
            "Give me examples of this in real life",
        ]
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": q,
                    "metadata": None,
                })
                st.rerun()
