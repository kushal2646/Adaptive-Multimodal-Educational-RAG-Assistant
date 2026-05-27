"""
ui/components.py
================
Reusable Streamlit UI components for EduRAG.
All components use the premium dark theme CSS.
"""

import streamlit as st
from pathlib import Path


def load_css() -> None:
    """Inject the custom CSS stylesheet."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the hero header section."""
    st.markdown(
        """
        <div class="edu-header">
            <h1>🎓 EduRAG Assistant</h1>
            <p>Adaptive Multimodal Educational AI · Powered by Groq Llama 3.3 & Neon PostgreSQL · Hybrid RAG Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str, metadata: dict = None) -> None:
    """
    Render a single chat message bubble.

    Args:
        role: "user" or "assistant"
        content: Message text
        metadata: Optional dict with intent, strategy, expertise, confidence
    """
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        
        # Build metadata badges
        if metadata and role == "assistant":
            badges = []
            if intent := metadata.get("intent"):
                intent_label = intent.replace("_", " ").title()
                badges.append(f'<span class="badge badge-intent">🎯 {intent_label}</span>')
            if strategy := metadata.get("strategy"):
                if strategy != "none":
                    badges.append(f'<span class="badge badge-strategy">🔍 {strategy.title()}</span>')
            if expertise := metadata.get("expertise"):
                badges.append(f'<span class="badge badge-expertise">📊 {expertise.title()}</span>')
            if confidence := metadata.get("confidence"):
                pct = int(confidence * 100)
                badge_cls = "badge-valid" if pct >= 60 else "badge-warning"
                badges.append(f'<span class="badge {badge_cls}">✅ {pct}% confident</span>')

            if badges:
                badges_html = f'<div class="meta-row">{"".join(badges)}</div>'
                st.markdown(badges_html, unsafe_allow_html=True)


def render_source_cards(chunks: list[dict]) -> None:
    """Render retrieved source chunk cards."""
    if not chunks:
        return

    st.markdown("#### 📚 Retrieved Sources")
    for i, chunk in enumerate(chunks[:5], 1):
        filename = chunk.get("filename", "Unknown source")
        subject = chunk.get("subject", "")
        page = chunk.get("page_num", "")
        score = chunk.get("final_score", chunk.get("fused_score", 0))
        content_preview = chunk.get("content", "")[:180] + "..."

        page_info = f"<span>📄 Page {page}</span>" if page else ""
        subject_info = f"<span>🏷️ {subject}</span>" if subject else ""
        
        # Limit score range to [0.0, 1.0] for the bar calculation
        clamped_score = max(0.0, min(1.0, score))
        score_pct = int(clamped_score * 100)

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">📄 {filename}</div>
                <div class="source-meta">
                    {subject_info}
                    {page_info}
                </div>
                <em style="color: #9ca3af; font-size: 0.82rem; line-height: 1.5; display: block; margin-bottom: 0.5rem;">
                    "{content_preview}"
                </em>
                <div class="source-score-container">
                    <div class="source-score-header">
                        <span>Relevance</span>
                        <strong>{score:.2f}</strong>
                    </div>
                    <div class="source-score-bar-bg">
                        <div class="source-score-bar-fill" style="width: {score_pct}%"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_pipeline_status(steps: list[dict]) -> None:
    """
    Render the RAG pipeline status indicator.

    Each step dict: {name, icon, status}  status: "done" | "active" | "pending" | "skipped"
    """
    html = '<div class="sidebar-section"><div class="sidebar-title">⚡ Pipeline Status</div>'
    for step in steps:
        icon = step.get("icon", "•")
        name = step.get("name", "")
        status = step.get("status", "pending")

        if status == "done":
            cls = "step-done"
            status_icon = "✓"
        elif status == "active":
            cls = "step-active"
            status_icon = "→"
        elif status == "skipped":
            cls = "step-pending"
            status_icon = "–"
        else:
            cls = "step-pending"
            status_icon = "○"

        html += f"""
            <div class="pipeline-step {cls}">
                <span class="step-icon">{icon}</span>
                <span>{name}</span>
                <span style="margin-left: auto; font-size: 0.75rem;">{status_icon}</span>
            </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_stats_cards(doc_count: int, chunk_count: int) -> None:
    """Render document/chunk count stat cards."""
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{doc_count}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{chunk_count:,}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Render the empty chat state."""
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem; color: #6e7681;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🎓</div>
            <h3 style="color: #8b949e; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                Welcome to EduRAG Assistant
            </h3>
            <p style="font-size: 0.9rem; max-width: 400px; margin: 0 auto; line-height: 1.6;">
                Upload educational materials in the sidebar, then ask me anything.<br/>
                I'll retrieve the most relevant context and generate an accurate, grounded response.
            </p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2); 
                             color: #58a6ff; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem;">
                    📄 Upload PDFs
                </span>
                <span style="background: rgba(188,140,255,0.1); border: 1px solid rgba(188,140,255,0.2); 
                             color: #bc8cff; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem;">
                    🖼️ Analyze Images
                </span>
                <span style="background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.2); 
                             color: #3fb950; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem;">
                    🎙️ Transcribe Audio
                </span>
                <span style="background: rgba(227,179,65,0.1); border: 1px solid rgba(227,179,65,0.2); 
                             color: #e3b341; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem;">
                    🧠 Adaptive RAG
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ingestion_result(result: dict, source_type: str) -> None:
    """Render ingestion success result."""
    st.success(
        f"✅ **{source_type.upper()} ingested successfully!**\n\n"
        f"- 📁 File: `{result.get('filename', 'N/A')}`\n"
        f"- 🧩 Chunks stored: **{result.get('total_chunks', 0)}**\n"
        f"- 🔑 Document ID: `{result.get('doc_id', 'N/A')[:16]}...`"
    )


def render_config_warning(missing: list[str]) -> None:
    """Render config missing warning."""
    st.markdown(
        f"""
        <div style="background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3);
                    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <strong style="color: #f85149;">⚠️ Configuration Required</strong><br/>
            <span style="color: #8b949e; font-size: 0.875rem;">
                Missing environment variables: <code>{', '.join(missing)}</code><br/>
                Copy <code>.env.example</code> to <code>.env</code> and fill in your credentials.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
