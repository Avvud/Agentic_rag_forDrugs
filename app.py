"""
app.py — Streamlit Web Interface for Medical Drug-Interaction Agent.

Features:
- Sleek modern styling with dark/light mode support and custom CSS.
- Interactive chat interface with conversation history.
- Verifiable citation cards for PDF, FDA labels (with DailyMed links), and Web sources.
- Collapsible "Agent Execution Trace" panel displaying tool invocations.
- Verification warning banners for unverified/dropped citations.
- Sidebar controls for re-running PDF ingestion and viewing collection metrics.
- Prominent educational medical disclaimer footer.
"""

import os
import streamlit as st
import chromadb

import config
import ingest
import agent
import tools
from models import AgentResult, PdfCitation, FdaLabelCitation, WebCitation

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug-Interaction Agentic RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich modern aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0d9488 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    .citation-card-pdf {
        background-color: #064e3b;
        color: #ecfdf5 !important;
        border-left: 5px solid #10b981;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    
    .citation-card-fda {
        background-color: #1e3a8a;
        color: #eff6ff !important;
        border-left: 5px solid #3b82f6;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }

    .citation-card-web {
        background-color: #78350f;
        color: #fffbeb !important;
        border-left: 5px solid #f59e0b;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }

    .citation-card-pdf *, .citation-card-fda *, .citation-card-web * {
        color: inherit;
    }

    .citation-card-pdf strong, .citation-card-fda strong, .citation-card-web strong {
        font-size: 0.98rem;
        font-weight: 600;
    }

    .citation-card-pdf code, .citation-card-fda code, .citation-card-web code {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
    }

    .citation-card-fda a, .citation-card-web a {
        color: #93c5fd !important;
        font-weight: 600 !important;
        text-decoration: underline !important;
    }
    
    .citation-card-fda a:hover, .citation-card-web a:hover {
        color: #bfdbfe !important;
    }
    
    .disclaimer-footer {
        margin-top: 3rem;
        padding: 1.2rem;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-size: 0.85rem;
        color: #94a3b8;
        text-align: center;
    }
    
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_chroma_chunk_count() -> int:
    """Return total number of chunks stored in ChromaDB collection."""
    try:
        if not os.path.exists(config.CHROMA_PATH):
            return 0
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collections = [c.name for c in client.list_collections()]
        if "drug_interactions" in collections:
            coll = client.get_collection("drug_interactions")
            return coll.count()
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pill.png", width=64)
    st.title("Settings & Status")
    
    chunk_count = get_chroma_chunk_count()
    st.metric("Indexed Chunks", f"{chunk_count}")

    if chunk_count == 0:
        st.warning("⚠️ ChromaDB is empty! Run ingestion to index the drug interactions PDF.")

    if st.button("🔄 Re-run PDF Ingestion", use_container_width=True):
        with st.spinner("Parsing PDF, embedding chunks with Gemini, and indexing in ChromaDB..."):
            try:
                count = ingest.ingest()
                st.success(f"Ingestion complete! Indexed {count} chunks.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### System Info")
    st.markdown(f"**LLM Model:** `{config.GEMINI_MODEL}`")
    st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Max Steps:** `{config.MAX_AGENT_STEPS}`")
    fda_key_status = "Configured ✅" if config.OPENFDA_API_KEY else "Not set (using free tier)"
    st.markdown(f"**openFDA Key:** {fda_key_status}")


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">💊 Agentic Medical Drug-Interaction Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Agentic RAG with verified citations over official FDA guidance and drug labels.</div>',
    unsafe_allow_html=True,
)

if chunk_count == 0:
    st.info(
        "👋 Welcome! ChromaDB has not been populated yet. "
        "Click **'Re-run PDF Ingestion'** in the sidebar to index the drug interaction PDF."
    )

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            result: AgentResult = msg["content"]
            st.markdown(result.answer.answer)
            
            # Display Verification Warning if present
            if result.verification_warning:
                st.warning(f"⚠️ **Citation Warning:** {result.verification_warning}")
            
            # Render Citations
            if result.answer.citations:
                st.markdown("#### 📚 Verifiable Sources")
                for cit in result.answer.citations:
                    if cit.type == "pdf":
                        st.markdown(
                            f"""
                            <div class="citation-card-pdf">
                                <strong>[{cit.id}] PDF Document (Page {cit.page}) - {cit.section}</strong><br/>
                                💬 "{cit.quote}"
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"📄 View PDF Page {cit.page} (Highlighted Line)", expanded=False):
                            img_bytes = tools.get_highlighted_pdf_page_image(cit.page, cit.quote)
                            if img_bytes:
                                st.image(
                                    img_bytes,
                                    caption=f"PDF Page {cit.page} — Highlighted Excerpt: \"{cit.quote}\"",
                                    use_container_width=True,
                                )
                                st.download_button(
                                    "📥 Download Highlighted Page PNG",
                                    data=img_bytes,
                                    file_name=f"PDF_Page_{cit.page}_Highlighted.png",
                                    mime="image/png",
                                    key=f"dl_{cit.id}_{cit.page}_{msg_idx if 'msg_idx' in locals() else 0}",
                                )
                            else:
                                st.info(f"Could not load page image for PDF Page {cit.page}.")
                    elif cit.type == "fda_label":
                        url_link = f'<a href="{cit.url}" target="_blank">View DailyMed Label ↗</a>' if cit.url else ""
                        st.markdown(
                            f"""
                            <div class="citation-card-fda">
                                <strong>[{cit.id}] FDA Official Label — {cit.drug.capitalize()} ({cit.section})</strong><br/>
                                <em>SET ID: <code>{cit.set_id}</code> | Effective Date: {cit.effective_time}</em> {url_link}<br/>
                                💬 "{cit.quote}"
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif cit.type == "web":
                        st.markdown(
                            f"""
                            <div class="citation-card-web">
                                <strong>[{cit.id}] Web Source: <a href="{cit.url}" target="_blank">{cit.title}</a></strong><br/>
                                🔗 <code>{cit.url}</code>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            
            # Render Agent Steps Trace
            if result.tool_calls:
                with st.expander("🛠️ Agent Execution Trace", expanded=False):
                    for idx, call in enumerate(result.tool_calls, start=1):
                        st.markdown(f"**Step {idx}: Call `{call.tool_name}`**")
                        st.json(call.arguments)
                        if call.error:
                            st.error(f"Error: {call.error}")
                        else:
                            st.caption(f"Returned {call.result_count or 0} result(s)")

# Chat Input
user_query = st.chat_input("Ask a question about drug interactions (e.g., 'What should I avoid with antihistamines?')...")

if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Searching drug interactions and verifying citations..."):
            result = agent.run_agent(user_query)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()

# Disclaimer Footer
st.markdown(
    """
    <div class="disclaimer-footer">
        ⚕️ <strong>Educational Disclaimer:</strong> This application provides educational information on drug interactions based on FDA reference materials. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult a licensed doctor or pharmacist for medical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)