


"""
app.py - Streamlit UI for the AUTOSAR HLD Document Analysis Assistant (RAG)

Run from the Code/ folder:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from ingestion import PROJECT_ROOT, load_config
from rag_pipeline import RAGPipeline, load_prompts

logger = logging.getLogger("hld_rag.app")
SAMPLE_PDF = PROJECT_ROOT / "Input_Data" / "Synthetic_AUTOSAR_BCM_HLD.pdf"

st.set_page_config(page_title="AUTOSAR HLD RAG Assistant", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # list of RAGAnswer
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "kb_docs" not in st.session_state:
    st.session_state.kb_docs = []

base_cfg = load_config()

# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Defaults come from Model_Prompts_Config/model_config.txt")

    st.subheader("Chunking (rebuild after changing)")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, int(base_cfg["CHUNK_SIZE"]), 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, int(base_cfg["CHUNK_OVERLAP"]), 10)
    if chunk_overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", 1, 10, int(base_cfg["TOP_K"]))
    threshold = st.slider("Similarity threshold", 0.0, 1.0, float(base_cfg["SIMILARITY_THRESHOLD"]), 0.01,
                          help="If the best chunk scores below this, the assistant says evidence was not found.")

    st.subheader("LLM")
    providers = ["gemini", "groq", "openai", "ollama", "none"]
    default_provider = str(base_cfg["LLM_PROVIDER"]).lower()
    provider = st.selectbox("Provider", providers, index=providers.index(default_provider) if default_provider in providers else 0)
    model_key = {"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL", "openai": "OPENAI_MODEL", "ollama": "OLLAMA_MODEL"}.get(provider)
    model_name = st.text_input("Model", str(base_cfg[model_key])) if model_key else None
    if provider in ("gemini", "groq", "openai"):
        env_var = f"{provider.upper()}_API_KEY"
        if os.getenv(env_var):
            st.success(f"{env_var} found in environment ✔")
        else:
            st.warning(f"{env_var} is not set. Set it as an environment variable or in Code/.env. "
                       "Until then an extractive fallback answer is shown.")
    elif provider == "none":
        st.info("No LLM: answers are the most relevant sentences from the evidence.")

cfg = dict(base_cfg)
cfg.update({"CHUNK_SIZE": chunk_size, "CHUNK_OVERLAP": chunk_overlap, "TOP_K": top_k,
            "SIMILARITY_THRESHOLD": threshold, "LLM_PROVIDER": provider})
if model_key and model_name:
    cfg[model_key] = model_name


@st.cache_resource(show_spinner=False)
def get_prompts():
    return load_prompts()


# ---------------------------------------------------------------------------
# Main area - knowledge base
# ---------------------------------------------------------------------------
st.title("📄 AUTOSAR HLD Document Analysis Assistant")
st.caption("Retrieval-Augmented Generation over AUTOSAR-style High-Level Design PDFs · "
           "answers are grounded in the uploaded document and cited by page/section.")

col_up, col_info = st.columns([3, 2])
with col_up:
    uploaded = st.file_uploader("Upload AUTOSAR HLD PDF(s)", type=["pdf"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    build_clicked = c1.button("🔨 Build knowledge base", type="primary", disabled=not uploaded or chunk_overlap >= chunk_size)
    sample_clicked = c2.button("📘 Use sample synthetic HLD", disabled=not SAMPLE_PDF.exists() or chunk_overlap >= chunk_size)


def build_kb(paths, names):
    pipe = RAGPipeline(cfg, get_prompts())
    with st.spinner("Extracting, chunking, embedding and indexing... (first run downloads the embedding model)"):
        meta = pipe.build(paths, names)
    st.session_state.pipeline = pipe
    st.session_state.kb_docs = names
    st.session_state.history = []
    st.success(f"Indexed {meta['num_chunks']} chunks from {len(names)} document(s).")


try:
    if build_clicked and uploaded:
        tmp_dir = Path(tempfile.mkdtemp(prefix="hld_upload_"))
        paths, names = [], []
        for f in uploaded:
            p = tmp_dir / Path(f.name).name
            p.write_bytes(f.getbuffer())
            paths.append(p)
            names.append(Path(f.name).name)
        logger.info("User uploaded: %s", names)
        build_kb(paths, names)
    elif sample_clicked:
        build_kb([SAMPLE_PDF], [SAMPLE_PDF.name])
except Exception as exc:
    logger.exception("Ingestion failed")
    st.error(f"Ingestion failed: {exc}")

pipe: RAGPipeline | None = st.session_state.pipeline
with col_info:
    if pipe and pipe.ready:
        st.markdown("**Knowledge base**")
        for name in st.session_state.kb_docs:
            pages = pipe.meta.get("page_counts", {}).get(name, "?")
            st.write(f"• {name} — {pages} pages")
        sections = sorted({c.section for c in pipe.chunks if c.section != 'N/A'})
        st.write(f"• {len(pipe.chunks)} chunks · {len(sections)} sections detected")
        with st.expander("Detected sections"):
            st.write(sections)
    else:
        st.info("Upload a PDF and click **Build knowledge base** (or use the sample).")

# keep runtime settings in sync without re-indexing (retrieval / LLM params only)
if pipe:
    for k in ("TOP_K", "SIMILARITY_THRESHOLD", "LLM_PROVIDER", "GEMINI_MODEL", "GROQ_MODEL", "OPENAI_MODEL", "OLLAMA_MODEL"):
        pipe.cfg[k] = cfg[k]
    sig = pipe.meta.get("signature", {})
    if sig and (sig.get("chunk_size") != chunk_size or sig.get("chunk_overlap") != chunk_overlap):
        st.warning("Chunk settings changed - click Build again to re-index with the new values.")

st.divider()

# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
st.subheader("💬 Ask a question")
examples = ["Which UDS diagnostic services are supported by the BCM?",
            "What happens if the VehicleSpeed signal times out?",
            "What is the cycle time and priority of OsTask_10ms?",
            "Which Bluetooth profile does the BCM use?"]
st.caption("Examples: " + " · ".join(f"_{e}_" for e in examples))

with st.form("ask", clear_on_submit=False):
    question = st.text_input("Your question", placeholder="e.g. Which DTC is stored when the wiper motor stalls?")
    asked = st.form_submit_button("Ask", disabled=not (pipe and pipe.ready))

if asked and question.strip():
    try:
        with st.spinner("Retrieving evidence and generating answer..."):
            st.session_state.history.insert(0, pipe.answer(question))
    except Exception as exc:
        logger.exception("Question failed")
        st.error(f"Error: {exc}")


def render(res, expanded: bool):
    st.markdown(f"#### ❓ {res.question}")
    if res.supported:
        st.markdown(res.answer)
    else:
        st.warning(res.answer)
    if res.note:
        st.caption(f"ℹ️ {res.note}")
    st.caption(f"Provider: {res.provider} · {res.latency_s:.2f}s")

    if res.citations:
        st.markdown("**Citations**")
        for c in res.citations:
            st.markdown(f"- **[{c['source']}]** {c['document']} — Page **{c['page']}** — Section: *{c['section']}* (similarity {c['score']})")

    with st.expander(f"🔎 Retrieved evidence ({len(res.evidence)} chunks)", expanded=expanded):
        thr = float(pipe.cfg["SIMILARITY_THRESHOLD"]) if pipe else 0
        for e in res.evidence:
            flag = "✅ used" if e.score >= thr else "⛔ below threshold"
            st.markdown(f"**[{e.tag}]** score `{e.score:.3f}` {flag} — {e.chunk.doc_name}, page {e.chunk.page}, *{e.chunk.section}*")
            st.text(e.chunk.text)


for i, res in enumerate(st.session_state.history):
    render(res, expanded=(i == 0))
    st.divider()

st.caption("Educational project · uses only synthetic AUTOSAR-style data · no proprietary documents.")