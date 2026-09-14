"""
Streamlit front-end for the RAG chatbot.

Run with:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline
from src.config import DEFAULT_CONFIG

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(DEFAULT_CONFIG)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

pipeline: RAGPipeline = st.session_state.pipeline

with st.sidebar:
    st.header("📁 Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, MD, or CSV files",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest documents", type="primary"):
        with st.spinner("Parsing → chunking → embedding → indexing…"):
            tmp_paths = []
            for f in uploaded:
                tmp_path = Path(tempfile.gettempdir()) / f.name
                tmp_path.write_bytes(f.getbuffer())
                tmp_paths.append(str(tmp_path))
            stats = pipeline.ingest(tmp_paths)
            st.session_state.ingested_files.extend(f.name for f in uploaded)
        st.success(f"Indexed {stats['_total_chunks']} chunks using **{stats['_embedder']}** embeddings.")
        st.json({k: v for k, v in stats.items() if not k.startswith("_")})

    if pipeline.store is not None:
        with st.expander("🔬 Inspect embeddings & vector store"):
            vectors = pipeline.store.all_vectors()
            st.write(f"**Embedder:** {pipeline.embedder.name}")
            st.write(f"**Embedding matrix shape:** {vectors.shape[0]} chunks × {vectors.shape[1]} dims")
            backend = "FAISS IndexFlatIP" if hasattr(pipeline.store, "_index") else "NumPy matrix"
            st.write(f"**Vector store backend:** {backend}")
            chunk_idx = st.number_input(
                "View vector for chunk #", min_value=0,
                max_value=len(pipeline.store.chunks) - 1, value=0,
            )
            st.caption(pipeline.store.chunks[chunk_idx].text[:200] + "…")
            st.write(f"L2 norm: {(vectors[chunk_idx] ** 2).sum() ** 0.5:.4f}")
            st.dataframe(vectors[chunk_idx].reshape(1, -1))

    if st.session_state.ingested_files:
        st.caption("Indexed so far:")
        for name in st.session_state.ingested_files:
            st.write(f"• {name}")

    st.divider()
    st.header("⚙️ Retrieval settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, DEFAULT_CONFIG.retrieval.top_k)
    use_mmr = st.checkbox("Diversify results (MMR)", value=DEFAULT_CONFIG.retrieval.use_mmr)
    pipeline.cfg.retrieval.top_k = top_k
    pipeline.cfg.retrieval.use_mmr = use_mmr

    generator_name = type(pipeline.generator).__name__
    st.divider()
    st.caption(f"Generator: **{generator_name}**" +
               ("" if generator_name == "AnthropicGenerator" else " — set ANTHROPIC_API_KEY to use Claude"))

    if st.button("Clear conversation"):
        pipeline.reset_history()
        st.session_state.messages = []
        st.rerun()

st.title("📚 RAG Chatbot")
st.caption("Ask questions grounded in the documents you've uploaded — every answer is cited back to its source chunk.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']}, similarity {s['score']})")
                    st.caption(s["preview"] + "…")

if query := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating an answer…"):
            result = pipeline.chat(query)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📎 Sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']}, similarity {s['score']})")
                    st.caption(s["preview"] + "…")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
