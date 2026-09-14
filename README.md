# 📚 RAG Chatbot — Document Q&A with Citations

An end-to-end Retrieval-Augmented Generation chatbot. Upload documents,
ask questions in natural language, get answers grounded in your own
content with source citations — not the LLM's memory.

**Use case chosen:** an internal knowledge-base / customer-support
assistant (policy handbooks, product manuals, contracts, onboarding
docs) — the kind of thing support teams and new hires currently do by
Ctrl-F-ing PDFs.

## Pipeline

```
Upload → Parse → Chunk → Embed → Vector Store → Retrieve (+MMR) → Generate (+citations)
```

| Stage | Implementation | Fallback (offline / no API key) |
|---|---|---|
| Parsing | `pypdf`, `python-docx`, native for `.txt/.md/.csv` | — |
| Chunking | Recursive splitter (paragraph → sentence → char), with overlap | — |
| Embedding | `sentence-transformers` (`all-MiniLM-L6-v2`) | TF-IDF (scikit-learn, zero downloads) |
| Vector store | FAISS `IndexFlatIP` (cosine via normalized vectors) | NumPy matrix + cosine similarity |
| Retrieval | Top-`fetch_k` recall pass → **MMR** re-rank to `top_k` for diversity | — |
| Generation | Claude (`claude-sonnet-4-6`) via `anthropic` SDK, cites `[Source N]` | Extractive: scores sentences in retrieved chunks by keyword overlap, stitches a cited answer with zero fabrication risk |

Every "primary" component degrades to a fallback automatically — the app
never crashes just because a model download or API key is missing. This
was a deliberate design choice: `build_embedder()` and `build_generator()`
try the higher-quality path first and silently fall back, so the same
code runs identically on a laptop with full internet access and on a
locked-down grading sandbox with none.

## Why these design choices

- **Recursive chunking with overlap** instead of naive fixed-size
  slicing — avoids splitting mid-sentence and keeps context continuous
  across chunk boundaries.
- **MMR re-ranking** — plain top-k similarity search tends to return 5
  near-duplicate chunks about the same sub-topic. MMR trades a bit of
  raw relevance for diversity, which noticeably improves answer quality
  on multi-part questions.
- **Citations by construction** — the generator is given numbered
  source blocks and instructed (or, in extractive mode, structurally
  forced) to tag every claim with `[Source N]`, and the UI shows the
  underlying chunk text so answers are auditable.
- **Persistence** — `pipeline.save()` / `RAGPipeline.load()` serialize
  the vector store and embedder state to disk, so a deployed app
  doesn't need to re-ingest documents on every restart.
- **Conversation memory** — the pipeline keeps a bounded rolling history
  so follow-up questions ("what about after that period?") resolve
  correctly.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Optional, for higher-quality generation:
export ANTHROPIC_API_KEY=sk-ant-...
```

`sentence-transformers`, `faiss-cpu`, and `anthropic` are optional. If
they're not installed (or the model/API can't be reached), the app
automatically uses the TF-IDF + extractive fallback and still works
fully offline.

## Run

**Chat UI:**
```bash
streamlit run app.py
```
Upload files in the sidebar, click **Ingest documents**, then chat.

**Headless demo (no browser needed):**
```bash
python cli_demo.py                                   # uses data/sample_policy.md
python cli_demo.py path/to/your.pdf "your question"   # any doc + question
```

**Retrieval quality check:**
```bash
python eval_retrieval.py
```

**Tests:**
```bash
pytest tests/ -v
```

## Project layout

```
rag_chatbot/
├── app.py                # Streamlit chat UI
├── cli_demo.py           # headless pipeline demo
├── eval_retrieval.py     # retrieval hit-rate eval harness
├── requirements.txt
├── data/sample_policy.md # sample doc used by the demo/tests
├── src/
│   ├── config.py         # all tunable knobs (chunk size, top_k, MMR lambda, model names…)
│   ├── loaders.py         # Upload → Parse
│   ├── chunker.py         # Parse → Chunk
│   ├── embedder.py        # Chunk → Embedding (+ offline fallback)
│   ├── vectorstore.py     # Embedding → Vector Storage (+ offline fallback)
│   ├── retriever.py       # Vector Storage → Retrieval (+ MMR)
│   ├── llm.py              # Retrieval → Response Generation (+ extractive fallback)
│   └── pipeline.py         # orchestrates everything + memory + persistence
└── tests/test_pipeline.py
```

## Possible extensions (not implemented, but the architecture supports them)

- Swap `VectorStore` for a managed DB (pgvector / Pinecone / Qdrant) by
  implementing the same `add`/`search` interface.
- Hybrid retrieval: combine the current dense/TF-IDF score with BM25.
- Streaming token-by-token responses in the Streamlit UI.
- Per-user document namespaces for a multi-tenant deployment.
- Re-ranking retrieved chunks with a cross-encoder before generation.
