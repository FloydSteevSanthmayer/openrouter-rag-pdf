# FLOWCHART_DETAILED.md

This document explains every step in the flowchart in detail — suitable for technical reviewers, architects, and maintainers.

## Overview
The system is a Streamlit-based RAG (Retrieval Augmented Generation) application. It accepts PDF uploads, extracts and chunks text, computes local embeddings with SentenceTransformers, indexes them in FAISS, retrieves relevant chunks for a user question, and uses OpenRouter as the chat LLM to answer.

---

## Step-by-step breakdown

### 1) Streamlit UI (User Input)
- `file_uploader()` receives the PDF bytes.
- The upload triggers the extraction stage. File validation occurs (MIME-type, extension).

### 2) PDF Extraction
- `extract_text_from_pdf(file_bytes)` uses PyMuPDF (`fitz.open(stream=..., filetype='pdf')`) to read pages and extract text via `page.get_text('text')`.
- Potential improvements: OCR fallback (Tesseract) if pages are images; header/footer stripping and column reflow for multi-column PDFs.

### 3) Chunking
- `chunk_text(text, chunk_size=500, overlap=50)` splits on whitespace into word-based chunks.
- Recommendation: token-aware chunking (using an encoder like tiktoken) to ensure model token limits are respected.

### 4) Embedding Model
- `load_embedder()` loads a `SentenceTransformer` model (cached via Streamlit).
- For large documents or production, consider running embeddings on GPU or using a managed embedding service if latency is a concern.

### 5) Embeddings and Indexing
- `embed_texts_local()` batches text encodings to avoid OOM and normalizes vectors for cosine via `faiss.normalize_L2()`.
- `build_faiss_index()` uses `faiss.IndexFlatIP` for exact inner-product search. For large-scale use, use IVF/HNSW + quantization.

### 6) Retrieval
- `retrieve_top_k()` embeds the question and performs `index.search()` to return top-k chunk ids and scores.
- Include score thresholds and de-duplication to avoid returning near-duplicate chunks.

### 7) Prompt Construction & LLM Call
- Construct a system message and user message with concatenated top chunks plus the question.
- `call_openrouter_chat()` posts JSON payload to OpenRouter; includes error handling, timeout, and retries for robustness.
- Advise the system prompt to instruct the LLM to cite chunk provenance and avoid hallucination.

### 8) Output and Follow-ups
- The answer is displayed in the Streamlit UI.
- Optionally a second LLM call suggests follow-up questions and can be used to drive conversation state.

---

## Operational considerations
- **Secrets**: never hard-code API keys. Use `.env` or `st.secrets` and rotate leaked keys immediately.
- **Scaling**: persist vector indices to disk; precompute embeddings for frequently used docs; consider approximate indexing.
- **Testing**: unit-test text extraction, chunking edge cases, FAISS retrieval results, and prompt formatting.
- **Monitoring**: log latency for embedding & LLM calls, error rates, and top-k retrieval distribution.
