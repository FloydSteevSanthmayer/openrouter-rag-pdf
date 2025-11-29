import os
import re
import math
import time
from typing import List, Tuple, Dict, Any

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# --------------------- Configuration ---------------------
CHAT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo-0613")
CHAT_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# API key handling: prefer environment or Streamlit secrets
API_KEY = os.environ.get("OPENROUTER_API_KEY") or (st.secrets.get("OPENROUTER_API_KEY") if "OPENROUTER_API_KEY" in st.secrets else None)

# --------------------- Helpers / Utilities ---------------------
@st.cache_resource
def load_embedder(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)


def extract_text_from_pdf(file_bytes: bytes) -> List[str]:
    """
    Extract plain text per-page from a PDF bytes object.
    Returns a list of page texts (index 0 == page 1).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        return pages
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return []


def chunk_text_pages(pages: List[str], chunk_size_words: int = 500, overlap_words: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk text by page, output list of dicts:
    {"text": ..., "page": page_number (1-indexed), "start_word": i, "end_word": j}
    """
    chunks = []
    for page_index, page_text in enumerate(pages):
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text": chunk_text,
                "page": page_index + 1,
                "start_word": start,
                "end_word": end
            })
            start += chunk_size_words - overlap_words
    return chunks


def embed_texts_local(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings locally using SentenceTransformer in batches.
    Returns normalized float32 numpy array with shape (n_texts, dim).
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    embedder = load_embedder()
    embs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        arr = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs_list.append(arr)
    embs = np.vstack(embs_list).astype(np.float32)
    # Normalize vectors (cosine similarity via inner product)
    faiss.normalize_L2(embs)
    return embs


def build_faiss_index(chunks: List[Dict[str, Any]]) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Build a FAISS IndexFlatIP from the chunk texts.
    Returns (index, chunks) where index contains embeddings.
    """
    texts = [c["text"] for c in chunks]
    embs = embed_texts_local(texts)
    if embs.size == 0:
        raise ValueError("No embeddings generated (empty document?)")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, chunks


def retrieve_top_k(index: faiss.IndexFlatIP, chunks: List[Dict[str, Any]], question: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
    """
    Retrieve top-k chunks for a question. Returns list of (chunk_dict, score).
    """
    q_emb = embed_texts_local([question])
    if q_emb.size == 0:
        return []
    scores, ids = index.search(q_emb, k)
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((chunks[idx], float(score)))
    return results


def call_openrouter_chat(messages: List[Dict[str, str]], timeout: int = 15) -> str:
    """
    Call OpenRouter chat completions endpoint and return the assistant content.
    Raises exception on HTTP error or unexpected format.
    """
    if not API_KEY:
        raise RuntimeError("OpenRouter API key is not set. Set OPENROUTER_API_KEY env or st.secrets.")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.0}
    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    # Defensive access
    try:
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response format: {e} / {j}")


def build_prompt_from_chunks(question: str, retrieved: List[Tuple[Dict[str, Any], float]], max_chunks: int = 4) -> List[Dict[str, str]]:
    """
    Build messages (system + user) for the chat model using retrieved chunks and a question.
    Each chunk includes provenance (page number).
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer using only the provided document excerpts. "
            "If the information is not present in the excerpts, say you don't know. "
            "When you reference content, include the page number of the excerpt."
        )
    }
    # Keep only up to max_chunks
    retrieved = retrieved[:max_chunks]
    context_parts = []
    for (chunk, score) in retrieved:
        context_parts.append(f"[Source: page {chunk['page']}] {chunk['text']}")
    user_content = "Use the following document excerpts to answer the question:\n\n" + "\n\n---\n\n".join(context_parts) + f"\n\nQuestion: {question}"
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]


# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("📄 RAG PDF Q&A — OpenRouter + Local Embeddings")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (words)", value=500, step=50)
    overlap = st.number_input("Chunk overlap (words)", value=50, step=10)
    top_k = st.slider("Top-k chunks to retrieve", min_value=1, max_value=8, value=4)
    batch_size = st.number_input("Embedding batch size", value=64, step=8)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
index_obj = None
chunks_meta = []

if uploaded_file:
    pages = extract_text_from_pdf(uploaded_file.read())
    if not pages:
        st.error("No text extracted from PDF.")
        st.stop()

    st.info(f"Extracted {len(pages)} pages.")
    # Chunk pages with provenance
    chunks_meta = chunk_text_pages(pages, chunk_size_words=int(chunk_size), overlap_words=int(overlap))
    if not chunks_meta:
        st.error("No chunks produced. Try reducing chunk size or checking the PDF content.")
        st.stop()

    with st.spinner("Creating vector index (embedding chunks)..."):
        try:
            # Reuse batch_size from UI by calling embed_texts_local indirectly in build_faiss_index
            # (we pass batch_size via monkeypatching the function call here)
            # Simpler: call embed_texts_local with UI batch_size directly
            texts = [c["text"] for c in chunks_meta]
            embs = embed_texts_local(texts, batch_size=int(batch_size))
            dim = embs.shape[1]
            index_obj = faiss.IndexFlatIP(dim)
            index_obj.add(embs)
            st.success(f"Indexed {len(chunks_meta)} chunks (dim={dim}).")
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            st.stop()

    # Show small preview of chunks
    if st.checkbox("Show chunk previews (first 5)"):
        for i, c in enumerate(chunks_meta[:5], 1):
            st.markdown(f"**Chunk {i} — page {c['page']} (words {c['start_word']}-{c['end_word']})**")
            st.write(c["text"][:1000] + ("..." if len(c["text"]) > 1000 else ""))

    # User question
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Retrieving relevant chunks..."):
            try:
                retrieved = retrieve_top_k(index_obj, chunks_meta, question, k=int(top_k))
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                retrieved = []

        if not retrieved:
            st.warning("No relevant chunks found.")
        else:
            # Build prompt and call LLM
            messages = build_prompt_from_chunks(question, retrieved, max_chunks=int(top_k))
            try:
                with st.spinner("Generating answer via OpenRouter..."):
                    answer = call_openrouter_chat(messages)
                st.subheader("Answer")
                st.write(answer)

                # Show sources used
                with st.expander("Context chunks and provenance"):
                    for idx, (chunk, score) in enumerate(retrieved, 1):
                        st.markdown(f"**{idx}. Page {chunk['page']} — score: {score:.4f}**")
                        st.write(chunk['text'][:1000] + ("..." if len(chunk['text']) > 1000 else ""))

                # Follow-up generation (fixed f-string usage)
                with st.expander("Follow-up Questions"):
                    try:
                        fu_prompt = f"Based on this answer, suggest three concise follow-up questions: {answer}"
                        fu_messages = [
                            {"role": "system", "content": "You are a helpful assistant. Provide concise numbered follow-up questions."},
                            {"role": "user", "content": fu_prompt}
                        ]
                        followups_text = call_openrouter_chat(fu_messages)
                        # split lines and clean numbers
                        followups = [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in followups_text.splitlines() if line.strip()]
                        # if model returned as one paragraph, try splitting on semicolons or periods
                        if len(followups) == 0:
                            # attempt lightweight split
                            candidates = re.split(r'\n|;|\u2022|\*', followups_text)
                            followups = [re.sub(r'^\s*\d+\.\s*', '', c).strip() for c in candidates if c.strip()][:3]
                        for i, q in enumerate(followups[:3], 1):
                            st.write(f"{i}. {q}")
                    except Exception:
                        st.info("Follow-up generation unavailable: ensure OPENROUTER_API_KEY is set and try again.")

            except Exception as e:
                st.error(f"LLM call failed: {e}")
else:
    st.info("Please upload a PDF to begin.")
