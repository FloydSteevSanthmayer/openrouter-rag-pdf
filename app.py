import os
import re
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
if not API_KEY:
    # we will not stop the app entirely to allow documentation viewing, but actions requiring the key will warn.
    st.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY in environment or Streamlit secrets to enable LLM calls.")

# --------------------- Helpers ---------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def extract_text_from_pdf(file_bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        return "\n".join(pages).strip()
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def chunk_text(text: str, chunk_size:int=500, overlap:int=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_texts_local(texts, batch_size=64):
    embedder = load_embedder()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        arr = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(arr)
    embs = np.vstack(embs).astype('float32')
    faiss.normalize_L2(embs)
    return embs

def build_faiss_index(chunks):
    embs = embed_texts_local(chunks)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, chunks

def retrieve_top_k(index, chunks, question, k=4):
    q_emb = embed_texts_local([question])
    scores, ids = index.search(q_emb, k)
    results = []
    for idx in ids[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def call_openrouter_chat(messages, timeout=15):
    if not API_KEY:
        raise RuntimeError("OpenRouter API key is missing. Set OPENROUTER_API_KEY.")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.0}
    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("📄 RAG PDF Q&A — OpenRouter + Local Embeddings")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file.read())
    if not raw_text:
        st.error("Failed to extract text from the uploaded PDF.")
        st.stop()

    chunks = chunk_text(raw_text)
    with st.spinner("Creating vector index..."):
        index, chunk_list = build_faiss_index(chunks)
    st.success(f"Indexed {len(chunk_list)} chunks.")

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Retrieving relevant context..."):
            top_chunks = retrieve_top_k(index, chunk_list, question, k=4)
        prompt = [
            {"role":"system","content":"You are a helpful assistant. Use the provided document excerpts only and cite provenance when possible."},
            {"role":"user","content":"Use the following document excerpts to answer the question:\\n\\n" + "\\n---\\n".join(top_chunks) + f"\\n---\\nQuestion: {question}"}
        ]
        try:
            with st.spinner("Generating answer via OpenRouter..."):
                answer = call_openrouter_chat(prompt)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"LLM call failed: {e}")

        with st.expander("Follow-up Questions"):
            try:
                fu_prompt = f\"Based on this answer, suggest three concise follow-up questions: {answer}\"
                followups = call_openrouter_chat([{\"role\":\"user\",\"content\":fu_prompt}]).splitlines()
                for i, q in enumerate(followups,1):
                    st.write(f\"{i}. {re.sub(r'^\\s*\\d+\\.\\s*','',q)}\")
            except Exception as e:
                st.info(\"Follow-up generation unavailable: set OPENROUTER_API_KEY and try again.\")

else:
    st.info("Please upload a PDF to begin.")
