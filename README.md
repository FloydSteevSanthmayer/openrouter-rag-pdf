# openrouter-rag-pdf

> **OpenRouter-style Retrieval-Augmented PDF Q&A**  
> Streamlit demo that extracts text from PDFs, creates local SentenceTransformer embeddings, indexes them with FAISS, and uses OpenRouter-compatible chat completions to answer user questions with provenance.

**Important:** This repository is **independently maintained** and is **not affiliated with, sponsored by, or endorsed by OpenRouter**. See **Disclaimer** at the bottom.

---

## Table of contents
1. [Quick badges](#quick-badges)  
2. [What this is](#what-this-is)  
3. [Features](#features)  
4. [Quick start](#quick-start)  
5. [Configuration / Environment](#configuration--environment)  
6. [Running with Docker](#running-with-docker)  
7. [How it works (high level)](#how-it-works-high-level)  
8. [Files of interest](#files-of-interest)  
9. [Troubleshooting](#troubleshooting)  
10. [Contributing & CI](#contributing--ci)  
11. [License](#license)  
12. [Disclaimer](#disclaimer)

---

## Quick badges
```
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]
```
(Replace or remove these badge lines as you prefer.)

---

## What this is
`openrouter-rag-pdf` is a developer-focused starter kit and demo for building a Retrieval-Augmented Generation (RAG) workflow around PDF documents. It demonstrates:

- ingesting PDF → extracting per-page text  
- chunking with provenance (page numbers + word offsets)  
- computing local embeddings (SentenceTransformers)  
- vector retrieval with FAISS  
- constructing prompts and calling an OpenRouter-compatible chat endpoint for answers  
- a minimal Streamlit UI for interaction

This repo is intended as a template for proof-of-concepts and small demos. Use it as a base to harden for production.

---

## Features
- Streamlit UI for uploading PDFs and asking questions  
- Per-page extraction via PyMuPDF (fast, reliable for text PDFs)  
- Chunking with overlap and provenance metadata (page number + offsets)  
- Local embeddings (`sentence-transformers`, default `all-MiniLM-L6-v2`) with batching support  
- FAISS `IndexFlatIP` with normalized vectors for cosine-style retrieval  
- OpenRouter-compatible chat completions integration for final answers and follow-ups  
- Dockerfile, CI scaffold, mermaid sources, placeholder diagrams, tests scaffold, and contributing guide

---

## Quick start

### Requirements
- Python 3.10+ (3.11 recommended)  
- `pip` (or use Docker)  
- An OpenRouter API key for LLM calls (`OPENROUTER_API_KEY`)

### Local (recommended)
```bash
# create and activate a virtual environment
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY (do NOT check .env into git)
streamlit run app.py
```
Open `http://localhost:8501` to use the app.

---

## Configuration / Environment

Supported environment variables (defaults shown):
- `OPENROUTER_API_KEY` — **(required for LLM calls)** your OpenRouter API key.
- `OPENROUTER_URL` — default: `https://openrouter.ai/api/v1/chat/completions`
- `OPENROUTER_MODEL` — default: `openai/gpt-3.5-turbo-0613`
- `EMBED_MODEL_NAME` — default: `all-MiniLM-L6-v2`

Use `.env` for local development and `st.secrets` for Streamlit Cloud. **Never commit secrets**.

`.env.example` is included as a template.

---

## Running with Docker
```bash
docker build -t openrouter-rag-pdf .
docker run --env-file .env -p 8501:8501 openrouter-rag-pdf
```
The container runs Streamlit on port `8501`. Ensure `.env` contains `OPENROUTER_API_KEY`.

---

## How it works (high level)
1. **Upload PDF** — user uploads a PDF via Streamlit.  
2. **Extract text** — per-page text is extracted using PyMuPDF. For scanned PDFs, add OCR (Tesseract) as a pre-step.  
3. **Chunking** — text is split into word-based chunks with overlap; each chunk stores page provenance.  
4. **Embedding** — chunks are embedded locally using SentenceTransformers; embeddings are normalized and batched.  
5. **Indexing** — FAISS `IndexFlatIP` stores vectors for fast retrieval.  
6. **Retrieval** — user question is embedded; top-k chunks are retrieved using inner product (cosine).  
7. **Prompting & LLM** — retrieved chunks + question are compiled into a system+user message and sent to OpenRouter-compatible chat completions. The assistant is instructed to only use provided excerpts and cite page provenance.  
8. **Output** — the answer and source chunks (with page numbers and scores) are shown; follow-up questions can be generated.

---

## Files of interest
- `app.py` — main Streamlit application  
- `flowchart_colored.mmd`, `architecture.mmd` — mermaid sources for diagrams  
- `docs/flowchart_colored.png`, `docs/architecture_colored.png` — placeholder diagram images (replace with Figma or mermaid exports)  
- `Dockerfile`, `requirements.txt`, `.env.example`, `.gitignore`  
- `.github/workflows/ci.yml`, `.github/dependabot.yml` — CI & dependency automation  
- `tests/` — pytest scaffold with basic smoke test  
- `LICENSE` — MIT (modify if needed)  
- `CONTRIBUTING.md` — contribution guidelines

---

## Troubleshooting (common issues)
- **SyntaxError from escaped quotes**  
  If you see `SyntaxError: unexpected character after line continuation character`, search for bad escaped f-strings like `f\"...\"` and replace with plain `f"..."`.

- **No text extracted**  
  PDF could be scanned images. Add OCR pre-processing or use an OCR-enabled extractor.

- **FAISS dimension mismatch / errors**  
  Ensure all embeddings are `float32`, normalized, and share the same vector dimension.

- **LLM call fails**  
  Verify `OPENROUTER_API_KEY` is set, the network can reach the endpoint, and `OPENROUTER_URL` is correct.

---

## Contributing & CI
- Use feature branches and open PRs.  
- A GitHub Actions CI workflow (`.github/workflows/ci.yml`) installs dependencies and runs tests.  
- Pre-commit config is provided in `.pre-commit-config.yaml` for basic checks (trailing whitespace, EOF fixer, flake8).

---

## License
This project is distributed under the **MIT License**. See `LICENSE` for full terms.

---

## Disclaimer
This repository is independently maintained and **is not affiliated with, endorsed by, or sponsored by OpenRouter**. The name OpenRouter is used to indicate compatibility only.
