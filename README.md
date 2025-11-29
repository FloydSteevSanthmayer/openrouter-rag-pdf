# RAG PDF Q&A — Streamlit (OpenRouter + Local Embeddings)

Professional, documented repository for a Retrieval-Augmented-Generation (RAG) PDF Q&A Streamlit application.
This package includes the app launcher, CI, docs, mermaid sources and a colored flowchart — ready to commit to GitHub.

## Contents
- `app.py` — Streamlit launcher (improved, safer defaults)
- `docs/flowchart_colored.png` — rendered flowchart (placeholder image)
- `docs/architecture_colored.png` — rendered architecture diagram (placeholder image)
- `flowchart_colored.mmd`, `architecture.mmd` — mermaid sources
- `FLOWCHART_DETAILED.md` — in-depth step-by-step explanation
- `Dockerfile`, `requirements.txt`, `.env.example`, `.gitignore`
- CI: `.github/workflows/ci.yml`, `.github/dependabot.yml`
- `tests/` — pytest scaffold
- `LICENSE` — MIT
- `CONTRIBUTING.md` — contribution guide

## Quick start (local)
1. Copy `.env.example` -> `.env` and fill `OPENROUTER_API_KEY` (do **not** commit `.env`).
2. Build and run locally:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
or with Docker:
```bash
docker build -t rag-pdf-qa .
docker run --env-file .env -p 8501:8501 rag-pdf-qa
```

## Files of note
- `flowchart_colored.mmd` contains the mermaid source used to create `docs/flowchart_colored.png`.
- `FLOWCHART_DETAILED.md` explains internal steps for technical reviewers.
- `app.py` includes safer handling for API keys and batching for embeddings.

---
_For visuals, the docs include placeholder PNGs that you can replace with exports from Figma or a real mermaid renderer._
