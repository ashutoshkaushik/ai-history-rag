# AI History Research Assistant

Citation-grounded RAG over a curated corpus of ~25 AI-history sources, plus an LLM-only baseline for comparison.
Course: The Gen Academy Week 2 (LangChain + LangGraph track). The phased plan and progress checkboxes are in `PROJECT_PLAN.md`; read it first.

## Stack
- Python 3.12 managed by `uv` (always `uv run ...`; add deps with `uv add`, never pip)
- LangChain + LangGraph, ChromaDB (local, `data/chroma/`)
- OpenAI: `gpt-4.1-mini` for chat/judge, `text-embedding-3-small` for embeddings. The API budget is about $5, so keep calls lean and cache where possible.
- All tunables live in `src/rag/config.py`. Experiments change config, not code.

## Commands
- Smoke test: `uv run python scripts/smoke_test.py`
- Download corpus: `uv run python scripts/download_corpus.py`
- Build/update index (incremental): `uv run python -m rag.ingest` (`--rebuild` for a full rebuild)
- Try retrieval: `uv run python -m rag.retrieve "your question"`
- Chunk viewer: `uv run python scripts/visualize_chunks.py`, then open `data/processed/chunk_viewer.html` (contains corpus text; never publish)
- Embedding viewer: `uv run python scripts/visualize_embeddings.py`, then open `data/processed/embedding_viewer.html` (2D map, raw vectors, question → top-5 rays)
- Export a doc's cleaned text (e.g. for ChunkViz): `uv run python scripts/export_clean_text.py <doc_id>`
- Chat UI + RAG Lab: `uv run streamlit run app.py` (pages registered with st.navigation in app.py in pipeline order; Lab page bodies in `lab/rag_lab.py`, embedded viewers in `lab/explore_pages.py`). Restart the server after editing `src/rag` or `lab/` modules: imported modules are not reloaded.
- Evaluate: `make eval LABEL=name` · re-grade saved answers: `uv run python -m rag.evaluate --rejudge <run_id> [--keep-faithfulness]`
- Ask (RAG): `uv run python -m rag.ask "question"` (`--compare` for RAG vs LLM-only, `--context` to show chunks)

## Code layout (mirrors the course app's structure)
- `models.py` get_llm / get_embeddings · `retrieve.py` get_vector_store, retrieve(question, strategy), RAG_STRATEGIES
- `prompts.py` RAG_PROMPT / GRADE_PROMPT / LLM_ONLY_PROMPT · `rag_graph.py` LangGraph flow + answer_question()
- `baseline.py` llm_only_answer() / llm_only_stream() · `ingest.py` load → clean → chunk → embed
- `retrieve.py` also has bm25_search(), rrf_fuse(), hybrid_search() (used by the Lab; not yet a chat strategy)

## Deployment
- Streamlit Community Cloud; see README "Deploy publicly". Public mode = `PUBLIC_DEPLOYMENT=1` (usage caps in `lab/usage.py`, only CC BY-SA full text). A fresh start runs `rag.bootstrap` to build data/.
- Theme: `.streamlit/config.toml` (serif headings/answers, warm palette, light + dark).

## Conventions
- Package code in `src/rag/`, one-off scripts in `scripts/`, eval data in `eval/`.
- `data/` is gitignored. Source PDFs are downloaded by script, never committed.
- `Mastering-Agentic-AI-Week2-Session1-main/` is course lesson material, for reference only. Don't modify or import from it.
- Every chunk carries full manifest metadata (title, authors, year, era, source_type, document_type, url, page, section).
- Keep it simple: add hybrid retrieval or reranking only when eval numbers justify it.
