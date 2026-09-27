# AI History Research Assistant

A citation-grounded RAG application that answers questions about the history of artificial intelligence from a curated corpus of primary papers, historical documents, and secondary sources. It also measures when RAG beats an LLM answering from memory.

## Setup
```bash
uv sync
cp .env.example .env   # then add your OPENAI_API_KEY
uv run python scripts/smoke_test.py
```

_More sections (architecture, results, usage) are added as the project progresses. See `PROJECT_PLAN.md`._

## Deploy publicly (Streamlit Community Cloud)

1. Push this repo to GitHub (it can be private). `data/` and `.env` are git-ignored: source documents are downloaded from their original hosts, never redistributed.
2. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the repo, set the main file to `app.py` and, under *Advanced settings*, the Python version to **3.12**.
3. In *Advanced settings → Secrets*, paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   PUBLIC_DEPLOYMENT = "1"
   MAX_QUESTIONS_PER_SESSION = "10"
   MAX_QUESTIONS_PER_DAY = "150"
   ```
4. Deploy. The first start builds the index (download, embed, viewers: about 3 minutes, about $0.01), and later starts reuse it until the app restarts.

Public mode caps visitor questions (each costs a fraction of a cent) and shows full document text only for openly licensed (CC BY-SA) sources. Also set a hard monthly limit at platform.openai.com → Limits.
