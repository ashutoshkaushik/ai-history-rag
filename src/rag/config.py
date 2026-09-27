"""Single place for every tunable setting, so experiments are config changes, not code changes."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Paths
CORPUS_MANIFEST = ROOT / "corpus" / "manifest.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
EVAL_DIR = ROOT / "eval"

# Models
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
# The judge is a different, stronger model than the generator. A gpt-4.1-mini judge approved
# all 57 claims in a manual audit where a human found 6 unsupported (eval/manual_audit.yaml).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE = 0.0

# Targets from the one-liner
FAITHFULNESS_TARGET = 0.90
LATENCY_P95_TARGET_S = 8.0

# Chunking (tokens measured with the embedding model's tokenizer, cl100k_base)
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 90
# Prefix each chunk with "[Title (Year) | Section]" so chunks that never name
# their source (e.g. a mid-paper paragraph) still match queries about it.
CHUNK_CONTEXT_HEADER = True

# Vector store
COLLECTION_NAME = "ai_history"

# Retrieval
TOP_K = 5
DEFAULT_STRATEGY = "vector"
# Cheap first gate for "I don't know": if even the best chunk is this dissimilar,
# refuse without calling the LLM. Phase 3 measurement: off-topic questions top out
# at ~0.21, answerable questions start at ~0.48.
MIN_SIMILARITY = 0.35

# Generation
REFUSAL_MESSAGE = (
    "I couldn't find sufficient evidence in the AI-history corpus to answer this question."
)

# Public deployment (Streamlit Community Cloud). Set PUBLIC_DEPLOYMENT=1 in the app's secrets to turn on
# visitor usage caps and hide full text of documents that aren't openly licensed. Local runs are unaffected.
PUBLIC_DEPLOYMENT = os.getenv("PUBLIC_DEPLOYMENT") == "1"
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "10"))
MAX_QUESTIONS_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "150"))
OPEN_LICENSES = ("CC BY-SA 4.0",)   # full document text may be shown publicly only for these
