"""Build everything under data/ on a fresh machine (e.g. the first start on Streamlit Community Cloud).

data/ is not in git: source documents are downloaded from their original hosts rather than
redistributed. This runs the same scripts you would run locally, once, only when the index is missing:
download -> ingest (embed into Chroma, ~$0.01) -> chunk viewer -> embedding viewer.
"""

import os
import subprocess
import sys

from rag import config

STEPS = [
    ("Downloading the 55 source documents", ["scripts/download_corpus.py"]),
    ("Cleaning, chunking and embedding them into Chroma", ["-m", "rag.ingest"]),
    ("Building the chunk viewer", ["scripts/visualize_chunks.py"]),
    ("Building the embedding viewer", ["scripts/visualize_embeddings.py"]),
]


def chunk_count() -> int:
    """Number of chunks in the current index (it can differ slightly between builds: Wikipedia is fetched live)."""
    path = config.PROCESSED_DIR / "chunks.jsonl"
    return sum(1 for _ in path.open()) if path.exists() else 0


def index_ready() -> bool:
    return (config.PROCESSED_DIR / "chunks.jsonl").exists() and config.CHROMA_DIR.exists()


def build(report=print) -> None:
    env = {**os.environ, "PYTHONPATH": str(config.ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for label, args in STEPS:
        report(label)
        # A failed download of one document is reported but not fatal; ingest skips missing files.
        subprocess.run([sys.executable, *args], cwd=config.ROOT, env=env, check=label.startswith("Cleaning"))


if __name__ == "__main__":
    build()
