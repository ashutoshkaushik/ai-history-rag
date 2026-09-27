"""Retrieval over the Chroma index.

Mirrors the course app: cached get_* factories plus retrieve(question, strategy).
Strategies are added here as they are measured (Phase 6 adds "hybrid").
"""

import json
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag import config
from rag.models import get_embeddings

RAG_STRATEGIES = {
    "vector": "Dense semantic search (text-embedding-3-small, cosine)",
}


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ------------------------------------------------------------------ keyword (BM25)

STOPWORDS = set("""a an and are as at be by did do does for from had has have how in into is it its of on or
that the their them then there these this those to was were what when where which who whom why with
would could should about after before between during than more most such also can may""".split())


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lowercase words/numbers (keeps "gpt-3", "1.3b", "13,500"), minus stopwords."""
    words = re.findall(r"[a-z0-9]+(?:[.,'\-][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """BM25 over the SAME chunks as the vector index (built from data/processed/chunks.jsonl),
    so both retrievers search identical units, as in the course's hybrid notebook."""
    docs = []
    with (config.PROCESSED_DIR / "chunks.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            docs.append(Document(page_content=c["text"], metadata=c["metadata"], id=c["id"]))
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize)


def bm25_search(question: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Top-k chunks with their raw BM25 score (unbounded; higher = more keyword overlap)."""
    bm25 = get_bm25_retriever()
    scores = bm25.vectorizer.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(bm25.docs[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------- hybrid (RRF)

def rrf_fuse(ranked: dict[str, list[Document]], weights: dict[str, float], c: int = 60,
             k: int = config.TOP_K) -> list[tuple[Document, float, dict[str, int]]]:
    """Weighted Reciprocal Rank Fusion: score(doc) = sum over lists of weight / (c + rank).

    Same formula as langchain_classic's EnsembleRetriever (used in the course's hybrid notebook).
    Returns (doc, rrf_score, {list_name: rank}) for the top k.
    """
    scores, ranks, by_id = {}, {}, {}
    for name, docs in ranked.items():
        for rank, doc in enumerate(docs, 1):
            by_id[doc.id] = doc
            scores[doc.id] = scores.get(doc.id, 0.0) + weights[name] / (c + rank)
            ranks.setdefault(doc.id, {})[name] = rank
    top = sorted(scores, key=lambda i: -scores[i])[:k]
    return [(by_id[i], scores[i], ranks[i]) for i in top]


def hybrid_search(question: str, k: int = config.TOP_K, pool: int = 20, vector_weight: float = 0.5,
                  c: int = 60) -> list[tuple[Document, float, dict[str, int]]]:
    """Fuse the top-`pool` vector and BM25 results with RRF and keep the top k."""
    vec = [d for d, _ in retrieve(question, "vector", k=pool)]
    kw = [d for d, _ in bm25_search(question, k=pool)]
    return rrf_fuse({"vector": vec, "bm25": kw}, {"vector": vector_weight, "bm25": 1 - vector_weight}, c=c, k=k)


def retrieve(question: str, strategy: str = config.DEFAULT_STRATEGY,
             k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Top-k chunks with a relevance score (cosine similarity for "vector": 1.0 = identical)."""
    if strategy == "vector":
        hits = get_vector_store().similarity_search_with_score(question, k=k)
        return [(doc, 1.0 - distance) for doc, distance in hits]
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(RAG_STRATEGIES)}")


if __name__ == "__main__":
    import sys

    for doc, score in retrieve(" ".join(sys.argv[1:]) or "What caused the first AI winter?"):
        m = doc.metadata
        print(f"{score:.3f}  {m['doc_id']:<32} p{m['page']:<3} {m['section'][:40]}")
