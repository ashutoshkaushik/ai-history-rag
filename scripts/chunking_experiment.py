"""Chunking experiment: how do chunk size and the context header affect retrieval?

    uv run python scripts/chunking_experiment.py

Re-chunks the corpus under several settings, embeds each variant in memory (the
Chroma index is not touched), and scores retrieval on the golden set. Embeddings
are cached in data/processed/exp_cache/, so re-runs are free.

Metrics (answerable questions only):
  hit@1 / hit@5   a gold document is among the top-k chunks (document-level)
  evidence@5      a top-5 chunk contains the answer text itself (chunk-level; stricter)
  ctx tokens      tokens of context the LLM must read per question (top-5 chunks)
"""

import re
import statistics

import numpy as np
import tiktoken
import yaml

from rag import config, ingest
from rag.models import get_embeddings

CACHE = config.PROCESSED_DIR / "exp_cache"
ENC = tiktoken.get_encoding("cl100k_base")

VARIANTS = [  # (name, chunk size, overlap, header)
    ("300 tok + header", 300, 45, True),
    ("600 tok + header (current)", 600, 90, True),
    ("600 tok, NO header", 600, 90, False),
    ("1200 tok + header", 1200, 180, True),
]

# Text that must appear in a retrieved chunk for it to count as containing the answer.
# Every pattern was checked against the corpus text.
EVIDENCE = {
    "q01": r"imitation game", "q02": r"Kasparov", "q07": r"fifty years", "q08": r"secretary",
    "q10": r"XOR", "q11": r"never actually used", "q12": r"GTX 580", "q13": r"biggest lesson",
    "q14": r"recurrence", "q15": r"1\.3B", "q16": r"13,500", "q17": r"10 ?120",
    "q18": r"3\.2 million", "q19": r"Advanced Automation", "q20": r"combinatorial explosion",
    "q21": r"1\.4 trillion", "q27": r"1987",
}


def build_chunks(size: int, overlap: int, header: bool):
    config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS, config.CHUNK_CONTEXT_HEADER = size, overlap, header
    docs = yaml.safe_load(config.CORPUS_MANIFEST.read_text())["documents"]
    chunks = []
    for doc in docs:
        loader, ext = ingest.LOADERS[doc["format"]]
        chunks += ingest.chunk_document(doc, loader(doc, config.RAW_DIR / f"{doc['id']}.{ext}"))
    return chunks


def embed(name: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (re.sub(r"\W+", "_", name) + ".npy")
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    vecs = np.array(get_embeddings().embed_documents(texts), dtype=np.float32)
    np.save(path, vecs)
    return vecs


def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main() -> None:
    questions = [q for q in yaml.safe_load((config.EVAL_DIR / "golden_set.yaml").read_text())["questions"]
                 if q["expected_behavior"] == "answer"]
    qvecs = normalize(embed("questions", [q["question"] for q in questions]))

    rows, q16 = [], []
    for name, size, overlap, header in VARIANTS:
        chunks = build_chunks(size, overlap, header)
        cvecs = normalize(embed(name, [c.page_content for c in chunks]))
        sims = qvecs @ cvecs.T                                  # cosine similarity, questions x chunks
        top = np.argsort(-sims, axis=1)[:, :5]

        h1 = h5 = ev = ev_n = 0
        ctx = []
        for qi, q in enumerate(questions):
            ids = [chunks[j].metadata["doc_id"] for j in top[qi]]
            h1 += ids[0] in q["gold_docs"]
            h5 += any(i in q["gold_docs"] for i in ids)
            ctx.append(sum(len(ENC.encode(chunks[j].page_content)) for j in top[qi]))
            if q["id"] in EVIDENCE:
                ev_n += 1
                ev += any(re.search(EVIDENCE[q["id"]], chunks[j].page_content) for j in top[qi])
            if q["id"] == "q16":  # where does the Dartmouth budget chunk rank?
                order = np.argsort(-sims[qi])
                rank = next(r + 1 for r, j in enumerate(order) if "13,500" in chunks[j].page_content
                            and chunks[j].metadata["doc_id"] == "dartmouth_proposal_1955")
                j = order[rank - 1]
                q16.append((name, rank, float(sims[qi, j])))

        sizes = [len(ENC.encode(c.page_content)) for c in chunks]
        n = len(questions)
        rows.append((name, len(chunks), statistics.median(sizes), f"{h1}/{n}", f"{h5}/{n}",
                     f"{ev}/{ev_n}", round(statistics.mean(ctx))))

    print(f"\n{'variant':<28}{'chunks':>7}{'median':>8}{'hit@1':>8}{'hit@5':>8}{'evid@5':>8}{'ctx tok':>9}")
    for r in rows:
        print(f"{r[0]:<28}{r[1]:>7}{r[2]:>8}{r[3]:>8}{r[4]:>8}{r[5]:>8}{r[6]:>9}")
    print("\nq16 'How much money did the 1955 Dartmouth proposal request?': rank of the $13,500 budget chunk")
    for name, rank, sim in q16:
        print(f"  {name:<28} rank {rank:<4} similarity {sim:.3f}")


if __name__ == "__main__":
    main()
