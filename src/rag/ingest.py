"""Ingestion pipeline: load -> clean -> chunk -> embed -> store.

    uv run python -m rag.ingest            # incremental: only new/changed docs are re-embedded
    uv run python -m rag.ingest --rebuild  # drop the collection and re-embed everything

Every chunk carries the full manifest metadata plus page and section, and is
also written to data/processed/chunks.jsonl for inspection and for BM25.
"""

import argparse
import bisect
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import lxml.html
import trafilatura
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.retrieve import get_vector_store

STATE_FILE = config.PROCESSED_DIR / "index_state.json"
CHUNKS_FILE = config.PROCESSED_DIR / "chunks.jsonl"
REPORT_FILE = config.PROCESSED_DIR / "ingest_report.json"

# Wikipedia sections that are navigation or citations, not content.
WIKI_DROP_SECTIONS = {
    "see also", "references", "notes", "external links", "further reading",
    "sources", "bibliography", "citations", "works cited", "footnotes",
}
REFERENCES_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|literature cited)\s*$", re.I)
NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+[A-Z][\w ,:&'()-]{2,70}$")
NAMED_HEADINGS = {
    "abstract", "introduction", "background", "related work", "method", "methods", "results",
    "discussion", "conclusion", "conclusions", "summary", "acknowledgments", "acknowledgements",
}


@dataclass
class Segment:
    """A run of cleaned text with the page and section it came from."""
    text: str
    page: int | None
    section: str | None


# ---------------------------------------------------------------- cleaning

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)           # ligatures (ﬁ -> fi), odd widths
    text = text.replace("­", "")                    # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # re-join words hyphenated across lines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)         # single newlines inside a paragraph
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def heading_of(block: str) -> str | None:
    line = block.strip()
    if len(line) > 80 or "\n" in line:
        return None
    numbered = NUMBERED_HEADING.match(line)
    if numbered and not line[0].isdigit() or numbered and int(re.match(r"\d+", line).group()) <= 15:
        return line.rstrip(".:")
    if line.lower().rstrip(".:") in NAMED_HEADINGS:
        return line.rstrip(".:")
    return None


# ----------------------------------------------------------------- loaders

def load_pdf(doc: dict, path: Path) -> list[Segment]:
    pdf = pymupdf.open(path)
    first, last = (map(int, doc["pages"].split("-")) if "pages" in doc else (1, len(pdf)))
    pages = [(i + 1, [b[4] for b in pdf[i].get_text("blocks") if b[6] == 0])
             for i in range(first - 1, min(last, len(pdf)))]

    # Running headers/footers: short lines repeated on many pages (digits ignored).
    def key(b):
        return re.sub(r"\d+", "#", b.strip().lower())
    counts = Counter(k for _, blocks in pages for k in {key(b) for b in blocks if len(b.strip()) < 100})
    repeated = {k for k, c in counts.items() if len(pages) >= 4 and c >= 0.4 * len(pages)}

    segments, section = [], None
    past_start = max(2, len(pages) // 3)  # ignore "References" in a table of contents
    for idx, (page_no, blocks) in enumerate(pages):
        kept = []
        for block in blocks:
            text = block.strip()
            if not text or key(text) in repeated or re.fullmatch(r"[\d\s\-–]+", text):
                continue
            if idx >= past_start and (REFERENCES_HEADING.match(text) or re.match(r"\[1\]\s", text)):
                if kept:
                    segments.append(Segment(normalize("\n\n".join(kept)), page_no, section))
                return segments
            if h := heading_of(normalize(text)):
                if kept:
                    segments.append(Segment(normalize("\n\n".join(kept)), page_no, section))
                    kept = []
                section = h
            kept.append(text)
        if kept:
            segments.append(Segment(normalize("\n\n".join(kept)), page_no, section))
    return segments


def load_html(doc: dict, path: Path) -> list[Segment]:
    """Walk the HTML in document order: h1-h4 (or a short fully-bold paragraph) sets
    the section, and text blocks become paragraphs. Short, link-heavy blocks
    (site navigation) are skipped. Falls back to trafilatura for pages without
    block markup."""
    html = path.read_text(errors="ignore")
    root = lxml.html.fromstring(html)
    for junk in root.xpath("//script|//style|//nav|//header|//footer|//noscript"):
        junk.drop_tree()

    segments, section, kept = [], None, []

    def flush():
        if kept:
            segments.append(Segment(normalize("\n\n".join(kept)), None, section))
            kept.clear()

    for el in root.iter("h1", "h2", "h3", "h4", "p", "pre", "blockquote", "li", "td"):
        if el.xpath("ancestor::p|ancestor::li|ancestor::blockquote|ancestor::td"):
            continue  # nested; already counted with its parent
        text = re.sub(r"\s+", " ", el.text_content()).strip()
        if not text:
            continue
        bold = "".join(b.text_content() for b in el.xpath("./b|./strong")).strip()
        if el.tag in ("h1", "h2", "h3", "h4") or (len(text) <= 80 and bold == text
                                                   and not re.search(r"\d", text)):  # not bold dates
            flush()
            section = text[:80]
        elif len(text) >= 40 and sum(len(a.text_content()) for a in el.iter("a")) < 0.5 * len(text):
            kept.append(text)
    flush()

    fallback = trafilatura.extract(html) or ""
    if sum(len(s.text) for s in segments) < 0.5 * len(fallback):
        return [Segment(normalize(fallback), None, None)]
    return segments


def load_wikipedia(doc: dict, path: Path) -> list[Segment]:
    text = json.loads(path.read_text())["text"]
    parts = re.split(r"^(={2,4})\s*(.+?)\s*\1\s*$", text, flags=re.M)
    segments = [Segment(normalize(parts[0]), None, "Introduction")]
    top = None
    for i in range(1, len(parts), 3):
        level, title, body = len(parts[i]), parts[i + 1], parts[i + 2]
        if level == 2:
            top = title
        if (top or "").lower() in WIKI_DROP_SECTIONS:
            continue
        section = title if level == 2 else f"{top} > {title}"
        if body.strip():
            segments.append(Segment(normalize(body), None, section))
    return [s for s in segments if s.text]


LOADERS = {"pdf": (load_pdf, "pdf"), "html": (load_html, "html"), "wikipedia": (load_wikipedia, "json")}


# ---------------------------------------------------------------- chunking

def splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        add_start_index=True,
    )


def doc_metadata(doc: dict) -> dict:
    """Manifest fields flattened to Chroma-compatible scalars."""
    return {
        "doc_id": doc["id"],
        "title": doc["title"],
        "authors": ", ".join(doc["authors"]),
        "year": int(doc["year"]),
        "era": doc["era"],
        "source_type": doc["source_type"],
        "document_type": doc["document_type"],
        "url": doc["url"],
    }


def split_with_offsets(segments: list[Segment]) -> tuple[str, list[int], list[tuple[str, int, int]]]:
    """Join segments into one text (so chunks can span pages) and split it.

    Returns (full_text, segment_start_offsets, [(chunk_text, start, end), ...]) where
    start/end are character offsets of each chunk in full_text.
    """
    full, starts = "", []
    for seg in segments:
        starts.append(len(full))
        full += seg.text + "\n\n"
    located, cursor = [], 0
    for piece in splitter().create_documents([full]):
        body = piece.page_content.strip()
        # LangChain's start_index is -1 whenever the merged chunk's whitespace differs from
        # the source (34% of chunks in practice), which mislabels page/section. Locate the
        # chunk ourselves by its opening text, searching forward from the previous chunk.
        start = full.find(body[:80], cursor)
        if start < 0:
            start = cursor
        cursor = start
        # search for the tail near the expected end, so repeated text (pull quotes) can't match early
        tail_at = full.find(body[-80:], max(start, start + int(0.8 * len(body)) - 80))
        end = tail_at + len(body[-80:]) if tail_at >= 0 else start + len(body)
        located.append((body, start, end))
    return full, starts, located


def chunk_document(doc: dict, segments: list[Segment]) -> list[Document]:
    """Chunk a document and map each chunk back to the page/section where it starts."""
    _, starts, located = split_with_offsets(segments)
    base = doc_metadata(doc)
    chunks = []
    for i, (body, start, _) in enumerate(located):
        seg = segments[bisect.bisect_right(starts, start) - 1]
        meta = {**base, "chunk_index": i, "page": seg.page or 0, "section": seg.section or ""}
        text = body
        if config.CHUNK_CONTEXT_HEADER:
            where = f" | {seg.section}" if seg.section else ""
            text = f"[{doc['title']} ({doc['year']}){where}]\n{text}"
        chunks.append(Document(page_content=text, metadata=meta, id=f"{doc['id']}::{i:04d}"))
    return chunks


# ------------------------------------------------------------------ driver

def fingerprint(doc: dict, raw: Path) -> str:
    """Changes when the source file, its manifest entry, or chunking settings change."""
    h = hashlib.sha256(raw.read_bytes())
    h.update(json.dumps(doc, sort_keys=True, default=str).encode())
    h.update(f"{config.CHUNK_SIZE_TOKENS}/{config.CHUNK_OVERLAP_TOKENS}/{config.CHUNK_CONTEXT_HEADER}".encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="drop the collection and re-embed everything")
    args = parser.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    docs = yaml.safe_load(config.CORPUS_MANIFEST.read_text())["documents"]
    store = get_vector_store()
    if args.rebuild:
        store.reset_collection()
        STATE_FILE.unlink(missing_ok=True)
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    # Docs removed from the manifest are removed from the index.
    for gone in set(state) - {d["id"] for d in docs}:
        store.delete(where={"doc_id": gone})
        del state[gone]
        print(f"removed  {gone}")

    all_chunks, report, embedded = [], {}, 0
    for doc in docs:
        loader, ext = LOADERS[doc["format"]]
        raw = config.RAW_DIR / f"{doc['id']}.{ext}"
        if not raw.exists():
            print(f"MISSING  {doc['id']} (run scripts/download_corpus.py)")
            continue
        chunks = chunk_document(doc, loader(doc, raw))
        all_chunks.extend(chunks)
        report[doc["id"]] = len(chunks)

        fp = fingerprint(doc, raw)
        if state.get(doc["id"]) != fp:
            store.delete(where={"doc_id": doc["id"]})
            store.add_documents(chunks, ids=[c.id for c in chunks])
            state[doc["id"]] = fp
            STATE_FILE.write_text(json.dumps(state, indent=1))  # save progress per doc
            embedded += 1
            print(f"embedded {doc['id']:<34} {len(chunks):>4} chunks")

    with CHUNKS_FILE.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps({"id": c.id, "text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")

    enc = splitter()._length_function
    sizes = [enc(c.page_content) for c in all_chunks]
    summary = {
        "documents": len(report), "chunks": len(all_chunks), "embedded_this_run": embedded,
        "tokens_total": sum(sizes), "tokens_per_chunk_median": statistics.median(sizes),
        "tokens_per_chunk_max": max(sizes), "chunks_per_doc": report,
    }
    REPORT_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['documents']} docs -> {summary['chunks']} chunks "
          f"({summary['tokens_total']:,} tokens, median {summary['tokens_per_chunk_median']}/chunk). "
          f"Re-embedded {embedded} doc(s); vectors in collection: {store._collection.count()}")


if __name__ == "__main__":
    main()
