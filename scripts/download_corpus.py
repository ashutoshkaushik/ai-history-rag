"""Download every document in corpus/manifest.yaml into data/raw/ and report what arrived.

    uv run python scripts/download_corpus.py            # fetch missing docs only
    uv run python scripts/download_corpus.py --force    # re-fetch everything
    uv run python scripts/download_corpus.py --only turing_1950 wiki_ai_winter

For each doc it records size, page count, and extractable text per page, so
scanned (image-only) PDFs are flagged before ingestion.
"""

import argparse
import json
import sys
import time
from urllib.parse import unquote, urlparse

import pymupdf
import httpx
import yaml

from rag import config

USER_AGENT = "AIHistoryRAG/0.1 (educational RAG course project; python-httpx)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SCANNED_CHARS_PER_PAGE = 200  # below this, a PDF page probably has no text layer


def load_manifest() -> list[dict]:
    return yaml.safe_load(config.CORPUS_MANIFEST.read_text())["documents"]


def raw_path(doc: dict):
    ext = {"pdf": "pdf", "html": "html", "wikipedia": "json"}[doc["format"]]
    return config.RAW_DIR / f"{doc['id']}.{ext}"


def fetch_wikipedia(client: httpx.Client, doc: dict) -> bytes:
    title = unquote(urlparse(doc["url"]).path.rsplit("/", 1)[-1])
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
        "prop": "extracts|revisions", "explaintext": 1, "rvprop": "ids|timestamp",
        "titles": title,
    }
    for attempt in range(4):  # back off on rate limiting (HTTP 429)
        r = client.get(WIKI_API, params=params)
        if r.status_code != 429:
            break
        time.sleep(float(r.headers.get("retry-after", 5 * (attempt + 1))))
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"Wikipedia page not found: {title}")
    rev = page["revisions"][0]
    return json.dumps({
        "title": page["title"], "text": page["extract"],
        "revision_id": rev["revid"], "revision_timestamp": rev["timestamp"],
    }, ensure_ascii=False).encode()


def inspect(doc: dict, path) -> dict:
    """Summarize a downloaded file: size, pages, and whether it has a text layer."""
    info = {"bytes": path.stat().st_size}
    if doc["format"] == "pdf":
        with pymupdf.open(path) as pdf:
            chars = [len(page.get_text().strip()) for page in pdf]
        info["pages"] = len(chars)
        info["avg_chars_per_page"] = round(sum(chars) / max(len(chars), 1))
        info["scanned"] = info["avg_chars_per_page"] < SCANNED_CHARS_PER_PAGE
    elif doc["format"] == "wikipedia":
        data = json.loads(path.read_text())
        info["chars"] = len(data["text"])
        info["revision_id"] = data["revision_id"]
    else:
        info["chars_html"] = info["bytes"]
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--only", nargs="*", help="restrict to these doc ids")
    args = parser.parse_args()

    docs = load_manifest()
    if args.only:
        docs = [d for d in docs if d["id"] in set(args.only)]
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    report, failures = {}, []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60) as client:
        for doc in docs:
            path = raw_path(doc)
            try:
                if args.force or not path.exists():
                    if doc["format"] == "wikipedia":
                        content = fetch_wikipedia(client, doc)
                    else:
                        r = client.get(doc["url"])
                        r.raise_for_status()
                        content = r.content
                        if doc["format"] == "pdf" and not content.startswith(b"%PDF"):
                            raise ValueError(f"not a PDF (content-type {r.headers.get('content-type')})")
                    path.write_bytes(content)
                    time.sleep(0.5)  # be polite to small academic hosts
                report[doc["id"]] = {"status": "ok", **inspect(doc, path)}
            except Exception as exc:  # report every failure, don't stop the batch
                path.unlink(missing_ok=True)
                report[doc["id"]] = {"status": "failed", "error": str(exc)[:200]}
                failures.append(doc["id"])

            r = report[doc["id"]]
            flag = "FAILED " + r["error"] if r["status"] == "failed" else (
                "SCANNED" if r.get("scanned") else "ok")
            detail = f"{r.get('pages', '')}p" if "pages" in r else f"{r.get('chars', r.get('chars_html', ''))} chars"
            print(f"{doc['id']:<34} {doc['format']:<9} {detail:>14}  {flag}")

    (config.RAW_DIR / "download_report.json").write_text(json.dumps(report, indent=2))
    ok = len(report) - len(failures)
    scanned = [k for k, v in report.items() if v.get("scanned")]
    print(f"\n{ok}/{len(report)} downloaded. Scanned PDFs: {scanned or 'none'}. Failed: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
