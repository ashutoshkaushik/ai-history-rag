"""Write one document's cleaned text (exactly what the chunker sees) to a .txt file,
e.g. to paste into ChunkViz (https://chunkviz.up.railway.app).

    uv run python scripts/export_clean_text.py lighthill_1973
"""

import sys

import tiktoken
import yaml

from rag import config
from rag.ingest import LOADERS

doc_id = sys.argv[1] if len(sys.argv) > 1 else "lighthill_1973"
doc = next(d for d in yaml.safe_load(config.CORPUS_MANIFEST.read_text())["documents"] if d["id"] == doc_id)
loader, ext = LOADERS[doc["format"]]
text = "".join(seg.text + "\n\n" for seg in loader(doc, config.RAW_DIR / f"{doc_id}.{ext}"))

out = config.PROCESSED_DIR / f"{doc_id}.clean.txt"
out.write_text(text)
tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
ratio = len(text) / tokens
print(f"{out}\n{len(text):,} chars, {tokens:,} tokens ({ratio:.1f} chars/token)")
print(f"ChunkViz 'Recursive Character' equivalent of our settings: "
      f"chunk size ~{round(config.CHUNK_SIZE_TOKENS * ratio, -1):.0f} chars, "
      f"overlap ~{round(config.CHUNK_OVERLAP_TOKENS * ratio, -1):.0f} chars")
