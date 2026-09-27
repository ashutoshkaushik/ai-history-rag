"""Build a local, self-contained chunk viewer (ChunkViz-style, but showing our exact chunks).

    uv run python scripts/visualize_chunks.py          # writes data/processed/chunk_viewer.html
    uv run python scripts/visualize_chunks.py --public # only openly licensed docs (the public site)
    open data/processed/chunk_viewer.html

It uses the same loaders and split_with_offsets() as ingestion, so what you see is
exactly what was embedded. The page contains corpus text, so it lives in data/
(gitignored) and should not be published.
"""

import json
import statistics

import tiktoken
import yaml

from rag import config
from rag.ingest import LOADERS, chunk_document, split_with_offsets

OUT = config.PROCESSED_DIR / "chunk_viewer.html"
ENC = tiktoken.get_encoding("cl100k_base")


def build_data(public: bool = False) -> dict:
    """public=True keeps only openly licensed documents (config.OPEN_LICENSES), for the public site."""
    manifest = yaml.safe_load(config.CORPUS_MANIFEST.read_text())
    docs = []
    for doc in manifest["documents"]:
        if public and doc.get("license") not in config.OPEN_LICENSES:
            continue
        loader, ext = LOADERS[doc["format"]]
        segments = loader(doc, config.RAW_DIR / f"{doc['id']}.{ext}")
        full, starts, located = split_with_offsets(segments)
        embedded = chunk_document(doc, segments)
        chunks = []
        for i, ((_, start, end), emb) in enumerate(zip(located, embedded)):
            header, _, body = emb.page_content.partition("\n") if config.CHUNK_CONTEXT_HEADER else ("", "", emb.page_content)
            prev_end = located[i - 1][2] if i else 0
            chunks.append({
                "id": emb.id, "start": start, "end": end,
                "tokens": len(ENC.encode(emb.page_content)),
                "overlap_chars": max(0, prev_end - start) if i else 0,
                "header": header, "body": body,
                "page": emb.metadata["page"], "section": emb.metadata["section"],
            })
        docs.append({
            "id": doc["id"], "title": doc["title"], "authors": ", ".join(doc["authors"]),
            "year": doc["year"], "era": doc["era"], "source_type": doc["source_type"],
            "document_type": doc["document_type"], "format": doc["format"], "url": doc["url"],
            "pages": doc.get("pages"), "notes": doc.get("notes"),
            "text": full,
            "segments": [{"start": s, "page": seg.page, "section": seg.section} for s, seg in zip(starts, segments)],
            "chunks": chunks,
            "tokens_total": len(ENC.encode(full)),
        })
    sizes = [c["tokens"] for d in docs for c in d["chunks"]]
    return {
        "settings": {
            "chunk_size": config.CHUNK_SIZE_TOKENS, "overlap": config.CHUNK_OVERLAP_TOKENS,
            "header": config.CHUNK_CONTEXT_HEADER, "embedding_model": config.EMBEDDING_MODEL,
            "separators": ["\\n\\n", "\\n", ". ", "; ", ", ", " ", ""],
        },
        "eras": manifest["eras"],
        "corpus": {"docs": len(docs), "chunks": len(sizes), "tokens": sum(sizes),
                   "median": statistics.median(sizes), "min": min(sizes), "max": max(sizes)},
        "docs": docs,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chunk Viewer</title>
<style>
:root {
  --bg: #f7f6f3; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6a66; --line: #e4e2dc;
  --accent: #2f5bd3; --ov: #9be29b; --gap: #ecebe6;
  --c0: #ffe08a; --c1: #9fd4ff; --c2: #ffb3c7; --c3: #c9b8ff; --c4: #ffc999;
  --hdr: #fff4c2; --sel: #1d1d1b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe6; --muted: #a09e97; --line: #34332f;
    --accent: #8aa8ff; --ov: #2f7d3a; --gap: #2a2926;
    --c0: #6b5a1c; --c1: #1f4f73; --c2: #6e2f41; --c3: #463a78; --c4: #74461f;
    --hdr: #4a4220; --sel: #ecebe6;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
header { padding: 14px 20px; border-bottom: 1px solid var(--line); background: var(--panel);
  display: flex; gap: 24px; align-items: baseline; flex-wrap: wrap; }
header h1 { font-size: 17px; margin: 0; }
header .settings { color: var(--muted); font-size: 12.5px; }
header code { background: var(--gap); padding: 1px 5px; border-radius: 4px; }
.layout { display: grid; grid-template-columns: 290px 1fr; height: calc(100vh - 54px); }
aside { border-right: 1px solid var(--line); overflow-y: auto; background: var(--panel); }
aside input { width: calc(100% - 24px); margin: 12px; padding: 7px 9px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--ink); }
.era { padding: 10px 14px 4px; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.doc { padding: 6px 14px; cursor: pointer; display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }
.doc:hover { background: var(--gap); }
.doc.active { background: var(--accent); color: #fff; }
.doc .n { color: var(--muted); font-variant-numeric: tabular-nums; }
.doc.active .n { color: #fff; }
main { overflow-y: auto; padding: 18px 24px 40px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
.meta h2 { margin: 0 0 4px; font-size: 18px; }
.meta .sub { color: var(--muted); }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.pill { background: var(--gap); border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 12px; }
.stat b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 12px; }
.tabs { display: flex; gap: 4px; margin: 4px 0 12px; }
.tab { border: 1px solid var(--line); background: var(--panel); color: var(--ink); padding: 6px 12px; border-radius: 6px; cursor: pointer; }
.tab.on { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 5px; }
#docview { white-space: pre-wrap; font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, monospace; }
#docview span.ck { cursor: pointer; border-radius: 2px; }
#docview span.ov { background: var(--ov); cursor: pointer; }
#docview span.gap { background: none; color: var(--muted); }
#docview span.sel { outline: 2px solid var(--sel); }
.marker { display: inline-block; font: 11px/1.4 -apple-system, sans-serif; color: var(--muted);
  border: 1px dashed var(--line); border-radius: 4px; padding: 0 6px; margin: 6px 0; }
.chunk { border-left: 5px solid var(--c0); }
.chunk .top { display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; font-size: 12.5px; color: var(--muted); margin-bottom: 8px; }
.chunk .top b { color: var(--ink); }
.chunk .hdr { background: var(--hdr); padding: 2px 6px; border-radius: 4px; font: 12.5px ui-monospace, Menlo, monospace; display: inline-block; margin-bottom: 6px; }
.chunk .body { white-space: pre-wrap; font: 12.5px/1.6 ui-monospace, Menlo, monospace; max-height: 260px; overflow-y: auto; }
.chunk .body mark { background: var(--ov); color: inherit; }
.chunk.flash { box-shadow: 0 0 0 3px var(--accent); }
.bars { display: flex; align-items: flex-end; gap: 2px; height: 120px; margin-top: 10px; }
.bars div { flex: 1; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; }
.axis { display: flex; justify-content: space-between; color: var(--muted); font-size: 11.5px; margin-top: 4px; }
.explain p { margin: 6px 0; }
.hidden { display: none; }
@media (max-width: 760px) {
  .layout { grid-template-columns: 1fr; height: auto; }
  aside { max-height: 40vh; border-right: 0; border-bottom: 1px solid var(--line); }
  main { padding: 14px 16px; }
}
</style>
</head>
<body>
<header>
  <h1>Chunk Viewer: AI History RAG</h1>
  <div class="settings" id="settings"></div>
</header>
<div class="layout">
  <aside>
    <input id="filter" placeholder="Filter documents…">
    <div id="doclist"></div>
  </aside>
  <main>
    <div class="card meta" id="meta"></div>
    <div class="tabs">
      <button class="tab on" data-tab="doc">Document view</button>
      <button class="tab" data-tab="chunks">Chunks as embedded</button>
      <button class="tab" data-tab="stats">Stats</button>
      <button class="tab" data-tab="about">How chunking works</button>
    </div>
    <section id="tab-doc">
      <div class="legend">
        <span><i class="sw" style="background:var(--c0)"></i><i class="sw" style="background:var(--c1)"></i><i class="sw" style="background:var(--c2)"></i>one colour per chunk</span>
        <span><i class="sw" style="background:var(--ov)"></i>overlap (text in two chunks)</span>
        <span><span class="marker">p.3 · Section</span> where a page or section starts</span>
        <span>Click any chunk to open it.</span>
      </div>
      <div class="card" id="docview"></div>
    </section>
    <section id="tab-chunks" class="hidden"><div id="chunklist"></div></section>
    <section id="tab-stats" class="hidden"><div id="stats"></div></section>
    <section id="tab-about" class="hidden"><div class="card explain" id="about"></div></section>
  </main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const COLORS = ['var(--c0)', 'var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)'];
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
let current = null;

const S = DATA.settings;
$('settings').innerHTML =
  `Recursive splitter · <code>${S.chunk_size}</code> tokens · overlap <code>${S.overlap}</code> tokens · ` +
  `header <code>${S.header ? 'on' : 'off'}</code> · ${esc(S.embedding_model)} · ` +
  `corpus: ${DATA.corpus.docs} docs → ${DATA.corpus.chunks.toLocaleString()} chunks`;

function renderList() {
  const q = $('filter').value.toLowerCase();
  const byEra = {};
  DATA.docs.forEach((d, i) => {
    if (q && !(d.title + d.id + d.authors).toLowerCase().includes(q)) return;
    (byEra[d.era] ||= []).push(i);
  });
  let html = '';
  for (const [era, info] of Object.entries(DATA.eras)) {
    if (!byEra[era]) continue;
    html += `<div class="era">${esc(info.label)} · ${esc(info.years)}</div>`;
    for (const i of byEra[era]) {
      const d = DATA.docs[i];
      html += `<div class="doc ${i === current ? 'active' : ''}" data-i="${i}"><span>${esc(d.title.length > 42 ? d.title.slice(0, 40) + '…' : d.title)}</span><span class="n">${d.chunks.length}</span></div>`;
    }
  }
  $('doclist').innerHTML = html;
  document.querySelectorAll('.doc').forEach((el) => el.onclick = () => select(+el.dataset.i));
}

function median(a) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }

function renderMeta(d) {
  const t = d.chunks.map((c) => c.tokens);
  const sections = new Set(d.chunks.map((c) => c.section).filter(Boolean)).size;
  $('meta').innerHTML = `
    <h2>${esc(d.title)}</h2>
    <div class="sub">${esc(d.authors)} · ${d.year} · <a href="${esc(d.url)}" target="_blank" rel="noopener">source</a></div>
    <div class="pills">
      <span class="pill">${esc(DATA.eras[d.era].label)}</span><span class="pill">${d.source_type}</span>
      <span class="pill">${d.document_type.replace('_', ' ')}</span><span class="pill">${d.format}</span>
      ${d.pages ? `<span class="pill">pages ${d.pages} only</span>` : ''}
      <span class="pill"><code>${d.id}</code></span>
    </div>
    ${d.notes ? `<p class="sub" style="margin:10px 0 0">Note: ${esc(d.notes)}</p>` : ''}
    <div class="stats">
      <div class="stat"><b>${d.text.length.toLocaleString()}</b><span>characters (cleaned)</span></div>
      <div class="stat"><b>${d.tokens_total.toLocaleString()}</b><span>tokens</span></div>
      <div class="stat"><b>${d.chunks.length}</b><span>chunks</span></div>
      <div class="stat"><b>${median(t)}</b><span>median tokens / chunk</span></div>
      <div class="stat"><b>${sections}</b><span>sections detected</span></div>
    </div>`;
}

function renderDocView(d) {
  // Split the text at every chunk start/end and every page/section start, then
  // colour each piece by how many chunks cover it.
  const cuts = new Set([0, d.text.length]);
  d.chunks.forEach((c) => { cuts.add(c.start); cuts.add(c.end); });
  const markers = {};
  let lastPage = null, lastSection = null;
  d.segments.forEach((s) => {
    const label = [];
    if (s.page && s.page !== lastPage) label.push('p.' + s.page);
    if (s.section && s.section !== lastSection) label.push(s.section);
    if (label.length) { markers[s.start] = label.join(' · '); cuts.add(s.start); }
    lastPage = s.page; lastSection = s.section;
  });
  const pts = [...cuts].filter((x) => x >= 0 && x <= d.text.length).sort((a, b) => a - b);
  let html = '';
  for (let k = 0; k < pts.length - 1; k++) {
    const a = pts[k], b = pts[k + 1];
    if (markers[a]) html += `<span class="marker">${esc(markers[a])}</span>\n`;
    const cover = [];
    d.chunks.forEach((c, i) => { if (c.start <= a && c.end >= b) cover.push(i); });
    const txt = esc(d.text.slice(a, b));
    if (cover.length === 0) html += `<span class="gap">${txt}</span>`;
    else if (cover.length > 1) html += `<span class="ov" data-c="${cover.join(',')}" title="Overlap: chunks ${cover.join(' & ')}">${txt}</span>`;
    else html += `<span class="ck" data-c="${cover[0]}" title="Chunk ${cover[0]}" style="background:${COLORS[cover[0] % COLORS.length]}">${txt}</span>`;
  }
  $('docview').innerHTML = html;
  $('docview').querySelectorAll('[data-c]').forEach((el) => el.onclick = () => openChunk(+el.dataset.c.split(',').pop()));
}

function renderChunks(d) {
  $('chunklist').innerHTML = d.chunks.map((c, i) => {
    const body = esc(c.body);
    // highlight the part repeated from the previous chunk (the overlap)
    const ovLen = Math.min(c.overlap_chars, c.body.length);
    const shown = ovLen > 0 ? `<mark>${esc(c.body.slice(0, ovLen))}</mark>${esc(c.body.slice(ovLen))}` : body;
    return `<div class="card chunk" id="chunk-${i}" style="border-left-color:${COLORS[i % COLORS.length]}">
      <div class="top"><span><b>${esc(c.id)}</b></span>
        <span><b>${c.tokens}</b> tokens · ${c.body.length.toLocaleString()} chars · ${c.page ? 'p.' + c.page + ' · ' : ''}${esc(c.section || 'no section')}${c.overlap_chars ? ` · <b>${c.overlap_chars}</b> chars overlap` : ''}</span></div>
      ${c.header ? `<div class="hdr">${esc(c.header)}</div>` : ''}
      <div class="body">${shown}</div></div>`;
  }).join('');
}

function histogram(values, lo, hi, bins) {
  const counts = new Array(bins).fill(0);
  values.forEach((v) => counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * bins)))]++);
  const max = Math.max(...counts, 1);
  return `<div class="bars">${counts.map((n, i) => `<div style="height:${n / max * 100}%" title="${Math.round(lo + i * (hi - lo) / bins)}–${Math.round(lo + (i + 1) * (hi - lo) / bins)} tokens: ${n} chunks"></div>`).join('')}</div>
    <div class="axis"><span>${lo} tokens</span><span>${Math.round((lo + hi) / 2)}</span><span>${hi}</span></div>`;
}

function renderStats(d) {
  const all = DATA.docs.flatMap((x) => x.chunks.map((c) => c.tokens));
  const perDoc = [...DATA.docs].sort((a, b) => b.chunks.length - a.chunks.length);
  const maxN = perDoc[0].chunks.length;
  $('stats').innerHTML = `
    <div class="card"><b>This document: chunk sizes</b>${histogram(d.chunks.map((c) => c.tokens), 0, 700, 28)}
      <p class="sub">Most chunks sit just under the ${S.chunk_size}-token limit. Smaller ones are the last chunk of a document or a short section.</p></div>
    <div class="card"><b>Whole corpus: ${DATA.corpus.chunks.toLocaleString()} chunks</b> (median ${DATA.corpus.median}, min ${DATA.corpus.min}, max ${DATA.corpus.max} tokens)
      ${histogram(all, 0, 700, 28)}</div>
    <div class="card"><b>Chunks per document</b>
      ${perDoc.map((x) => `<div style="display:flex;gap:8px;align-items:center;font-size:12.5px;margin:3px 0">
        <span style="width:230px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">${esc(x.title)}</span>
        <span style="flex:1;background:var(--gap);border-radius:3px"><span style="display:block;height:10px;border-radius:3px;background:${x.id === d.id ? 'var(--accent)' : 'var(--muted)'};width:${x.chunks.length / maxN * 100}%"></span></span>
        <span style="width:30px;text-align:right">${x.chunks.length}</span></div>`).join('')}
      <p class="sub">Documents with many chunks take up more of the search space. That's one reason the longest papers were cut to their main body.</p></div>`;
}

const _b = DATA.docs.flatMap((d) => d.chunks.slice(1));
const overlapPct = Math.round(100 * _b.filter((c) => c.overlap_chars > 0).length / _b.length);
$('about').innerHTML = `
  <h3 style="margin-top:0">How this corpus is chunked</h3>
  <p><b>1. Clean first.</b> The splitter never sees the raw PDF. It gets text with hyphenated line breaks rejoined, running headers and footers removed, and reference lists cut. The Document view shows exactly that cleaned text.</p>
  <p><b>2. Recursive splitting.</b> LangChain's <code>RecursiveCharacterTextSplitter</code> tries separators in order (${S.separators.map((s) => `<code>${esc(s || '(character)')}</code>`).join(' → ')}): paragraphs first, then lines, sentences, clauses and words. It only falls back to a finer separator when a piece is still too big, so chunks end at natural boundaries where possible.</p>
  <p><b>3. Measured in tokens.</b> Length is counted with the <code>cl100k_base</code> tokenizer, not characters. A chunk holds up to ${S.chunk_size} tokens, about 450 words or 3,000 characters of prose.</p>
  <p><b>4. Overlap.</b> Each chunk repeats up to ${S.overlap} tokens (about 15%) from the end of the previous one, shown in green. A sentence split across two chunks is then still whole in at least one of them.
     <br><b>But "overlap ${S.overlap}" is a maximum, not a guarantee.</b> The recursive splitter builds overlap only from <i>whole pieces</i> (paragraphs, sentences…) at the end of the previous chunk. If that last piece is longer than ${S.overlap} tokens, nothing fits, and the next chunk starts with no overlap at all.
     In this corpus, <b>${overlapPct}%</b> of chunk boundaries have overlap. It's lowest for HTML documents, where paragraphs are long.</p>
  <p><b>5. Context header.</b> Before embedding, each chunk gets a header line like <code>[Title (Year) | Section]</code>, shown in yellow in the Chunks view. A mid-paper paragraph that never names its paper can then still match a question about that paper.</p>
  <p><b>6. Page and section.</b> Each chunk is labelled with the page and section where it <i>starts</i>. That's what citations show.</p>
  <p><b>Things to look for:</b> math-heavy Wikipedia articles (Backpropagation, RLHF), where formulas extract one symbol per line; OCR noise in 1950s–60s scans (ELIZA, Minsky's "Steps"); and the Lighthill Report, whose sections (Category A / B / C) come from the HTML headings.</p>`;

function openChunk(i) {
  setTab('chunks');
  const el = $('chunk-' + i);
  el.scrollIntoView({behavior: 'smooth', block: 'center'});
  el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1400);
}

function setTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t.dataset.tab === name));
  ['doc', 'chunks', 'stats', 'about'].forEach((t) => $('tab-' + t).classList.toggle('hidden', t !== name));
}
document.querySelectorAll('.tab').forEach((t) => t.onclick = () => setTab(t.dataset.tab));

function select(i) {
  current = i;
  const d = DATA.docs[i];
  renderList(); renderMeta(d); renderDocView(d); renderChunks(d); renderStats(d);
  try { localStorage.setItem('chunkviewer.doc', d.id); } catch (e) {}
  document.querySelector('main').scrollTop = 0;
}
$('filter').oninput = renderList;

let start = DATA.docs.findIndex((d) => d.id === 'lighthill_1973');
try { const saved = localStorage.getItem('chunkviewer.doc'); const j = DATA.docs.findIndex((d) => d.id === saved); if (j >= 0) start = j; } catch (e) {}
select(Math.max(0, start));
</script>
</body>
</html>
"""


def main() -> None:
    import sys
    public = "--public" in sys.argv or config.PUBLIC_DEPLOYMENT
    data = build_data(public)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", payload))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): "
          f"{data['corpus']['docs']} docs, {data['corpus']['chunks']} chunks")


if __name__ == "__main__":
    main()
