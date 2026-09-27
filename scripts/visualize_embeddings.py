"""Build a local embedding viewer: see the vectors behind retrieval.

    uv run python scripts/visualize_embeddings.py      # writes data/processed/embedding_viewer.html
    open data/processed/embedding_viewer.html

- Map: every chunk's 1,536-dim embedding projected to 2D (PCA and t-SNE), so chunks with
  similar meaning sit close together.
- Vector: a chunk's raw 1,536 numbers as a colour strip, plus its nearest neighbours.
- Questions: where each golden-set question lands, and lines to the 5 chunks retrieval returns.

Chunk vectors are read from Chroma (free); only the 34 questions are embedded (~500 tokens).
The page contains corpus text, so it stays in data/ (gitignored) and should not be published.
"""

import base64
import json

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rag import config
from rag.models import get_embeddings
from rag.retrieve import get_vector_store

OUT = config.PROCESSED_DIR / "embedding_viewer.html"


def scale01(xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return (xy - lo) / (hi - lo)


def build_data() -> dict:
    got = get_vector_store()._collection.get(include=["embeddings", "metadatas", "documents"])
    order = np.argsort(got["ids"])  # stable order: doc, then chunk index
    ids = [got["ids"][i] for i in order]
    metas = [got["metadatas"][i] for i in order]
    texts = [got["documents"][i] for i in order]
    vecs = np.array(got["embeddings"], dtype=np.float32)[order]

    # Nearest neighbours by cosine similarity (vectors are unit length, so cosine = dot product).
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    neighbours = np.argsort(-sims, axis=1)[:, :5]

    # 2D layouts. PCA is a linear projection (keeps global structure, loses most detail);
    # t-SNE keeps local neighbourhoods (clusters) but distances between clusters mean little.
    pca = PCA(n_components=2, random_state=0).fit(vecs)
    pca_xy = pca.transform(vecs)
    tsne_xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=42).fit_transform(vecs)

    # Golden-set questions: embed, retrieve top-5 by cosine, place on the map.
    questions = yaml.safe_load((config.EVAL_DIR / "golden_set.yaml").read_text())["questions"]
    qvecs = np.array(get_embeddings().embed_documents([q["question"] for q in questions]), dtype=np.float32)
    qsims = qvecs @ vecs.T
    qtop = np.argsort(-qsims, axis=1)[:, :5]
    q_pca = pca.transform(qvecs)
    # t-SNE cannot place new points, so a question is drawn at the similarity-weighted
    # centre of its top-5 chunks (approximate, and labelled as such in the UI).
    w = np.take_along_axis(qsims, qtop, axis=1)
    q_tsne = np.einsum("qk,qkd->qd", w / w.sum(axis=1, keepdims=True), tsne_xy[qtop])

    all_pca = scale01(np.vstack([pca_xy, q_pca]))
    all_tsne = scale01(np.vstack([tsne_xy, q_tsne]))
    n = len(ids)

    # Raw vectors for the colour strip, compressed to int8 (value / max|value| * 127).
    vmax = float(np.abs(vecs).max())
    strip = base64.b64encode(np.clip(np.round(vecs / vmax * 127), -127, 127).astype(np.int8).tobytes()).decode()

    chunks = [{
        "id": ids[i], "doc_id": m["doc_id"], "title": m["title"], "year": m["year"], "era": m["era"],
        "source_type": m["source_type"], "page": m.get("page") or None, "section": m.get("section") or "",
        "preview": texts[i].split("\n", 1)[-1][:280],
        "pca": [round(float(v), 4) for v in all_pca[i]], "tsne": [round(float(v), 4) for v in all_tsne[i]],
        "nn": [[int(j), round(float(sims[i, j]), 3)] for j in neighbours[i]],
    } for i, m in enumerate(metas)]

    qs = [{
        "id": q["id"], "question": q["question"], "category": q["category"],
        "expected": q["expected_behavior"], "gold": q["gold_docs"],
        "pca": [round(float(v), 4) for v in all_pca[n + k]], "tsne": [round(float(v), 4) for v in all_tsne[n + k]],
        "top": [[int(j), round(float(qsims[k, j]), 3)] for j in qtop[k]],
    } for k, q in enumerate(questions)]

    manifest = yaml.safe_load(config.CORPUS_MANIFEST.read_text())
    return {
        "dims": int(vecs.shape[1]), "vmax": vmax, "strip": strip, "model": config.EMBEDDING_MODEL,
        "pca_var": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "min_similarity": config.MIN_SIMILARITY,
        "eras": manifest["eras"], "chunks": chunks, "questions": qs,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding Viewer</title>
<style>
:root {
  --bg: #f7f6f3; --panel: #fff; --ink: #1d1d1b; --muted: #6b6a66; --line: #e4e2dc; --gap: #ecebe6; --accent: #2f5bd3;
  --e0: #e8a33d; --e1: #d9594c; --e2: #8e5bd0; --e3: #6b7280; --e4: #2e9d8f; --e5: #3a7bd5; --e6: #c2418a; --e7: #7a9a2e;
  --q: #111; --pos: #c0392b; --neg: #2f5bd3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe6; --muted: #a09e97; --line: #34332f; --gap: #2a2926; --accent: #8aa8ff;
    --q: #fff; --pos: #ff7a6b; --neg: #7fa2ff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
header { padding: 12px 20px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
header h1 { font-size: 17px; margin: 0 8px 0 0; }
label { font-size: 13px; color: var(--muted); }
select, button { font: inherit; font-size: 13px; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--ink); }
.seg button.on { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.layout { display: grid; grid-template-columns: 1fr 380px; height: calc(100vh - 58px); }
#mapwrap { position: relative; padding: 12px; min-height: 420px; }
svg { width: 100%; height: 100%; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
circle.pt { stroke: var(--panel); stroke-width: .5; cursor: pointer; }
circle.pt.dim { opacity: .12; }
circle.pt.hit { stroke: var(--q); stroke-width: 2; }
circle.pt.sel { stroke: var(--q); stroke-width: 2.5; }
line.ray { stroke: var(--q); stroke-width: 1.3; opacity: .7; }
line.nnray { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.star { fill: var(--q); }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px;
  font-size: 12.5px; max-width: 320px; box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none; }
aside { border-left: 1px solid var(--line); background: var(--panel); overflow-y: auto; padding: 14px 16px 30px; }
aside h3 { margin: 16px 0 6px; font-size: 14px; }
aside h3:first-child { margin-top: 0; }
.muted { color: var(--muted); font-size: 12.5px; }
.legend div { display: flex; align-items: center; gap: 6px; font-size: 12.5px; cursor: pointer; padding: 1px 0; }
.legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.legend div.off { opacity: .35; }
canvas#strip { width: 100%; height: 34px; border: 1px solid var(--line); border-radius: 4px; image-rendering: pixelated; display: block; }
.nn { font-size: 12.5px; padding: 5px 0; border-bottom: 1px solid var(--line); cursor: pointer; }
.nn b { font-variant-numeric: tabular-nums; }
.bar { height: 6px; background: var(--gap); border-radius: 3px; margin-top: 3px; }
.bar span { display: block; height: 100%; border-radius: 3px; background: var(--accent); }
.preview { font: 12px/1.55 ui-monospace, Menlo, monospace; white-space: pre-wrap; background: var(--gap); padding: 8px; border-radius: 6px; max-height: 150px; overflow-y: auto; }
.tag { display: inline-block; font-size: 11.5px; padding: 1px 7px; border-radius: 999px; background: var(--gap); margin-right: 4px; }
.good { color: #1f8a4c; } .bad { color: #c0392b; }
details summary { cursor: pointer; font-weight: 600; font-size: 13.5px; }
details p { font-size: 13px; margin: 6px 0; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; height: auto; } #mapwrap { height: 70vh; } aside { border-left: 0; border-top: 1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <h1>Embedding Viewer</h1>
  <span class="seg"><label>Layout </label><button data-l="tsne" class="on">t-SNE</button><button data-l="pca">PCA</button></span>
  <label>Colour <select id="color"><option value="era">era</option><option value="source_type">source type</option><option value="doc">one document</option></select></label>
  <label>Question <select id="qsel"><option value="">none</option></select></label>
</header>
<div class="layout">
  <div id="mapwrap"><svg id="map"></svg><div id="tip"></div></div>
  <aside id="side"></aside>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const ERA_KEYS = Object.keys(D.eras);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const ERA_COL = Object.fromEntries(ERA_KEYS.map((k, i) => [k, `var(--e${i})`]));
const SRC_COL = {primary: 'var(--e5)', secondary: 'var(--e0)', reference: 'var(--e1)'};
const bytes = Uint8Array.from(atob(D.strip), (c) => c.charCodeAt(0));
const vec = (i) => new Int8Array(bytes.buffer, i * D.dims, D.dims);

let layout = 'tsne', colorBy = 'era', selChunk = null, selQ = null, focusDoc = D.chunks[0].doc_id;
const hidden = new Set();
const NS = 'http://www.w3.org/2000/svg';

function colorOf(c) {
  if (colorBy === 'era') return ERA_COL[c.era];
  if (colorBy === 'source_type') return SRC_COL[c.source_type];
  return c.doc_id === focusDoc ? 'var(--e1)' : 'var(--gap)';
}

function draw() {
  const svg = $('map'); const W = svg.clientWidth, H = svg.clientHeight, pad = 24;
  const X = (p) => pad + p[0] * (W - 2 * pad), Y = (p) => pad + (1 - p[1]) * (H - 2 * pad);
  svg.innerHTML = '';
  const q = selQ !== null ? D.questions[selQ] : null;
  const hits = new Set(q ? q.top.map((t) => t[0]) : []);
  const nnSet = new Set(selChunk !== null ? D.chunks[selChunk].nn.map((t) => t[0]) : []);
  D.chunks.forEach((c, i) => {
    const key = colorBy === 'era' ? c.era : colorBy === 'source_type' ? c.source_type : null;
    if (key && hidden.has(key)) return;
    const el = document.createElementNS(NS, 'circle');
    el.setAttribute('cx', X(c[layout])); el.setAttribute('cy', Y(c[layout]));
    el.setAttribute('r', hits.has(i) || i === selChunk ? 6 : 3.6);
    el.setAttribute('fill', colorOf(c));
    let cls = 'pt';
    if ((q && !hits.has(i)) || (selChunk !== null && !q && i !== selChunk && !nnSet.has(i))) cls += ' dim';
    if (hits.has(i)) cls += ' hit';
    if (i === selChunk) cls += ' sel';
    el.setAttribute('class', cls);
    el.onmouseenter = (e) => tip(e, c); el.onmouseleave = () => $('tip').style.display = 'none';
    el.onclick = () => { selChunk = i; selQ = null; $('qsel').value = ''; focusDoc = c.doc_id; draw(); side(); };
    svg.appendChild(el);
  });
  if (selChunk !== null && !q) D.chunks[selChunk].nn.forEach(([j]) => line(svg, X(D.chunks[selChunk][layout]), Y(D.chunks[selChunk][layout]), X(D.chunks[j][layout]), Y(D.chunks[j][layout]), 'nnray'));
  if (q) {
    q.top.forEach(([j]) => line(svg, X(q[layout]), Y(q[layout]), X(D.chunks[j][layout]), Y(D.chunks[j][layout]), 'ray'));
    const s = document.createElementNS(NS, 'path');
    const cx = X(q[layout]), cy = Y(q[layout]), r = 11;
    let d = '';
    for (let k = 0; k < 10; k++) { const a = Math.PI / 5 * k - Math.PI / 2, rr = k % 2 ? r * .45 : r; d += (k ? 'L' : 'M') + (cx + rr * Math.cos(a)) + ',' + (cy + rr * Math.sin(a)); }
    s.setAttribute('d', d + 'Z'); s.setAttribute('class', 'star'); svg.appendChild(s);
  }
}
function line(svg, x1, y1, x2, y2, cls) {
  const l = document.createElementNS(NS, 'line');
  Object.entries({x1, y1, x2, y2}).forEach(([k, v]) => l.setAttribute(k, v));
  l.setAttribute('class', cls); svg.insertBefore(l, svg.firstChild);
}
function tip(e, c) {
  const t = $('tip'), r = $('mapwrap').getBoundingClientRect();
  t.innerHTML = `<b>${esc(c.title)}</b> (${c.year})<br><span class="muted">${esc(c.section || 'no section')}${c.page ? ' · p.' + c.page : ''} · ${esc(c.id)}</span><br>${esc(c.preview.slice(0, 160))}…`;
  t.style.display = 'block';
  t.style.left = Math.min(e.clientX - r.left + 14, r.width - 330) + 'px';
  t.style.top = (e.clientY - r.top + 14) + 'px';
}

function legend() {
  if (colorBy === 'doc') return `<p class="muted">Highlighting <b>${esc(D.chunks.find((c) => c.doc_id === focusDoc).title)}</b>. Click any point to switch documents. Do its chunks cluster together, or spread across topics?</p>`;
  const entries = colorBy === 'era' ? ERA_KEYS.map((k) => [k, D.eras[k].label + ' · ' + D.eras[k].years, ERA_COL[k]]) : Object.entries(SRC_COL).map(([k, v]) => [k, k, v]);
  return `<div class="legend">${entries.map(([k, label, col]) => `<div data-k="${k}" class="${hidden.has(k) ? 'off' : ''}"><i style="background:${col}"></i>${esc(label)}</div>`).join('')}</div><p class="muted">Click a legend entry to hide or show it.</p>`;
}

function stripHtml() { return `<canvas id="strip" width="${D.dims}" height="1"></canvas>
  <div class="muted" style="display:flex;justify-content:space-between"><span>dim 1</span><span><span style="color:var(--neg)">■</span> negative · <span style="color:var(--pos)">■</span> positive</span><span>dim ${D.dims}</span></div>`; }
function paintStrip(i) {
  const cv = $('strip'); if (!cv) return;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(D.dims, 1), v = vec(i);
  const pos = [192, 57, 43], neg = [47, 91, 211], bg = [246, 246, 243];
  for (let k = 0; k < D.dims; k++) {
    const t = Math.min(1, Math.abs(v[k]) / 60), c = v[k] >= 0 ? pos : neg;
    for (let ch = 0; ch < 3; ch++) img.data[k * 4 + ch] = bg[ch] + (c[ch] - bg[ch]) * t;
    img.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function nnRow(j, s, label) {
  const c = D.chunks[j];
  return `<div class="nn" data-j="${j}"><b>${s.toFixed(3)}</b> ${label || ''} ${esc(c.title.length > 48 ? c.title.slice(0, 46) + '…' : c.title)} <span class="muted">(${c.year})${c.section ? ' · ' + esc(c.section.slice(0, 30)) : ''}</span>
    <div class="bar"><span style="width:${Math.max(0, s) * 100}%"></span></div></div>`;
}

function side() {
  let html = '';
  if (selQ !== null) {
    const q = D.questions[selQ], best = q.top[0][1];
    const goldHit = q.top.some(([j]) => q.gold.includes(D.chunks[j].doc_id));
    html += `<h3>${esc(q.id)}: ${esc(q.question)}</h3>
      <span class="tag">category ${q.category}</span><span class="tag">should ${q.expected}</span>
      <p class="muted">The star is the question's position. Lines go to the 5 chunks with the highest cosine similarity, which is exactly what retrieval returns.${layout === 'tsne' ? ' (In t-SNE the star is placed approximately, at the centre of its matches.)' : ''}</p>
      <p>Best similarity <b>${best.toFixed(3)}</b> ${best < D.min_similarity ? `<span class="bad">< ${D.min_similarity}: refused at the score gate</span>` : `<span class="muted">≥ ${D.min_similarity}: goes on to the evidence check</span>`}<br>
      ${q.expected === 'answer' ? (goldHit ? '<span class="good">✓ a correct (gold) document is in the top 5</span>' : '<span class="bad">✗ no gold document in the top 5</span>') : '<span class="muted">No gold document: this question should be refused.</span>'}</p>
      ${q.top.map(([j, s]) => nnRow(j, s, q.gold.includes(D.chunks[j].doc_id) ? '✓' : '')).join('')}`;
  } else if (selChunk !== null) {
    const c = D.chunks[selChunk];
    html += `<h3>${esc(c.title)} (${c.year})</h3>
      <span class="tag">${esc(D.eras[c.era].label)}</span><span class="tag">${c.source_type}</span><span class="tag">${esc(c.id)}</span>
      <p class="muted">${esc(c.section || 'no section')}${c.page ? ' · p.' + c.page : ''}</p>
      <div class="preview">${esc(c.preview)}…</div>
      <h3>Its embedding: ${D.dims} numbers</h3>${stripHtml()}
      <p class="muted">Each column is one dimension. No single dimension means anything readable; meaning is spread across the whole pattern. Similar chunks have similar patterns.</p>
      <h3>Nearest neighbours (cosine similarity)</h3>
      ${c.nn.map(([j, s]) => nnRow(j, s)).join('')}
      <p class="muted">Dashed lines on the map. Neighbours from the <i>same</i> document show the context header pulling a document's chunks together.</p>`;
  } else {
    html += `<h3>What you're looking at</h3>
      <p class="muted">Each dot is one of the ${D.chunks.length.toLocaleString()} chunks. Its position comes from its ${D.dims}-number embedding (${esc(D.model)}), squashed down to 2D. Dots close together have similar meaning. Hover over a dot to read it, click it to see its vector and neighbours, or pick a question above to watch retrieval.</p>`;
  }
  html += `<h3>Legend</h3>${legend()}
    <h3>Concepts</h3>
    <details><summary>What is an embedding?</summary><p>A list of ${D.dims} numbers that the embedding model computes from a text. Texts with similar meaning get similar lists, even with no words in common ("combinatorial explosion" ≈ "too many possibilities to search"). Every vector here has length 1.0.</p></details>
    <details><summary>Cosine similarity</summary><p>How closely two vectors point in the same direction: 1.0 identical, ~0 unrelated. Because the vectors have length 1, it's just the sum of the products of their numbers. Retrieval computes it between the question and every chunk and keeps the top 5. Our score gate refuses when the best is below ${D.min_similarity}.</p></details>
    <details><summary>Why the map is only a rough guide</summary><p>Going from ${D.dims} dimensions to 2 throws most of the information away. PCA's two axes keep only ${(100 * (D.pca_var[0] + D.pca_var[1])).toFixed(1)}% of the variance, so a chunk can be a nearest neighbour in ${D.dims}D and still look far away here. t-SNE keeps local neighbourhoods (clusters) intact, but distances <i>between</i> clusters mean little. Trust the similarity numbers; use the map for intuition.</p></details>
    <details><summary>Things to try</summary><p>• Pick <b>q16</b> (Dartmouth budget) and <b>q33</b> (Pluto): one lands among its sources, the other in empty space.<br>• Pick <b>q31</b> (GANs): it scores high but hits no gold doc. That's why the evidence check exists.<br>• Pick <b>q05</b>: a retrieval miss, and where the question drifted to.<br>• Colour by <b>one document</b>, then click an AI100 chunk: a broad report spreads across the map.<br>• Switch to <b>PCA</b>: does the era order show up along an axis?</p></details>`;
  $('side').innerHTML = html;
  if (selChunk !== null && selQ === null) paintStrip(selChunk);
  document.querySelectorAll('.legend div[data-k]').forEach((el) => el.onclick = () => { hidden.has(el.dataset.k) ? hidden.delete(el.dataset.k) : hidden.add(el.dataset.k); draw(); side(); });
  document.querySelectorAll('.nn').forEach((el) => el.onclick = () => { selChunk = +el.dataset.j; selQ = null; $('qsel').value = ''; focusDoc = D.chunks[selChunk].doc_id; draw(); side(); });
}

D.questions.forEach((q, k) => $('qsel').insertAdjacentHTML('beforeend', `<option value="${k}">${q.id} [${q.category}] ${esc(q.question.slice(0, 60))}</option>`));
$('qsel').onchange = (e) => { selQ = e.target.value === '' ? null : +e.target.value; draw(); side(); };
$('color').onchange = (e) => { colorBy = e.target.value; hidden.clear(); draw(); side(); };
document.querySelectorAll('.seg button').forEach((b) => b.onclick = () => {
  layout = b.dataset.l; document.querySelectorAll('.seg button').forEach((x) => x.classList.toggle('on', x === b)); draw(); side();
});
window.addEventListener('resize', draw);
draw(); side();
</script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(data['chunks'])} chunks, "
          f"{len(data['questions'])} questions; PCA 2D keeps {100 * sum(data['pca_var']):.1f}% of variance")


if __name__ == "__main__":
    main()
