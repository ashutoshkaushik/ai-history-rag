"""RAG Lab: live, interactive views of each pipeline step (understanding, not production).

Each step is its own page in the sidebar of `uv run streamlit run app.py`, registered in app.py with
st.navigation in pipeline order: tokens -> chunks -> embeddings -> retrieval -> score gate & top k ->
prompts & generation. This module holds the page bodies and the shared pipeline header.
"""

import html
import re
import statistics
import time

import streamlit as st
import yaml
from langchain_core.callbacks import BaseCallbackHandler

from lab.learn import render_learning
from lab.nav import tour_footer
from lab.usage import remaining_note, try_spend
from rag import config
from rag.retrieve import get_bm25_retriever, retrieve, rrf_fuse, tokenize


COLORS = {"vector": "#2f5bd3", "bm25": "#d97706", "hybrid": "#16a34a"}
LABELS = {"vector": "🧭 Vector (semantic)", "bm25": "🔤 BM25 (keyword)", "hybrid": "🔀 Hybrid (RRF)"}

def lab_css() -> None:
    st.markdown("""
    <style>
      .lab-sub { opacity: .75; margin-bottom: .6rem; }
      .res { border: 1px solid rgba(127,127,127,.25); border-radius: 8px; padding: .45rem .6rem; margin-bottom: .4rem; }
      .res .t { font-weight: 600; font-size: .9rem; }
      .res .m { opacity: .68; font-size: .78rem; }
      .res.gold { border-left: 4px solid #16a34a; }
      .score { font-variant-numeric: tabular-nums; font-weight: 700; }
      .both { font-size: .7rem; background: rgba(22,163,74,.15); color: #16a34a; border-radius: 99px; padding: 0 6px; margin-left: 4px; }
      mark { background: rgba(217,119,6,.28); color: inherit; padding: 0 1px; border-radius: 2px; }
      .snip { font-size: .8rem; opacity: .8; margin-top: .25rem; }
            .crumbs { font-size: .78rem; margin-bottom: .2rem; line-height: 2; }
      .crumb { padding: 2px 8px; border-radius: 99px; border: 1px solid rgba(127,127,127,.3); white-space: nowrap; }
      .crumb { opacity: .75; }
      .crumb.on { background: #C2603E; border-color: #C2603E; color: #fff; font-weight: 600; opacity: 1; }
      .arrow { opacity: .5; }
    </style>""", unsafe_allow_html=True)



@st.cache_data
def golden() -> list[dict]:
    return yaml.safe_load((config.EVAL_DIR / "golden_set.yaml").read_text())["questions"]


@st.cache_data(show_spinner=False)
def vector_pool(question: str, pool: int) -> list[tuple[str, float]]:
    return [(d.id, s) for d, s in retrieve(question, "vector", k=pool)]


def bm25_pool(question: str, pool: int) -> tuple[list[tuple[int, float]], list[str]]:
    bm = get_bm25_retriever()
    tokens = tokenize(question)
    scores = bm.vectorizer.get_scores(tokens)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
    return [(i, float(scores[i])) for i in top], tokens


@st.cache_resource
def doc_index() -> dict:
    bm = get_bm25_retriever()
    return {d.id: i for i, d in enumerate(bm.docs)}


def snippet(text: str, terms: set[str], width: int = 260) -> str:
    """Body text around the first matched term, with matched terms highlighted."""
    body = text.split("\n", 1)[-1]
    low = body.lower()
    pos = min([low.find(t) for t in terms if low.find(t) >= 0] or [0])
    start = max(0, pos - 80)
    piece = html.escape(body[start:start + width].replace("\n", " "))
    for t in sorted(terms, key=len, reverse=True):
        piece = re.sub(rf"(?i)\b({re.escape(html.escape(t))})\b", r"<mark>\1</mark>", piece)
    return ("…" if start else "") + piece + "…"


def result_card(doc, score_txt: str, gold: set[str], also_in: list[str], terms: set[str]) -> None:
    m = doc.metadata
    is_gold = m["doc_id"] in gold
    badges = "".join(f"<span class='both'>also in {n}</span>" for n in also_in)
    where = f" · {m['section'][:40]}" if m.get("section") else ""
    st.markdown(
        f"<div class='res{' gold' if is_gold else ''}'><span class='score'>{score_txt}</span> "
        f"{'✓ ' if is_gold else ''}<span class='t'>{html.escape(m['title'][:60])}</span> ({m['year']}){badges}"
        f"<div class='m'>{m['source_type']}{html.escape(where)}</div>"
        f"<div class='snip'>{snippet(doc.page_content, terms)}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------- step 4 · retrieval

def retrieval_tab() -> None:
    st.markdown("<div class='lab-sub'>The same question, three ways of finding chunks. <b>Vector</b> search "
                "compares meaning (embeddings, cosine similarity). <b>BM25</b> counts shared keywords, weighting "
                "rare words higher. <b>Hybrid</b> fuses both ranked lists with Reciprocal Rank Fusion, the "
                "course's notebook-3 approach. ✓ and a green edge mark a correct (gold) document for test "
                "questions.</div>", unsafe_allow_html=True)

    qs = golden()
    labels = {f"{q['id']} [{q['category']}] {q['question']}": q for q in qs}
    c1, c2 = st.columns([3, 2])
    with c1:
        pick = st.selectbox("Pick a test question", list(labels), index=4)   # q05: the known vector miss
    with c2:
        typed = st.text_input("…or type your own", placeholder="e.g. ORD-style exact terms: 'GTX 580'")
    question = typed.strip() or labels[pick]["question"]
    gold = set() if typed.strip() else set(labels[pick]["gold_docs"])

    s1, s2, s3, s4 = st.columns(4)
    k = s1.slider("Top k shown", 3, 10, config.TOP_K)
    pool = s2.slider("Candidates fused per list", 5, 50, 20, help="How deep into each list RRF looks.")
    w = s3.slider("Vector weight", 0.0, 1.0, 0.5, 0.05, help="BM25 weight = 1 − vector weight.")
    c = s4.number_input("RRF constant c", 1, 200, 60, help="Larger c flattens the gap between ranks 1, 2, 3…")

    with st.spinner("Retrieving…"):
        vec = vector_pool(question, pool)
        bm, tokens = bm25_pool(question, pool)
    bm25 = get_bm25_retriever()
    docs_by_id = {d.id: d for d in bm25.docs}
    idx = doc_index()
    vec_docs = [docs_by_id[i] for i, _ in vec]
    bm_docs = [bm25.docs[i] for i, _ in bm]
    fused = rrf_fuse({"vector": vec_docs, "bm25": bm_docs}, {"vector": w, "bm25": 1 - w}, c=c, k=k)

    top_ids = {"vector": [d.id for d in vec_docs[:k]], "bm25": [d.id for d in bm_docs[:k]],
               "hybrid": [d.id for d, _, _ in fused]}
    qterms = set(tokens)

    def others(doc_id: str, me: str) -> list[str]:
        return [n for n in ("vector", "bm25") if n != me and doc_id in top_ids[n]]

    cols = st.columns(3, gap="medium")
    for col, name in zip(cols, ("vector", "bm25", "hybrid")):
        with col:
            hit = next((r for r, i in enumerate(top_ids[name], 1)
                        if docs_by_id[i].metadata["doc_id"] in gold), None) if gold else None
            verdict = ("" if not gold else f" · first ✓ at rank {hit}" if hit else " · ✗ no gold doc in top k")
            st.markdown(f"<div style='border-bottom:3px solid {COLORS[name]};font-weight:700'>{LABELS[name]}"
                        f"<span style='font-weight:400;opacity:.68;font-size:.8rem'>{verdict}</span></div>",
                        unsafe_allow_html=True)
            if name == "vector":
                for d, (_, s) in zip(vec_docs[:k], vec[:k]):
                    result_card(d, f"{s:.3f}", gold, others(d.id, name), qterms)
            elif name == "bm25":
                for d, (_, s) in zip(bm_docs[:k], bm[:k]):
                    matched = {t for t in qterms if t in bm25.vectorizer.doc_freqs[idx[d.id]]}
                    result_card(d, f"{s:.1f}", gold, others(d.id, name), matched)
            else:
                for d, s, _ in fused:
                    result_card(d, f"{s:.4f}", gold, [], qterms)

    # ---- why BM25 ranked what it did
    with st.expander("🔤 How BM25 saw this question: tokens, rarity (IDF), and which chunks contain them"):
        n_docs = len(bm25.docs)
        rows = []
        for t in dict.fromkeys(tokens):
            df = sum(1 for f in bm25.vectorizer.doc_freqs if t in f)
            rows.append({"token": t, "chunks containing it": df, "of all chunks": f"{100 * df / n_docs:.1f}%",
                         "IDF (rarity weight)": round(bm25.vectorizer.idf.get(t, 0.0), 2)})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("BM25 lowercases, splits into words, and drops stopwords (what, the, of…). A rare token (high IDF) "
                   "that appears in a chunk pushes it up strongly; a common one barely matters. Tokens in no chunk "
                   "contribute nothing. Exact names and numbers are where BM25 shines, and paraphrases are where it fails.")

    # ---- the RRF arithmetic
    with st.expander("🔀 How hybrid fused the lists: the RRF arithmetic, step by step"):
        st.latex(r"\text{RRF}(d) = \sum_{\text{list}} \frac{w_{\text{list}}}{c + \text{rank}_{\text{list}}(d)}")
        rows = []
        for d, s, ranks in fused:
            rv, rb = ranks.get("vector"), ranks.get("bm25")
            rows.append({"chunk": f"{d.metadata['title'][:45]} ({d.metadata['year']})",
                         "vector rank": rv, "BM25 rank": rb,
                         f"{w:.2f}/({c}+rank)": round(w / (c + rv), 5) if rv else 0,
                         f"{1 - w:.2f}/({c}+rank)": round((1 - w) / (c + rb), 5) if rb else 0,
                         "RRF score": round(s, 5)})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(f"RRF ignores the raw scores (cosine 0–1 and BM25 0–∞ aren't comparable) and uses only ranks. "
                   f"A chunk ranked well by *both* lists beats one ranked 1st by only one. An empty rank means it wasn't in that "
                   f"list's top {pool}.")

    # ---- whole golden set
    st.divider()
    st.markdown("#### Across all 28 answerable test questions")
    st.caption("Does the right document appear? Computed with the settings above (weight, c, candidates). "
               "It embeds the 28 questions once, a fraction of a cent.")
    if st.button("Compare vector vs BM25 vs hybrid on the golden set"):
        answerable = [q for q in qs if q["expected_behavior"] == "answer"]
        table, disagreements = {n: {"hit@1": 0, f"hit@{k}": 0, "ranks": []} for n in LABELS}, []
        with st.spinner("Retrieving for 28 questions…"):
            for q in answerable:
                v = [docs_by_id[i] for i, _ in vector_pool(q["question"], pool)]
                b = [bm25.docs[i] for i, _ in bm25_pool(q["question"], pool)[0]]
                h = [d for d, _, _ in rrf_fuse({"vector": v, "bm25": b}, {"vector": w, "bm25": 1 - w}, c=c, k=pool)]
                row = {"id": q["id"]}
                for name, lst in (("vector", v), ("bm25", b), ("hybrid", h)):
                    r = next((i for i, d in enumerate(lst, 1) if d.metadata["doc_id"] in q["gold_docs"]), None)
                    table[name]["hit@1"] += r == 1
                    table[name][f"hit@{k}"] += bool(r and r <= k)
                    table[name]["ranks"].append(1 / r if r else 0)
                    row[name] = str(r) if r else "miss"
                if len({"hit" if row[n] != "miss" and int(row[n]) <= k else "miss" for n in LABELS}) > 1:
                    disagreements.append({**row, "question": q["question"][:80]})
        n = len(answerable)
        st.dataframe([{"method": LABELS[m], "hit@1": f"{t['hit@1']}/{n}", f"hit@{k}": f"{t[f'hit@{k}']}/{n}",
                       "MRR": round(statistics.mean(t["ranks"]), 3)} for m, t in table.items()],
                     hide_index=True, width="stretch")
        if disagreements:
            st.markdown(f"**Where the methods disagree** (gold-document rank; a miss is beyond {pool}):")
            st.dataframe(disagreements, hide_index=True, width="stretch")


# ------------------------------------------ step 5 · score gate & top k

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs CVD-safe). Slot 3 (aqua) is
# below 3:1 contrast on white, so every chart also has shapes + a table view (relief rule).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GROUPS = {"answer": "Answerable (28)", "refuse": "Not in corpus (4)", "offtopic": "Off-topic (2)"}
METHODS = {"vector": "Vector", "bm25": "BM25", "hybrid": "Hybrid (RRF 50/50)"}


def group_of(q: dict) -> str:
    return "offtopic" if q["category"] == "H" else ("answer" if q["expected_behavior"] == "answer" else "refuse")


@st.cache_data(show_spinner="Retrieving the top 50 chunks for all 34 test questions…")
def retrieval_table(depth: int = 50) -> list[dict]:
    """For every golden question: top-`depth` lists from each method (doc ids + chunk texts) and vector scores."""
    bm25 = get_bm25_retriever()
    by_id = {d.id: d for d in bm25.docs}
    rows = []
    for q in golden():
        vec = vector_pool(q["question"], depth)
        v_docs = [by_id[i] for i, _ in vec]
        b_docs = [bm25.docs[i] for i, _ in bm25_pool(q["question"], depth)[0]]
        h_docs = [d for d, _, _ in rrf_fuse({"vector": v_docs, "bm25": b_docs}, {"vector": .5, "bm25": .5}, k=depth)]
        rows.append({"q": q, "top_score": vec[0][1],
                     "lists": {m: [(d.metadata["doc_id"], d.page_content) for d in docs]
                               for m, docs in (("vector", v_docs), ("bm25", b_docs), ("hybrid", h_docs))}})
    return rows


@st.cache_data
def last_eval_stages() -> dict[str, str]:
    """Refusal stage per question from the most recent evaluation run (for context in the tooltip)."""
    runs = sorted((config.EVAL_DIR / "results").glob("2*.json"))
    if not runs:
        return {}
    import json
    res = json.loads(runs[-1].read_text())["results"]
    return {r["id"]: (r["rag"]["refusal_stage"] or "answered") for r in res}


def threshold_k_tab() -> None:
    import altair as alt
    st.markdown("<div class='lab-sub'>Two design numbers: the <b>score gate</b> (refuse without an LLM call if even the "
                "best chunk is dissimilar) and <b>top k</b> (how many chunks the LLM reads). Both are set in "
                "<code>config.py</code>. Here you can see the data behind them.</div>", unsafe_allow_html=True)
    rows = retrieval_table()
    stages = last_eval_stages()

    # ---------- A. the score gate
    st.markdown("#### 1 · The score gate: where should “I don't know” start?")
    thr = st.slider("Score gate threshold (MIN_SIMILARITY)", 0.0, 0.8, float(config.MIN_SIMILARITY), 0.01,
                    help=f"Currently {config.MIN_SIMILARITY} in config.py. Moving the slider here changes nothing in the app.")
    pts = [{"id": r["q"]["id"], "question": r["q"]["question"], "group": GROUPS[group_of(r["q"])],
            "best similarity": round(r["top_score"], 3), "category": r["q"]["category"],
            "at this threshold": "refused by gate" if r["top_score"] < thr else "passes to evidence check",
            "last eval run": stages.get(r["q"]["id"], "—")} for r in rows]
    n_off = sum(p["group"] == GROUPS["offtopic"] for p in pts)
    n_ref = sum(p["group"] == GROUPS["refuse"] for p in pts)
    n_ans = sum(p["group"] == GROUPS["answer"] for p in pts)
    caught_off = sum(p["group"] == GROUPS["offtopic"] and p["best similarity"] < thr for p in pts)
    caught_ref = sum(p["group"] == GROUPS["refuse"] and p["best similarity"] < thr for p in pts)
    wrong = sum(p["group"] == GROUPS["answer"] and p["best similarity"] < thr for p in pts)
    m1, m2, m3 = st.columns(3)
    m1.metric("Off-topic caught by the gate", f"{caught_off}/{n_off}")
    m2.metric("Not-in-corpus caught by the gate", f"{caught_ref}/{n_ref}",
              help="These usually score high (related topics), so the LLM evidence check has to catch them.")
    m3.metric("Answerable wrongly refused", f"{wrong}/{n_ans}", delta=None if not wrong else "false refusals",
              delta_color="inverse")

    domain = list(GROUPS.values())
    color = alt.Color("group:N", scale=alt.Scale(domain=domain, range=SERIES), legend=alt.Legend(orient="top", title=None, labelLimit=400))
    shape = alt.Shape("group:N", scale=alt.Scale(domain=domain, range=["circle", "diamond", "triangle-up"]), legend=None)
    base = alt.Chart(alt.Data(values=pts))
    zone = alt.Chart(alt.Data(values=[{"x0": 0, "x1": thr}])).mark_rect(opacity=.08, color="#8a867c").encode(
        x="x0:Q", x2="x1:Q")
    dots = base.mark_point(size=110, filled=True, stroke="white", strokeWidth=1.5).encode(
        x=alt.X("best similarity:Q", scale=alt.Scale(domain=[0, .9]), title="best (top-1) cosine similarity"),
        y=alt.Y("group:N", sort=domain, title=None, axis=alt.Axis(labelLimit=320)),
        yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1, 1])), color=color, shape=shape,
        tooltip=["id:N", "question:N", "best similarity:Q", "at this threshold:N", "last eval run:N"],
    ).transform_calculate(jitter="random()*1.6-0.8")
    rule = alt.Chart(alt.Data(values=[{"t": thr}])).mark_rule(strokeDash=[4, 3], strokeWidth=2, color="#8a867c").encode(x="t:Q")
    label = rule.mark_text(align="left", dx=6, dy=-8, fontSize=12, color="#8a867c").encode(
        y=alt.value(0), text=alt.value(f"gate = {thr:.2f}  (refuse ←)"))
    st.altair_chart((zone + dots + rule + label).properties(height=300), width="stretch")
    st.caption("Each dot is one test question at its best retrieved similarity; hover for details. Left of the dashed "
               "line is refused with no LLM call. Off-topic questions sit far left, but the not-in-corpus ones sit "
               "among the answerable ones, which is why the pipeline has a second, LLM-based evidence check.")

    sweep = []
    for i in range(0, 81):
        x = i / 100
        sweep.append({"threshold": x, "series": "Should-refuse caught",
                      "percent": 100 * sum(p["best similarity"] < x for p in pts if p["group"] != GROUPS["answer"]) / (n_off + n_ref)})
        sweep.append({"threshold": x, "series": "Answerable wrongly refused",
                      "percent": 100 * sum(p["best similarity"] < x for p in pts if p["group"] == GROUPS["answer"]) / n_ans})
    sdomain = ["Should-refuse caught", "Answerable wrongly refused"]
    sc = alt.Chart(alt.Data(values=sweep)).encode(
        x=alt.X("threshold:Q", title="score gate threshold"),
        y=alt.Y("percent:Q", title="% of questions", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("series:N", scale=alt.Scale(domain=sdomain, range=SERIES[1::-1]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=400)))
    hover = alt.selection_point(fields=["threshold"], nearest=True, on="pointerover", empty=False)
    lines = sc.mark_line(strokeWidth=2, interpolate="step-after")
    hits = sc.mark_point(size=60, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                                                       tooltip=["threshold:Q", "series:N", alt.Tooltip("percent:Q", format=".0f")]).add_params(hover)
    now = alt.Chart(alt.Data(values=[{"t": thr}])).mark_rule(strokeDash=[4, 3], color="#8a867c").encode(x="t:Q")
    st.altair_chart((lines + hits + now).properties(height=240), width="stretch")
    st.caption("Sweeping the threshold: the gate only ever catches the 2 off-topic questions before it starts refusing "
               "answerable ones. The safe zone is roughly 0.22–0.47; 0.35 sits in the middle of it.")
    with st.expander("Table view"):
        st.dataframe(sorted(pts, key=lambda p: p["best similarity"]), hide_index=True, width="stretch")

    # ---------- B. the k curve
    st.markdown("#### 2 · Top k: how many chunks should the LLM read?")
    level = st.radio("Count a hit when…", ["the right document is in the top k", "the answer text itself is in the top k"],
                     horizontal=True, help="Document-level uses gold_docs (28 questions). Answer-text level uses each "
                                           "question's evidence pattern (21 questions), which is stricter.")
    chunk_level = level.startswith("the answer text")
    answerable = [r for r in rows if r["q"]["expected_behavior"] == "answer" and (not chunk_level or r["q"].get("evidence"))]
    curve, first_hit = [], {}
    for m in METHODS:
        for r in answerable:
            q = r["q"]
            hit_at = next((i for i, (doc_id, text) in enumerate(r["lists"][m], 1)
                           if (re.search(q["evidence"], text) if chunk_level else doc_id in q["gold_docs"])), None)
            first_hit.setdefault(q["id"], {})[METHODS[m]] = hit_at
        for k_ in range(1, 21):
            got = sum(bool(first_hit[r["q"]["id"]][METHODS[m]]) and first_hit[r["q"]["id"]][METHODS[m]] <= k_
                      for r in answerable)
            curve.append({"k": k_, "method": METHODS[m], "percent": round(100 * got / len(answerable), 1), "hits": got})
    kdomain = list(METHODS.values())
    kc = alt.Chart(alt.Data(values=curve)).encode(
        x=alt.X("k:Q", title="top k chunks retrieved", scale=alt.Scale(domain=[1, 20])),
        y=alt.Y("percent:Q", title=f"% of {len(answerable)} questions with a hit", scale=alt.Scale(domain=[40, 100])),
        color=alt.Color("method:N", scale=alt.Scale(domain=kdomain, range=SERIES), legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        shape=alt.Shape("method:N", scale=alt.Scale(domain=kdomain, range=["circle", "diamond", "triangle-up"]), legend=None))
    khover = alt.selection_point(fields=["k"], nearest=True, on="pointerover", empty=False)
    klines = kc.mark_line(strokeWidth=2) + kc.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(khover, alt.value(1), alt.value(.25)),
        tooltip=["method:N", "k:Q", "hits:Q", alt.Tooltip("percent:Q", format=".0f")]).add_params(khover)
    know = alt.Chart(alt.Data(values=[{"k": config.TOP_K}])).mark_rule(strokeDash=[4, 3], color="#8a867c").encode(x="k:Q")
    st.altair_chart((klines + know).properties(height=300), width="stretch")
    ctx = config.TOP_K * 567
    st.caption(f"The dashed line is the current TOP_K = {config.TOP_K}. Every extra chunk adds ~567 tokens that "
               f"<b>both</b> LLM calls must read (k = {config.TOP_K} ≈ {ctx:,} tokens per call; k = 10 ≈ {10 * 567:,}), "
               "so a higher k costs money and latency, and buries the relevant sentence among more noise. "
               "Where the curve flattens is the sensible k. Switch to 'answer text' to see how much stricter "
               "chunk-level retrieval is.", unsafe_allow_html=True)
    with st.expander("Table view: rank of the first hit per question (empty = not in the top 50)"):
        st.dataframe([{"id": qid, **v} for qid, v in first_hit.items()], hide_index=True, width="stretch")


# ---------------------------------------- step 6 · prompts & generation

STEP_LABEL = {"retrieve": "retrieve (Chroma)", "grade_evidence": "evidence check (LLM call 1)",
              "generate": "generate (LLM call 2)"}
PROMPT_RULES = [
    ("Answer using ONLY the numbered sources", "Grounding: the model must not fill gaps from memory. This is what separates RAG from the LLM-only column."),
    ("Every factual sentence must cite its source(s) inline, like [1]", "Makes every claim checkable; the app turns [n] into source cards, and the faithfulness judge checks each one."),
    ("Do not add outside knowledge, even if you know it", "gpt-4.1-mini knows plenty of AI history, some of it wrong ('$500'). This rule keeps answers inside the corpus."),
    ("If the sources only partly answer, say what is missing", "Prefers an honest partial answer over refusing outright or padding it with guesses."),
    ("Keep numbers, names and dates exactly as the sources give them", "Guards the long-tail details (13,500; 3.2 million; 10^120) where the LLM alone is weakest."),
    ("Be concise", "Fewer sentences mean fewer claims that can go wrong, and lower latency."),
]


class CallRecorder(BaseCallbackHandler):
    """LangChain callback handler that records every LLM call inside the graph, exactly as sent."""

    def __init__(self):
        self.calls, self._open = [], {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        self._open[run_id] = {"step": (metadata or {}).get("langgraph_node", "?"),
                              "model": (kw.get("invocation_params") or {}).get("model", "?"),
                              "prompt": messages[0][-1].content, "t0": time.perf_counter()}

    def on_llm_end(self, response, *, run_id, **kw):
        call = self._open.pop(run_id)
        msg = response.generations[0][0].message
        call.update(seconds=round(time.perf_counter() - call.pop("t0"), 3),
                    output=msg.content or str(msg.additional_kwargs), usage=msg.usage_metadata or {})
        self.calls.append(call)


def run_inspected(question: str) -> dict:
    import tiktoken
    from rag.rag_graph import GRAPH, format_context
    rec = CallRecorder()
    t0 = time.perf_counter()
    state = GRAPH.invoke({"question": question, "strategy": config.DEFAULT_STRATEGY, "timings": {}},
                         config={"callbacks": [rec]})
    total = round(time.perf_counter() - t0, 3)
    enc = tiktoken.get_encoding("o200k_base")          # the gpt-4.1 family's tokenizer
    context = format_context(state.get("retrieved", [])) if state.get("retrieved") else ""
    for call in rec.calls:
        api_in = call["usage"].get("input_tokens", len(enc.encode(call["prompt"])))
        src = len(enc.encode(context)) if context and context[:200] in call["prompt"] else 0
        q = len(enc.encode(question))
        call["parts"] = {"instructions + formatting": max(0, api_in - src - q), "retrieved sources": src,
                         "question": q, "output": call["usage"].get("output_tokens", 0)}
        call["cached"] = (call["usage"].get("input_token_details") or {}).get("cache_read", 0)
    return {"question": question, "state": state, "calls": rec.calls, "total": total,
            "retrieved": [{"rank": i, "similarity": round(s, 3), "document": d.metadata["title"][:60],
                           "year": d.metadata["year"], "section": d.metadata.get("section") or "",
                           "tokens": len(enc.encode(d.page_content))}
                          for i, (d, s) in enumerate(state.get("retrieved", []), 1)]}


def prompt_inspector_tab() -> None:
    import altair as alt
    st.markdown("<div class='lab-sub'>Runs the real pipeline once and records every LLM call exactly as it was sent: "
                "the full prompt, the raw response, the token counts OpenAI reports, and how long each step took. "
                "One run costs about as much as one chat question.</div>", unsafe_allow_html=True)
    labels = {f"{q['id']} [{q['category']}] {q['question']}": q for q in golden()}
    c1, c2 = st.columns([3, 2])
    pick = c1.selectbox("Pick a test question", list(labels), index=15, key="pi_pick")   # q16 Dartmouth budget
    typed = c2.text_input("…or type your own", key="pi_typed")
    question = typed.strip() or labels[pick]["question"]
    if st.button("▶ Run the pipeline and capture every LLM call", type="primary"):
        if blocked := try_spend():
            st.warning(blocked, icon="⏳")
        else:
            with st.spinner("Running retrieve → evidence check → generate…"):
                st.session_state.pi_result = run_inspected(question)
    if note := remaining_note():
        st.caption(note)
    res = st.session_state.get("pi_result")
    if not res:
        st.info("Press the button to run one question through the pipeline.")
        return
    if res["question"] != question:
        st.caption(f"Showing the last run: “{res['question']}”. Press the button to inspect the new question.")

    s, calls = res["state"], res["calls"]
    m = st.columns(5)
    m[0].metric("Total time", f"{res['total']:.2f}s")
    m[1].metric("LLM calls", len(calls))
    m[2].metric("Input tokens", f"{sum(c['usage'].get('input_tokens', 0) for c in calls):,}")
    m[3].metric("…of which cached", f"{sum(c['cached'] for c in calls):,}",
                help="OpenAI caches the beginning of a prompt (1,024+ tokens) that matches a recent request, e.g. the "
                     "same question asked again. Cached input is cheaper and faster. The two calls here start "
                     "differently, so they don't share a cache entry with each other.")
    m[4].metric("Output tokens", f"{sum(c['usage'].get('output_tokens', 0) for c in calls):,}")
    outcome = (f"🛑 Refused at the **{s['refusal_stage'].replace('_', ' ')}**" if s["refused"]
               else f"✅ Answered, citing {len(s['sources'])} source(s)")
    st.markdown(outcome)

    # ---- timeline
    st.markdown("#### Where the time goes")
    segs, start = [], 0.0
    for key, name in (("retrieve_s", "retrieve"), ("grade_s", "grade_evidence"), ("generate_s", "generate")):
        if key in s["timings"]:
            d = s["timings"][key]
            segs.append({"step": STEP_LABEL[name], "start": round(start, 3), "end": round(start + d, 3),
                         "seconds": d, "label": f"{d:.2f}s"})
            start += d
    order = [STEP_LABEL[n] for n in ("retrieve", "grade_evidence", "generate")]
    gantt = alt.Chart(alt.Data(values=segs)).encode(y=alt.Y("step:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
    bars = gantt.mark_bar(cornerRadius=4, height=22).encode(
        x=alt.X("start:Q", title="seconds since the question arrived"), x2="end:Q",
        color=alt.Color("step:N", scale=alt.Scale(domain=order, range=SERIES), legend=None),
        tooltip=["step:N", "seconds:Q"])
    text = gantt.mark_text(align="left", dx=6, color="#8a867c").encode(x="end:Q", text="label:N")
    st.altair_chart((bars + text).properties(height=150), width="stretch")
    st.caption("Steps run one after another. Retrieval is one embedding call plus a Chroma lookup; each LLM "
               "step has to read the whole prompt before it writes anything. In the chat, the generate step "
               "streams, so its first words appear well before this bar ends.")

    # ---- token budget
    if calls:
        st.markdown("#### What each LLM call reads and writes (tokens)")
        parts = ["instructions + formatting", "retrieved sources", "question", "output"]
        rows = [{"call": STEP_LABEL.get(c["step"], c["step"]), "part": k, "tokens": c["parts"][k], "order": i}
                for c in calls for i, k in enumerate(parts)]
        present = [STEP_LABEL.get(c["step"], c["step"]) for c in calls]
        budget = alt.Chart(alt.Data(values=rows)).mark_bar(stroke="white", strokeWidth=2, cornerRadius=3).encode(
            y=alt.Y("call:N", sort=present, title=None, axis=alt.Axis(labelLimit=260),
                    scale=alt.Scale(paddingInner=0.35)),
            x=alt.X("sum(tokens):Q", title="tokens"),
            color=alt.Color("part:N", scale=alt.Scale(domain=parts, range=SERIES + ["#eda100"]),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("order:Q"), tooltip=["call:N", "part:N", "tokens:Q"])
        st.altair_chart(budget.properties(height=70 * len(calls) + 20), width="stretch")
        st.caption("The retrieved sources dominate: " + (
            "both calls read the same ~5 chunks, so top k is the main cost lever. 'Cached input' comes from "
            "OpenAI's prompt caching: when a prompt starts with the same 1,024+ tokens as a recent request (e.g. "
            "the same question asked again), that part is reused at a discount." if len(calls) > 1 else
            "the evidence check reads all ~5 chunks just to decide. A refusal here still costs one LLM call, "
            "whereas the score gate refuses for free."))
        st.dataframe([{"call": STEP_LABEL.get(c["step"], c["step"]), "model": c["model"], **c["parts"],
                       "cached input": c["cached"], "API input total": c["usage"].get("input_tokens"),
                       "seconds": c["seconds"]} for c in calls], hide_index=True, width="stretch")

    # ---- the exact prompts
    st.markdown("#### The exact prompts")
    for i, c in enumerate(calls, 1):
        with st.expander(f"LLM call {i} · {STEP_LABEL.get(c['step'], c['step'])} · {c['model']} · "
                         f"{c['usage'].get('input_tokens', '?'):,} tokens in → {c['usage'].get('output_tokens', '?')} out"):
            st.markdown("**Prompt as sent**")
            st.code(c["prompt"], language="markdown", wrap_lines=True)
            st.markdown("**Raw response**")
            st.code(c["output"], language="json" if c["output"].lstrip().startswith("{") else "markdown", wrap_lines=True)
    if not calls:
        st.caption("No LLM calls: the score gate refused before any prompt was built.")
    with st.expander("Why each rule in the answer prompt is there"):
        for rule, why in PROMPT_RULES:
            st.markdown(f"- **{rule}.** {why}")
    with st.expander(f"The {len(res['retrieved'])} retrieved chunks (in prompt order)"):
        st.dataframe(res["retrieved"], hide_index=True, width="stretch")


# ------------------------------------------------- step 1 · tokens

TOKENIZERS = {"cl100k_base": "text-embedding-3-small · chunk sizes", "o200k_base": "gpt-4.1 / gpt-4.1-mini"}
PRESETS = {  # (doc_id, regex) -> a real passage from the corpus, found at runtime in chunks.jsonl
    "Shannon 1950: an exponent lost in PDF extraction": ("shannon_chess_1950", r"10120"),
    "Turing 1950: “about 109” (really 10^9)": ("turing_1950", r"about 109"),
    "Dartmouth 1955: the budget table": ("dartmouth_proposal_1955", r"13,500"),
    "ELIZA 1966: OCR noise from an old scan": ("weizenbaum_eliza_1966", r"usclls|Tcchnu"),
    "Wikipedia RLHF: math extracted as LaTeX": ("wiki_rlhf", r"pretrain\n"),
}
TOKEN_TINTS = ["rgba(42,120,214,.16)", "rgba(235,104,52,.16)"]   # alternate only to show boundaries


@st.cache_data
def chunk_texts() -> list[dict]:
    import json
    with (config.PROCESSED_DIR / "chunks.jsonl").open() as f:
        return [json.loads(line) for line in f]


def preset_text(doc_id: str, pattern: str) -> str:
    for c in chunk_texts():
        if c["metadata"]["doc_id"] == doc_id and (m := re.search(pattern, c["text"])):
            body = c["text"]
            return body[max(0, m.start() - 160): m.end() + 220].strip()
    return ""


@st.cache_resource
def encoder(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


def token_html(text: str, enc) -> tuple[str, list[dict]]:
    ids = enc.encode(text)
    spans, rows = [], []
    for i, tid in enumerate(ids):
        piece = enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace")
        shown = html.escape(piece).replace(" ", "<span class='ws'>·</span>").replace("\n", "<span class='ws'>↵</span><br>")
        spans.append(f"<span class='tok' style='background:{TOKEN_TINTS[i % 2]}' title='id {tid}'>{shown}</span>")
        rows.append({"#": i + 1, "token": repr(piece), "id": tid})
    return "".join(spans), rows


@st.cache_data(show_spinner="Tokenizing all 1,145 chunks…")
def corpus_efficiency() -> list[dict]:
    enc = encoder("cl100k_base")
    per = {}
    for c in chunk_texts():
        body = c["text"].split("\n", 1)[-1]
        d = per.setdefault(c["metadata"]["doc_id"], {"doc": c["metadata"]["title"][:48], "chars": 0, "tokens": 0,
                                                    "type": c["metadata"]["document_type"]})
        d["chars"] += len(body)
        d["tokens"] += len(enc.encode(body))
    return [{**v, "chars per token": round(v["chars"] / v["tokens"], 2)} for v in per.values()]


def tokenizer_tab() -> None:
    import altair as alt
    st.markdown("<div class='lab-sub'>Models never see characters. They see <b>tokens</b>, pieces of text from a "
                "fixed vocabulary. Chunk sizes, costs, context limits and even what a model can \"see\" in a number "
                "are all counted in tokens. Two tokenizers are in play here: the embedding model's and the chat "
                "model's. Everything on this page runs locally, with no API calls.</div>", unsafe_allow_html=True)
    st.markdown("""<style>
      .tokbox { font: 13px/2 ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-word; }
      .tok { border-radius: 3px; padding: 1px 0; margin-right: 1px; box-shadow: inset 0 0 0 1px rgba(127,127,127,.25); }
      .ws { opacity: .45; }
    </style>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox("Example from the corpus", ["Type your own…"] + list(PRESETS), index=1)
    default = "What does the 'T' in GPT stand for? ChatGPT, GPT-3, $13,500, 10^120 and 10120." \
        if choice == "Type your own…" else preset_text(*PRESETS[choice])
    text = c2.text_area("Text to tokenize", value=default, height=120, key=f"tok_{choice}")
    if not text:
        return

    cols = st.columns(2, gap="large")
    for col, (name, used_by) in zip(cols, TOKENIZERS.items()):
        enc = encoder(name)
        spans, rows = token_html(text, enc)
        with col:
            st.markdown(f"**`{name}`** · <span style='opacity:.68'>{used_by}</span>", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Tokens", len(rows))
            b.metric("Characters", len(text))
            c.metric("Chars per token", f"{len(text) / max(len(rows), 1):.2f}")
            st.markdown(f"<div class='tokbox'>{spans}</div>", unsafe_allow_html=True)
            with st.expander("Token list"):
                st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Each shaded box is one token (hover for its id). · marks a space, ↵ a line break. Alternating "
               "shades only show where one token ends and the next begins.")

    with st.expander("What to notice"):
        st.markdown(
            "- **Numbers lose their meaning.** `10120` becomes `101` + `20`. Once the PDF dropped the superscript, "
            "nothing in the tokens says “10 to the power 120”, so the model has to rely on context. Question q17 "
            "tests exactly this.\n"
            "- **The tokenizers disagree.** `cl100k_base` splits `GPT` into `G` + `PT`, while `o200k_base` keeps "
            "`GPT` whole. The embedding model (cl100k) never sees “GPT” as one unit, which helps explain why "
            "vector search struggles with q05 (“What does the T in GPT stand for?”).\n"
            "- **Noise is expensive.** OCR garbage and LaTeX leftovers tokenize into many short pieces: more tokens "
            "per useful word, so they use up chunk space and prompt budget while adding almost no meaning.\n"
            "- **Spaces belong to the next word.** ` the` (with its leading space) is one token, which is why "
            "tokens often start with ·.")

    st.divider()
    st.markdown("#### How efficiently each document tokenizes")
    eff = sorted(corpus_efficiency(), key=lambda r: r["chars per token"])
    show = eff[:8] + eff[-8:]
    chart = alt.Chart(alt.Data(values=show)).mark_bar(cornerRadius=3, color=SERIES[0]).encode(
        y=alt.Y("doc:N", sort=[r["doc"] for r in show], title=None,
                axis=alt.Axis(labelLimit=340, labelOverlap=False)),
        x=alt.X("chars per token:Q", title="characters per token (cl100k_base; higher = cleaner prose)",
                axis=alt.Axis(tickCount=6)),
        tooltip=["doc:N", "type:N", "chars:Q", "tokens:Q", "chars per token:Q"])
    labels = chart.mark_text(align="left", dx=4, color="#8a867c").encode(text=alt.Text("chars per token:Q", format=".2f"))
    st.altair_chart(chart + labels, height=16 * 30, width="stretch")
    avg = sum(r["chars"] for r in eff) / sum(r["tokens"] for r in eff)
    st.caption(f"The 8 least and 8 most efficient of {len(eff)} documents (corpus average {avg:.2f} characters per "
               "token). The least efficient are all math-heavy (equations, LaTeX leftovers from Wikipedia, logic "
               "notation); the most efficient are plain prose such as the Lighthill Report and the Bitter Lesson. "
               "A 600-token chunk of a math-heavy document holds roughly 40% less text than one of clean prose.")
    with st.expander("Table view: all documents"):
        st.dataframe(eff, hide_index=True, width="stretch")


# ------------------------------------------------------------ pipeline pages

PIPELINE = [  # (key, icon, name, what happens at this step) in the order data flows through the system
    ("tokens", "🔡", "Tokens", "Text is split into tokens, the unit every size, cost and limit is measured in."),
    ("chunks", "🧩", "Chunks", "Each cleaned document is split into ~600-token chunks, each with a title/section header."),
    ("embeddings", "🌌", "Embeddings", "Each chunk becomes a 1,536-number vector; similar meanings end up close together."),
    ("retrieval", "🔀", "Retrieval", "A question is matched against all chunks by meaning (vector), keywords (BM25) or both."),
    ("gate", "🎚️", "Score gate & top k", "Decide whether to refuse without an LLM call, and how many chunks the LLM reads."),
    ("prompts", "🧾", "Prompts & generation", "The evidence check and the answer: exactly what each LLM call receives and returns."),
]


def pipeline_header(key: str) -> None:
    """Page title with its step number, plus a breadcrumb of the whole pipeline with this step highlighted."""
    lab_css()
    i = next(n for n, step in enumerate(PIPELINE) if step[0] == key)
    _, icon, name, what = PIPELINE[i]
    crumbs = " <span class='arrow'>→</span> ".join(
        f"<span class='crumb{' on' if n == i else ''}'>{n + 1} · {s[2]}</span>" for n, s in enumerate(PIPELINE))
    st.markdown(f"<div class='crumbs'>{crumbs}</div>", unsafe_allow_html=True)
    st.title(f"{icon} {i + 1} · {name}")
    st.markdown(f"<div class='lab-sub'><b>This step:</b> {what}</div>", unsafe_allow_html=True)


def tokens_page() -> None:
    pipeline_header("tokens")
    tokenizer_tab()
    render_learning("tokens")
    tour_footer("tokens")


def retrieval_page() -> None:
    pipeline_header("retrieval")
    retrieval_tab()
    render_learning("retrieval")
    tour_footer("retrieval")


def gate_page() -> None:
    pipeline_header("gate")
    threshold_k_tab()
    render_learning("gate")
    tour_footer("gate")


def prompts_page() -> None:
    pipeline_header("prompts")
    prompt_inspector_tab()
    render_learning("prompts")
    tour_footer("prompts")
