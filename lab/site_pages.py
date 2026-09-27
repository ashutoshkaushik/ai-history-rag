"""Site pages: the 'Start here' landing page and the Results page (scoreboard + LLM-errors gallery).

All numbers come from the final evaluation run (eval/results/*-rejudged.json, judged by gpt-4.1)
and the corpus manifest, so the pages update automatically after a new eval run.
"""

import html
import json
import statistics

import streamlit as st
import yaml

from lab.nav import link
from rag import config

RAG_COLOR, LLM_COLOR = "#2a78d6", "#eb6834"      # validated palette slots 1 and 2

CSS = """
<style>
  .eyebrow { font: 600 .74rem/1 "Hanken Grotesk", sans-serif; letter-spacing: .12em; text-transform: uppercase;
             color: #C2603E; margin-bottom: .6rem; }
  .hero { font-family: "Source Serif 4", Georgia, serif; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12;
          font-weight: 600; margin: 0 0 .8rem; text-wrap: balance; }
  .lede { font-size: 1.12rem; line-height: 1.6; opacity: .82; max-width: 62ch; margin-bottom: 1.2rem; }
  .duel { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: .4rem 0 1rem; }
  .duel > div { border: 1px solid rgba(127,127,127,.28); border-radius: .6rem; padding: .9rem 1rem; }
  .duel .who { font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .7; }
  .duel .ans { font-family: "Source Serif 4", Georgia, serif; font-size: 1.9rem; font-weight: 600; margin: .15rem 0; }
  .duel .note { font-size: .86rem; opacity: .75; }
  .duel .bad { border-color: rgba(235,104,52,.55); }
  .duel .good { border-color: rgba(42,120,214,.55); }
  .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .steps > div { border-top: 3px solid #C2603E; padding-top: .5rem; }
  .steps b { display: block; font-size: .95rem; margin-bottom: .2rem; }
  .steps span { font-size: .86rem; opacity: .75; }
  .foot { font-size: .8rem; opacity: .6; }
  .card { border: 1px solid rgba(127,127,127,.28); border-radius: .6rem; padding: .9rem 1rem; margin-bottom: .9rem; }
  .card .q { font-family: "Source Serif 4", Georgia, serif; font-size: 1.12rem; font-weight: 600; margin-bottom: .5rem; }
  .card .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .card .lab { font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .7; margin-bottom: .2rem; }
  .card .txt { font-family: "Source Serif 4", Georgia, serif; font-size: .98rem; line-height: 1.55; }
  .card .wrong { background: rgba(235,104,52,.12); border-left: 3px solid #eb6834; padding: .35rem .6rem; margin-top: .45rem;
                 font-size: .86rem; border-radius: 3px; }
  .card .ref { font-size: .84rem; opacity: .72; margin-top: .6rem; }
  @media (max-width: 760px) { .duel, .card .cols { grid-template-columns: 1fr; } }
</style>"""


@st.cache_data
def final_run() -> dict:
    """The latest re-judged evaluation run (the project's final v1 numbers)."""
    runs = sorted((config.EVAL_DIR / "results").glob("*-rejudged.json"))
    return json.loads(runs[-1].read_text())


@st.cache_data
def manifest() -> dict:
    return yaml.safe_load(config.CORPUS_MANIFEST.read_text())


def pct(a: float, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


# ------------------------------------------------------------------ Start here

def home_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    run, man = final_run(), manifest()
    S, R, L = run["summary"], run["summary"]["rag"], run["summary"]["llm"]
    n, m = S["n_answerable"], S["n_refuse"]
    q16 = next((r for r in run["results"] if r["id"] == "q16"), None)

    st.markdown("<div class='eyebrow'>AI history · retrieval-augmented generation</div>"
                "<div class='hero'>Ask about the history of AI, and see exactly where each answer comes from.</div>"
                f"<div class='lede'>A research assistant grounded in {len(man['documents'])} curated sources, from "
                "Turing's 1950 paper and the Dartmouth proposal to the Lighthill Report and GPT-4. It answers only "
                "from those documents, cites every claim, and says so when the sources don't have the answer. Next "
                "to it runs the same model with no sources, so you can see what grounding changes.</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        link("chat", "Try the assistant", "💬", width="stretch")
    with c2:
        link("tokens", "Take the pipeline tour", "🔬", width="stretch")
    with c3:
        link("results", "See the results", "📊", width="stretch")

    st.markdown("### The $13,500 question")
    llm_ans = html.escape(q16["llm"]["answer"]) if q16 else "The 1955 Dartmouth proposal requested $500."
    st.markdown(
        "<div class='duel'>"
        f"<div class='bad'><div class='who'>🤖 {config.CHAT_MODEL}, no sources</div><div class='ans'>$500</div>"
        f"<div class='note'>“{llm_ans}” The same answer in 5 of 5 runs: confident, and wrong.</div></div>"
        f"<div class='good'><div class='who'>🧠 {config.CHAT_MODEL} + RAG</div><div class='ans'>$13,500</div>"
        "<div class='note'>Itemized from the budget table in the 1955 proposal itself, with a citation "
        "to the primary source.</div></div></div>", unsafe_allow_html=True)
    st.caption("“How much money did the 1955 Dartmouth proposal request?” is one of 34 test questions. Larger models "
               "such as gpt-4.1 do know this one ($13,500 in 5 of 5 runs). The point is that retrieval lets a small, "
               "cheap model match them on obscure details, and adds a source you can check instead of trusting "
               "any model's memory.")

    st.markdown(f"### Measured on {S['n']} test questions")
    k = st.columns(4)
    k[0].metric("Fully correct answers", pct(R["correct"], n),
                delta=f"{100 * (R['correct'] - L['correct']) / n:+.0f} pts vs no sources")
    k[1].metric("Claims supported by a source", f"{100 * R['faithfulness']:.0f}%",
                help="Faithfulness, graded claim by claim by gpt-4.1 and checked against a manual audit.")
    k[2].metric("Correctly said “I don't know”", f"{R['correct_refusals']}/{m}",
                delta=f"no sources: {L['correct_refusals']}/{m}", delta_color="off")
    k[3].metric("Answer time (95th percentile)", f"{S['latency']['total_s']['p95']:.1f}s",
                help="Target: under 8 seconds.")

    st.markdown("### How it works")
    st.markdown(
        "<div class='steps'>"
        "<div><b>1 · Retrieve</b><span>Find the 5 most relevant passages among 1,145 chunks of the corpus.</span></div>"
        "<div><b>2 · Gate</b><span>If nothing is even close, refuse instantly, with no model call.</span></div>"
        "<div><b>3 · Check</b><span>A model reads the passages and decides whether they actually answer.</span></div>"
        "<div><b>4 · Answer</b><span>Write from those passages only, citing each claim as [1]…[5].</span></div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("")
    link("diagram", "See the full system diagram", "🗺️")
    st.markdown(f"<div class='foot'>Built with LangChain, LangGraph, Chroma and OpenAI "
                f"({config.CHAT_MODEL} for answers, {config.EMBEDDING_MODEL} for search, {config.JUDGE_MODEL} for "
                f"grading). Results from evaluation run <code>{run['run_id']}</code>.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------- Results

def results_page() -> None:
    import altair as alt
    st.markdown(CSS, unsafe_allow_html=True)
    run = final_run()
    S, R, L = run["summary"], run["summary"]["rag"], run["summary"]["llm"]
    n, m = S["n_answerable"], S["n_refuse"]
    rows = run["results"]

    st.title("📊 Results: when does RAG beat the model alone?")
    st.markdown(f"<div class='lede'>Both systems answered the same {S['n']} test questions: {n} with answers in "
                f"the corpus and {m} that should be refused. The chat model is {run['settings']['chat_model']} in "
                f"both; answers were graded by {run['settings']['judge_model']}, a different and stronger model. "
                "RAG wins where memory is weakest (obscure details), and it's the only one of the two that knows "
                "when to say “I don't know”.</div>", unsafe_allow_html=True)

    st.markdown("### Scoreboard")
    board = [
        ("Fully correct answers", f"{R['correct']}/{n} ({pct(R['correct'], n)})", f"{L['correct']}/{n} ({pct(L['correct'], n)})"),
        ("Key facts covered (average)", f"{100 * R['coverage']:.0f}%", f"{100 * L['coverage']:.0f}%"),
        ("Answers containing a wrong claim", f"{R['hallucinated']}/{n} *", f"{L['hallucinated']}/{n}"),
        ("Claims supported by the sources (faithfulness)", f"{100 * R['faithfulness']:.0f}% of {R['claims']}", "no sources to check"),
        ("Citations that support their claim", f"{100 * R['citation_accuracy']:.0f}%", "no citations"),
        ("Correctly refused (should refuse)", f"{R['correct_refusals']}/{m}", f"{L['correct_refusals']}/{m}"),
        ("Wrongly refused (answer was in the corpus)", f"{R['false_refusals']}/{n}", f"{L['false_refusals']}/{n}"),
        ("Answer time, median / 95th percentile", f"{S['latency']['total_s']['p50']:.1f}s / {S['latency']['total_s']['p95']:.1f}s",
         f"{S['latency']['llm_total_s']['p50']:.1f}s / {S['latency']['llm_total_s']['p95']:.1f}s"),
    ]
    st.dataframe([{"": a, "🧠 RAG": b, "🤖 Model alone": c} for a, b, c in board], hide_index=True, width="stretch")
    st.caption("* The one RAG flag (q03, “the 1956 Dartmouth project”) is a known judge false positive: the "
               "workshop did take place in 1956. The 3 wrong refusals are retrieval misses (q05, q07, q24): RAG said "
               "“I don't know” rather than guess.")

    st.markdown("### Where RAG adds the most")
    by = st.radio("Group questions by", ["audience", "category"], horizontal=True,
                  format_func=lambda x: {"audience": "who would ask it", "category": "question type"}[x])
    names = {"A": "A · direct lookup", "B": "B · multi-document", "C": "C · comparison", "D": "D · specific evidence",
             "E": "E · temporal", "G": "G · long-tail detail",
             "popular": "popular", "curious": "curious", "expert": "expert"}
    ans = [r for r in rows if r["expected_behavior"] == "answer"]
    data = []
    for key in sorted({r[by] for r in ans}):
        g = [r for r in ans if r[by] == key]
        for side, label in (("rag", "🧠 RAG"), ("llm", "🤖 Model alone")):
            cov = 100 * statistics.mean(r[side]["grade"]["coverage"] or 0 for r in g)
            data.append({"group": f"{names.get(key, key)} ({len(g)})", "system": label, "coverage": round(cov)})
    order = [d["group"] for d in data[::2]]
    base = alt.Chart(alt.Data(values=data)).encode(
        y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        yOffset=alt.YOffset("system:N"),
        x=alt.X("coverage:Q", title="key facts covered (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("system:N", scale=alt.Scale(domain=["🧠 RAG", "🤖 Model alone"], range=[RAG_COLOR, LLM_COLOR]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["group:N", "system:N", "coverage:Q"])
    bars = base.mark_bar(cornerRadiusEnd=4, height=14)
    text = base.mark_text(align="left", dx=4, color="#8a867c").encode(text=alt.Text("coverage:Q", format=".0f"))
    st.altair_chart(bars + text, height=60 + 44 * len(order), width="stretch")
    st.caption("The gap is widest on long-tail details (+56 points) and expert questions (+54): the Dartmouth "
               "budget, ImageNet's size, Lighthill's categories. It is narrowest on comparisons (+8) and timelines "
               "(+9), where the model's general knowledge already covers the broad story.")

    st.markdown("### Questions that should be refused")
    ref = [r for r in rows if r["expected_behavior"] == "refuse"]
    st.dataframe([{"question": r["question"],
                   "type": {"F1": "impossible", "F2": "not in the corpus", "H": "not about AI"}[r["category"]],
                   "🧠 RAG": ("refused, " + r["rag"]["refusal_stage"].replace("_", " ")) if r["rag"]["refused"] else "answered",
                   "🤖 Model alone": "declined" if r["llm"]["grade"]["declined"] else "answered from memory: "
                   + r["llm"]["answer"][:70]} for r in ref], hide_index=True, width="stretch")
    st.caption("The model alone answers most of these from memory, sometimes correctly (Tombaugh, 1930), but with "
               "nothing a reader could check. RAG refuses all six: two for free at the score gate, four at the "
               "evidence check.")

    st.markdown("### Where the model alone got it wrong")
    st.caption("Every answer below was graded against a reference answer verified in the corpus.")
    wrong = [r for r in ans if r["llm"]["grade"]["incorrect_claims"]]
    for r in wrong:
        rag_txt = html.escape(r["rag"]["answer"][:420] + ("…" if len(r["rag"]["answer"]) > 420 else ""))
        claims = "".join(f"<div class='wrong'>✗ {html.escape(c)}</div>" for c in r["llm"]["grade"]["incorrect_claims"])
        st.markdown(
            f"<div class='card'><div class='q'>{html.escape(r['question'])}</div><div class='cols'>"
            f"<div><div class='lab'>🤖 Model alone · {r['llm']['grade']['verdict']}</div>"
            f"<div class='txt'>{html.escape(r['llm']['answer'][:420])}</div>{claims}</div>"
            f"<div><div class='lab'>🧠 RAG · {r['rag']['grade']['verdict']}</div><div class='txt'>{rag_txt}</div></div>"
            f"</div><div class='ref'>Reference: {html.escape(r['reference_answer'])}</div></div>",
            unsafe_allow_html=True)

    st.markdown("### How the answers were graded")
    st.markdown(
        "Grading used an LLM as judge, which needed its own testing. It took three rounds:\n"
        "1. **Same model as the generator (gpt-4.1-mini):** it rated faithfulness **100%**. A manual audit of 57 "
        "claims found 6 unsupported (blended facts, invented dates, editorial add-ons), so it was too lenient.\n"
        "2. **A stronger model (gpt-4.1) with a stricter prompt:** faithfulness **91%**, in line with the audit. But "
        "its correctness grades relied on its own knowledge, and flagged true statements as wrong (for example, "
        "ELIZA's code being rediscovered in 2021).\n"
        "3. **Correctness graded against the reference answer only**, with grounding left to the faithfulness "
        "judge. That gave sensible verdicts, apart from one known false positive (q03).")
    st.caption(f"Full per-question results: eval/results/{run['run_id']}.md · manual audit: eval/manual_audit.yaml")
