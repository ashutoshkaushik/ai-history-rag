"""AI History Research Assistant: Streamlit chat UI.

    uv run streamlit run app.py

Structure follows the course's company_kb_viewer.py: main() builds the page and tabs,
render_research_assistant() is the chat, render_corpus_browser() lists the corpus.
All RAG logic lives in src/rag; this file only calls stream_answer() and llm_only_stream(),
so pipeline improvements show up here without UI changes.

Side-by-side mode runs RAG and LLM-only at the same time: each runs in a worker thread that
pushes streamed tokens into a queue, and the main script thread (the only one allowed to draw
in Streamlit) drains both queues and updates the two columns as text arrives.
"""

import queue
import re
import sys
import threading
import time
import warnings
from pathlib import Path

# Make the `rag` package importable without installing the project (Streamlit Community Cloud
# installs requirements.txt only). Locally, `uv run` has already installed it; this is harmless.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import yaml

from lab.explore_pages import chunk_viewer_page, diagram_page, embedding_viewer_page
from lab.nav import PAGES, TOUR
from lab.usage import remaining_note, try_spend
from lab.rag_lab import gate_page, prompts_page, retrieval_page, tokens_page
from lab.site_pages import home_page, results_page
from rag import config
from rag.baseline import llm_only_stream
from rag.rag_graph import stream_answer

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

EXAMPLES = {
    "Popular": [
        "Who proposed the Turing test, and what did he originally call it?",
        "Which computer defeated Garry Kasparov at chess, and when?",
        "What training technique made models like ChatGPT follow instructions?",
    ],
    "Curious": [
        "What is the 'ELIZA effect', and who first showed it?",
        "Why was the MYCIN expert system never used in clinical practice?",
        "What were the main causes of the first AI winter?",
    ],
    "Expert": [
        "How much money did the 1955 Dartmouth proposal request?",
        "What were the three categories A, B and C of AI research in the Lighthill Report?",
        "How large was ImageNet when it was first published in 2009?",
    ],
    "Should refuse": [
        "Who invented generative adversarial networks (GANs)?",
        "What did Galileo discover when he pointed his telescope at Jupiter?",
    ],
}

SOURCE_ICON = {"primary": "📜", "secondary": "📖", "reference": "🗂️"}
STAGE_LABEL = {"retrieve_s": "retrieve", "grade_s": "evidence check", "generate_s": "generate",
               "first_token_s": "first token", "total_s": "total"}
STAGE_STATUS = {"retrieve": "🧪 Checking the evidence…", "grade_evidence": "✍️ Writing a cited answer…"}
CURSOR = " ▌"


# --------------------------------------------------------------------- styling

def inject_css() -> None:
    st.markdown("""
    <style>
      /* Answers read like prose: serif, slightly larger, as in Claude's own chat */
      [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        font-family: "Source Serif 4", Georgia, serif; font-size: 1.04rem; line-height: 1.62; }
      .app-title { font-family: "Source Serif 4", Georgia, serif; font-size: 2rem; font-weight: 600; margin-bottom: 0; }
      .app-sub { opacity: .72; margin-bottom: 0.6rem; }
      .side-head { font-weight: 700; font-size: 1.02rem; margin-bottom: 0.1rem; }
      .side-sub { opacity: .68; font-size: 0.8rem; margin-bottom: 0.5rem; }
      .rag-head { border-bottom: 3px solid #C2603E; padding-bottom: 0.2rem; }
      .llm-head { border-bottom: 3px solid #9c988c; padding-bottom: 0.2rem; }
      .status { color: #C2603E; font-size: 0.85rem; }
      .refusal { border-left: 4px solid #b7791f; padding: 0.4rem 0.8rem; background: rgba(183,121,31,0.10);
                 border-radius: 4px; }
      .fallback { border-left: 4px solid #9c988c; padding: 0.4rem 0.8rem; background: rgba(156,152,140,0.14);
                  border-radius: 4px; }
      .src-meta { opacity: .68; font-size: 0.85rem; }
      .timing { opacity: .6; font-size: 0.8rem; margin-top: 0.3rem; }
      sup.cite { font-size: 0.72em; font-weight: 600; color: #C2603E; }
    </style>""", unsafe_allow_html=True)


@st.cache_data
def load_manifest() -> dict:
    return yaml.safe_load(config.CORPUS_MANIFEST.read_text())


# ------------------------------------------------------------------ formatting

def safe_md(text: str) -> str:
    """Escape $ so Streamlit doesn't render "$13,500 ... $1,200" as a LaTeX formula."""
    return text.replace("$", "\\$")


def cite_markup(answer: str) -> str:
    """Escape $ and turn [1] / [2][3] into small superscript markers."""
    return re.sub(r"\[(\d+)\]", r"<sup class='cite'>[\1]</sup>", safe_md(answer))


def timing_line(timings: dict, extra: str = "") -> str:
    parts = " · ".join(f"{STAGE_LABEL.get(k, k)} {v:.2f}s" for k, v in timings.items())
    return f"<div class='timing'>⏱ {parts}{' · ' + extra if extra else ''}</div>"


def to_message(state: dict, strategy: str) -> dict:
    """Keep only what the UI needs from the final RAG state (it goes into session history)."""
    if state.get("refused"):
        reason = (f"Refused by the score gate: the best-matching chunk scored {state['top_score']:.2f}, "
                  f"below {config.MIN_SIMILARITY}. The question looks unrelated to the corpus."
                  if state["refusal_stage"] == "score_gate"
                  else f"Refused by the evidence check: {state.get('grade_reason')}")
    else:
        reason = ""
    return {
        "answer": state.get("answer", ""), "refused": state.get("refused", False), "refusal_reason": reason,
        "sources": state.get("sources", []), "timings": state.get("timings", {}), "strategy": strategy,
        "retrieved": [{"n": i, "score": s, "title": d.metadata["title"], "year": d.metadata["year"],
                       "section": d.metadata.get("section"),
                       "preview": d.page_content.split("\n", 1)[-1][:350].replace("\n", " ")}
                      for i, (d, s) in enumerate(state.get("retrieved", []), 1)],
    }


# ------------------------------------------------------------------- rendering

def side_header(kind: str) -> None:
    if kind == "rag":
        st.markdown("<div class='side-head rag-head'>🧠 RAG: grounded in the corpus</div>"
                    "<div class='side-sub'>Retrieves sources, checks the evidence, cites every claim</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='side-head llm-head'>🤖 LLM-only: model memory</div>"
                    "<div class='side-sub'>Same model, no retrieval, no citations, can't be verified</div>",
                    unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources cited ({len(sources)})", expanded=True):
        for s in sources:
            where = " · ".join(x for x in [f"p.{s['page']}" if s.get("page") else "", s.get("section") or ""] if x)
            st.markdown(
                f"**[{s['n']}]** {SOURCE_ICON.get(s['source_type'], '📄')} [{s['title']}]({s['url']}) ({s['year']})  \n"
                f"<span class='src-meta'>{s['authors']} · {s['source_type']}{' · ' + where if where else ''}</span>",
                unsafe_allow_html=True)


def render_retrieved(retrieved: list[dict]) -> None:
    with st.expander(f"🔎 Retrieved context ({len(retrieved)} chunks)"):
        st.caption(f"The top {len(retrieved)} chunks by cosine similarity. The answer may only use these. "
                   f"The score gate refuses if the best score is below {config.MIN_SIMILARITY}.")
        for r in retrieved:
            st.markdown(f"**[{r['n']}] {r['score']:.3f}** · {r['title']} ({r['year']})"
                        f"{' · ' + r['section'] if r['section'] else ''}")
            st.markdown(f"<div class='src-meta'>{safe_md(r['preview'])}…</div>", unsafe_allow_html=True)


def render_rag(msg: dict) -> None:
    if msg["refused"]:
        st.markdown(f"<div class='refusal'>🛑 {msg['answer']}</div>", unsafe_allow_html=True)
        st.caption(msg["refusal_reason"])
    else:
        st.markdown(cite_markup(msg["answer"]), unsafe_allow_html=True)
        if msg["sources"]:
            render_sources(msg["sources"])
    if msg.get("fallback"):
        st.markdown(
            f"<div class='fallback'><b>⚠️ Unverified answer from the model's general knowledge</b> "
            f"(not from the corpus, no citations):<br>{safe_md(msg['fallback'])}</div>", unsafe_allow_html=True)
    if msg["retrieved"]:
        render_retrieved(msg["retrieved"])
    st.markdown(timing_line(msg["timings"]), unsafe_allow_html=True)


def render_llm(llm: dict) -> None:
    st.markdown(safe_md(llm["answer"]))
    st.markdown(timing_line(llm["timings"]), unsafe_allow_html=True)


def render_turn(turn: dict) -> None:
    """One assistant turn: RAG alone, or RAG and LLM-only side by side."""
    if turn.get("llm"):
        left, right = st.columns(2, gap="large")
        with left:
            side_header("rag")
            render_rag(turn["rag"])
        with right:
            side_header("llm")
            render_llm(turn["llm"])
    else:
        render_rag(turn["rag"])


# ------------------------------------------------------------- live streaming

def _pump(gen, q: queue.Queue, tag: str) -> None:
    """Worker thread: forward every event from a generator into the shared queue."""
    try:
        for item in gen:
            q.put((tag, item))
    except Exception as e:  # surface errors in the UI instead of dying silently
        q.put((tag, {"type": "error", "error": str(e)}))
    q.put((tag, {"type": "done"}))


def stream_turn(question: str, strategy: str, compare: bool, fallback: bool) -> dict:
    """Stream RAG (and, if compare, LLM-only) into the page at the same time. Returns the finished turn."""
    q: queue.Queue = queue.Queue()
    t0 = time.perf_counter()
    rag_state, rag_text, llm_text, llm_first, llm_total = {}, "", "", None, None
    errors = {}

    if compare:
        left, right = st.columns(2, gap="large")
        with left:
            side_header("rag")
            rag_status, rag_slot = st.empty(), st.empty()
        with right:
            side_header("llm")
            llm_slot = st.empty()
    else:
        rag_status, rag_slot, llm_slot = st.empty(), st.empty(), None

    rag_status.markdown("<div class='status'>🔎 Retrieving sources…</div>", unsafe_allow_html=True)
    threading.Thread(target=_pump, args=(stream_answer(question, strategy), q, "rag"), daemon=True).start()
    running = {"rag"}
    if compare:
        threading.Thread(target=_pump, args=((({"type": "token", "text": t} for t in llm_only_stream(question))),
                                             q, "llm"), daemon=True).start()
        running.add("llm")

    while running:
        try:
            tag, ev = q.get(timeout=0.05)
        except queue.Empty:
            continue
        kind = ev["type"]
        if kind == "done":
            running.discard(tag)
            if tag == "llm":  # finalize the LLM column as soon as its own stream ends
                llm_total = round(time.perf_counter() - t0, 3)
                with llm_slot.container():
                    render_llm({"answer": llm_text or f"⚠️ {errors.get('llm', 'no answer')}",
                                "timings": {"first_token_s": llm_first or 0.0, "total_s": llm_total}})
        elif kind == "error":
            errors[tag] = ev["error"]
        elif tag == "llm":
            llm_first = llm_first or round(time.perf_counter() - t0, 3)
            llm_text += ev["text"]
            llm_slot.markdown(safe_md(llm_text) + CURSOR)
        elif kind == "stage":
            status = STAGE_STATUS.get(ev["node"])
            if status and not ev["state"].get("refused"):
                rag_status.markdown(f"<div class='status'>{status}</div>", unsafe_allow_html=True)
        elif kind == "token":
            rag_status.empty()
            rag_text += ev["text"]
            rag_slot.markdown(cite_markup(rag_text) + CURSOR, unsafe_allow_html=True)
        elif kind == "final":
            rag_state = ev["state"]

    # Replace the live text with the finished, fully formatted turn.
    rag_status.empty()
    if "rag" in errors:
        rag_state = {"answer": f"⚠️ Something went wrong: {errors['rag']}", "refused": True, "refusal_stage": "",
                     "timings": {}, "sources": [], "retrieved": []}
    rag_msg = to_message(rag_state, strategy)
    turn = {"role": "assistant", "rag": rag_msg}
    if compare:
        turn["llm"] = {"answer": llm_text or f"⚠️ {errors.get('llm', 'no answer')}",
                       "timings": {"first_token_s": llm_first or 0.0, "total_s": llm_total or 0.0}}
        if fallback and rag_msg["refused"] and llm_text:
            rag_msg["fallback"] = llm_text                # same answer the right column streamed; no extra call
        with rag_slot.container():
            render_rag(rag_msg)
        with llm_slot.container():
            render_llm(turn["llm"])
    else:
        if fallback and rag_msg["refused"]:
            rag_status.markdown("<div class='status'>🤖 Asking the model's general knowledge…</div>",
                                unsafe_allow_html=True)
            rag_msg["fallback"] = "".join(llm_only_stream(question))
            rag_status.empty()
        with rag_slot.container():
            render_rag(rag_msg)
    return turn


# ------------------------------------------------------------------ the tabs

def render_research_assistant() -> None:
    st.markdown("<div class='app-sub'>Ask about the history of AI. Answers come only from the curated "
                "corpus, with citations. If the corpus doesn't have the answer, the assistant says so.</div>",
                unsafe_allow_html=True)

    # Always side by side, with the configured retrieval strategy (config.DEFAULT_STRATEGY).
    strategy, compare = config.DEFAULT_STRATEGY, True
    left, right = st.columns(2, gap="large")              # aligned with the RAG / LLM-only columns below
    with left:
        fallback = st.toggle("🧠 RAG: fall back to the LLM's answer when refused", value=False,
                             help="When the RAG pipeline refuses (the corpus has no evidence), also show the "
                                  "model's general-knowledge answer in the RAG column, clearly labelled as "
                                  "unverified and uncited.")
    with right:
        if st.columns([2, 1])[1].button("🗑️ Clear chat", width="stretch"):
            st.session_state.chat_messages = []
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    chat_area = st.container()                            # messages render here, above the input box
    for msg in st.session_state.chat_messages:           # replay history
        with chat_area, st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(safe_md(msg["content"]))
            else:
                render_turn(msg)

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Ask a question about AI history…") or pending
    if prompt and (blocked := try_spend()):
        st.warning(blocked, icon="⏳")
    elif prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with chat_area, st.chat_message("user"):
            st.markdown(safe_md(prompt))
        with chat_area, st.chat_message("assistant"):
            turn = stream_turn(prompt, strategy, compare, fallback)
        st.session_state.chat_messages.append(turn)
    if note := remaining_note():
        st.caption(note)


def render_corpus_timeline(manifest: dict) -> None:
    """Primary sources placed on a 1943–2023 timeline, one lane per era. Secondary sources (mostly
    Wikipedia, dated by retrieval year) are left out, since their year isn't when the history happened."""
    import altair as alt
    eras = manifest["eras"]
    lanes = [e["label"] for k, e in eras.items() if k != "cross_era"]
    docs = [{"title": d["title"], "year": d["year"], "era": eras[d["era"]]["label"],
             "authors": ", ".join(d["authors"][:3]) + (" et al." if len(d["authors"]) > 3 else ""),
             "type": d["document_type"].replace("_", " ")}
            for d in manifest["documents"] if d["source_type"] == "primary" and d["era"] != "cross_era"]
    bands = [{"era": e["label"], "start": int(e["years"].split("-")[0]),
              "end": 2026 if e["years"].endswith("present") else int(e["years"].split("-")[1])}
             for k, e in eras.items() if k != "cross_era"]
    y = alt.Y("era:N", sort=lanes, title=None, axis=alt.Axis(labelLimit=260))
    x = alt.X("year:Q", title=None, scale=alt.Scale(domain=[1940, 2026]), axis=alt.Axis(format="d", tickCount=9))
    band = alt.Chart(alt.Data(values=bands)).mark_bar(opacity=.13, color="#C2603E", cornerRadius=4, height=22).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[1940, 2026])), x2="end:Q", y=y)
    dots = alt.Chart(alt.Data(values=docs)).mark_circle(size=120, color="#C2603E", opacity=.95,
                                                         stroke="white", strokeWidth=1.5).encode(
        x=x, y=y, tooltip=["title:N", "authors:N", "year:Q", "type:N"])
    st.markdown("#### The primary sources on a timeline")
    st.altair_chart(band + dots, height=320, width="stretch")
    st.caption(f"{len(docs)} primary sources (original papers, proposals and reports), each at its publication year; "
               "shaded bands are the era's span. Hover a dot for the document. Eras overlap in time on purpose: "
               "expert systems and the AI winters, for example, happened together.")


def render_corpus_browser() -> None:
    manifest = load_manifest()
    docs = manifest["documents"]
    counts = {t: sum(d["source_type"] == t for d in docs) for t in ("primary", "secondary", "reference")}
    st.markdown(f"<div class='app-sub'>{len(docs)} documents across {len(manifest['eras'])} eras · "
                f"{counts['primary']} primary · {counts['secondary']} secondary · {counts['reference']} reference. "
                f"Questions outside these documents are refused.</div>", unsafe_allow_html=True)
    render_corpus_timeline(manifest)
    st.markdown("#### All documents by era")
    query = st.text_input("Filter", placeholder="🔍 Filter by title or author…", label_visibility="collapsed")
    for era, info in manifest["eras"].items():
        era_docs = [d for d in docs if d["era"] == era and
                    (not query or query.lower() in (d["title"] + " ".join(d["authors"])).lower())]
        if not era_docs:
            continue
        st.markdown(f"#### {info['label']} · {info['years']}")
        for d in sorted(era_docs, key=lambda d: d["year"]):
            st.markdown(
                f"{SOURCE_ICON.get(d['source_type'], '📄')} **[{d['title']}]({d['url']})** ({d['year']})  \n"
                f"<span class='src-meta'>{', '.join(d['authors'])} · {d['source_type']} · "
                f"{d['document_type'].replace('_', ' ')} · {d['why']}</span>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Try a question")
        for group, questions in EXAMPLES.items():
            st.markdown(f"**{group}**")
            for q in questions:
                if st.button(q, key=f"ex_{q}", width="stretch"):
                    st.session_state.pending_question = q
                    st.rerun()
        st.divider()
        st.markdown("### How it works")
        st.caption(
            f"1. **Retrieve** the top {config.TOP_K} chunks from {len(load_manifest()['documents'])} AI-history "
            f"documents ({config.EMBEDDING_MODEL}, Chroma).\n"
            f"2. **Score gate:** refuse if the best match is below {config.MIN_SIMILARITY}.\n"
            f"3. **Evidence check:** an LLM confirms the chunks actually answer the question.\n"
            f"4. **Generate** an answer from those chunks only, citing [n] ({config.CHAT_MODEL}), streamed.\n\n"
            "The right-hand column runs the same model **without** retrieval in parallel, for comparison. "
            "Each question is answered independently (no chat memory).")


def chat_page() -> None:
    inject_css()
    st.markdown("<div class='app-title'>🧠 AI History Research Assistant</div>", unsafe_allow_html=True)
    render_sidebar()
    tab_chat, tab_corpus = st.tabs(["💬 Research Assistant", "📚 Corpus"])
    with tab_chat:
        render_research_assistant()
    with tab_corpus:
        render_corpus_browser()


@st.cache_resource(show_spinner=False)
def ensure_index() -> bool:
    """On a fresh machine (e.g. the first cloud start), build data/: download, embed, viewers."""
    from rag import bootstrap
    if bootstrap.index_ready():
        return True
    with st.status("First start: building the search index. This takes about 3 minutes, once.", expanded=True) as s:
        bootstrap.build(report=lambda step: s.write(f"• {step}…"))
        s.update(label="Index ready", state="complete", expanded=False)
    return bootstrap.index_ready()


def main() -> None:
    st.set_page_config(page_title="AI History Research Assistant", page_icon="🧠", layout="wide")
    if not ensure_index():
        st.error("The search index could not be built. Check the app logs (OPENAI_API_KEY set in secrets?).")
        st.stop()
    pages = {
        "home": st.Page(home_page, title="Start here", icon="✨", url_path="start", default=True),
        "chat": st.Page(chat_page, title="Research Assistant", icon="💬", url_path="assistant"),
        "results": st.Page(results_page, title="Results: RAG vs the model alone", icon="📊", url_path="results"),
        "diagram": st.Page(diagram_page, title="System diagram", icon="🗺️", url_path="diagram"),
        "tokens": st.Page(tokens_page, title="1 · Tokens", icon="🔡", url_path="tokens"),
        "chunks": st.Page(chunk_viewer_page, title="2 · Chunks", icon="🧩", url_path="chunks"),
        "embeddings": st.Page(embedding_viewer_page, title="3 · Embeddings", icon="🌌", url_path="embeddings"),
        "retrieval": st.Page(retrieval_page, title="4 · Retrieval", icon="🔀", url_path="retrieval"),
        "gate": st.Page(gate_page, title="5 · Score gate & top k", icon="🎚️", url_path="gate"),
        "prompts": st.Page(prompts_page, title="6 · Prompts & generation", icon="🧾", url_path="prompts"),
    }
    PAGES.update(pages)
    st.navigation({
        "App": [pages["home"], pages["chat"]],
        "Overview": [pages["results"], pages["diagram"]],
        "RAG Lab · the pipeline, step by step": [pages[k] for k in TOUR],
    }).run()


if __name__ == "__main__":
    main()
