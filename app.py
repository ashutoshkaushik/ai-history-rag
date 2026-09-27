"""AI History Research Assistant: Streamlit chat UI.

    uv run streamlit run app.py

Structure: main() builds the page and tabs,
render_research_assistant() is the chat, render_corpus_browser() lists the corpus.
All RAG logic lives in src/rag; this file only calls stream_answer() and llm_only_stream(),
so pipeline improvements show up here without UI changes.

Side-by-side mode runs RAG and LLM-only at the same time: each runs in a worker thread that
pushes streamed tokens into a queue, and the main script thread (the only one allowed to draw
in Streamlit) drains both queues and updates the two columns as text arrives.
"""

import html
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
from lab.code_page import code_page
from ui import chrome, theme
from ui.components import chunk_card
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

STAGE_LABEL = {"retrieve_s": "retrieve", "grade_s": "evidence check", "generate_s": "generate",
               "first_token_s": "first token", "total_s": "total"}
STAGE_STATUS = {"retrieve": "Checking the evidence…", "grade_evidence": "Writing a cited answer…"}
CURSOR = " ▌"


# --------------------------------------------------------------------- styling

def inject_css() -> None:
    """Chat-page layout. Colours and fonts come from ui/theme.py tokens (var(--…))."""
    st.markdown("""
    <style>
      .app-sub { color: var(--muted); margin-bottom: .6rem; }
      .question { font-family: var(--font-heading); font-size: 1.25rem; font-weight: 600; margin: 1.2rem 0 .6rem; }
      .side-head { font-weight: 700; font-size: 1rem; padding-bottom: .25rem; }
      .side-head.rag { border-bottom: 3px solid var(--rag); }
      .side-head.base { border-bottom: 3px solid var(--baseline); }
      .side-sub { color: var(--muted); font-size: .8rem; margin-top: .25rem; }
      .timing { color: var(--muted); font-size: .78rem; margin: .15rem 0 .6rem; min-height: 1.1rem; }
      .status { color: var(--rag); font-size: .82rem; margin: .15rem 0 .6rem; min-height: 1.1rem; }
      .refusal { border-left: 4px solid var(--caution); padding: .5rem .8rem; background: var(--caution-soft);
                 border-radius: 4px; }
      .fallback { border-left: 4px solid var(--baseline); padding: .5rem .8rem; background: var(--baseline-soft);
                  border-radius: 4px; margin-top: .6rem; }
      .src-meta { color: var(--muted); font-size: .85rem; }
      sup.cite { font-size: .72em; font-weight: 600; color: var(--rag); }
      /* answers: sans body at 16px / 1.6 for long-form reading */
      [class*="st-key-turn"] [data-testid="stMarkdownContainer"] p,
      [class*="st-key-turn"] [data-testid="stMarkdownContainer"] li { font-family: var(--font-body);
        font-size: 16px; line-height: 1.6; }
      /* under 900px, stack the two answers, RAG first */
      @media (max-width: 900px) {
        [class*="st-key-turn"] [data-testid="stHorizontalBlock"] { flex-direction: column; }
        [class*="st-key-turn"] [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important;
                                                          min-width: 100% !important; }
      }
      .empty-lede { font-size: 1.02rem; color: var(--muted); max-width: 70ch; margin: .4rem 0 .8rem; }
      /* question chips: full text, wrapped and left-aligned (Streamlit buttons are single-line by default) */
      .st-key-starters button { height: 100%; min-height: 3.4rem; justify-content: flex-start; text-align: left; }
      .st-key-starters button * { white-space: normal !important; overflow: visible !important;
                                  text-overflow: clip !important; text-align: left; }
      .chip-group { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
                    color: var(--muted); margin: .3rem 0 .15rem; }
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


def timing_line(timings: dict) -> str:
    parts = " · ".join(f"{STAGE_LABEL.get(k, k)} {v:.2f}s" for k, v in timings.items())
    return f"<div class='timing'>{parts}</div>"


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
                       "section": d.metadata.get("section"), "source_type": d.metadata["source_type"],
                       "text": d.page_content}
                      for i, (d, s) in enumerate(state.get("retrieved", []), 1)],
    }


# ------------------------------------------------------------------- rendering

def side_header(kind: str) -> None:
    if kind == "rag":
        st.markdown(f"<div class='side-head rag'>RAG · grounded in the corpus"
                    f"<div class='side-sub'>{config.CHAT_MODEL} + retrieval: checks the evidence, cites every claim"
                    f"</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='side-head base'>Model alone · from memory"
                    f"<div class='side-sub'>{config.CHAT_MODEL}, no retrieval: no citations, can't be verified"
                    f"</div></div>", unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources cited ({len(sources)})", expanded=True):
        for s in sources:
            where = " · ".join(x for x in [f"p.{s['page']}" if s.get("page") else "", s.get("section") or ""] if x)
            st.markdown(
                f"**[{s['n']}]** [{s['title']}]({s['url']}) ({s['year']}) "
                f"<span class='badge base'>{s['source_type']}</span>  \n"
                f"<span class='src-meta'>{s['authors']}{' · ' + where if where else ''}</span>",
                unsafe_allow_html=True)


def render_retrieved(retrieved: list[dict], key: str) -> None:
    with st.expander(f"Retrieved context ({len(retrieved)} chunks)"):
        st.caption(f"The top {len(retrieved)} chunks by cosine similarity, numbered as the answer cites them. The "
                   f"answer may only use these. The score gate refuses if the best score is below "
                   f"{config.MIN_SIMILARITY}.")
        cols = st.columns(2, gap="small")
        for i, r in enumerate(retrieved):
            with cols[i % 2]:
                chunk_card(rank=r["n"], title=r["title"], year=r["year"], text=r.get("text") or r.get("preview", ""),
                           score=r["score"], score_kind="cosine", section=r.get("section"),
                           source_type=r.get("source_type"), key=f"{key}_ctx_{r['n']}")


def render_rag_body(msg: dict) -> None:
    if msg["refused"]:
        st.markdown(f"<div class='refusal'><span class='badge refuse'>Refused</span> {msg['answer']}</div>",
                    unsafe_allow_html=True)
        st.caption(msg["refusal_reason"])
    else:
        st.markdown(cite_markup(msg["answer"]), unsafe_allow_html=True)
    if msg.get("fallback"):
        st.markdown(
            f"<div class='fallback'><span class='badge base'>Unverified · model memory</span> "
            f"Not from the corpus, no citations:<br>{safe_md(msg['fallback'])}</div>", unsafe_allow_html=True)


def render_details(msg: dict, key: str) -> None:
    """Full-width row under both answers: what RAG cited and everything it retrieved."""
    if msg["sources"]:
        render_sources(msg["sources"])
    if msg["retrieved"]:
        render_retrieved(msg["retrieved"], key)


def render_question(text: str) -> None:
    st.markdown(f"<div class='question'>{html.escape(text).replace('$', '&#36;')}</div>", unsafe_allow_html=True)


def render_turn(turn: dict, key: str) -> None:
    """One finished turn: question, both answers side by side, then sources across the full width."""
    with st.container(key=key):
        render_question(turn["question"])
        left, right = st.columns(2, gap="medium")
        with left:
            side_header("rag")
            st.markdown(timing_line(turn["rag"]["timings"]), unsafe_allow_html=True)
            render_rag_body(turn["rag"])
        with right:
            side_header("llm")
            st.markdown(timing_line(turn["llm"]["timings"]), unsafe_allow_html=True)
            st.markdown(safe_md(turn["llm"]["answer"]))
        render_details(turn["rag"], key)


# ------------------------------------------------------------- live streaming

def _pump(gen, q: queue.Queue, tag: str) -> None:
    """Worker thread: forward every event from a generator into the shared queue."""
    try:
        for item in gen:
            q.put((tag, item))
    except Exception as e:  # surface errors in the UI instead of dying silently
        q.put((tag, {"type": "error", "error": str(e)}))
    q.put((tag, {"type": "done"}))


def stream_turn(question: str, strategy: str, fallback: bool, key: str) -> dict:
    """Stream RAG and LLM-only into the page at the same time. Returns the finished turn."""
    q: queue.Queue = queue.Queue()
    t0 = time.perf_counter()
    rag_state, rag_text, llm_text, llm_first, llm_total = {}, "", "", None, None
    errors = {}

    with st.container(key=key):
        render_question(question)
        left, right = st.columns(2, gap="medium")
        with left:
            side_header("rag")
            rag_status, rag_slot = st.empty(), st.empty()
        with right:
            side_header("llm")
            llm_status, llm_slot = st.empty(), st.empty()
        details_slot = st.empty()

    rag_status.markdown("<div class='status'>Retrieving sources…</div>", unsafe_allow_html=True)
    llm_status.markdown("<div class='status'>&nbsp;</div>", unsafe_allow_html=True)
    threading.Thread(target=_pump, args=(stream_answer(question, strategy), q, "rag"), daemon=True).start()
    threading.Thread(target=_pump, args=((({"type": "token", "text": t} for t in llm_only_stream(question))),
                                         q, "llm"), daemon=True).start()
    running = {"rag", "llm"}

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
                llm_status.markdown(timing_line({"first_token_s": llm_first or 0.0, "total_s": llm_total}),
                                    unsafe_allow_html=True)
                llm_slot.markdown(safe_md(llm_text or f"Something went wrong: {errors.get('llm', 'no answer')}"))
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
            rag_text += ev["text"]
            rag_slot.markdown(cite_markup(rag_text) + CURSOR, unsafe_allow_html=True)
        elif kind == "final":
            rag_state = ev["state"]

    # Replace the live text with the finished, fully formatted turn.
    if "rag" in errors:
        rag_state = {"answer": f"Something went wrong: {errors['rag']}", "refused": True, "refusal_stage": "",
                     "timings": {}, "sources": [], "retrieved": []}
    rag_msg = to_message(rag_state, strategy)
    llm = {"answer": llm_text or f"Something went wrong: {errors.get('llm', 'no answer')}",
           "timings": {"first_token_s": llm_first or 0.0, "total_s": llm_total or 0.0}}
    if fallback and rag_msg["refused"] and llm_text:
        rag_msg["fallback"] = llm_text                    # same answer the right column streamed; no extra call
    rag_status.markdown(timing_line(rag_msg["timings"]), unsafe_allow_html=True)
    with rag_slot.container():
        render_rag_body(rag_msg)
    with details_slot.container():
        render_details(rag_msg, key)
    return {"role": "assistant", "question": question, "rag": rag_msg, "llm": llm}


# ------------------------------------------------------------------ the tabs

def render_research_assistant() -> None:
    st.markdown("<div class='app-sub'>Ask about the history of AI. Answers come only from the curated "
                "corpus, with citations. If the corpus doesn't have the answer, the assistant says so.</div>",
                unsafe_allow_html=True)

    strategy = config.DEFAULT_STRATEGY                   # always side by side, with the configured strategy
    left, right = st.columns(2, gap="large")              # aligned with the RAG / model-alone columns below
    with left:
        fallback = st.toggle("Allow fallback to model memory", value=False,
                             help="Applies to the RAG column. When RAG refuses because the corpus has no evidence, "
                                  "also show the model's own general-knowledge answer there, clearly labelled as "
                                  "unverified and uncited.")
    with right, st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Clear chat", icon=":material/delete:", width="content", help="Remove all messages"):
            st.session_state.chat_messages = []
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    turns = [m for m in st.session_state.chat_messages if m.get("role") == "assistant" and "question" in m]
    st.session_state.chat_messages = turns                # drop entries from the older message format

    chat_area = st.container()                            # turns render here, above the input box
    for i, turn in enumerate(turns):
        with chat_area:
            render_turn(turn, key=f"turn_{i}")
    empty_slot = chat_area.empty()                        # the empty state, until a conversation exists

    pending = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Ask about the history of AI, e.g. who coined 'artificial intelligence'?") or pending
    if not turns and not prompt:
        with empty_slot.container():
            render_empty_state()
    if prompt and (blocked := try_spend()):
        st.warning(blocked, icon=":material/hourglass_top:")
    elif prompt:
        with chat_area:
            turn = stream_turn(prompt, strategy, fallback, key=f"turn_{len(turns)}")
        st.session_state.chat_messages.append(turn)
    if note := remaining_note():
        st.caption(note)


STARTERS = [  # (group, question): shown as clickable chips when the chat is empty
    ("Popular", "Who proposed the Turing test, and what did he originally call it?"),
    ("Popular", "Which computer defeated Garry Kasparov at chess, and when?"),
    ("Curious", "What is the 'ELIZA effect', and who first showed it?"),
    ("Curious", "Why was the MYCIN expert system never used in clinical practice?"),
    ("Expert", "How much money did the 1955 Dartmouth proposal request?"),
    ("Should refuse", "What did Alan Turing think about ChatGPT?"),
]


def render_empty_state() -> None:
    """What a new visitor sees: what the assistant does, and questions to click."""
    st.markdown("<div class='empty-lede'>Pick a question to start, or type your own below. The right-hand column "
                "shows the same model answering from memory, so you can compare.</div>", unsafe_allow_html=True)
    with st.container(key="starters"):                    # keyed so the CSS can let these buttons wrap
        for row in (STARTERS[:3], STARTERS[3:]):
            for col, (group, question) in zip(st.columns(3), row):
                with col:
                    st.markdown(f"<div class='chip-group'>{group}</div>", unsafe_allow_html=True)
                    if st.button(question, key=f"starter_{question}", width="stretch"):
                        st.session_state.pending_question = question
                        st.rerun()


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
    data_color = theme.tokens()["data-1"]                 # corpus documents: a neutral data colour, not RAG
    band = alt.Chart(alt.Data(values=bands)).mark_bar(opacity=.13, color=data_color, cornerRadius=4, height=22).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[1940, 2026])), x2="end:Q", y=y)
    dots = alt.Chart(alt.Data(values=docs)).mark_circle(size=120, color=data_color, opacity=.95,
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
    query = st.text_input("Filter", placeholder="Filter by title or author…", label_visibility="collapsed")
    for era, info in manifest["eras"].items():
        era_docs = [d for d in docs if d["era"] == era and
                    (not query or query.lower() in (d["title"] + " ".join(d["authors"])).lower())]
        if not era_docs:
            continue
        st.markdown(f"#### {info['label']} · {info['years']}")
        for d in sorted(era_docs, key=lambda d: d["year"]):
            st.markdown(
                f"**[{d['title']}]({d['url']})** ({d['year']}) <span class='badge base'>{d['source_type']}</span>  \n"
                f"<span class='src-meta'>{', '.join(d['authors'])} · {d['source_type']} · "
                f"{d['document_type'].replace('_', ' ')} · {d['why']}</span>", unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        with st.expander("More sample questions", expanded=False):
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
    st.title("AI History Research Assistant")
    render_sidebar()
    render_research_assistant()


def corpus_page() -> None:
    inject_css()
    st.title("Corpus")
    render_corpus_browser()


@st.cache_resource
def _build_lock() -> threading.Lock:
    return threading.Lock()                                # one shared lock for all visitor sessions


def ensure_index() -> bool:
    """On a fresh machine (e.g. the first cloud start), build data/: download, embed, viewers.

    The check runs on every page load but outside any cache: Streamlit replays elements drawn inside
    a cached function on every later run, which previously left the "building…" box on screen forever.
    """
    from rag import bootstrap
    if bootstrap.index_ready():
        return True
    with _build_lock():                                    # a second visitor waits instead of building twice
        if bootstrap.index_ready():
            return True
        with st.status("First start: building the search index. This takes about 3 minutes, once.",
                       expanded=True) as s:
            bootstrap.build(report=lambda step: s.write(f"• {step}…"))
            s.update(label="Index ready", state="complete", expanded=False)
    return bootstrap.index_ready()


def main() -> None:
    st.set_page_config(page_title="AI History Research Assistant", page_icon=":material/history_edu:",
                       layout="wide")
    theme.apply_theme()                                   # colour/type tokens for every page (ui/theme.py)
    if not ensure_index():
        st.error("The search index could not be built. Check the app logs (OPENAI_API_KEY set in secrets?).")
        st.stop()
    pages = {
        "home": st.Page(home_page, title="Start here", icon=":material/home:", default=True),
        "chat": st.Page(chat_page, title="Research Assistant", icon=":material/forum:", url_path="assistant"),
        "results": st.Page(results_page, title="Results: RAG vs the model alone", icon=":material/leaderboard:", url_path="results"),
        "corpus": st.Page(corpus_page, title=f"Corpus: the {len(load_manifest()['documents'])} sources", icon=":material/library_books:", url_path="corpus"),
        "code": st.Page(code_page, title="How it's built: the code", icon=":material/code:", url_path="code"),
        "diagram": st.Page(diagram_page, title="System diagram", icon=":material/account_tree:", url_path="diagram"),
        "tokens": st.Page(tokens_page, title="1 · Tokens", icon=":material/text_fields:", url_path="tokens"),
        "chunks": st.Page(chunk_viewer_page, title="2 · Chunks", icon=":material/view_agenda:", url_path="chunks"),
        "embeddings": st.Page(embedding_viewer_page, title="3 · Embeddings", icon=":material/scatter_plot:", url_path="embeddings"),
        "retrieval": st.Page(retrieval_page, title="4 · Retrieval", icon=":material/manage_search:", url_path="retrieval"),
        "gate": st.Page(gate_page, title="5 · Score gate & top k", icon=":material/tune:", url_path="gate"),
        "prompts": st.Page(prompts_page, title="6 · Prompts & generation", icon=":material/receipt_long:", url_path="prompts"),
    }
    PAGES.update(pages)
    page = st.navigation({
        "App": [pages["home"], pages["chat"]],
        "Overview": [pages["results"], pages["corpus"], pages["diagram"], pages["code"]],
        "RAG Lab · the pipeline, step by step": [pages[k] for k in TOUR],
    })
    page.run()
    chrome.author_card()                                   # bottom of the sidebar, on every page
    chrome.footer()                                        # bottom of every page


if __name__ == "__main__":
    main()
