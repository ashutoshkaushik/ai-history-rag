"""The RAG application as a LangGraph flow, with "I don't know" designed in from the start.

    START -> retrieve -> (score gate) -> grade_evidence -> generate -> END
                             |                 |
                             +------> refuse <-+

1. retrieve        top-k chunks for the question
2. score gate      if even the best chunk is below MIN_SIMILARITY, refuse without an LLM call
                   (catches off-topic questions for free)
3. grade_evidence  one small LLM call: do these chunks actually answer the question?
                   (catches "related but not in corpus" questions that score deceptively high)
4. generate        answer only from numbered sources, citing [n]
5. refuse          a fixed "I couldn't find sufficient evidence" message

Public entry point: answer_question(question, strategy).
"""

import re
import time
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag import config
from rag.models import get_llm
from rag.prompts import GRADE_PROMPT, RAG_PROMPT
from rag.retrieve import retrieve


class RAGState(TypedDict, total=False):
    question: str
    strategy: str
    retrieved: list[tuple[Document, float]]
    top_score: float
    sufficient: bool
    grade_reason: str
    answer: str
    refused: bool
    refusal_stage: str          # "score_gate" | "evidence_check" | ""
    sources: list[dict]         # only the sources the answer actually cites
    timings: dict[str, float]


class EvidenceGrade(BaseModel):
    sufficient: bool = Field(description="True if the sources directly answer the question, fully or partially")
    reason: str = Field(description="One short sentence explaining the decision")


# ------------------------------------------------------------------ helpers

def format_context(retrieved: list[tuple[Document, float]]) -> str:
    """Numbered source blocks: [n] Title (Year), p.X, section, then the chunk text."""
    blocks = []
    for i, (doc, _) in enumerate(retrieved, 1):
        m = doc.metadata
        where = ", ".join(x for x in [f"p.{m['page']}" if m.get("page") else "", m.get("section", "")] if x)
        blocks.append(f"[{i}] {m['title']} ({m['year']}){', ' + where if where else ''}\n{doc.page_content}")
    return "\n\n".join(blocks)


def source_info(n: int, doc: Document, score: float) -> dict:
    m = doc.metadata
    return {
        "n": n, "doc_id": m["doc_id"], "title": m["title"], "authors": m["authors"], "year": m["year"],
        "era": m["era"], "source_type": m["source_type"], "page": m.get("page") or None,
        "section": m.get("section") or None, "url": m["url"], "score": round(score, 3),
    }


def _timed(state: RAGState, key: str, start: float) -> dict[str, float]:
    return {**state.get("timings", {}), key: round(time.perf_counter() - start, 3)}


# -------------------------------------------------------------------- nodes

def retrieve_node(state: RAGState) -> RAGState:
    t = time.perf_counter()
    retrieved = retrieve(state["question"], state.get("strategy", config.DEFAULT_STRATEGY))
    top = max((s for _, s in retrieved), default=0.0)
    return {"retrieved": retrieved, "top_score": top, "timings": _timed(state, "retrieve_s", t)}


def grade_node(state: RAGState) -> RAGState:
    t = time.perf_counter()
    grader = get_llm().with_structured_output(EvidenceGrade)
    grade = grader.invoke(GRADE_PROMPT.invoke({
        "question": state["question"], "context": format_context(state["retrieved"]),
    }))
    return {"sufficient": grade.sufficient, "grade_reason": grade.reason,
            "timings": _timed(state, "grade_s", t)}


def generate_node(state: RAGState) -> RAGState:
    t = time.perf_counter()
    retrieved = state["retrieved"]
    response = get_llm().invoke(RAG_PROMPT.invoke({
        "question": state["question"], "context": format_context(retrieved),
    }))
    answer = response.content.strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(retrieved)})
    sources = [source_info(n, *retrieved[n - 1]) for n in cited]
    return {"answer": answer, "sources": sources, "refused": False, "refusal_stage": "",
            "timings": _timed(state, "generate_s", t)}


def refuse_node(state: RAGState) -> RAGState:
    stage = "score_gate" if state.get("top_score", 0) < config.MIN_SIMILARITY else "evidence_check"
    return {"answer": config.REFUSAL_MESSAGE, "sources": [], "refused": True, "refusal_stage": stage}


# ------------------------------------------------------------------- routing

def after_retrieve(state: RAGState) -> str:
    return "grade_evidence" if state["top_score"] >= config.MIN_SIMILARITY else "refuse"


def after_grade(state: RAGState) -> str:
    return "generate" if state["sufficient"] else "refuse"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_evidence", grade_node)
    g.add_node("generate", generate_node)
    g.add_node("refuse", refuse_node)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["grade_evidence", "refuse"])
    g.add_conditional_edges("grade_evidence", after_grade, ["generate", "refuse"])
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


GRAPH = build_graph()


def answer_question(question: str, strategy: str = config.DEFAULT_STRATEGY) -> RAGState:
    """Run the full RAG flow. Returns the final state: answer, sources, refusal info,
    retrieved chunks and per-stage timings."""
    t = time.perf_counter()
    state = GRAPH.invoke({"question": question, "strategy": strategy, "timings": {}})
    state["timings"]["total_s"] = round(time.perf_counter() - t, 3)
    return state


def stream_answer(question: str, strategy: str = config.DEFAULT_STRATEGY):
    """Run the same graph, but yield events as they happen (for a streaming UI):

        {"type": "stage", "node": "retrieve" | "grade_evidence" | "generate" | "refuse", "state": {...}}
        {"type": "token", "text": "..."}          # only tokens from the generate node
        {"type": "final", "state": {...}}         # same shape as answer_question()
    """
    t = time.perf_counter()
    state: dict = {"question": question, "strategy": strategy, "timings": {}}
    first_token = None
    for mode, chunk in GRAPH.stream(state, stream_mode=["updates", "messages"]):
        if mode == "messages":
            message, meta = chunk
            if meta.get("langgraph_node") == "generate" and message.content:
                if first_token is None:
                    first_token = round(time.perf_counter() - t, 3)
                yield {"type": "token", "text": message.content}
        else:  # "updates": {node_name: partial state}
            for node, update in chunk.items():
                state.update(update or {})
                yield {"type": "stage", "node": node, "state": state}
    if first_token is not None:
        state["timings"]["first_token_s"] = first_token
    state["timings"]["total_s"] = round(time.perf_counter() - t, 3)
    yield {"type": "final", "state": state}
