"""Ask a question from the command line.

    uv run python -m rag.ask "What was the Lighthill Report?"
    uv run python -m rag.ask "..." --llm-only      # baseline: LLM with no retrieval
    uv run python -m rag.ask "..." --compare       # RAG and LLM-only side by side
    uv run python -m rag.ask "..." --context       # also print the retrieved chunks
"""

import argparse
import textwrap

from rag import config
from rag.baseline import llm_only_answer
from rag.rag_graph import answer_question


def wrap(text: str) -> str:
    return "\n".join(textwrap.fill(p, 100) for p in text.split("\n"))


def print_rag(state: dict, show_context: bool) -> None:
    print(wrap(state["answer"]))
    if state["refused"]:
        why = (f"best similarity {state['top_score']:.2f} < {config.MIN_SIMILARITY}"
               if state["refusal_stage"] == "score_gate" else f"evidence check: {state.get('grade_reason')}")
        print(f"\n(refused at {state['refusal_stage']}: {why})")
    for s in state["sources"]:
        where = ", ".join(x for x in [f"p.{s['page']}" if s["page"] else "", s["section"] or ""] if x)
        print(f"  [{s['n']}] {s['title']} ({s['year']}){' - ' + where if where else ''}  [{s['source_type']}]")
    if show_context:
        print("\n--- retrieved context ---")
        for i, (doc, score) in enumerate(state.get("retrieved", []), 1):
            print(f"[{i}] {score:.3f} {doc.metadata['doc_id']}\n{textwrap.shorten(doc.page_content, 400)}\n")
    t = state["timings"]
    print("\n" + "  ".join(f"{k}={v:.2f}s" for k, v in t.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()

    if not args.llm_only:
        print("=== RAG " + "=" * 92)
        print_rag(answer_question(args.question), args.context)
    if args.llm_only or args.compare:
        base = llm_only_answer(args.question)
        print("\n=== LLM only " + "=" * 87)
        print(wrap(base["answer"]))
        print(f"\ntotal_s={base['timings']['total_s']:.2f}s")


if __name__ == "__main__":
    main()
