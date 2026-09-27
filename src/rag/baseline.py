"""System A of the experiment: the same LLM answering from its own knowledge, with no retrieval."""

import time

from rag.models import get_llm
from rag.prompts import LLM_ONLY_PROMPT


def llm_only_answer(question: str) -> dict:
    t = time.perf_counter()
    response = get_llm().invoke(LLM_ONLY_PROMPT.invoke({"question": question}))
    return {"answer": response.content.strip(), "timings": {"total_s": round(time.perf_counter() - t, 3)}}


def llm_only_stream(question: str):
    """Stream the LLM-only answer token by token (for the side-by-side UI)."""
    yield from (chunk.content for chunk in get_llm().stream(LLM_ONLY_PROMPT.invoke({"question": question}))
                if chunk.content)
