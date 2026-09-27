"""Usage caps for the public deployment: every visitor question spends the owner's OpenAI credit.

A question is anything that calls the chat model: a chat turn (RAG + LLM-only side by side) or a
Prompt-inspector run. Caps apply only when config.PUBLIC_DEPLOYMENT is on. The daily count is kept
in memory for the running app, so it also resets if the app restarts.
"""

import datetime
import threading

import streamlit as st

from rag import config


@st.cache_resource
def _daily_counter() -> dict:
    return {"date": None, "count": 0, "lock": threading.Lock()}


def _today_count(counter: dict) -> int:
    today = datetime.date.today().isoformat()
    if counter["date"] != today:
        counter["date"], counter["count"] = today, 0
    return counter["count"]


def try_spend() -> str | None:
    """Reserve one question. Returns None if allowed, or a message explaining why not."""
    if not config.PUBLIC_DEPLOYMENT:
        return None
    used = st.session_state.get("questions_used", 0)
    if used >= config.MAX_QUESTIONS_PER_SESSION:
        return (f"You've reached this visit's limit of {config.MAX_QUESTIONS_PER_SESSION} questions. This demo runs on "
                "a small personal API budget; the RAG Lab pages that don't call the model still work.")
    counter = _daily_counter()
    with counter["lock"]:
        if _today_count(counter) >= config.MAX_QUESTIONS_PER_DAY:
            return ("The demo has reached today's question limit across all visitors. Please come back tomorrow; "
                    "the RAG Lab pages that don't call the model still work.")
        counter["count"] += 1
    st.session_state.questions_used = used + 1
    return None


def remaining_note() -> str:
    """A short 'N of M questions left' note for the public site (empty locally)."""
    if not config.PUBLIC_DEPLOYMENT:
        return ""
    left = config.MAX_QUESTIONS_PER_SESSION - st.session_state.get("questions_used", 0)
    return f"{max(left, 0)} of {config.MAX_QUESTIONS_PER_SESSION} questions left in this visit."
