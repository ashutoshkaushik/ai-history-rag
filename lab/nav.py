"""Page registry, so any page can link to any other (landing-page buttons, Lab Previous / Next).

app.py builds the st.Page objects and registers them here before running the navigation.
"""

import streamlit as st

PAGES: dict = {}

TOUR = ["tokens", "chunks", "embeddings", "retrieval", "gate", "prompts"]   # the pipeline, in order


def link(key: str, label: str, icon: str | None = None, **kw) -> None:
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon, **kw)


def tour_footer(key: str) -> None:
    """Previous / Next buttons at the bottom of a pipeline page."""
    i = TOUR.index(key)
    prev_key = TOUR[i - 1] if i > 0 else "diagram"
    next_key = TOUR[i + 1] if i < len(TOUR) - 1 else "results"
    st.divider()
    left, _, right = st.columns([2, 1, 2])
    with left:
        prev_page = PAGES.get(prev_key)
        if prev_page:
            st.page_link(prev_page, label=f"Previous: {prev_page.title}", icon="⬅️", width="stretch")
    with right:
        next_page = PAGES.get(next_key)
        if next_page:
            st.page_link(next_page, label=f"Next: {next_page.title}", icon="➡️", width="stretch")
