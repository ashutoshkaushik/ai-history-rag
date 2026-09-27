"""Site chrome shown on every page: the author line in the sidebar and the page footer.

app.main() calls both after the selected page has run, so they land at the very bottom of the
sidebar and of the page. Styles come from ui/theme.py tokens.
"""

import streamlit as st

AUTHOR = "Ashutosh Kaushik"
LINKEDIN_URL = "https://www.linkedin.com/in/ashutosh-kaushik/"


def author_card() -> None:
    with st.sidebar:
        st.divider()
        st.markdown(f"<div class='author-name'>{AUTHOR}</div>", unsafe_allow_html=True)
        st.link_button("Connect on LinkedIn", LINKEDIN_URL, icon=":material/person_add:", width="stretch")


def footer() -> None:
    st.markdown(
        "<div class='site-footer'>Built with Streamlit, LangChain and LangGraph, using Claude Code and other AI "
        "tools, with a human in the loop.</div>", unsafe_allow_html=True)
