"""Shared UI components. Styles live in ui/theme.py (apply_theme), so these only emit markup.

chunk_card() is the one way a retrieved chunk is shown anywhere in the app: the Retrieval lab,
the Prompts lab, and the Research Assistant's "Retrieved context".
"""

import html
import re

import streamlit as st

SCORE_FORMAT = {"cosine": "{:.3f}", "BM25": "{:.1f}", "RRF": "{:.4f}"}


def _strip_braced(text: str, opener: str) -> str:
    """Remove every `opener{…}` block, matching nested braces (e.g. {\\displaystyle D_{\\text{x}}})."""
    out, i = [], 0
    while (j := text.find(opener, i)) != -1:
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1
    out.append(text[i:])
    return "".join(out)


def clean_display_text(text: str) -> str:
    """Text for display only (the index is unchanged): drop the [Title | Section] header, LaTeX
    leftovers, runs of bare numbers (tables) and one-symbol-per-line math, then collapse whitespace."""
    if text.startswith("[") and "]\n" in text[:400]:
        text = text.split("]\n", 1)[1]
    text = _strip_braced(text, "{\\displaystyle")
    text = re.sub(r"(?:(?<![\w.])[-+]?\d[\d.,]*%?\s+){5,}[-+]?\d[\d.,]*%?", " … ", text)   # 6+ numbers in a row
    text = re.sub(r"(?:(?<!\S)\S{1,2}\s+){6,}", " … ", text)                             # math shreds: "x ∼ D ( ) ]"
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:\s*…\s*){2,}", " … ", text)
    return text.strip()


def _highlight(escaped: str, terms: set[str]) -> str:
    for t in sorted(terms, key=len, reverse=True):
        escaped = re.sub(rf"(?i)\b({re.escape(html.escape(t))})\b", r"<mark>\1</mark>", escaped)
    return escaped


def chunk_card(*, rank: int, title: str, year, text: str, score: float | None = None, score_kind: str = "cosine",
               section: str | None = None, source_type: str | None = None, gold: bool = False,
               notes: list[str] | None = None, terms: set[str] | None = None, key: str) -> None:
    """One retrieved chunk: prominent rank, typed score, title, 3-line preview and a 'Show more' expander.
    gold=True adds a green border and a '✓ Gold' badge (a document that contains the answer)."""
    terms = {w for w in (terms or set()) if len(w) > 1}      # single letters (e.g. "t") only add noise
    body = clean_display_text(text)
    # Start the preview near the first highlighted term so the reason for the match is visible.
    low = body.lower()
    hits = [low.find(t) for t in terms if low.find(t) >= 0]
    start = max(0, min(hits) - 60) if hits else 0
    preview = ("…" if start else "") + body[start:]
    score_html = (f"<span class='cc-score'>{score_kind} {SCORE_FORMAT.get(score_kind, '{:.3f}').format(score)}</span>"
                  if score is not None else "")
    badges = ("<span class='badge ok'>✓ Gold</span>" if gold else "") + \
             "".join(f"<span class='badge base'>{html.escape(n)}</span>" for n in (notes or []))
    meta = " · ".join(x for x in [source_type, section[:48] if section else None] if x)
    with st.container(border=True, key=f"cc{'gold' if gold else ''}_{key}"):
        st.markdown(
            f"<div class='cc-top'><span class='cc-rank'>#{rank}</span>{score_html}{badges}</div>"
            f"<div class='cc-title'>{html.escape(title)} <span class='cc-year'>({year})</span></div>"
            + (f"<div class='cc-meta'>{html.escape(meta)}</div>" if meta else "")
            + f"<div class='cc-text'>{_highlight(html.escape(preview), terms)}</div>", unsafe_allow_html=True)
        if len(body) > 260:
            with st.expander("Show more"):
                st.markdown(f"<div class='cc-full'>{_highlight(html.escape(body), terms)}</div>",
                            unsafe_allow_html=True)


def compare_table(columns: dict[str, list[tuple[str, bool]]], k: int) -> None:
    """Compact comparison: rows = rank, one column per method, cells = title (✓ marks gold)."""
    head = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    rows = []
    for r in range(k):
        cells = []
        for items in columns.values():
            if r < len(items):
                title, gold = items[r]
                cells.append(f"<td class='{'gold' if gold else ''}'>{'✓ ' if gold else ''}{html.escape(title)}</td>")
            else:
                cells.append("<td></td>")
        rows.append(f"<tr><th class='rk'>#{r + 1}</th>{''.join(cells)}</tr>")
    st.markdown(f"<div class='tablewrap'><table class='ctable'><thead><tr><th class='rk'>Rank</th>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)
