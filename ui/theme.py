"""The site's single source of truth for colour and type.

Every page gets these tokens through apply_theme(), which app.main() calls once per run before the
selected page renders. Page CSS refers to them as var(--token); Altair charts read the same values
from tokens() / categorical(). No other file should hard-code a colour or font.

Colour tokens have fixed meanings, so a colour always says the same thing:
  --accent    terracotta: buttons, links and the active nav item ONLY (also Streamlit's primaryColor)
  --rag       RAG everywhere: answer-column header, cards, chart bars, citation markers, pipeline steps
  --baseline  the model alone (no retrieval) everywhere
  --success   correct / gold / "✓" markers
  --error     wrong / "✗" markers
  --caution   refusals ("I don't know")
  --data-1..3 categories in Lab charts that are neither RAG nor right/wrong (methods, question groups,
              pipeline steps). Validated as a CVD-safe set in both modes.
"""

import streamlit as st

FONT_HEADING = '"Source Serif 4", Georgia, "Times New Roman", serif'
FONT_BODY = '"Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

LIGHT = {
    "accent": "#C2603E", "on-accent": "#ffffff",
    "rag": "#2a78d6", "rag-soft": "rgba(42,120,214,.10)", "on-rag": "#ffffff",
    "baseline": "#8a867c", "baseline-soft": "rgba(138,134,124,.14)",
    "success": "#1f8a4c", "success-soft": "rgba(31,138,76,.13)",
    "error": "#c4521f", "error-soft": "rgba(196,82,31,.12)",
    "caution": "#a86a12", "caution-soft": "rgba(183,121,31,.11)",
    "muted": "#6f6b62",
    "line": "rgba(31,30,29,.14)",
    "surface": "#FAF9F5", "sidebar": "#F3F1EA",
    "highlight": "rgba(237,161,0,.30)",
    "data-1": "#4a3aa7", "data-2": "#eda100", "data-3": "#e87ba4",
}
DARK = {
    "accent": "#D97757", "on-accent": "#1F1E1D",
    "rag": "#3987e5", "rag-soft": "rgba(57,135,229,.16)", "on-rag": "#ffffff",
    "baseline": "#9c988c", "baseline-soft": "rgba(156,152,140,.18)",
    "success": "#3fb67a", "success-soft": "rgba(63,182,122,.18)",
    "error": "#e66a3d", "error-soft": "rgba(230,106,61,.18)",
    "caution": "#d9a441", "caution-soft": "rgba(217,164,65,.16)",
    "muted": "#a8a397",
    "line": "rgba(242,240,232,.16)",
    "surface": "#262624", "sidebar": "#1F1E1D",
    "highlight": "rgba(201,133,0,.40)",
    "data-1": "#9085e9", "data-2": "#c98500", "data-3": "#d55181",
}


def mode() -> str:
    """'light' or 'dark', following the visitor's current Streamlit theme."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit or no browser context
        return "light"


def tokens() -> dict[str, str]:
    return DARK if mode() == "dark" else LIGHT


def categorical() -> list[str]:
    """The three non-RAG category colours, in fixed order (for Altair chart ranges)."""
    t = tokens()
    return [t["data-1"], t["data-2"], t["data-3"]]


def apply_theme() -> None:
    """Inject the tokens and the shared typography / component styles. Call once per run."""
    t = tokens()
    variables = "\n".join(f"    --{k}: {v};" for k, v in t.items())
    st.markdown(f"""<style>
  :root {{
{variables}
    --font-heading: {FONT_HEADING};
    --font-body: {FONT_BODY};
    --font-mono: {FONT_MONO};
  }}

  /* ---- Typography: one serif for headings, one sans for everything else, same scale on every page */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ font-family: var(--font-body); }}
  h1, h2, h3, h4, [data-testid="stHeading"] * {{ font-family: var(--font-heading) !important; }}
  h1 {{ font-size: 2.1rem !important; font-weight: 600 !important; line-height: 1.2 !important; }}
  h2 {{ font-size: 1.55rem !important; font-weight: 600 !important; }}
  h3 {{ font-size: 1.22rem !important; font-weight: 600 !important; }}
  h4 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
  code, pre {{ font-family: var(--font-mono); }}

  /* ---- Layout: a comfortable max width, so side-by-side answers get room */
  [data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-left: 1.5rem; padding-right: 1.5rem; }}

  /* ---- Labels wrap instead of being cut off with an ellipsis */
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"],
  [data-testid="stMetricDelta"] *, [data-testid^="stBaseButton"] *, [data-testid="stWidgetLabel"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
  [data-testid^="stBaseButton"] {{ height: auto; min-height: 2.5rem; }}

  /* ---- Shared components */
  .muted {{ color: var(--muted); }}
  .lede {{ font-size: 1.1rem; line-height: 1.6; color: var(--muted); max-width: 64ch; }}
  .badge {{ display: inline-block; font-size: .78rem; font-weight: 600; padding: 1px 9px; border-radius: 99px;
            white-space: nowrap; }}
  .badge.ok {{ background: var(--success-soft); color: var(--success); }}
  .badge.no {{ background: var(--error-soft); color: var(--error); }}
  .badge.refuse {{ background: var(--caution-soft); color: var(--caution); }}
  .badge.rag {{ background: var(--rag-soft); color: var(--rag); }}
  .badge.base {{ background: var(--baseline-soft); color: var(--muted); }}
  .badge + .badge, .cc-score + .badge {{ margin-left: .3rem; }}

  /* ---- Site chrome (ui/chrome.py) */
  .author-name {{ font-family: var(--font-heading); font-size: 1.05rem; font-weight: 600; margin-bottom: .45rem;
                  text-align: center; }}
  /* LinkedIn's own button style: LinkedIn blue, white "in" logo, pill shape, darker blue on hover.
     Fixed brand colours (not theme tokens), so it reads as LinkedIn in both light and dark mode. */
  a.li-btn {{ display: flex; align-items: center; justify-content: center; gap: .55rem; width: 100%;
             box-sizing: border-box; padding: .55rem 1rem; border-radius: 999px; background: #0A66C2;
             color: #ffffff !important; text-decoration: none !important; font: 600 .95rem/1.2 var(--font-body);
             transition: background-color .15s ease, box-shadow .15s ease; }}
  a.li-btn:hover {{ background: #004182; box-shadow: 0 2px 8px rgba(10,102,194,.35); }}
  a.li-btn:focus-visible {{ outline: 2px solid #70B5F9; outline-offset: 2px; }}
  a.li-btn .li-logo {{ width: 1.15rem; height: 1.15rem; flex: none; }}
  /* Author card pinned to the bottom of the sidebar at any window height. The sidebar's scroll area is
     Streamlit's own, so the card's block is position: fixed, and `contain: layout` makes the sidebar (not the
     window) its containing block: it spans the sidebar's width and stays put while the menu scrolls behind it. */
  [data-testid="stSidebar"] {{ contain: layout; }}
  [data-testid="stSidebarContent"] {{ padding-bottom: 7.5rem; }}
  [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.author-card) {{
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 5; margin: 0; width: auto !important;
      background: var(--sidebar); border-top: 1px solid var(--line); padding: .8rem 1.5rem 1rem; }}
  .author-card {{ max-width: 22rem; margin: 0 auto; }}

  /* Footer, pages without a chat box: the page column fills at least the window, the footer is pushed to its
     end, and it sticks to the bottom of the window while scrolling. Streamlit's large default bottom padding
     is removed so the footer sits flush with the bottom edge. */
  [data-testid="stApp"]:not(:has([data-testid="stChatInput"])) [data-testid="stMainBlockContainer"] {{
      padding-bottom: 0 !important; min-height: 100dvh; display: flex; flex-direction: column; }}
  [data-testid="stApp"]:not(:has([data-testid="stChatInput"])) [data-testid="stMainBlockContainer"]
      > [data-testid="stVerticalBlock"] {{ flex: 1 0 auto; }}
  [data-testid="stApp"]:not(:has([data-testid="stChatInput"])) [data-testid="stElementContainer"]:has(.site-footer) {{
      margin-top: auto; position: sticky; bottom: 0; z-index: 4; }}
  /* Footer, pages with a chat box: Streamlit pins the chat box to the bottom with an empty strip under it.
     The footer is fixed into that strip, above the chat box's layer, using the main area (made a containing
     block) for its width so it lines up with the page, not the window. */
  :has(> [data-testid="stAppScrollToBottomContainer"]) {{ contain: layout; }}
  [data-testid="stApp"]:has([data-testid="stChatInput"]) [data-testid="stElementContainer"]:has(.site-footer) {{
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 100; margin: 0; width: auto !important; }}
  [data-testid="stApp"]:has([data-testid="stChatInput"]) .site-footer {{
      background: transparent; backdrop-filter: none; border-top: 0; padding: .35rem 1rem .55rem; }}
  .site-footer {{ padding: .55rem 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem;
                  text-align: center; line-height: 1.4;
                  background: color-mix(in srgb, var(--surface) 88%, transparent); backdrop-filter: blur(6px); }}
  .site-footer a {{ color: inherit; text-decoration: underline; }}
  .site-footer .sf-short {{ display: none; }}
  @media (max-width: 640px) {{ .site-footer .sf-long {{ display: none; }} .site-footer .sf-short {{ display: inline; }} }}
  [data-testid="stApp"]:has([data-testid="stChatInput"]) .site-footer .sf-long {{ display: none; }}
  [data-testid="stApp"]:has([data-testid="stChatInput"]) .site-footer .sf-short {{ display: inline; }}
  /* Streamlit gives markdown blocks a -1rem bottom margin; without this the pinned card and footer overhang
     the bottom edge */
  [data-testid="stElementContainer"]:has(.site-footer) [data-testid="stMarkdownContainer"],
  [data-testid="stElementContainer"]:has(.author-card) [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}

  /* ---- Chunk card (ui/components.chunk_card): rank, typed score, title, 3-line preview */
  [class*="st-key-ccgold_"] {{ border: 2px solid var(--success) !important; background: var(--success-soft); }}
  .cc-top {{ display: flex; align-items: center; flex-wrap: wrap; gap: .35rem .5rem; margin-bottom: .2rem; }}
  .cc-rank {{ font-family: var(--font-heading); font-size: 1.35rem; font-weight: 600; line-height: 1; }}
  .cc-score {{ font-family: var(--font-mono); font-size: .78rem; color: var(--muted); font-variant-numeric: tabular-nums; }}
  .cc-title {{ font-weight: 600; font-size: .92rem; line-height: 1.35; }}
  .cc-year {{ font-weight: 400; color: var(--muted); }}
  .cc-meta {{ font-size: .76rem; color: var(--muted); margin-top: .1rem; }}
  .cc-text {{ font-size: .84rem; line-height: 1.5; margin-top: .35rem; display: -webkit-box; -webkit-line-clamp: 3;
              -webkit-box-orient: vertical; overflow: hidden; }}
  .cc-full {{ font-size: .86rem; line-height: 1.55; }}
  mark {{ background: var(--highlight); color: inherit; padding: 0 1px; border-radius: 2px; }}

  /* ---- Compact comparison table (ui/components.compare_table) */
  .tablewrap {{ overflow-x: auto; }}
  .ctable {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  .ctable th, .ctable td {{ text-align: left; vertical-align: top; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }}
  .ctable thead th {{ border-bottom: 2px solid var(--line); }}
  .ctable .rk {{ width: 3.5rem; font-family: var(--font-heading); color: var(--muted); }}
  .ctable td.gold {{ color: var(--success); font-weight: 600; background: var(--success-soft); }}

  /* ---- Pipeline breadcrumb: one row (scrolls sideways on narrow screens instead of wrapping) */
  .crumbs {{ display: flex; align-items: center; gap: .35rem; flex-wrap: nowrap; overflow-x: auto;
             white-space: nowrap; font-size: .78rem; padding-bottom: .2rem; margin-bottom: .4rem; }}
  .crumb {{ padding: 2px 9px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); flex: none; }}
  .crumb.on {{ background: var(--accent); border-color: var(--accent); color: var(--on-accent); font-weight: 600; }}
  .crumbs .arrow {{ color: var(--muted); flex: none; }}
</style>""", unsafe_allow_html=True)
