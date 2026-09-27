"""'How it's built' page: the LangChain and LangGraph code behind each box of the system diagram.

Snippets are read from the live source with inspect.getsource(), so this page always shows the code
that is actually running (not a copy that can drift). Each block links to the file on GitHub.
"""

import inspect
import re

import streamlit as st

from rag import config, evaluate, ingest, models, prompts, rag_graph, retrieve

REPO = "https://github.com/ashutoshkaushik/ai-history-rag/blob/main/"


def _source_of(obj) -> str:
    return inspect.getsource(obj).rstrip()


def _prompt_source(name: str) -> str:
    """A module-level prompt constant, cut from prompts.py (inspect can't fetch variables)."""
    text = inspect.getsource(prompts)
    m = re.search(rf"^{name} = PromptTemplate\.from_template\(\n.*?^\)\n", text, re.S | re.M)
    return m.group(0).rstrip() if m else f"# {name} not found"


def _snippet(title: str, why: str, code: str, path: str) -> None:
    st.markdown(f"**{title}**")
    st.markdown(f"<div class='code-why'>{why}</div>", unsafe_allow_html=True)
    st.code(code, language="python", line_numbers=False, wrap_lines=True)
    st.markdown(f"<div class='code-src'><a href='{REPO}{path}' target='_blank'>{path}</a></div>",
                unsafe_allow_html=True)


def code_page() -> None:
    st.markdown("""<style>
      .code-why { color: var(--muted); font-size: .92rem; margin: -.2rem 0 .4rem; max-width: 75ch; }
      .code-src { font-size: .78rem; margin: -.6rem 0 1.4rem; }
      .who { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: .4rem 0 1rem; }
      .who > div { border: 1px solid var(--line); border-radius: .6rem; padding: .8rem 1rem; }
      .who b { font-family: var(--font-heading); font-size: 1.05rem; }
      .who ul { margin: .4rem 0 0 1rem; padding: 0; font-size: .9rem; }
      @media (max-width: 760px) { .who { grid-template-columns: 1fr; } }
    </style>""", unsafe_allow_html=True)

    st.title("How it's built: LangChain + LangGraph")
    st.markdown("<div class='lede'>The code behind each box of the system diagram, in the order data flows. "
                "Every snippet is read from the running source, so it's exactly what the app executes.</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='who'>"
        "<div><b>LangChain: the building blocks</b><ul>"
        "<li>Text splitting (<code>RecursiveCharacterTextSplitter</code>)</li>"
        f"<li>Embeddings and chat models (<code>OpenAIEmbeddings</code>, <code>ChatOpenAI</code>: "
        f"{config.EMBEDDING_MODEL}, {config.CHAT_MODEL})</li>"
        "<li>Vector store (<code>Chroma</code>) and keyword search (<code>BM25Retriever</code>)</li>"
        "<li>Prompts (<code>PromptTemplate</code>) and structured output (<code>with_structured_output</code>)</li>"
        "</ul></div>"
        "<div><b>LangGraph: the control flow</b><ul>"
        "<li>A typed state (<code>RAGState</code>) passed between steps</li>"
        "<li>Nodes: retrieve → grade_evidence → generate, or refuse</li>"
        "<li>Conditional edges: the two “I don't know” gates</li>"
        "<li>Streaming the answer node's tokens to the chat (<code>stream_mode=\"messages\"</code>)</li>"
        "</ul></div></div>", unsafe_allow_html=True)

    st.header("1 · Ingestion (offline)")
    _snippet("Split into ~600-token chunks, measured with the embedding model's tokenizer",
             "LangChain's recursive splitter tries paragraph breaks first, then lines, sentences and words, so "
             "chunks end at natural boundaries. Sizes are in tokens, the unit that limits and costs are counted in.",
             _source_of(ingest.splitter), "src/rag/ingest.py")
    _snippet("Embed and store in Chroma",
             "One Chroma collection with cosine distance. Every chunk carries its document's metadata (title, "
             "year, era, page, section), which becomes the citation.",
             _source_of(retrieve.get_vector_store), "src/rag/retrieve.py")

    st.header("2 · Retrieval")
    _snippet("Vector search: the strategy the chat uses",
             "Chroma returns cosine distance; 1 − distance gives the similarity the score gate checks.",
             _source_of(retrieve.retrieve), "src/rag/retrieve.py")
    _snippet("Keyword search with BM25, over the same chunks",
             "Built from the same chunk file as the vector index, so both retrievers search identical units.",
             _source_of(retrieve.get_bm25_retriever), "src/rag/retrieve.py")
    _snippet("Hybrid search: Reciprocal Rank Fusion",
             "Merges ranked lists by rank, not score. Verified to give the same results as LangChain's "
             "EnsembleRetriever; compare it live on the RAG Lab's Retrieval page.",
             _source_of(retrieve.rrf_fuse), "src/rag/retrieve.py")

    st.header("3 · The LangGraph flow: answer or refuse")
    _snippet("The state that flows between nodes",
             "A TypedDict: each node returns only the keys it changes, and LangGraph merges them.",
             _source_of(rag_graph.RAGState), "src/rag/rag_graph.py")
    _snippet("Wiring the graph",
             f"Four nodes and two conditional edges. The score gate refuses below {config.MIN_SIMILARITY} with "
             "no LLM call; the evidence check catches related-but-unanswerable questions.",
             _source_of(rag_graph.after_retrieve) + "\n\n\n" + _source_of(rag_graph.after_grade) + "\n\n\n"
             + _source_of(rag_graph.build_graph), "src/rag/rag_graph.py")
    _snippet("The evidence check: an LLM call with structured output",
             "with_structured_output(EvidenceGrade) makes the model return a validated {sufficient, reason} "
             "object instead of free text, so the graph can branch on it.",
             _source_of(rag_graph.EvidenceGrade) + "\n\n\n" + _source_of(rag_graph.grade_node),
             "src/rag/rag_graph.py")
    _snippet("Generating the cited answer",
             "Sources are numbered [1]…[5] in the prompt; only the ones the answer actually cites come back as "
             "source cards.",
             _source_of(rag_graph.generate_node), "src/rag/rag_graph.py")

    st.header("4 · Prompts")
    _snippet("The answer prompt",
             "Every rule targets a failure seen in testing: outside knowledge, uncited claims, paraphrased numbers.",
             _prompt_source("RAG_PROMPT"), "src/rag/prompts.py")

    st.header("5 · Streaming into the chat")
    _snippet("Running the same graph, token by token",
             "stream_mode=[\"updates\", \"messages\"] yields both the state after each node (for the live "
             "“Checking the evidence…” status) and the generate node's tokens (for the typing effect).",
             _source_of(rag_graph.stream_answer), "src/rag/rag_graph.py")
    _snippet("One place that creates the models",
             "Cached clients; max_retries makes calls wait out rate limits instead of failing.",
             _source_of(models.get_llm), "src/rag/models.py")

    st.header("6 · Evaluation: LLM as judge")
    _snippet("Claim-by-claim faithfulness, graded by a stronger model",
             f"{config.JUDGE_MODEL} splits each answer into claims and marks each supported, partial or "
             "unsupported against the retrieved sources. It was checked against a manual audit first.",
             _source_of(evaluate.judge_faithfulness), "src/rag/evaluate.py")
