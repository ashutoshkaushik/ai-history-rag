"""All prompts in one place, so prompt iterations are easy to track and document."""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------- RAG answer
RAG_PROMPT = PromptTemplate.from_template(
    """You are an AI-history research assistant. Answer the question using ONLY the numbered
sources below, which come from a curated corpus of AI-history papers, reports and articles.

Rules:
- Every factual sentence must cite its source(s) inline, like [1] or [2][3].
- Use only facts stated in the sources. Do not add outside knowledge, even if you know it.
- If the sources only partly answer the question, answer that part and say clearly what is missing.
- Keep numbers, names and dates exactly as the sources give them.
- Be concise: a short paragraph, or a few bullet points for comparisons and timelines.

Sources:
{context}

Question: {question}

Answer:"""
)

# ------------------------------------------------------------ evidence check
GRADE_PROMPT = PromptTemplate.from_template(
    """You check whether retrieved sources contain evidence to answer a question about AI history.

Question: {question}

Sources:
{context}

Decide whether the sources contain information that directly answers the question, fully or partially.
- "sufficient": true if at least part of the answer is stated in the sources.
- "sufficient": false if the sources are only about related topics, merely mention the same words,
  or the question is outside AI history.
Judge ONLY from the sources, not from your own knowledge."""
)

# ------------------------------------------------------------ LLM-only baseline
LLM_ONLY_PROMPT = PromptTemplate.from_template(
    """You are an AI-history research assistant. Answer the question concisely and accurately.
If you are not sure of the answer, say so rather than guessing.

Question: {question}

Answer:"""
)

# ============================================================ evaluation judges
# Used only by rag.evaluate (LLM-as-judge). Reference answers and key facts come from
# eval/golden_set.yaml, which were verified against the corpus.

CORRECTNESS_JUDGE_PROMPT = PromptTemplate.from_template(
    """You are grading an answer to a question about AI history against a verified reference.

Question: {question}

Reference answer (ground truth): {reference}

Key facts to check:
{key_facts}

Candidate answer:
{answer}

Grade the candidate answer:
- declined: true if the answer declines, says it doesn't know, or says it can't find the information,
  instead of answering.
- key_facts: for EACH key fact above, in the same order, whether the candidate states it (same meaning
  counts; exact wording is not needed; "10^9" and "about a billion" are the same).
- incorrect_claims: statements in the candidate that directly CONTRADICT the reference answer or a key
  fact. Use ONLY the reference as ground truth, never your own knowledge: if the reference does not
  address a statement, it is not incorrect (grounding is checked separately). Before listing one, check
  it really contradicts the reference and is not just phrased differently, less precise, extra detail,
  or about a closely related fact (e.g. an event's date vs. the date of the document that proposed it).
  Omissions and missing detail are NEVER incorrect claims.
- verdict: "correct" (all key facts, no incorrect claims), "partial" (some key facts, or all key facts
  plus a minor error), "incorrect" (no key facts, or a major error), or "declined"."""
)

FAITHFULNESS_JUDGE_PROMPT = PromptTemplate.from_template(
    """You are a strict fact-checker. Decide whether each claim in an answer is supported by the
numbered sources the answer was given. You must judge ONLY from the source text, never from your own
knowledge, even when you know a claim is true.

Sources:
{context}

Answer:
{answer}

Split the answer into atomic factual claims (one checkable fact each). Skip statements that only say
information is missing from the sources. For each claim give:
- claim: the claim, briefly restated
- cited: the source numbers the answer cites for this claim ([] if none)
- supported_by: the source numbers that support it ([] if none)
- verdict:
  "supported": a source states it, or it follows directly with no added information. Every part must
     be backed, including names, numbers, dates, time periods, causes and comparisons.
  "partial": the core is supported but some detail is not (an added date, qualifier, cause, or
     characterisation that no source states).
  "unsupported": no source states it, OR it combines facts from the sources into a new relationship
     the sources do not state (e.g. source says "X is distinct from Y" and the claim says "X replaced
     Y"), OR it assigns a date or period the source does not give, OR it adds interpretation or
     attribution ("as the author implied", "this also enabled...") that no source makes.
- note: one short phrase explaining any verdict other than "supported"."""
)
