"""'Learn from this step' content shown at the bottom of each RAG Lab page.

Every number here was measured on this project (Lab pages, eval runs, manual audit), so the
insights stay true to this corpus. Questions render as expanders so a reader can think first.
Edit the text here; the pages pick it up automatically.
"""

import streamlit as st

LEARN: dict[str, dict] = {
    # ------------------------------------------------------------------ 1 · Tokens
    "tokens": {
        "insights": [
            "**Everything is counted in tokens.** Chunk size (600), embedding limits, prompt size, cost and "
            "latency are all measured in tokens, not characters or words.",
            "**Two tokenizers are in play.** `cl100k_base` for the embedding model and chunk sizing, and "
            "`o200k_base` for gpt-4.1 / gpt-4.1-mini. They often give similar counts but split some words "
            "differently: `GPT` is `G` + `PT` in cl100k but one token in o200k.",
            "**Numbers are split into pieces.** `$13,500` is 4 tokens (`$`, `13`, `,`, `500`), and `10120` is "
            "`101` + `20`. Once the PDF lost the superscript in 10^120, nothing in the tokens can bring it back.",
            "**Characters per token measure how clean the text is.** The corpus averages about 4.5. Clean prose "
            "(Lighthill Report, The Bitter Lesson) reaches 5.2–5.5, while math-heavy documents (SVM, "
            "Backpropagation, LSTM, RLHF) drop to 3.3–4.0 because equations and LaTeX leftovers tokenize into "
            "many short pieces.",
            "**A space belongs to the word after it.** ` the` (with its leading space) is a single token, "
            "which is why so many tokens start with ·.",
        ],
        "qa": [
            ("Why is chunk size measured in tokens instead of characters?",
             "Because every limit and price the models have is in tokens. The same 600-token chunk can be "
             "~2,000 characters of math or ~3,300 characters of prose, so a character limit would give chunks "
             "of very different weight for the model. Measuring in tokens keeps each chunk's cost and "
             "capacity predictable."),
            ("A 600-token chunk of the Backpropagation article holds less information than one from the Lighthill "
             "Report. Why?",
             "Both are capped at 600 tokens, but Backpropagation tokenizes at ~3.3 characters per token against "
             "~5.5 for Lighthill, so its chunks hold roughly 40% fewer characters. Some of what's left is LaTeX "
             "noise like `{\\displaystyle ...}` that carries almost no meaning."),
            ("Why can't the model recover “10^120” from the text “10120”?",
             "The exponent was lost during PDF text extraction, before tokenization. The tokens `101` and `20` "
             "look exactly like the number 10,120. The model can only guess from context (“variations… 10^90 "
             "years”), which is what test question q17 checks. The real fix belongs in cleaning, not in the model."),
            ("Why does it matter that the embedding tokenizer splits “GPT” into “G” + “PT”?",
             "The embedding model never sees GPT as one unit, so the question's vector is built from generic "
             "pieces shared with many other words. That is one reason vector search ranks the right document for "
             "q05 (“What does the T in GPT stand for?”) only 19th, while keyword search (BM25), which matches "
             "whole words, ranks it 3rd."),
            ("Which tokenizer decides what a chat question costs?",
             "The chat model's, `o200k_base`: OpenAI bills gpt-4.1-mini by its own token counts. `cl100k_base` "
             "only decides how the corpus was cut into chunks and how the embedding call is billed (about "
             "15 tokens per question)."),
        ],
    },

    # ------------------------------------------------------------------ 2 · Chunks
    "chunks": {
        "insights": [
            "**55 documents became 1,145 chunks**, with a median of ~567 tokens (range 82–634, header included). "
            "Documents range from 1 chunk (General Problem Solver) to 86 (the AI100 report).",
            "**The splitter prefers natural boundaries.** It tries paragraph breaks first, then lines, sentences, "
            "clauses and finally words, so most chunks end at the end of a paragraph or sentence.",
            "**An overlap of 90 tokens is a maximum, not a guarantee.** Only 66% of chunk boundaries actually "
            "overlap (43% for HTML documents), because overlap is built only from whole pieces.",
            "**Each chunk carries a header** like `[Title (Year) | Section]`, added before embedding, so a chunk "
            "that never names its source can still be found by questions about it.",
            "**Citations come from where a chunk starts.** An early bug gave 34% of chunks the wrong page or "
            "section label (LangChain reported their position as −1). It was found by reading a demo answer, "
            "not by any metric.",
        ],
        "qa": [
            ("Why do most chunks come out at 560–600 tokens rather than exactly 600?",
             "The splitter adds whole paragraphs or sentences until the next piece wouldn't fit, then stops, so "
             "each chunk ends short by up to one piece. The header is added afterwards, which is why a few "
             "chunks go slightly over 600."),
            ("Why do some chunk boundaries have no overlap at all?",
             "Overlap is made only of whole pieces from the end of the previous chunk. If the last paragraph is "
             "longer than 90 tokens, nothing fits in the overlap window and the next chunk starts clean. That's "
             "common in the Lighthill Report, whose paragraphs are long."),
            ("The Dartmouth budget chunk never says “Dartmouth”. How does q16 still find it?",
             "Through its header: `[A Proposal for the Dartmouth Summer Research Project on Artificial "
             "Intelligence (1955) | …]`. That puts “Dartmouth”, “proposal” and “1955” into the chunk's embedding "
             "and into BM25's keyword index. Without the header, the chunk is just an anonymous budget table."),
            ("What would doubling the chunk size to 1,200 tokens do?",
             "About half as many chunks. Each embedding would average over more topics, so specific facts like "
             "“$13,500” would stand out less. Every LLM call would read about twice as many tokens (more cost and "
             "latency). On the plus side, fewer answers would be split across two chunks."),
            ("Why were reference lists removed and long papers cut to their main body?",
             "Reference lists share keywords with many questions but never answer them, so they would take up "
             "retrieval slots. Very long papers (GPT-3 has 75 pages) would crowd out shorter sources, so only "
             "their main body is kept."),
            ("How does a citation know which page and section to show?",
             "Each chunk is mapped back to the part of the document where it starts, and inherits that part's "
             "page number and section heading. A chunk that spans pages 4–5 is therefore cited as p.4."),
        ],
    },

    # ------------------------------------------------------------------ 3 · Embeddings
    "embeddings": {
        "insights": [
            "**An embedding is 1,536 numbers** computed from a chunk's text by `text-embedding-3-small`. Texts "
            "with similar meaning get similar numbers, even with no words in common.",
            "**Every vector has length 1**, so cosine similarity (how closely two vectors point the same way) is "
            "simply the sum of their numbers multiplied pairwise.",
            "**Clusters by era appear on their own.** The model never saw the era labels, yet chunks from the same "
            "period and topic land near each other.",
            "**The 2D map is only a rough guide.** Squashing 1,536 dimensions to 2 keeps only ~18% of the "
            "information (PCA). t-SNE keeps close neighbours together, but distances between clusters mean little.",
            "**Related ideas end up near each other across documents.** Lighthill's “Category B” chunk has three "
            "other Lighthill chunks as nearest neighbours (0.78–0.72), then Sutherland's rebuttal (0.70).",
        ],
        "qa": [
            ("What does a single dimension of an embedding mean?",
             "Nothing you can read on its own. Meaning is spread across the pattern of all 1,536 numbers. That's "
             "why the vector strip looks like noise, yet similar chunks have similar strips."),
            ("q31 (“Who invented GANs?”) scores 0.56 against real chunks, about as high as answerable questions. "
             "Why isn't that enough to answer it?",
             "Similarity measures relatedness, not whether a text contains the answer. The GANs question shares "
             "words and topics with chunks about generative models and adversarial examples, but none of them say "
             "who invented GANs. That's why the pipeline has an LLM evidence check after retrieval."),
            ("Why do chunks from the same document tend to cluster?",
             "They share a topic, vocabulary and the same header (title and year), which the embedding includes. "
             "It helps retrieval (a question about a paper finds several of its chunks) but can also fill the "
             "top 5 with one document."),
            ("Should you trust the map or the similarity numbers?",
             "The numbers. Two chunks can be nearest neighbours in 1,536 dimensions and still look far apart on "
             "the map, because 2D keeps only a fraction of the information. Use the map for intuition and the "
             "scores for decisions."),
            ("How much did it cost to embed the whole corpus?",
             "About 608K tokens at roughly $0.02 per million, so about one cent, once. The vectors are stored "
             "locally in Chroma. Each question then costs a ~15-token embedding, a tiny fraction of a cent."),
        ],
    },

    # ------------------------------------------------------------------ 4 · Retrieval
    "retrieval": {
        "insights": [
            "**On the 28 answerable test questions:** vector search puts the right document first 22 times and in "
            "the top 5 26 times (MRR 0.843). BM25 manages 18 and 26 (MRR 0.766). Hybrid 50/50 manages 22 and 26 "
            "(MRR 0.854).",
            "**The methods fail on different questions.** For q05 (“T in GPT”) the right document ranks 19th for "
            "vector, 3rd for BM25 and 7th for hybrid. For q07 (Turing's prediction) it ranks 2nd, 9th and 5th.",
            "**BM25 weights rare words.** Common words like `gpt` (in many chunks) count for little; rare names "
            "and numbers count for a lot. Stopwords (what, the, of) are ignored entirely.",
            "**RRF fuses by rank, not score.** Each list adds `weight ÷ (60 + rank)` per chunk. The hand-written "
            "version here gives exactly the same results as LangChain's `EnsembleRetriever` from the course.",
            "**Hybrid isn't automatically better.** At 50/50 it matched vector search on top-5 hits. Whether a "
            "different weight helps overall has to be measured, not assumed.",
        ],
        "qa": [
            ("When does BM25 beat vector search, and when does it lose?",
             "BM25 wins on exact terms: names, acronyms, numbers, rare words (q05's “GPT”). Vector search wins on "
             "paraphrase and meaning: a question and a passage that use different words for the same idea. BM25 "
             "has no notion that “year 2000” and “fifty years' time” are related."),
            ("Why does RRF ignore the raw scores?",
             "They aren't comparable: cosine similarity runs from 0 to 1, while BM25 scores are unbounded and vary "
             "with the question. Ranks are always comparable, and a chunk ranked well by both lists reliably "
             "rises to the top."),
            ("Work it out: a chunk is ranked 1st by vector and 4th by BM25, with weights 0.5 / 0.5. What is its "
             "RRF score?",
             "0.5 ÷ (60 + 1) + 0.5 ÷ (60 + 4) = 0.00820 + 0.00781 = **0.01601**. A chunk ranked 1st by only one "
             "list scores 0.00820, so appearing in both lists roughly doubles its score."),
            ("What does the constant c (60) do?",
             "It flattens the difference between top ranks. With c = 60, rank 1 (1/61) is only about 5% ahead of "
             "rank 4 (1/64). A small c would make rank 1 dominate; a large c makes the fusion reward chunks that "
             "appear in both lists."),
            ("If hybrid fixes q05, why does the chat still use vector search?",
             "At 50/50 hybrid didn't improve top-5 hits overall (26/28 either way), and a BM25-heavier weight might "
             "rescue q05 but push other questions out. The project's rule is to switch only when a measurement "
             "shows a net gain."),
            ("What's the difference between “right document” and “answer text” hits?",
             "Retrieving the right document isn't enough if it's the wrong part of it. For q07, a Turing chunk was "
             "retrieved (a document-level hit), but not the one containing the “fifty years” prediction, so the "
             "evidence check correctly refused."),
        ],
    },

    # ------------------------------------------------------------------ 5 · Score gate & top k
    "gate": {
        "insights": [
            "**Off-topic questions score very low.** The two astronomy questions top out at ~0.21, while every "
            "answerable question starts at ~0.48. Any threshold between ~0.22 and ~0.47 separates them; 0.35 sits "
            "in the middle.",
            "**“Not in corpus” questions can't be caught by a threshold.** They score 0.36–0.57, among the "
            "answerable ones, because they're about related topics. The gate catches 0 of those 4.",
            "**The two refusal layers work together.** In the evaluation, all 6 should-refuse questions were "
            "refused: 2 by the free score gate and 4 by the LLM evidence check.",
            "**Retrieving more chunks helps less and less.** All three methods reach ~93% of questions with the "
            "right document by k = 5, and the curve flattens after that.",
            "**Each extra chunk costs ~567 tokens in both LLM calls**, so k is the main lever on cost and latency.",
        ],
        "qa": [
            ("Why does the pipeline need two refusal layers?",
             "The score gate is free and instant but only catches clearly unrelated questions. Related-but-"
             "unanswerable questions score as high as real ones, so only an LLM reading the chunks can tell "
             "“related” from “answers the question”. The gate saves that LLM call when it isn't needed."),
            ("Why is 0.35 safe but 0.50 a bad threshold?",
             "At 0.50 the gate would wrongly refuse answerable questions that score just below it, such as q05 "
             "(0.48) and q25 (0.48), both real questions with answers in the corpus. 0.35 has a wide margin on "
             "both sides."),
            ("Why not simply retrieve 20 chunks to be safe?",
             "Cost and latency roughly quadruple (both LLM calls read every chunk), and the relevant sentence gets "
             "buried among irrelevant text. Models attend less reliably to details in the middle of long contexts. "
             "The curve shows little gain after k = 5."),
            ("Why is the “answer text” curve lower than the “right document” curve?",
             "It's a stricter test: the specific passage containing the answer must be retrieved, not just any "
             "chunk from the right document. The gap shows how often retrieval finds the right source but the "
             "wrong part of it."),
            ("Could a better threshold also catch the “not in corpus” questions?",
             "No. They overlap with answerable questions (0.36–0.57 vs 0.48–0.78), so any threshold high enough to "
             "catch them would also refuse real questions. That's a limit of similarity scores, not of the "
             "chosen number."),
        ],
    },

    # ------------------------------------------------------------------ 6 · Prompts & generation
    "prompts": {
        "insights": [
            "**An answered question makes two LLM calls** to gpt-4.1-mini: the evidence check (returns JSON "
            "`{sufficient, reason}`) and the answer. Each reads ~2,700 input tokens.",
            "**About 94% of those input tokens are the retrieved sources.** The instructions and the question "
            "are small, so top k drives cost.",
            "**Repeated questions are cheaper.** OpenAI caches the beginning of a prompt (1,024+ tokens) that "
            "matches a recent request, so asking the same question again reuses most of the input at a discount. "
            "The two calls start with different instructions, so they don't share a cache entry with each other.",
            "**Time is spent in sequence:** retrieval ~0.6s, evidence check ~1s, answer ~1–1.5s. In the chat the "
            "answer streams, so the first words appear before generation finishes.",
            "**A refusal at the evidence check still costs one LLM call**, reading all 5 chunks. A refusal at the "
            "score gate costs nothing.",
        ],
        "qa": [
            ("Why a separate evidence check instead of just telling the answer prompt to refuse when needed?",
             "The decision is explicit and testable (a yes/no with a reason), cheap (a short structured output), "
             "and happens before any answer text exists, so a half-grounded answer can't slip through. The "
             "answer prompt can still say which part is missing when the sources only partly answer."),
            ("Why are the sources numbered [1] to [5] in the prompt?",
             "So the model can cite them inline. The app then turns each [n] into a source card with title, year, "
             "page and section, and the faithfulness judge checks that each cited source supports its claim."),
            ("The temperature is 0, yet the same question sometimes gets a different answer. Why?",
             "Temperature 0 means the model always picks its top choice, but hardware-level floating-point "
             "differences can change which choice is on top. The LLM-only answer to q07 changed between runs this "
             "way, which is why evaluations and demos should not rely on a single run."),
            ("What does the rule “Keep numbers, names and dates exactly as the sources give them” protect?",
             "The long-tail details where RAG adds the most value. Asked for the Dartmouth budget, the LLM alone "
             "said “$500”; RAG quoted $13,500 from the proposal. Paraphrasing numbers is where errors creep in."),
            ("Why does the answer prompt say “Do not add outside knowledge, even if you know it”?",
             "gpt-4.1-mini knows a lot of AI history, some of it wrong. If it mixes its own memory into a cited "
             "answer, the citation makes unsupported claims look verified. The manual audit found exactly that "
             "kind of blending in 6 of 57 audited claims (~10%)."),
            ("How much does one answered question cost?",
             "Roughly 5.4K input and ~100–150 output tokens across the two gpt-4.1-mini calls, plus a ~15-token "
             "embedding: a fraction of a cent, and less when the prompt is cached."),
        ],
    },
}


def render_learning(key: str) -> None:
    """The '🎓 Learn from this step' section at the bottom of a Lab page."""
    content = LEARN.get(key)
    if not content:
        return
    st.divider()
    st.markdown("## 🎓 Learn from this step")
    st.markdown("#### Key insights")
    st.markdown("\n".join(f"- {line}" for line in content["insights"]))
    st.markdown("#### Test your understanding")
    st.caption("Think about each question first, then open it to check your answer.")
    for question, answer in content["qa"]:
        with st.expander(f"❓ {question}"):
            st.markdown(answer)
