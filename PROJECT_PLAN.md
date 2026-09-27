# AI History Research Assistant: Project Plan

A citation-grounded RAG system over AI-history sources, built with LangChain + LangGraph and measured against an LLM answering alone.

**Research question:** When does retrieval-augmented generation measurably beat an LLM answering from its own pretrained knowledge on AI-history questions?

---

## Requirements

| Requirement | Where it lands |
|---|---|
| One-liner: user, question type, corpus, surface, % faithfulness and/or % relevance | Phase 0 |
| Design summary: use case, corpus, ingestion + cleaning, freshness, chunking + embedding, retrieval (1–2 sentences each) | Phase 0 draft, finalized in Phase 8 |
| Latency ceiling picked up front | Phase 0 |
| Explicit "I don't know" path, designed first | Phase 4 |
| Cited answers | Phase 4 |
| Evaluation of retrieval quality and failures | Phases 5–6 |
| Chat UI on top of the RAG system | Phase 7 |
| Write-up and demo | Phase 8 |

---

## Phase 0: Scoping and setup ✅
**Goal:** get decisions on paper and a working environment.

- [x] Draft the one-liner, including a faithfulness target (e.g. ≥90%) and a latency ceiling (e.g. p95 ≤ 8s end to end)
- [ ] Draft the design summary (a first pass is fine; it gets revised in Phase 8)
- [x] `git init`, `.gitignore`, `README.md` stub
- [x] Python 3.12 via `uv` (the system Python is 3.9.6, which is too old for current LangChain)
- [x] Model stack: OpenAI `gpt-4.1-mini` (generation, LLM-only baseline, judge) + `text-embedding-3-small` (a full corpus embed costs cents). Local cross-encoder reranker in Phase 6 if it's needed.
- [x] All providers are switchable in `config.py` (LangChain `init_chat_model`), so moving to Ollama or another provider is a config change, not a code change
- [ ] `OPENAI_API_KEY` in `.env` (gitignored), with a hard usage limit set in the OpenAI dashboard
- [x] Smoke test: one chat call and one local embedding call
- [x] `CLAUDE.md` holding the project conventions, so Claude Code in VSCode has the context

**Exit:** `uv run python scripts/smoke_test.py` prints one LLM reply and one embedding vector length.

## Phase 1: Corpus curation ✅
**Goal:** ~50 documents (expanded from 20–30), chosen on purpose, each with metadata.

- [x] `corpus/manifest.yaml` with one entry per document: id, title, authors, year, era, source_type (primary/secondary/reference), document_type, url, license note, why it's included, and the question types it enables
- [x] Balance: about 60% primary sources, 40% secondary/reference. Primary papers rarely explain *why* something happened (AI winters, for example), so the "cause" questions need secondary sources.
- [x] Coverage check: every era has at least 2 docs, and every eval question category has at least one doc that can answer it
- [x] Licensing: **commit the manifest and the download script, not the PDFs.** arXiv and Wikipedia are fine; journal PDFs such as Turing's 1950 paper in *Mind* should not be redistributed on a public GitHub repo.
- [x] Flag scanned or image-only PDFs (Lighthill 1973, the Dartmouth proposal, older reports), which need OCR or an alternative text source

**Exit:** the manifest is reviewed and `scripts/download_corpus.py` fetches every document into `data/raw/` (gitignored).

**Outcome:** 55 docs (33 primary / 21 secondary / 1 reference), all downloaded. Text-layer checks caught 2 broken PDFs (Dartmouth, LSTM) and 1 table-only Wikipedia page, all replaced with clean sources. 3 hosts block automated download and were swapped for alternatives (documented in `notes:` fields).

## Phase 2: Golden evaluation set (built *before* any tuning) ✅
**Goal:** a fixed test set, so every later decision is measured against it and not judged by feel.

- [x] 25–30 questions, each with a category, a reference answer, **gold source doc id(s)**, and a key-facts checklist
- [x] Categories: A direct lookup · B multi-document · C cross-era comparison · D specific evidence/quote · E temporal · **G long-tail detail** (obscure specifics where the LLM is likely to hallucinate, e.g. the budget requested in the Dartmouth proposal) · F unanswerable
- [x] Split unanswerable (F) into two kinds:
  - F1 "impossible in reality" (Turing on ChatGPT). A bare LLM also refuses these, so they don't separate the two systems.
  - F2 "real AI-history facts that are *not in the corpus*". These are the real test of grounding, because the LLM will answer and RAG should refuse.
- [x] Store as `eval/golden_set.yaml`

**Exit:** the golden set is reviewed, and each question has gold doc ids that exist in the manifest.

**Outcome:** 34 questions in `eval/golden_set.yaml`, each tagged by audience (popular / curious / expert). There are 28 answerable and 6 that should be refused: 1 impossible, 3 real-but-not-in-corpus, and 2 off-topic astronomy questions, one of them an adversarial Galileo trap. All answers were verified against the corpus text, and absent topics were confirmed by full-corpus search.

## Phase 3: Ingestion, cleaning, chunking, indexing ✅
**Goal:** raw documents become clean, metadata-rich chunks in a vector store.

- [x] Loaders: PDF (PyMuPDF), HTML (trafilatura or BeautifulSoup), with OCR fallback for scanned docs
- [x] Cleaning: strip headers, footers, and page numbers; fix hyphenated line breaks; drop reference lists and boilerplate; normalize unicode and entities
- [x] Save cleaned docs to `data/processed/*.jsonl` so they can be inspected by eye
- [x] Chunking v1: recursive splitter, ~500–800 tokens, ~15% overlap, section-aware where headings exist. Carry all manifest metadata plus page and section onto each chunk.
- [x] Embed with `text-embedding-3-small` and store in **Chroma** (local and persistent, with no server to run)
- [x] Freshness: the manifest hash plus a per-doc content hash drive an **incremental re-index** (only new or changed docs get re-embedded). SLA: a document added to the manifest is searchable after one `make ingest` run.
- [x] Ingestion report: doc count, chunk count, token distribution, and docs that failed or were OCR'd

**Exit:** `uv run python -m rag.ingest` builds the index idempotently, and a retrieval query from a notebook or CLI returns sensible chunks with metadata.

**Outcome:** 55 docs → 1,145 chunks (608K tokens, median 565 tokens per chunk, with a contextual header), embedded in 42s for about $0.01. A re-run re-embeds 0 docs.
Dense baseline on the golden set: **hit@1 22/28, hit@5 26/28 (93%), hit@10 27/28**. The remaining misses are q05 (the acronym "GPT", an exact-match case where BM25 should help) and q28 (a multi-hop question spanning 2012–2017).
Score separation: off-topic questions top out at a similarity of about 0.21 (easy threshold), but not-in-corpus questions score 0.54–0.56, overlapping answerable ones (minimum 0.48). So refusal needs an LLM evidence check, not just a threshold.
Fixes made along the way: false numbered headings (addresses, dates), a bibliography with no heading (Minsky), and HTML headings lost by trafilatura (replaced with an lxml walker). Gold labels were widened for 4 questions where secondary docs also held the answer.

## Phase 4: RAG v1 (dense retrieval + cited generation + refusal), in LangGraph ✅
**Goal:** the simplest complete pipeline, with the "I don't know" path designed first.

- [x] Graph: `retrieve → assess_evidence → (generate_with_citations | refuse)`
- [x] Refusal triggers: the top retrieval score falls below a threshold, or the LLM's evidence check says the context is insufficient
- [x] Generation prompt: answer only from the numbered context, cite as `[n]`, and say explicitly when evidence is missing
- [x] Output schema: answer, citations (doc id, title, year, page), retrieved chunks, latency per stage
- [x] CLI: `uv run python -m rag.ask "What was the Lighthill Report?"`
- [x] LLM-only baseline: the same model and question with no context, using a neutral prompt

**Exit:** both systems answer all golden questions end to end from the CLI.

**Outcome:** the code is organised around small factories and entry points: `get_llm` / `get_vector_store` factories, `RAG_PROMPT` built with `PromptTemplate.from_template`, `retrieve(question, strategy)` and `answer_question(question, strategy)`, with the LangGraph flow in `rag_graph.py`.
Behavior on all 34 questions: **31/34 correct, all 6 refusals correct** (2 at the free score gate, 4 at the evidence check). Latency p50 2.3s, **p95 3.5s** (target ≤ 8s).
3 false refusals, all caused by retrieval, not by the grader: q05 (acronym, so BM25), q07 (the "year 2000" query doesn't match Turing's "fifty years" wording, and the right Turing chunk wasn't retrieved), q24 (a comparison, so one query finds only one side).
Lesson: doc-level hit@k overstates retrieval quality (q07 counted as a hit), so Phase 5 adds chunk-level evidence checks.
Bug found while spot-checking a demo answer: LangChain's `start_index` was -1 for 388/1,145 chunks (34%), so those chunks got the *last* section's page and section label, giving wrong citations. Fixed by locating each chunk's opening text ourselves (0 unlocated now) and re-embedding. Behavior after the rebuild is unchanged (31/34, p95 4.0s).
Demo moment: asked for the Dartmouth budget, the LLM alone confidently said "$500"; RAG answered "$13,500" with the itemized budget and citations.

## Interlude: Understanding tools (done, between Phases 4 and 5) ✅
- [x] `scripts/visualize_chunks.py` → `data/processed/chunk_viewer.html`: exact chunks per document, overlaps, headers, page/section, token stats
- [x] `scripts/visualize_embeddings.py` → `data/processed/embedding_viewer.html`: t-SNE/PCA map of all chunks, raw vectors, nearest neighbours, question → top-5 retrieval
- [x] `scripts/export_clean_text.py`: a document's cleaned text for pasting into ChunkViz
- Findings: "overlap 90" is a maximum, not a guarantee (only 66% of chunk boundaries overlap; 43% for HTML); Wikipedia math extracts as one symbol per line; 2D PCA keeps only 18% of the embedding variance
- Not run yet: `scripts/chunking_experiment.py` (300/600/1200 tokens, header on/off; ~$0.05). Candidate Phase 6 ablation.

## Phase 5: Evaluation harness + LLM vs RAG comparison ✅
**Goal:** the core experiment, with numbers.

- [x] Retrieval metrics (no LLM needed): hit@k and recall@k against the gold doc ids, plus MRR
- [x] Answer metrics (LLM-as-judge using a *different* model than the generator, plus manual spot-checks of about 20%):
  - Correctness against the reference answer and key facts
  - Faithfulness: every claim is supported by the retrieved context (RAG only)
  - Citation accuracy: the cited chunk actually supports the sentence
  - Hallucination rate: unsupported or false claims
  - Refusal: correct refusals on F, and **false refusals** on A–G
- [x] Latency: p50/p95 per stage and end to end
- [x] Output: `eval/results/<run_id>.json` plus a markdown summary table, broken down by category

**Exit:** one command (`make eval`) produces the LLM-only vs RAG v1 table.

**Run `20260926-1134-v1-vector`** (34 questions, ~290K tokens):
| | RAG | LLM-only |
|---|---|---|
| Fully correct | 14/28 (50%) | 5/28 (18%) |
| Key-fact coverage | 76% | 52% |
| ≥1 incorrect claim | 1/28 (4%)* | 8/28 (29%) |
| Correct refusals | 6/6 | 2/6 |
| False refusals | 3/28 | 0/28 |
| p50 / p95 latency | 2.25s / 3.97s | 0.80s / 1.86s |
| Faithfulness (judge) | 100% (147 claims) | n/a |
| **Faithfulness (manual audit, 57 claims)** | **~89.5%** | n/a |

*The single RAG "incorrect claim" (q03, 1956 vs 1955) is a judge error: the project *was* in 1956.
**Judge audit:** the gpt-4.1-mini judge approved all 57 audited claims; a human found 6 unsupported (blends, mis-dating, editorial inferences). Same-model judging is lenient, so the judge needs to be stricter or stronger (see `eval/manual_audit.yaml`).
Other judge errors: q22/q27 LLM-only flagged for omissions or near-matches. LLM-only output also varied between runs at temperature 0 (q07).

**Judge iteration (same saved answers, re-graded):**
1. gpt-4.1-mini judge: faithfulness 100%, which the manual audit disproved (~89.5%).
2. gpt-4.1 + strict 3-level faithfulness prompt: **faithfulness 91%** strict (99% counting partial), in line with the audit; 5/7 answer-level agreement with the human (missed the q14 "additive attention" blend and a q22 editorial phrase). But the correctness judge then used its own (sometimes outdated) knowledge and flagged 5 true RAG statements, e.g. ELIZA's code being rediscovered in 2021.
3. The correctness judge was restricted to the reference answer only (grounding is the faithfulness judge's job). `--rejudge --keep-faithfulness` re-graded correctness alone.

**Final v1 results (`20260926-1159-v1-vector-rejudged`, judge gpt-4.1):**
| | RAG | LLM-only |
|---|---|---|
| Fully correct | **17/28 (61%)** | 5/28 (18%) |
| Key-fact coverage | **78%** | 53% |
| ≥1 incorrect claim | 1/28 (a known judge false positive, q03) | **5/28 (18%)**: $500 budget, 12,000 synsets, wrong Lighthill B/C, reversed Chinchilla finding, "coined 1956" |
| Faithfulness (strict / incl. partial) | **91% / 99%** (145 claims) | n/a |
| Citation accuracy | 98% | n/a |
| Correct refusals | **6/6** | 1/6 (answers GANs, Goostman, Pluto, Galileo from memory) |
| False refusals | 3/28 (q05, q07, q24: retrieval misses) | 0/28 |
| Latency p50 / p95 | 2.25s / **3.97s** | 0.80s / 1.86s |
Both targets met: faithfulness ≥ 90% (91%), p95 ≤ 8s (3.97s). Judge cost: ~165K gpt-4.1 tokens, paced for the 30K TPM limit.

## Phase 6: Measured improvements (ablations)
**Goal:** add complexity only where the numbers justify it.

- [ ] v2: hybrid retrieval (BM25 + dense, fused with RRF)
- [ ] v3: rerank with a cross-encoder (e.g. bge-reranker) over the top ~20, keeping 5
- [ ] Optional: a chunk size variant, a metadata filter for year or era ("primary sources 2010–2020"), or query rewriting for comparison questions
- [ ] Ablation table: each version × each metric × latency. Keep a change only if it helps.
- [ ] Failure analysis: 5–8 concrete cases covering retrieval misses, chunking problems, bad citations, false refusals, and cases where the LLM alone was just as good

**Exit:** a final config is chosen with evidence behind it, plus a written "when does RAG help" analysis.

## Phase 7: Chat UI ✅ (built before Phase 6 at the user's request; strategies plug in automatically)
- [x] Streamlit chat: answer with inline `[n]` citations, a sources list, an expandable "Retrieved context" panel with scores, and a latency badge
- [x] Optional toggle: show the LLM-only answer side by side, which works well for demos

**Exit:** `uv run streamlit run app.py` works locally.

**Outcome:** `app.py` is structured as main → tabs → render_* functions, with chat_input + history and expanders. Features: [n] superscript citations with source cards (primary/secondary icon, authors, page/section, link); refusal banner with the reason (score gate vs evidence check); "Compare with LLM-only" and "LLM fallback on refusal" toggles (the fallback is clearly labelled as unverified); retrieved-context expander with scores; per-stage timings; example questions by audience; a corpus browser tab.
**v2 of the UI: side-by-side streaming.** RAG and LLM-only run at the same time and stream token by token into two columns. Each runs in a worker thread that pushes into a queue, and Streamlit's script thread drains both queues and draws. RAG streams only the `generate` node's tokens via LangGraph `stream_mode=["updates","messages"]` (`stream_answer()` in rag_graph.py), and shows its stage live (🔎 Retrieving → 🧪 Checking the evidence → ✍️ Writing). Each column shows time-to-first-token and total. A measured run: the LLM-only column streams from 0.2s and finishes at ~1.2s, while RAG retrieves and checks the evidence until ~1.6s, then streams until 2.8s. That makes the cost of grounding visible in the demo. RAG-only mode keeps the labelled "LLM fallback on refusal".
**UI simplification (user request):** removed the retrieval-strategy radio, the side-by-side toggle and the fallback toggle. The layout is always RAG vs LLM-only side by side, using `config.DEFAULT_STRATEGY`; the fallback toggle was then restored above the RAG column, labelled as a RAG setting ("🧠 RAG: fall back to the LLM's answer when refused"); on a refusal it shows the LLM-only column's answer inside the RAG column, labelled unverified, with no extra API call. To demo a Phase 6 strategy, change `DEFAULT_STRATEGY` in config.py.
Bug found in testing: Streamlit rendered "$13,500 … $1,200" as a LaTeX formula, so all model/user text now escapes `$`.

## Phase 8: Write-up and demo
- [ ] Project write-up and a short demo walkthrough

---

## Phase 9: RAG Lab visualizations (moved ahead of Phase 6 by user choice: understanding first)
A `🔬 RAG Lab` Streamlit page next to the chat, with live, interactive views:
- [x] 1. Retrieval comparison: BM25 vs vector vs hybrid for any question, matched terms and IDF, RRF fusion arithmetic, and hit@k across the golden set
  - `lab/rag_lab.py`, registered via `st.navigation` in app.py (pages: 💬 Research Assistant, 🔬 RAG Lab). BM25 = LangChain `BM25Retriever` over the same chunks.jsonl, custom tokenizer (lowercase, keeps "gpt-3"/"13,500", drops stopwords). The hand-written weighted RRF (`rrf_fuse`) was verified identical to `langchain_classic` `EnsembleRetriever` on 3 questions.
  - Golden set, 50/50, c = 60, 20 candidates: vector hit@1 22 / hit@5 26 / MRR 0.843 · BM25 18 / 26 / 0.766 · hybrid 22 / 26 / 0.854. No clear win for hybrid at default settings. q05: vector rank 19, **BM25 rank 3**, hybrid 7 (diluted). q07: 2 / 9 / 5. q28: 10 / miss / 15. Lesson: "hybrid is usually right" still has to be measured and tuned per corpus.
- [x] 2. Threshold slider + k curve: similarity distributions for answer vs refuse questions; refusals at each threshold; hit@k for k = 1–20
  - Altair charts (validated dataviz palette slots 1–3 plus shapes and table views). Findings: off-topic tops out at ~0.21 and answerable starts at ~0.48, so the gate's safe zone is ~0.22–0.47 and 0.35 sits mid-zone; the gate catches 0/4 not-in-corpus questions (0.36–0.57), which is why the LLM evidence check exists. k curve: all methods reach ~93% document-level by k = 5 and flatten; each extra chunk adds ~567 tokens to both LLM calls.
- [x] Understand-the-system pages: `st.navigation` sections, App (💬 Research Assistant) and Understand the system (🗺️ System diagram from assets/system_diagram.html, 🧩 Chunk viewer, 🌌 Embedding viewer, 🔬 RAG Lab). The viewers are embedded with `components.html` from data/processed/ (regenerate with `make viz`; they show build instructions if missing).
- [x] 3. Prompt inspector + step trace: the fully assembled prompts, token budget, and a per-node LangGraph timeline
  - A LangChain callback handler (`CallRecorder`) passed to `GRAPH.invoke(config={"callbacks": [...]})` records each LLM call with its LangGraph node, model, exact prompt, raw response and API usage. The token budget splits input into instructions / retrieved sources / question (o200k_base, reconciled to the API total) plus output; a timeline shows each step; there's an explainer for each RAG_PROMPT rule.
  - Findings: a 2-call answer is ~5.4K input tokens, ~94% of it retrieved sources (so top k is the cost lever). OpenAI prompt caching served ~5.1K of those tokens from cache. That came from earlier runs of the *same question* (an identical prompt start across requests), not from the two calls sharing a prefix; their prompts start differently. This corrects an earlier explanation. An evidence-check refusal still costs one LLM call; the score gate refuses for free.
- [x] System diagram updated (local `assets/system_diagram.html` and the claude.ai artifact v2, the same file): chat app + streaming, gpt-4.1 judge, BM25/hybrid available in the Lab, DEFAULT_STRATEGY knob.
- [x] 7. Tokenizer view: text coloured by token (exponents, OCR noise, cost)
  - Side-by-side cl100k_base (embeddings / chunk sizes) vs o200k_base (gpt-4.1 family), presets pulled live from real corpus chunks, a token list, and a corpus-wide characters-per-token chart. All local, no API calls.
  - Findings: "10120" tokenizes as `101` + `20` (the exponent is unrecoverable); cl100k splits "GPT" into `G` + `PT` while o200k keeps it whole, which may contribute to q05's vector-search miss; Wikipedia math chunks still contain LaTeX leftovers (`{\displaystyle ...}`), a cleaning fix for Phase 6; math-heavy docs tokenize at 3.3–4.0 characters per token vs 5.2–5.5 for prose, so ~40% less text per chunk.
- Housekeeping: `use_container_width` replaced with `width="stretch"` (deprecated); mixed-type table columns fixed; the Streamlit preview is bound to 127.0.0.1.
- [x] Navigation restructured in pipeline order: sections App · Overview (System diagram) · "RAG Lab · the pipeline, step by step" with pages 1 Tokens → 2 Chunks → 3 Embeddings → 4 Retrieval → 5 Score gate & top k → 6 Prompts & generation. Each page has a breadcrumb highlighting its step (`pipeline_header()` in lab/rag_lab.py); the Lab module exposes page functions instead of tabs.
- [x] "🎓 Learn from this step" at the bottom of all 6 Lab pages (`lab/learn.py`): 5 key insights each, backed by this project's measured numbers, plus 33 questions in total as expanders (think first, then open). Writing them caught a wrong explanation of prompt caching, corrected on the page and in this plan.
Later (paired with Phase 6): query-rewriting view, reranker before/after, metadata filters, chunk-size chart.

## Phase 6 roadmap (after the Lab views): each change gets an eval run vs v1
1. Hybrid search (BM25 + vector, RRF) → q05 · 2. Re-tune MIN_SIMILARITY / TOP_K with Lab data · 3. Stricter answer prompt → faithfulness · 4. Query rewriting / sub-questions → q07, q24, q28 · optional: reranker, metadata filters, chunk-size ablation

## Planned repo layout
```
ai-history-rag/
├── PROJECT_PLAN.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── corpus/manifest.yaml
├── data/               # gitignored: raw/, processed/, chroma/
├── eval/
│   ├── golden_set.yaml
│   └── results/
├── scripts/            # smoke_test, download_corpus
├── src/rag/
│   ├── config.py       # models, chunk sizes, top-k: one place for experiments
│   ├── ingest.py       # load → clean → chunk → embed → store
│   ├── retrieve.py     # dense / hybrid / rerank
│   ├── graph.py        # LangGraph: retrieve → assess → generate | refuse
│   ├── baseline.py     # LLM-only
│   └── evaluate.py
└── app.py              # Streamlit UI
```

### Re-test of the "$500" claim (2026-09-26, user challenge)
Same baseline prompt, temperature 0, 5 runs each: **gpt-4.1-mini → "$500" 5/5; gpt-4.1 → "$13,500" 5/5 (correct).** gpt-4.1-mini asked plainly at temperature 0.7: "$120,000", "$500", "$120,000". So the failure is reproducible but model-specific: larger models (and Gemini, per the user) know this fact. The site now names the model and frames the finding as "RAG lets a small, cheap model match larger ones on long-tail facts, with a checkable source". Possible follow-up: re-run the LLM-only baseline with gpt-4.1 to measure how much of RAG's advantage survives against a stronger model.

## Public deployment (Streamlit Community Cloud), prepared 2026-09-26
- Claude-like theme in `.streamlit/config.toml`: Source Serif 4 headings and answers, Hanken Grotesk UI, IBM Plex Mono code, warm light and dark palettes, terracotta accent, validated chart palette.
- New pages: ✨ Start here (default), 📊 Results (scoreboard, coverage by audience/category, refusal table, "where the model alone got it wrong" gallery, judge story), Lab Previous/Next tour buttons, and a corpus timeline in the Corpus tab (primary sources by year on era bands).
- Public mode (`PUBLIC_DEPLOYMENT=1` in secrets): caps of 10 questions per visit and 150 per day across visitors (`lab/usage.py`, covering chat and Prompt-inspector runs); the Chunks viewer shows only the 19 CC BY-SA Wikipedia documents.
- First start builds data/ via `rag.bootstrap` (download → ingest → viewers). Tested on a clean copy: 55/55 downloaded, 2 minutes. It produced 1,146 chunks vs 1,145 locally because a live Wikipedia article had changed, so the cloud index can drift slightly from the evaluated one.
- `requirements.txt` is pinned from uv.lock; app.py adds `src/` to sys.path, so no package install is needed.
- Results caption corrected to the final data: the gap is widest on long-tail (+56 pts) and expert (+54) questions and narrowest on comparisons (+8) and timelines (+9).

### UI polish (2026-09-26)
- Home action cards with real buttons ("Try the assistant" is the primary one); Lab Previous/Next are buttons.
- Truncation fixed site-wide, verified with an ellipsis/overflow detector at 1024px and 1280px on all pages: short metric labels with full descriptions in tooltips, a global CSS rule so metric labels, deltas and button text wrap, "Allow fallback to model memory" toggle, "Clear chat" icon button, and the Results refusal table as a wrapping HTML table with ✓/✗ badges and word-boundary excerpts.
- Chat empty state: 6 starter questions (2 popular, 2 curious, 1 expert, 1 should-refuse) that submit on click and disappear once a conversation starts; the sidebar sample list is collapsed by default; new input placeholder.

### Design system + layout pass (2026-09-26)
- `ui/theme.py`: one token set (light + dark, chosen from `st.context.theme`), serif headings / sans body at a fixed H1–H4 scale on every page, shared badges, main width capped at 1240px. 47 hard-coded colours/fonts removed from app.py, lab/rag_lab.py and lab/site_pages.py.
- Lab category charts moved to a validated violet / yellow / magenta set, so blue always means RAG and green always means correct.
- Home: new title ("A RAG pipeline you can use, and look inside."), and the $13,500 comparison states the correct answer, with ✓ Correct on RAG and ✗ Wrong / struck-through $500 on the model alone.
- Research Assistant: each turn is question → two columns (header, timing, answer in sans 16px/1.6) → full-width Sources cited + Retrieved context; the columns stack under 900px, RAG first. Measured: 453px per column at 1280px with the sidebar open, 533px at 1440px, ~580px at 1280px with the sidebar collapsed.
- Emoji removed from titles, headers, tabs and expanders; Material icons in the sidebar.

### Chunk card component + Retrieval lab readability (2026-09-26)
- `ui/components.py`: `chunk_card()` (large rank #n, typed score "cosine 0.479" / "BM25 10.3" / "RRF 0.0138", title/year/section, cleaned 3-line preview centred on BM25-matched terms, "Show more" expander, green border + "✓ Gold" badge) and `compare_table()`. `clean_display_text()` strips `{\displaystyle …}` blocks (nested braces), runs of 6+ numbers and one/two-character math shreds, and collapses whitespace (display only; the index is unchanged). An RLHF math chunk goes from 1,391 to 715 characters of readable text.
- Retrieval page: a per-column gold summary ("✓ Gold found at rank 3" / "✗ Gold not in top 5"), a note that scores aren't comparable across methods, and a "Compact view" toggle (rank × method table, gold cells highlighted).
- The same card is used on the Prompts page (retrieved chunks, with token counts) and in the Research Assistant's "Retrieved context" (2-column grid). Chat history now stores full chunk text for "Show more".
- Breadcrumb is a single row (scrolls sideways when narrow); Previous/Next buttons are at the bottom of all six Lab pages.

### Public app live: https://ai-history-rag.streamlit.app/ (2026-09-26)
- Checked on the live site: pages render, public mode works (the Chunks viewer lists 19 CC BY-SA docs with the licensing notice, no errors).
- Bug: the "First start: building the search index…" box stayed visible forever. It was drawn inside an `st.cache_resource` function, and Streamlit replays a cached function's elements on every later run. Fix: `ensure_index()` checks `index_ready()` outside any cache and shows `st.status` only during a real build, with a shared lock so concurrent first visitors don't build twice. Verified on a clean copy: the box shows during the ~2.5-minute build, ends as "Index ready", and is gone after reload.
- The chunk count on the home page and the Embeddings caption now comes from the index (`rag.bootstrap.chunk_count()`): the cloud build has 1,146 chunks vs 1,145 locally (live Wikipedia).
- The live app was still on the previous commit when checked; it needs a push and possibly a reboot to pick up the design-system release.

### Corpus page, code page, author link, footer (2026-09-26)
- Corpus moved from a tab on the chat page to its own Overview page ("Corpus: the N sources", count from the manifest); the chat page has no tabs, so its input is now pinned to the bottom of the window.
- New Overview page "How it's built: the code" (`lab/code_page.py`): what LangChain vs LangGraph do here, then 13 snippets in system-diagram order (splitter, Chroma, retrieve, BM25, RRF, RAGState, graph wiring, structured evidence check, generate, RAG_PROMPT, stream_answer, get_llm, faithfulness judge). Snippets come from `inspect.getsource()` on the running code, each linked to its file on GitHub.
- `ui/chrome.py`: author name + "Connect on LinkedIn" at the bottom of the sidebar, and a footer ("Built with Streamlit, LangChain and LangGraph, using Claude Code and other AI tools, with a human in the loop."), both on every page (called after `page.run()`).

### "What is RAG?" infographic on Start here (2026-09-27)
- `rag_explainer()` in `lab/site_pages.py`, first thing on the home page, all colours from `ui/theme.py` tokens (new `--on-rag`).
- Left, "Where it fits in AI": nested layers AI ⊃ machine learning ⊃ deep learning ⊃ LLMs, with the note that RAG is not a new model but a way of using an LLM at question time.
- Right, "What changes": LLM alone vs LLM + RAG (answers from, knowledge, citations, when it doesn't know, cost to update).
- Bottom: a 4-step flow (Question → Retrieve → Augment → Generate) using the $13,500 Dartmouth example and the live source count and top-k.
- Checked at 1280/1100 in light and dark mode, and at 420px (everything stacks to one column, no sideways scroll).
