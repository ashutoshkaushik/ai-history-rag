# Evaluation run `20260926-1159-v1-vector-rejudged`

Strategy **vector** · chat `gpt-4.1-mini` · judge `gpt-4.1` · top-k 5 · score gate 0.35 · 34 questions (28 answerable, 6 should be refused)

## Headline: RAG vs LLM-only

| Metric | RAG | LLM-only |
|---|---|---|
| Fully correct answers | 17/28 (61%) | 5/28 (18%) |
| Partially correct | 8/28 (29%) | 19/28 (68%) |
| Key-fact coverage (mean) | 78% | 53% |
| Answers with ≥1 incorrect claim | 1/28 (4%) | 5/28 (18%) |
| **Faithfulness** (claims fully supported by sources) | **132/145 (91%)** | n/a (no sources) |
| Faithfulness incl. partially supported | 144/145 (99%) | n/a |
| Fully faithful answers | 20/25 (80%) | n/a |
| Citation accuracy (cited source supports claim) | 98% of 143 cited claims | n/a |
| Correct refusals (should refuse) | 6/6 (100%) | 1/6 (17%) |
| False refusals (answerable) | 3/28 (11%) | 0/28 (0%) |
| Latency p50 / p95 | 2.25s / 3.97s | 0.80s / 1.86s |

Targets: faithfulness ≥ 90% → **MET** (91.0%) · p95 ≤ 8s → **MET** (3.97s)

## By category

| category | n | RAG correct / partial | LLM correct / partial | RAG coverage | LLM coverage | RAG ≥1 wrong claim | LLM ≥1 wrong claim |
|---|---|---|---|---|---|---|---|
| A | 5 | 3 / 2 | 0 / 5 | 83% | 60% | 1 | 1 |
| B | 3 | 1 / 1 | 1 / 1 | 60% | 40% | 0 | 0 |
| C | 4 | 2 / 1 | 1 / 3 | 69% | 60% | 0 | 0 |
| D | 7 | 4 / 2 | 2 / 4 | 77% | 56% | 0 | 1 |
| E | 3 | 2 / 1 | 1 / 2 | 83% | 74% | 0 | 0 |
| G | 6 | 5 / 1 | 0 / 4 | 89% | 33% | 0 | 3 |

## By audience

| audience | n | RAG correct / partial | LLM correct / partial | RAG coverage | LLM coverage | RAG ≥1 wrong claim | LLM ≥1 wrong claim |
|---|---|---|---|---|---|---|---|
| curious | 13 | 6 / 5 | 2 / 11 | 70% | 59% | 0 | 0 |
| expert | 8 | 6 / 2 | 1 / 4 | 89% | 34% | 0 | 4 |
| popular | 7 | 5 / 1 | 2 / 4 | 82% | 63% | 1 | 1 |

## Questions that should be refused

| id | cat | question | RAG | LLM-only |
|---|---|---|---|---|
| q29 | F1 | What did Alan Turing think about ChatGPT? | refused (evidence_check) | declined |
| q30 | F2 | Which chatbot was claimed to have passed the Turing test in 2014? | refused (evidence_check) | **answered from memory**: The chatbot claimed to have passed the Turing test in 2014 was "Eugene Goostman." |
| q31 | F2 | Who invented generative adversarial networks (GANs)? | refused (evidence_check) | **answered from memory**: Generative adversarial networks (GANs) were invented by Ian Goodfellow and his colleagues  |
| q32 | F2 | What is the Loebner Prize? | refused (evidence_check) | **answered from memory**: The Loebner Prize is an annual competition that awards the most human-like computer progra |
| q33 | H | Who discovered the dwarf planet Pluto, and in what year? | refused (score_gate) | **answered from memory**: Pluto was discovered by Clyde Tombaugh in 1930. |
| q34 | H | What did Galileo discover when he pointed his telescope at Jupiter? | refused (score_gate) | **answered from memory**: When Galileo pointed his telescope at Jupiter in 1610, he discovered four moons orbiting t |

## Retrieval (answerable questions)

- Gold document ranked 1st: 22/28 (79%)
- Gold document in top 5: 26/28 (93%) · MRR 0.84
- Answer text itself in a top-5 chunk (chunk-level): 17/21 (81%)

| stage | p50 | p95 |
|---|---|---|
| retrieve_s | 0.24s | 0.41s |
| grade_s | 0.81s | 1.04s |
| generate_s | 1.46s | 3.62s |
| total_s | 2.25s | 3.97s |
| llm_total_s | 0.80s | 1.86s |

## Per question

| id | cat | aud | gold rank | evidence | RAG | RAG faithful | LLM | notes |
|---|---|---|---|---|---|---|---|---|
| q01 | A | popular | 1 | yes | correct | 100% (4) | partial |  |
| q02 | A | popular | 1 | yes | correct | 100% (4) | partial |  |
| q03 | A | popular | 1 | yes | partial | 100% (4) | partial | RAG wrong: The proposal was for the 1956 Dartmouth Summer Research Project (the proposal was written in 1955, not 1956). · LLM wrong: The term 'artificial intelligence' was coined by John McCarthy in 1956 for the Dartmouth Conference |
| q04 | A | popular | 1 | yes | correct | 86% (7) | partial |  |
| q05 | B | popular | **miss** | **no** | declined | - | correct | RAG refused at evidence_check |
| q06 | B | popular | 1 | yes | correct | 100% (3) | incorrect |  |
| q07 | D | curious | 2 | **no** | declined | - | partial | RAG refused at evidence_check |
| q08 | D | curious | 1 | yes | correct | 100% (4) | partial |  |
| q09 | B | curious | 1 | - | partial | 100% (9) | partial |  |
| q10 | D | curious | 1 | yes | partial | 100% (3) | partial |  |
| q11 | D | curious | 1 | yes | correct | 100% (5) | partial |  |
| q12 | G | curious | 1 | yes | correct | 100% (3) | partial |  |
| q13 | A | curious | 1 | yes | partial | 80% (5) | partial |  |
| q14 | D | curious | 1 | **no** | partial | 100% (4) | correct |  |
| q15 | G | curious | 1 | yes | correct | 100% (3) | partial |  |
| q16 | G | expert | 1 | yes | correct | 100% (7) | incorrect | LLM wrong: The 1955 Dartmouth proposal requested $500. |
| q17 | G | expert | 1 | yes | partial | 100% (2) | partial |  |
| q18 | G | expert | 1 | yes | correct | 100% (2) | partial | LLM wrong: organized into roughly 12,000 categories (synsets) |
| q19 | D | expert | 2 | yes | correct | 100% (16) | incorrect | LLM wrong: Category B: The study of computer-based models of human cognition and problem-solving.; Category C: The creation of prog |
| q20 | D | expert | 5 | yes | correct | 100% (4) | correct |  |
| q21 | G | expert | 1 | **no** | correct | 100% (5) | incorrect | LLM wrong: the optimal balance involves increasing the amount of training tokens proportionally more than model parameters |
| q22 | C | expert | 1 | - | correct | 100% (10) | partial |  |
| q23 | C | curious | 1 | - | correct | 100% (7) | correct |  |
| q24 | C | curious | 4 | - | declined | - | partial | RAG refused at evidence_check |
| q25 | C | expert | 1 | - | partial | 0% (5) | partial |  |
| q26 | E | curious | 1 | - | correct | 90% (20) | partial |  |
| q27 | E | popular | 1 | yes | correct | 100% (1) | correct |  |
| q28 | E | curious | **miss** | - | partial | 50% (8) | partial |  |

Tokens used this run: {'gpt-4.1-2025-04-14': {'output_tokens': 4085, 'input_tokens': 43032}}

## Judge vs manual audit

Human audit of run `20260926-1134-v1-vector`: 57 claims in 7 answers. An answer counts as flagged if it has ≥1 claim not fully supported.

| id | human: unsupported | judge: unsupported / partial | agree? | judge-flagged claims |
|---|---|---|---|---|
| q09 | 0 | 0 / 0 | yes |  |
| q10 | 0 | 0 / 0 | yes |  |
| q14 | 1 | 0 / 0 | **no** |  |
| q22 | 1 | 0 / 0 | **no** |  |
| q25 | 1 | 0 / 5 | yes | [partial] The Chinchilla paper revised the earlier Kaplan et al. scaling laws by; [partial] Chinchilla (70B parameters) was trained on about 4 times more tokens t; [partial] The Chinchilla work found a power-law relationship between compute bud; [partial] This led to the conclusion that dataset scaling (increasing training t; [partial] Chinchilla revised Kaplan et al. by emphasizing training on more data  |
| q26 | 2 | 0 / 2 | yes | [partial] Statistical machine translation (SMT) emerged in the 1980s to early 19; [partial] Internet-based services like AltaVista's Babel Fish (using SYSTRAN tec |
| q28 | 1 | 1 / 3 | yes | [unsupported] The developments in datasets, compute, and training methods that enabl; [partial] The fast weight controller concept (1992) introduced dynamic, input-de; [partial] The attention mechanism was introduced as an additive method distinct ; [partial] The transition from AlexNet to Transformer involved moving from CNNs f |

Answer-level agreement: **5/7**
