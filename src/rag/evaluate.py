"""Evaluation harness: RAG vs LLM-only on the golden set.

    uv run python -m rag.evaluate --label v1-vector              # full run
    uv run python -m rag.evaluate --label test --only q16 q31    # a few questions
    uv run python -m rag.evaluate --rejudge 20260926-1134-v1-vector   # re-grade saved answers only

Stage 1 runs both systems question by question (sequential, so latency is honest).
Stage 2 grades every answer with an LLM judge (parallel):
  - correctness judge (both systems): key facts covered, incorrect claims, declined?
  - faithfulness judge (RAG answers): atomic claims -> which sources support them, and
    whether the cited [n] is one of them.

Writes eval/results/<run_id>.json (per-question detail + summary) and
eval/results/<run_id>.md (tables for the write-up), plus an HTML claim-level report in
data/processed/ (it quotes corpus text, so it is not committed).

LLM-only answers and their grades are cached in eval/results/baseline_cache.json:
they don't change when the RAG pipeline changes, so Phase 6 re-runs don't pay for them.
"""

import argparse
import contextvars
import hashlib
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal

import yaml
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel, Field

from rag import config
from rag.baseline import llm_only_answer
from rag.models import get_llm
from rag.prompts import CORRECTNESS_JUDGE_PROMPT, FAITHFULNESS_JUDGE_PROMPT, LLM_ONLY_PROMPT
from rag.rag_graph import answer_question, format_context

RESULTS_DIR = config.EVAL_DIR / "results"
BASELINE_CACHE = RESULTS_DIR / "baseline_cache.json"
REFUSE_CATEGORIES = {"F1", "F2", "H"}


# ------------------------------------------------------------ judge rate limiting

JUDGE_TPM_BUDGET = 27_000          # stay under this account's 30K tokens/min limit for gpt-4.1
_window: list[tuple[float, int]] = []
_window_lock = threading.Lock()


def _reserve_tokens(estimate: int) -> None:
    """Block until `estimate` tokens fit in the rolling 60-second budget."""
    while True:
        with _window_lock:
            now = time.monotonic()
            _window[:] = [(ts, n) for ts, n in _window if now - ts < 60]
            if sum(n for _, n in _window) + estimate <= JUDGE_TPM_BUDGET or not _window:
                _window.append((now, estimate))
                return
            wait = 60 - (now - _window[0][0]) + 0.1
        time.sleep(wait)


def call_judge(schema, prompt_value):
    """Structured judge call with pacing and patient retries on 429 rate limits."""
    text = prompt_value.to_string()
    for attempt in range(12):
        _reserve_tokens(len(text) // 3 + 1500)          # rough input + output estimate
        try:
            return get_llm(config.JUDGE_MODEL).with_structured_output(schema).invoke(prompt_value)
        except Exception as exc:                        # openai.RateLimitError via langchain
            if "rate_limit" not in str(exc) and "429" not in str(exc):
                raise
            m = re.search(r"try again in ([\d.]+)(ms|s)", str(exc))
            delay = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else 10.0
            time.sleep(delay + 1 + attempt)
    raise RuntimeError("judge call kept hitting the rate limit")


# ------------------------------------------------------------------ judge schemas

class KeyFactCheck(BaseModel):
    fact: str
    present: bool


class CorrectnessGrade(BaseModel):
    declined: bool
    key_facts: list[KeyFactCheck] = Field(default_factory=list)
    incorrect_claims: list[str] = Field(default_factory=list)
    verdict: Literal["correct", "partial", "incorrect", "declined"]


class Claim(BaseModel):
    claim: str
    cited: list[int] = Field(default_factory=list)
    supported_by: list[int] = Field(default_factory=list)
    verdict: Literal["supported", "partial", "unsupported"]
    note: str = ""


class FaithfulnessGrade(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


def judge_correctness(q: dict, answer: str) -> dict:
    facts = q.get("key_facts") or ["(none: this question should be declined)"]
    grade = call_judge(CorrectnessGrade,
        CORRECTNESS_JUDGE_PROMPT.invoke({
            "question": q["question"], "reference": q["reference_answer"],
            "key_facts": "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1)), "answer": answer,
        }))
    present = [k.present for k in grade.key_facts][: len(q.get("key_facts", []))]
    n = len(q.get("key_facts", []))
    return {
        "declined": grade.declined, "verdict": grade.verdict,
        "key_facts_present": present, "coverage": (sum(present) / n) if n else None,
        "incorrect_claims": grade.incorrect_claims,
    }


def judge_faithfulness(context: str, answer: str) -> dict:
    grade = call_judge(FaithfulnessGrade, FAITHFULNESS_JUDGE_PROMPT.invoke({"context": context, "answer": answer}))
    claims = [c.model_dump() for c in grade.claims]
    supported = [c for c in claims if c["verdict"] == "supported"]       # strict: only fully supported
    cited = [c for c in claims if c["cited"]]
    cite_ok = [c for c in cited if c["verdict"] != "unsupported" and set(c["cited"]) & set(c["supported_by"])]
    return {
        "claims": claims, "n_claims": len(claims),
        "faithfulness": len(supported) / len(claims) if claims else None,
        "faithfulness_lenient": sum(c["verdict"] != "unsupported" for c in claims) / len(claims) if claims else None,
        "citation_accuracy": len(cite_ok) / len(cited) if cited else None,
        "uncited_claims": len(claims) - len(cited),
    }


# ------------------------------------------------------------------- stage 1: run

def run_rag(q: dict, strategy: str) -> dict:
    state = answer_question(q["question"], strategy)
    retrieved = state.get("retrieved", [])
    return {
        "answer": state["answer"], "refused": state["refused"], "refusal_stage": state["refusal_stage"],
        "grade_reason": state.get("grade_reason"), "top_score": round(state.get("top_score", 0.0), 3),
        "sources": state["sources"], "timings": state["timings"],
        "retrieved": [{"n": i, "id": d.id, "doc_id": d.metadata["doc_id"], "score": round(s, 3),
                       "title": d.metadata["title"], "section": d.metadata.get("section")}
                      for i, (d, s) in enumerate(retrieved, 1)],
        "_context": format_context(retrieved) if retrieved else "",
        "_texts": [d.page_content for d, _ in retrieved],
    }


def retrieval_metrics(q: dict, rag: dict) -> dict:
    docs = [r["doc_id"] for r in rag["retrieved"]]
    rank = next((i for i, d in enumerate(docs, 1) if d in q["gold_docs"]), None)
    out = {"gold_rank": rank, "hit@1": rank == 1, "hit@5": rank is not None and rank <= 5,
           "rr": 1 / rank if rank else 0.0}
    if q.get("evidence"):
        out["evidence@5"] = any(re.search(q["evidence"], t) for t in rag["_texts"])
    return out


def baseline_key(q: dict) -> str:
    raw = f"{config.CHAT_MODEL}|{config.JUDGE_MODEL}|{LLM_ONLY_PROMPT.template}|{CORRECTNESS_JUDGE_PROMPT.template}|{q['question']}|{q['reference_answer']}|{q.get('key_facts')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --------------------------------------------------------------------- summaries

def pct(num: float, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[max(0, int(round(0.95 * len(s))) - 1)] if s else 0.0


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected_behavior"] == "answer"]
    ref = [r for r in rows if r["expected_behavior"] == "refuse"]

    def side(key: str) -> dict:
        a_grades = [r[key]["grade"] for r in ans]
        return {
            "correct": sum(g["verdict"] == "correct" for g in a_grades),
            "partial": sum(g["verdict"] == "partial" for g in a_grades),
            "coverage": statistics.mean(g["coverage"] or 0 for g in a_grades) if a_grades else 0,
            "hallucinated": sum(bool(g["incorrect_claims"]) for g in a_grades),
            "false_refusals": sum(g["declined"] for g in a_grades),
            "correct_refusals": sum(r[key]["grade"]["declined"] for r in ref),
            "latency": [r[key]["timings"]["total_s"] for r in rows],
        }

    rag, llm = side("rag"), side("llm")
    rag["false_refusals"] = sum(r["rag"]["refused"] for r in ans)     # deterministic for RAG
    rag["correct_refusals"] = sum(r["rag"]["refused"] for r in ref)
    faith = [r["rag"]["faithfulness"] for r in ans if r["rag"].get("faithfulness")]
    claims = sum(f["n_claims"] for f in faith)
    supported = sum(sum(c["verdict"] == "supported" for c in f["claims"]) for f in faith)
    partial = sum(sum(c["verdict"] == "partial" for c in f["claims"]) for f in faith)
    cited = [c for f in faith for c in f["claims"] if c["cited"]]
    cite_ok = sum(c["verdict"] != "unsupported" and bool(set(c["cited"]) & set(c["supported_by"])) for c in cited)
    rag.update({
        "answered": len(faith), "claims": claims, "supported_claims": supported, "partial_claims": partial,
        "faithfulness": supported / claims if claims else 0,
        "fully_faithful": sum(f["faithfulness"] == 1.0 for f in faith),
        "citation_accuracy": cite_ok / len(cited) if cited else 0, "cited_claims": len(cited),
    })
    ret = [r["retrieval"] for r in ans]
    ev = [x["evidence@5"] for x in ret if "evidence@5" in x]
    stages = {s: [r["rag"]["timings"][s] for r in rows if s in r["rag"]["timings"]]
              for s in ("retrieve_s", "grade_s", "generate_s", "total_s")}
    return {
        "n": len(rows), "n_answerable": len(ans), "n_refuse": len(ref), "rag": rag, "llm": llm,
        "retrieval": {"hit@1": sum(x["hit@1"] for x in ret), "hit@5": sum(x["hit@5"] for x in ret),
                      "mrr": statistics.mean(x["rr"] for x in ret) if ret else 0,
                      "evidence@5": sum(ev), "evidence_n": len(ev)},
        "latency": {s: {"p50": statistics.median(v), "p95": p95(v)} for s, v in stages.items() if v}
                   | {"llm_total_s": {"p50": statistics.median(llm["latency"]), "p95": p95(llm["latency"])}},
    }


def render_markdown(run: dict) -> str:
    S, rows = run["summary"], run["results"]
    R, L, n, m = S["rag"], S["llm"], S["n_answerable"], S["n_refuse"]
    lat = S["latency"]
    out = [f"# Evaluation run `{run['run_id']}`", "",
           f"Strategy **{run['settings']['strategy']}** · chat `{run['settings']['chat_model']}` · judge "
           f"`{run['settings']['judge_model']}` · top-k {run['settings']['top_k']} · score gate "
           f"{run['settings']['min_similarity']} · {S['n']} questions ({n} answerable, {m} should be refused)", "",
           "## Headline: RAG vs LLM-only", "",
           "| Metric | RAG | LLM-only |", "|---|---|---|",
           f"| Fully correct answers | {pct(R['correct'], n)} | {pct(L['correct'], n)} |",
           f"| Partially correct | {pct(R['partial'], n)} | {pct(L['partial'], n)} |",
           f"| Key-fact coverage (mean) | {100 * R['coverage']:.0f}% | {100 * L['coverage']:.0f}% |",
           f"| Answers with ≥1 incorrect claim | {pct(R['hallucinated'], n)} | {pct(L['hallucinated'], n)} |",
           f"| **Faithfulness** (claims fully supported by sources) | **{pct(R['supported_claims'], R['claims'])}** | n/a (no sources) |",
           f"| Faithfulness incl. partially supported | {pct(R['supported_claims'] + R['partial_claims'], R['claims'])} | n/a |",
           f"| Fully faithful answers | {pct(R['fully_faithful'], R['answered'])} | n/a |",
           f"| Citation accuracy (cited source supports claim) | {100 * R['citation_accuracy']:.0f}% of {R['cited_claims']} cited claims | n/a |",
           f"| Correct refusals (should refuse) | {pct(R['correct_refusals'], m)} | {pct(L['correct_refusals'], m)} |",
           f"| False refusals (answerable) | {pct(R['false_refusals'], n)} | {pct(L['false_refusals'], n)} |",
           f"| Latency p50 / p95 | {lat['total_s']['p50']:.2f}s / {lat['total_s']['p95']:.2f}s | "
           f"{lat['llm_total_s']['p50']:.2f}s / {lat['llm_total_s']['p95']:.2f}s |", "",
           f"Targets: faithfulness ≥ {100 * config.FAITHFULNESS_TARGET:.0f}% → "
           f"**{'MET' if R['faithfulness'] >= config.FAITHFULNESS_TARGET else 'NOT MET'}** "
           f"({100 * R['faithfulness']:.1f}%) · p95 ≤ {config.LATENCY_P95_TARGET_S:.0f}s → "
           f"**{'MET' if lat['total_s']['p95'] <= config.LATENCY_P95_TARGET_S else 'NOT MET'}** "
           f"({lat['total_s']['p95']:.2f}s)", ""]

    def group_table(field: str, title: str) -> list[str]:
        t = [f"## By {title}", "", f"| {title} | n | RAG correct / partial | LLM correct / partial | RAG coverage | LLM coverage | RAG ≥1 wrong claim | LLM ≥1 wrong claim |",
             "|---|---|---|---|---|---|---|---|"]
        for key in sorted({r[field] for r in rows}):
            g = [r for r in rows if r[field] == key and r["expected_behavior"] == "answer"]
            if not g:
                continue
            def c(side, v): return sum(r[side]["grade"]["verdict"] == v for r in g)
            def cov(side): return 100 * statistics.mean(r[side]["grade"]["coverage"] or 0 for r in g)
            def wr(side): return sum(bool(r[side]["grade"]["incorrect_claims"]) for r in g)
            t.append(f"| {key} | {len(g)} | {c('rag','correct')} / {c('rag','partial')} | {c('llm','correct')} / {c('llm','partial')} "
                     f"| {cov('rag'):.0f}% | {cov('llm'):.0f}% | {wr('rag')} | {wr('llm')} |")
        return t + [""]

    out += group_table("category", "category") + group_table("audience", "audience")
    ref = [r for r in rows if r["expected_behavior"] == "refuse"]
    out += ["## Questions that should be refused", "", "| id | cat | question | RAG | LLM-only |", "|---|---|---|---|---|"]
    for r in ref:
        rag = f"refused ({r['rag']['refusal_stage']})" if r["rag"]["refused"] else "**answered**"
        llm = "declined" if r["llm"]["grade"]["declined"] else "**answered from memory**: " + r["llm"]["answer"][:90].replace("\n", " ").replace("|", "/")
        out.append(f"| {r['id']} | {r['category']} | {r['question']} | {rag} | {llm} |")
    X = S["retrieval"]
    out += ["", "## Retrieval (answerable questions)", "",
            f"- Gold document ranked 1st: {pct(X['hit@1'], n)}",
            f"- Gold document in top 5: {pct(X['hit@5'], n)} · MRR {X['mrr']:.2f}",
            f"- Answer text itself in a top-5 chunk (chunk-level): {pct(X['evidence@5'], X['evidence_n'])}", "",
            "| stage | p50 | p95 |", "|---|---|---|"]
    out += [f"| {s} | {v['p50']:.2f}s | {v['p95']:.2f}s |" for s, v in S["latency"].items()]
    out += ["", "## Per question", "",
            "| id | cat | aud | gold rank | evidence | RAG | RAG faithful | LLM | notes |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["expected_behavior"] != "answer":
            continue
        rg, lg, rt = r["rag"]["grade"], r["llm"]["grade"], r["retrieval"]
        f = r["rag"].get("faithfulness")
        faith = f"{100 * f['faithfulness']:.0f}% ({f['n_claims']})" if f and f["faithfulness"] is not None else "-"
        notes = []
        if r["rag"]["refused"]:
            notes.append(f"RAG refused at {r['rag']['refusal_stage']}")
        if rg["incorrect_claims"]:
            notes.append("RAG wrong: " + "; ".join(rg["incorrect_claims"])[:120])
        if lg["incorrect_claims"]:
            notes.append("LLM wrong: " + "; ".join(lg["incorrect_claims"])[:120])
        ev = {True: "yes", False: "**no**"}.get(rt.get("evidence@5"), "-")
        out.append(f"| {r['id']} | {r['category']} | {r['audience']} | {rt['gold_rank'] or '**miss**'} | {ev} | {rg['verdict']} | {faith} | {lg['verdict']} | "
                   + " · ".join(notes).replace("|", "/").replace("\n", " ") + " |")
    u = run["usage"]
    out += ["", f"Tokens used this run: {u}", ""]
    return "\n".join(out)


def render_html(run: dict) -> str:
    """Claim-level report: each RAG answer's claims coloured by support, next to the LLM-only answer."""
    def esc(s): return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cards = []
    for r in run["results"]:
        f = r["rag"].get("faithfulness") or {}
        claims = "".join(
            f"<li class='{ {'supported': 'ok', 'partial': 'part'}.get(c.get('verdict'), 'bad') }'>{esc(c['claim'])} "
            f"<span class='m'>{c.get('verdict', '')}{' (' + esc(c.get('note')) + ')' if c.get('note') else ''} · cited {c['cited'] or '-'} · supported by {c['supported_by'] or 'nothing'}"
            f"{' · citation mismatch' if c['cited'] and not set(c['cited']) & set(c['supported_by']) else ''}</span></li>"
            for c in f.get("claims", []))
        srcs = "".join(f"<li>[{x['n']}] {x['score']:.2f} {esc(x['title'])} <span class='m'>{esc(x['section'] or '')}</span>"
                       f"{' ✓' if x['doc_id'] in r['gold_docs'] else ''}</li>" for x in r["rag"]["retrieved"])
        rg, lg = r["rag"]["grade"], r["llm"]["grade"]
        cards.append(f"""<div class="card"><h3>{r['id']} <span class="tag">{r['category']}</span><span class="tag">{r['audience']}</span>
          <span class="tag">should {r['expected_behavior']}</span></h3><p class="q">{esc(r['question'])}</p>
          <p class="m">Reference: {esc(r['reference_answer'])}</p>
          <div class="cols"><div><h4>RAG · <span class="v {rg['verdict']}">{rg['verdict']}</span>
          {f"· faithfulness {100 * f['faithfulness']:.0f}%" if f.get('faithfulness') is not None else ''}</h4>
          <div class="a">{esc(r['rag']['answer'])}</div>{f'<ul class="claims">{claims}</ul>' if claims else ''}
          {'<p class="m">Refused at ' + esc(r['rag']['refusal_stage']) + ': ' + esc(r['rag']['grade_reason'] or f"best similarity {r['rag']['top_score']}") + '</p>' if r['rag']['refused'] else ''}
          <details><summary>Retrieved chunks</summary><ol class="src">{srcs}</ol></details></div>
          <div><h4>LLM-only · <span class="v {lg['verdict']}">{lg['verdict']}</span></h4><div class="a">{esc(r['llm']['answer'])}</div>
          {'<p class="bad m">Wrong: ' + esc('; '.join(lg['incorrect_claims'])) + '</p>' if lg['incorrect_claims'] else ''}</div></div></div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval Report</title><style>
:root{{--bg:#f7f6f3;--p:#fff;--i:#1d1d1b;--m:#6b6a66;--l:#e4e2dc;--ok:#1f8a4c;--bad:#c0392b;--g:#ecebe6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--p:#1f1f1d;--i:#ecebe6;--m:#a09e97;--l:#34332f;--ok:#4cc38a;--bad:#ff7a6b;--g:#2a2926}}}}
body{{margin:0;padding:20px;background:var(--bg);color:var(--i);font:14px/1.5 -apple-system,Segoe UI,sans-serif}}
.card{{background:var(--p);border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:0 auto 14px;max-width:1200px}}
h3{{margin:0 0 4px;font-size:15px}} h4{{margin:8px 0 4px;font-size:13.5px}} .q{{font-weight:600;margin:4px 0}}
.m{{color:var(--m);font-size:12.5px}} .tag{{font-size:11.5px;background:var(--g);border-radius:99px;padding:1px 8px;margin-left:4px;font-weight:400}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .a{{white-space:pre-wrap;background:var(--g);border-radius:6px;padding:8px;font-size:13px}}
.claims li{{margin:3px 0;font-size:13px}} .claims li.ok::marker{{color:var(--ok)}} .claims li.bad{{color:var(--bad)}} .claims li.part{{color:#b7791f}}
.v.correct{{color:var(--ok)}} .v.incorrect,.bad{{color:var(--bad)}} .src{{font-size:12.5px}}
@media (max-width:800px){{.cols{{grid-template-columns:1fr}}body{{padding:12px}}}}
</style></head><body><div class="card"><h2 style="margin:0">Eval report · {esc(run['run_id'])}</h2>
<p class="m">Claims: default colour = fully supported by a retrieved source, amber = partially supported, red = unsupported. ✓ marks a gold document.</p></div>{''.join(cards)}</body></html>"""


# --------------------------------------------------------------------------- main

def rebuild_context(rag: dict) -> tuple[str, list[str]]:
    """Recreate the exact numbered context a saved RAG answer was generated from (chunk ids -> Chroma)."""
    from langchain_core.documents import Document
    from rag.retrieve import get_vector_store
    ids = [r["id"] for r in rag["retrieved"]]
    if not ids:
        return "", []
    got = get_vector_store()._collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = {i: Document(page_content=d, metadata=m, id=i)
             for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    retrieved = [(by_id[r["id"]], r["score"]) for r in rag["retrieved"]]
    return format_context(retrieved), [d.page_content for d, _ in retrieved]


def audit_agreement(run: dict) -> str | None:
    """Compare the judge with the human audit in eval/manual_audit.yaml (answer level + flagged claims)."""
    path = config.EVAL_DIR / "manual_audit.yaml"
    if not path.exists():
        return None
    audit = yaml.safe_load(path.read_text())
    if audit["run"].split("-", 2)[2] not in run["run_id"]:          # same answers (original or re-judged)
        return None
    rows = {r["id"]: r for r in run["results"]}
    lines = ["## Judge vs manual audit", "",
             f"Human audit of run `{audit['run']}`: {sum(a['claims'] for a in audit['answers'].values())} claims in "
             f"{len(audit['answers'])} answers. An answer counts as flagged if it has ≥1 claim not fully supported.", "",
             "| id | human: unsupported | judge: unsupported / partial | agree? | judge-flagged claims |", "|---|---|---|---|---|"]
    agree = 0
    for qid, a in audit["answers"].items():
        f = rows[qid]["rag"].get("faithfulness") or {"claims": []}
        uns = [c for c in f["claims"] if c["verdict"] == "unsupported"]
        par = [c for c in f["claims"] if c["verdict"] == "partial"]
        human = len(a["unsupported"])
        ok = (human > 0) == bool(uns or par)
        agree += ok
        flagged = "; ".join(f"[{c['verdict']}] {c['claim'][:70]}" for c in uns + par).replace("|", "/")
        lines.append(f"| {qid} | {human} | {len(uns)} / {len(par)} | {'yes' if ok else '**no**'} | {flagged} |")
    lines += ["", f"Answer-level agreement: **{agree}/{len(audit['answers'])}**", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--strategy", default=config.DEFAULT_STRATEGY)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=2, help="parallel judge calls (gpt-4.1 has a 30K tokens/min limit)")
    ap.add_argument("--rejudge", metavar="RUN_ID", help="re-grade a saved run's answers without regenerating them")
    ap.add_argument("--keep-faithfulness", action="store_true",
                    help="with --rejudge: re-run only the correctness judge, reuse the run's faithfulness grades")
    args = ap.parse_args()

    questions = yaml.safe_load((config.EVAL_DIR / "golden_set.yaml").read_text())["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(BASELINE_CACHE.read_text()) if BASELINE_CACHE.exists() else {}
    t0 = time.perf_counter()
    source = json.loads((RESULTS_DIR / f"{args.rejudge}.json").read_text()) if args.rejudge else None
    label = f"{source['run_id'].split('-', 2)[2].removesuffix('-rejudged')}-rejudged" if source else args.label
    run_id = f"{datetime.now():%Y%m%d-%H%M}-{label}"

    with get_usage_metadata_callback() as usage:
        rows = []
        if source:
            # Re-judge: keep answers, retrieval and timings; rebuild each answer's context from Chroma.
            wanted = {q["id"] for q in questions}
            for r in source["results"]:
                if r["id"] not in wanted:
                    continue
                r["rag"]["_context"], r["rag"]["_texts"] = rebuild_context(r["rag"])
                if args.keep_faithfulness and r["rag"].get("faithfulness"):
                    r["rag"]["_kept_faithfulness"] = r["rag"]["faithfulness"]
                r["rag"].pop("faithfulness", None)
                r["llm"]["grade"] = None
                r["_key"] = baseline_key(next(q for q in questions if q["id"] == r["id"]))
                rows.append(r)
            print(f"Re-judging {len(rows)} saved answers from {source['run_id']}")
        else:
            # Stage 1: both systems, one question at a time (honest latency)
            for i, q in enumerate(questions, 1):
                rag = run_rag(q, args.strategy)
                key = baseline_key(q)
                llm = cache[key] if key in cache else {**llm_only_answer(q["question"]), "grade": None}
                rows.append({**{k: q.get(k) for k in ("id", "question", "category", "audience", "expected_behavior",
                                                       "gold_docs", "reference_answer", "key_facts")},
                             "rag": rag, "llm": llm, "_key": key,
                             "retrieval": retrieval_metrics(q, rag) if q["expected_behavior"] == "answer" else {}})
                state = "refused" if rag["refused"] else "answered"
                print(f"[{i:>2}/{len(questions)}] {q['id']} RAG {state:<8} {rag['timings']['total_s']:.1f}s | LLM {llm['timings']['total_s']:.1f}s")

        # Stage 2: judges, in parallel
        def grade(row):
            q = next(x for x in questions if x["id"] == row["id"])
            row["rag"]["grade"] = judge_correctness(q, row["rag"]["answer"])
            if row["rag"].get("_kept_faithfulness"):
                row["rag"]["faithfulness"] = row["rag"].pop("_kept_faithfulness")
            elif not row["rag"]["refused"]:
                row["rag"]["faithfulness"] = judge_faithfulness(row["rag"]["_context"], row["rag"]["answer"])
            if row["llm"].get("grade") is None:
                row["llm"]["grade"] = judge_correctness(q, row["llm"]["answer"])
            if q["expected_behavior"] == "refuse":
                # No ground-truth answer exists in the corpus, so "incorrect claims" would just be
                # the judge flagging true-but-ungrounded facts (e.g. "Tombaugh, 1930"). Only
                # declining is meaningful; a non-declined answer is labelled as answered from memory.
                for side in ("rag", "llm"):
                    g = row[side]["grade"]
                    g["incorrect_claims"] = []
                    g["verdict"] = "declined" if g["declined"] else "answered_from_memory"
            return row

        print(f"\nJudging {len(rows)} questions with {config.JUDGE_MODEL}…")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # copy the context into each task so the usage callback also counts judge calls
            futures = [pool.submit(contextvars.copy_context().run, grade, row) for row in rows]
            rows = [f.result() for f in futures]

    for r in rows:  # update the LLM-only cache; drop private fields
        cache[r.pop("_key")] = r["llm"]
        r["rag"].pop("_context", None)
        r["rag"].pop("_texts", None)
    BASELINE_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    settings = source["settings"] if source else {
        "strategy": args.strategy, "chat_model": config.CHAT_MODEL, "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K, "min_similarity": config.MIN_SIMILARITY, "chunk_size": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP_TOKENS, "context_header": config.CHUNK_CONTEXT_HEADER}
    run = {
        "run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
        **({"rejudged_from": source["run_id"]} if source else {}),
        "settings": {**settings, "judge_model": config.JUDGE_MODEL},
        "usage": {m: {k: v for k, v in u.items() if k in ("input_tokens", "output_tokens")}
                  for m, u in usage.usage_metadata.items()},
        "duration_s": round(time.perf_counter() - t0, 1),
        "summary": summarize(rows), "results": rows,
    }
    (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    md = render_markdown(run)
    agreement = audit_agreement(run)
    if agreement:
        md += "\n" + agreement
    (RESULTS_DIR / f"{run_id}.md").write_text(md)
    html_path = config.PROCESSED_DIR / f"eval_report_{run_id}.html"
    html_path.write_text(render_html(run))

    print("\n" + md.split("## By category")[0])
    if agreement:
        print(agreement)
    print(f"Saved eval/results/{run_id}.json and .md · claim-level report: {html_path}")
    print(f"Duration {run['duration_s']}s · tokens {run['usage']}")


if __name__ == "__main__":
    main()
