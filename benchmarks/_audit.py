#!/usr/bin/env python3
"""Audit script — strict FP/FN review of all ≥140q benchmark runs.

Reads every *.jsonl, classifies each question into sure_correct /
sure_FP / sure_FN / sure_wrong / unknown using the same heuristics as
OVERVIEW.md and aggregates the result.

For evidence: counts per-run "✓ans + ✗ev" as FN signal (each such
question proves the gold-supporting chunks were retrieved even if the
verbatim-window detector missed them).
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(s: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(s or ""):
        clean = m.replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(clean))
        except ValueError:
            pass
    return out


def starts_yes_no(s: str) -> str | None:
    s = (s or "").strip().lower()
    for w in ("yes", "no"):
        if s.startswith(w):
            return w
    return None


def first_yes_no_in(s: str, n: int = 200) -> str | None:
    head = (s or "")[:n].lower()
    yi = head.find("yes")
    ni = head.find("no")
    cand = [(yi, "yes"), (ni, "no")]
    cand = [(i, w) for i, w in cand if i >= 0]
    if not cand:
        return None
    cand.sort()
    return cand[0][1]


def classify_question(q: dict) -> str:
    gold = (q.get("answer") or "").strip()
    pred = (q.get("predicted_answer") or "").strip()
    correct = bool(q.get("correct"))
    pred_lower = pred.lower()
    last400 = pred[-400:]
    last300 = pred[-300:]
    last150 = pred[-150:]
    head200 = pred[:200].lower()

    not_found_phrases = [
        "not found in context",
        "i cannot determine",
        "the provided text does not",
        "the provided context does not",
        "no relevant context",
        "context does not contain",
        "information is not provided",
        "cannot be calculated",
        "unable to determine",
    ]
    pred_says_not_found = any(p in pred_lower for p in not_found_phrases)

    gold_yn = starts_yes_no(gold)
    pred_yn = starts_yes_no(pred) or first_yes_no_in(pred, 200)

    if correct:
        # sure_FP: graded ✓ but pred admits not-found OR yes/no flip
        if pred_says_not_found:
            return "sure_FP"
        if gold_yn and pred_yn and gold_yn != pred_yn:
            return "sure_FP"
        return "sure_correct"

    # graded as wrong — check FN heuristics
    # 1) gold answer text (≥10 chars) appears verbatim in last 400 chars of pred
    if len(gold) >= 10 and gold.lower() in last400.lower():
        return "sure_FN"

    # 2) Yes/No gold (<80 chars), pred starts with same Y/N, no contradictory Y/N early
    if gold_yn and len(gold) < 80:
        if pred_yn == gold_yn:
            return "sure_FN"

    # 3) all gold numbers present in last 300 AND at least 1 in last 150
    gold_nums = extract_numbers(gold)
    if gold_nums:
        last300_nums = extract_numbers(last300)
        last150_nums = extract_numbers(last150)
        all_in_last300 = all(
            any(abs(g - p) <= max(0.015 * abs(g), 0.01) for p in last300_nums)
            for g in gold_nums
        )
        any_in_last150 = any(
            any(abs(g - p) <= max(0.015 * abs(g), 0.01) for p in last150_nums)
            for g in gold_nums
        )
        if all_in_last300 and any_in_last150:
            return "sure_FN"

    if pred_says_not_found:
        return "sure_wrong"

    return "unknown"


def audit_run(path: Path) -> dict | None:
    cfg = None
    summary = None
    questions = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type")
        if t == "run_config":
            cfg = r
        elif t == "summary":
            summary = r
        elif t == "question":
            questions.append(r)
    if not (cfg and summary and len(questions) >= 140):
        return None

    buckets = {
        "sure_correct": 0,
        "sure_FP": 0,
        "sure_FN": 0,
        "sure_wrong": 0,
        "unknown": 0,
    }
    # 4-quadrant: correct vs evidence_hit
    quad = {"faithful": 0, "hallucinated": 0, "model_fail": 0, "retrieval_fail": 0}
    ev_fn_questions = 0  # questions where correct=True AND evidence_hit=False
    ev_fn_spans = 0  # spans those questions cover
    for q in questions:
        b = classify_question(q)
        buckets[b] += 1
        c = bool(q.get("correct"))
        e = bool(q.get("evidence_hit"))
        if c and e:
            quad["faithful"] += 1
        elif c and not e:
            quad["hallucinated"] += 1
            ev_fn_questions += 1
            ev_fn_spans += int(q.get("gold_evidence_count") or 0)
        elif (not c) and e:
            quad["model_fail"] += 1
        else:
            quad["retrieval_fail"] += 1

    total = len(questions)
    correct = summary.get("answers_correct", 0)
    real_correct = correct - buckets["sure_FP"] + buckets["sure_FN"]
    real_ans_pct = real_correct / total

    ev_hit = summary.get("evidence_spans_hit", 0)
    ev_total = summary.get("evidence_spans_total", 0)
    real_ev_hit = ev_hit + ev_fn_spans
    real_ev_pct = min(1.0, real_ev_hit / ev_total) if ev_total else 0.0

    return {
        "file": path.name,
        "embed": cfg.get("embed_model"),
        "chunking": cfg.get("chunking"),
        "llm": cfg.get("llm_model"),
        "k": cfg.get("top_k"),
        "total": total,
        "correct": correct,
        "ans_pct": correct / total,
        "real_ans_pct": real_ans_pct,
        "sure_correct": buckets["sure_correct"],
        "sure_FP": buckets["sure_FP"],
        "sure_FN": buckets["sure_FN"],
        "sure_wrong": buckets["sure_wrong"],
        "unknown": buckets["unknown"],
        "doc_hit_pct": summary.get("doc_hit_pct", 0),
        "ev_hit": ev_hit,
        "ev_total": ev_total,
        "ev_pct": summary.get("evidence_hit_pct", 0),
        "real_ev_pct": real_ev_pct,
        "ev_fn_q": ev_fn_questions,
        "ev_fn_spans": ev_fn_spans,
        "quad_faithful": quad["faithful"],
        "quad_hallucinated": quad["hallucinated"],
        "quad_model_fail": quad["model_fail"],
        "quad_retrieval_fail": quad["retrieval_fail"],
        "mrr": summary.get("mrr", 0),
        "noise_avg": summary.get("noise_avg", 0),
        "wall_s": summary.get("wall_time_s", 0),
        "answer_avg_s": summary.get("answer_avg_s", 0),
    }


def main():
    rows = []
    for f in sorted(glob.glob("*.jsonl")):
        r = audit_run(Path(f))
        if r:
            rows.append(r)
    rows.sort(key=lambda x: (-x["real_ans_pct"], -x["real_ev_pct"]))
    Path("_audit.json").write_text(json.dumps(rows, indent=2))
    # also print compact summary
    print(f"# {len(rows)} runs ≥140q audited")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:2d}. {r['embed']:30s} {r['chunking']:30s} {r['llm']:30s} k={r['k']:>3} "
            f"Ans={r['ans_pct']*100:5.1f}%→{r['real_ans_pct']*100:5.1f}% "
            f"(+{r['sure_FN']}/-{r['sure_FP']}, ?={r['unknown']}) "
            f"Ev={r['ev_pct']*100:5.1f}%→{r['real_ev_pct']*100:5.1f}% "
            f"(+{r['ev_fn_q']}q/{r['ev_fn_spans']}sp)"
        )


if __name__ == "__main__":
    main()
