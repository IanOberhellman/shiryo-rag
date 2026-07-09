"""Evaluation harness: runs a fixed exam of questions with known answers against
the RAG pipeline and reports a score. Run: python eval.py"""

import json
import os

from rag import answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def grade(case: dict, result: dict) -> tuple[bool, str]:
    if not case["answerable"]:
        # trick question: the system must refuse, not hallucinate
        if not result["answerable"]:
            return True, "correctly refused (not in documents)"
        return False, f"HALLUCINATION RISK: answered a question the docs don't cover: {result['answer'][:80]}"

    if not result["answerable"]:
        return False, "wrongly refused an answerable question"
    missing = [
        kw for kw in case["expected_keywords"]
        if kw.lower() not in result["answer"].lower()
    ]
    if missing:
        return False, f"answer missing expected keyword(s): {missing}"
    if not result["citations"]:
        return False, "no citations provided"
    return True, "answer contains expected facts, with citations"


def main():
    with open(os.path.join(BASE_DIR, "eval_set.json")) as f:
        cases = json.load(f)

    results, passed = [], 0
    total_in = total_out = 0
    for i, case in enumerate(cases, 1):
        result = answer(case["question"])
        ok, reason = grade(case, result)
        passed += ok
        total_in += result["usage"]["input"]
        total_out += result["usage"]["output"]
        status = "PASS" if ok else "FAIL"
        print(f"[{i:2d}/{len(cases)}] {status}  {case['question'][:60]} — {reason}")
        results.append((case, result, ok, reason))

    score = passed / len(cases) * 100
    print(f"\nScore: {passed}/{len(cases)} ({score:.0f}%)  |  tokens: {total_in} in / {total_out} out")

    with open(os.path.join(BASE_DIR, "eval_results.md"), "w") as f:
        f.write("# Eval results\n\n")
        f.write(f"**Score: {passed}/{len(cases)} ({score:.0f}%)** — "
                f"{sum(1 for c, _, ok, _ in results if not c['answerable'] and ok)}/"
                f"{sum(1 for c in cases if not c['answerable'])} trick questions correctly refused\n\n")
        f.write("| # | Question | Type | Result | Notes |\n|---|---|---|---|---|\n")
        for i, (case, result, ok, reason) in enumerate(results, 1):
            qtype = "answerable" if case["answerable"] else "trick (not in docs)"
            f.write(f"| {i} | {case['question']} | {qtype} | {'✅ PASS' if ok else '❌ FAIL'} | {reason} |\n")
    print("Wrote eval_results.md")


if __name__ == "__main__":
    main()
