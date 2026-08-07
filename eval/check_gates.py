"""CI quality gate: fails (exit 1) if any metric in eval/results_v3.json
falls below its configured threshold. Used by .github/workflows/ragas_eval.yml
to block merges on regressions in RAG/SQL/doc quality.

Gate keys are dotted paths into the (possibly nested) results dict, e.g.
"rag.faithfulness" or "sql.accuracy". A section whose value is
`{"status": "skipped: ..."}` (no DATABASE_URL, empty testset, ragas
unavailable, ...) is reported as SKIP rather than FAIL -- an
unconfigured dependency in a given environment shouldn't block merges
that don't touch that agent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GATES = {
    "rag.faithfulness": float(os.getenv("RAGAS_FAITHFULNESS_GATE", "0.75")),
    "rag.answer_relevancy": float(os.getenv("RAGAS_ANSWER_RELEVANCY_GATE", "0.70")),
    "rag.context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_GATE", "0.70")),
    "rag.context_recall": float(os.getenv("RAGAS_CONTEXT_RECALL_GATE", "0.60")),
    "sql.accuracy": float(os.getenv("SQL_ACCURACY_GATE", "0.80")),
    "doc.accuracy": float(os.getenv("DOC_ACCURACY_GATE", "0.80")),
}

# Legacy flat results.json (RAG-only, pre-v3.1) used bare metric names
# with no "rag." prefix -- fall back to a matching bare key when the
# dotted lookup below finds nothing, so old results files still work.
_LEGACY_GATES = {
    "rag.faithfulness": "faithfulness",
    "rag.answer_relevancy": "answer_relevancy",
    "rag.context_precision": "context_precision",
}


def _lookup(results: dict, dotted_key: str) -> object | None:
    node = results
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _section_status(results: dict, dotted_key: str) -> str | None:
    """Return the `status` string for the top-level section a gate belongs
    to (e.g. "sql" for "sql.accuracy"), if that section was skipped."""

    section = results.get(dotted_key.split(".")[0])
    if isinstance(section, dict) and isinstance(section.get("status"), str):
        return section["status"]
    return None


def check_gates(results_path: str, gates: dict[str, float] | None = None) -> bool:
    gates = gates or GATES
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    passed = True
    for metric, threshold in gates.items():
        score = _lookup(results, metric)
        if score is None and metric in _LEGACY_GATES:
            score = results.get(_LEGACY_GATES[metric])

        if score is None:
            status = _section_status(results, metric)
            if status is not None:
                print(f"SKIP: {metric} ({status})")
                continue
            print(f"MISSING metric '{metric}' in {results_path}")
            passed = False
            continue

        if not isinstance(score, (int, float)):
            print(f"MISSING metric '{metric}' in {results_path} (got {score!r})")
            passed = False
            continue

        gate_passed = score >= threshold
        if not gate_passed:
            passed = False
        status = "PASS" if gate_passed else "FAIL"
        print(f"{status}: {metric}={score:.3f} (gate={threshold:.3f})")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval results against quality gates")
    parser.add_argument("--results", default="eval/results_v3.json")
    args = parser.parse_args()

    if not check_gates(args.results):
        print("Quality gate FAILED")
        sys.exit(1)
    print("Quality gate PASSED")


if __name__ == "__main__":
    main()
