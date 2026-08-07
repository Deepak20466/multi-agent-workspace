"""CI quality gate: fails (exit 1) if any RAGAS metric in eval/results.json
falls below its configured threshold. Used by .github/workflows/ragas_eval.yml
to block merges on regressions in retrieval/generation quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GATES = {
    "faithfulness": float(os.getenv("RAGAS_FAITHFULNESS_GATE", "0.75")),
    "answer_relevancy": float(os.getenv("RAGAS_ANSWER_RELEVANCY_GATE", "0.70")),
    "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_GATE", "0.70")),
}


def check_gates(results_path: str, gates: dict[str, float] | None = None) -> bool:
    gates = gates or GATES
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    passed = True
    for metric, threshold in gates.items():
        score = results.get(metric)
        if score is None:
            print(f"MISSING metric '{metric}' in {results_path}")
            passed = False
            continue
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"{status}: {metric}={score:.3f} (gate={threshold:.3f})")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RAGAS results against quality gates")
    parser.add_argument("--results", default="eval/results.json")
    args = parser.parse_args()

    if not check_gates(args.results):
        print("Quality gate FAILED")
        sys.exit(1)
    print("Quality gate PASSED")


if __name__ == "__main__":
    main()
