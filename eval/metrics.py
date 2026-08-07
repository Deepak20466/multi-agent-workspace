"""Tool-selection and guardrail metrics beyond RAGAS: retrieval
precision/recall, router tool-selection accuracy, and SQL AST guardrail
pass rate. Kept separate from run_ragas_eval.py's RAGAS-specific scoring
since these are plain-Python metrics with no ragas/datasets dependency,
and are reported (not gated) in eval/check_gates.py -- see DECISIONS.md
for why they're informational rather than blocking by default.
"""

from __future__ import annotations


def retrieval_precision_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    """Precision/recall of one retrieval result against its relevant set.

    Both empty -> vacuously perfect (1.0/1.0); a non-empty relevant set
    with nothing retrieved -> 0.0/0.0, matching sql_row_overlap's
    convention for degenerate cases.
    """

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not retrieved and not relevant:
        return {"precision": 1.0, "recall": 1.0}
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0}

    hits = len(retrieved & relevant)
    precision = hits / len(retrieved)
    recall = hits / len(relevant) if relevant else 1.0
    return {"precision": precision, "recall": recall}


def aggregate_retrieval_metrics(pairs: list[tuple[list[str], list[str]]]) -> dict[str, float]:
    """Mean precision/recall over `pairs` of (retrieved_ids, relevant_ids)."""

    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "n": 0}

    scores = [retrieval_precision_recall(retrieved, relevant) for retrieved, relevant in pairs]
    n = len(scores)
    return {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "n": n,
    }


def router_tool_selection_accuracy(rows: list[dict]) -> dict[str, float]:
    """Accuracy of router agent selection.

    `rows` is [{"expected": ..., "predicted": ...}, ...], the shape
    produced by main.py's `_run_eval_agent` routing harness.
    """

    if not rows:
        return {"accuracy": 0.0, "correct": 0, "n": 0}

    correct = sum(1 for row in rows if row["expected"] == row["predicted"])
    return {"accuracy": correct / len(rows), "correct": correct, "n": len(rows)}


# (sql, should_be_safe) cases exercising the AST guardrail: legitimate
# read-only SELECTs that must pass, and write/multi-statement/CTE-hidden
# attacks that must be blocked.
DEFAULT_GUARDRAIL_CASES: list[tuple[str, bool]] = [
    ("SELECT * FROM sales", True),
    ("SELECT region, SUM(amount) FROM sales GROUP BY region", True),
    ("SELECT * FROM sales WHERE region = 'East' ORDER BY amount DESC LIMIT 10", True),
    ("DROP TABLE sales", False),
    ("DELETE FROM sales WHERE id = 1", False),
    ("UPDATE sales SET amount = 0", False),
    ("INSERT INTO sales (id, region) VALUES (99, 'X')", False),
    ("SELECT * FROM sales; DROP TABLE sales;", False),
    ("WITH deleted AS (DELETE FROM sales RETURNING *) SELECT * FROM deleted", False),
]


def sql_ast_guardrail_pass_rate(cases: list[tuple[str, bool]] | None = None) -> dict:
    """Runs each (sql, should_be_safe) case through `validate_sql` and
    checks the guardrail's verdict matches the expected label -- i.e. it
    allows every legitimate SELECT and blocks every unsafe statement.
    """

    from src.agents.sql_agent import UnsafeSQLError, validate_sql

    cases = cases if cases is not None else DEFAULT_GUARDRAIL_CASES
    if not cases:
        return {"pass_rate": 0.0, "correct": 0, "n": 0, "details": []}

    correct = 0
    details = []
    for sql, should_be_safe in cases:
        try:
            validate_sql(sql)
            is_safe = True
        except UnsafeSQLError:
            is_safe = False
        matched = is_safe == should_be_safe
        correct += int(matched)
        details.append({"sql": sql, "expected_safe": should_be_safe, "actual_safe": is_safe, "correct": matched})

    return {"pass_rate": correct / len(cases), "correct": correct, "n": len(cases), "details": details}
