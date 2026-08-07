from eval.metrics import (
    DEFAULT_GUARDRAIL_CASES,
    aggregate_retrieval_metrics,
    retrieval_precision_recall,
    router_tool_selection_accuracy,
    sql_ast_guardrail_pass_rate,
)

# ---------------------------------------------------------------------------
# retrieval_precision_recall / aggregate_retrieval_metrics
# ---------------------------------------------------------------------------


def test_retrieval_precision_recall_perfect_match():
    result = retrieval_precision_recall(["a", "b"], ["a", "b"])
    assert result == {"precision": 1.0, "recall": 1.0}


def test_retrieval_precision_recall_partial_overlap():
    result = retrieval_precision_recall(["a", "b", "c"], ["a", "z"])
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1 / 2


def test_retrieval_precision_recall_no_overlap():
    result = retrieval_precision_recall(["a"], ["z"])
    assert result == {"precision": 0.0, "recall": 0.0}


def test_retrieval_precision_recall_both_empty_is_vacuously_perfect():
    assert retrieval_precision_recall([], []) == {"precision": 1.0, "recall": 1.0}


def test_retrieval_precision_recall_nothing_retrieved_but_relevant_exists():
    assert retrieval_precision_recall([], ["a"]) == {"precision": 0.0, "recall": 0.0}


def test_aggregate_retrieval_metrics_averages_across_pairs():
    pairs = [(["a", "b"], ["a", "b"]), (["a"], ["z"])]
    result = aggregate_retrieval_metrics(pairs)
    assert result["n"] == 2
    assert result["precision"] == (1.0 + 0.0) / 2
    assert result["recall"] == (1.0 + 0.0) / 2


def test_aggregate_retrieval_metrics_empty_input():
    assert aggregate_retrieval_metrics([]) == {"precision": 0.0, "recall": 0.0, "n": 0}


# ---------------------------------------------------------------------------
# router_tool_selection_accuracy
# ---------------------------------------------------------------------------


def test_router_tool_selection_accuracy_all_correct():
    rows = [{"expected": "sql", "predicted": "sql"}, {"expected": "rag", "predicted": "rag"}]
    result = router_tool_selection_accuracy(rows)
    assert result == {"accuracy": 1.0, "correct": 2, "n": 2}


def test_router_tool_selection_accuracy_partial():
    rows = [{"expected": "sql", "predicted": "sql"}, {"expected": "rag", "predicted": "web"}]
    result = router_tool_selection_accuracy(rows)
    assert result["accuracy"] == 0.5
    assert result["correct"] == 1


def test_router_tool_selection_accuracy_empty_rows():
    assert router_tool_selection_accuracy([]) == {"accuracy": 0.0, "correct": 0, "n": 0}


# ---------------------------------------------------------------------------
# sql_ast_guardrail_pass_rate
# ---------------------------------------------------------------------------


def test_sql_ast_guardrail_pass_rate_default_cases_all_pass():
    result = sql_ast_guardrail_pass_rate()
    assert result["pass_rate"] == 1.0
    assert result["n"] == len(DEFAULT_GUARDRAIL_CASES)


def test_sql_ast_guardrail_pass_rate_flags_wrong_verdict():
    # A write statement mislabeled as "should be safe" -- the guardrail
    # correctly blocks it, so the *expectation* is wrong, and the case
    # counts against the pass rate.
    cases = [("DROP TABLE sales", True)]
    result = sql_ast_guardrail_pass_rate(cases)
    assert result["pass_rate"] == 0.0
    assert result["details"][0]["actual_safe"] is False


def test_sql_ast_guardrail_pass_rate_empty_cases():
    assert sql_ast_guardrail_pass_rate([]) == {"pass_rate": 0.0, "correct": 0, "n": 0, "details": []}
