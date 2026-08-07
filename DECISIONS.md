# Architecture Decision Log

Trade-offs behind the non-obvious choices in this codebase, newest first.
Each entry: the decision, the alternatives considered, and why the chosen
option won. See inline docstrings in the referenced files for the
mechanical "how"; this file is the "why."

## 1. Three-tier reranker cascade: Cohere -> flashrank -> local cross-encoder

**Decision:** `Reranker.arerank` (`src/reranking.py`) tries the Cohere
Rerank API first if `COHERE_API_KEY` is set, then a local `flashrank`
ONNX cross-encoder if `use_flashrank=True`, then falls all the way back
to a `sentence-transformers` `CrossEncoder`. Each tier catches its own
exceptions and falls through to the next rather than raising.

**Why a cascade instead of one reranker:** RRF-fused retrieval alone
gets roughly ~61% top-k accuracy on our benchmark queries because its
score only reflects rank position, not real query-document semantic
interaction. A cross-encoder reranker that scores the query against
each candidate's actual text closes most of that gap (~85%). But no
single reranker is right for every deployment:

- **Cohere** is the strongest of the three but costs a network round
  trip, a per-call fee, and an API key -- unacceptable for the offline/
  air-gapped deployments this project also targets (see decision #4).
- **flashrank** is the new default local tier: an ONNX-runtime
  cross-encoder with no `torch` dependency, sub-100ms inference, and no
  API key. It's the right default for "good accuracy, zero external
  dependency."
- **sentence-transformers CrossEncoder** (the original implementation)
  stays as the last-resort fallback since it needs no extra optional
  package beyond what the project already installs, even though it's
  heavier (pulls in `torch`) and slower to cold-start than flashrank.

**Why fall through instead of erroring:** a reranker failure (network
blip, missing optional dependency, model download failure) must never
take down the whole RAG pipeline -- degraded ranking quality is a much
better failure mode than a 500.

**Trade-off accepted:** `use_flashrank` defaults to `True` in
`config.yaml`, but `Reranker.__init__` itself defaults to `False` so
existing callers/tests that construct a bare `Reranker()` keep the
original two-tier (Cohere/local) behavior unless they opt in. This means
the flashrank tier is invisible unless something explicitly threads the
config value through (as `main.py` does) -- a deliberate choice to keep
the default constructor's behavior stable rather than changing it
silently under existing callers.

## 2. RRF (not a weighted score blend) for hybrid retrieval fusion

**Decision:** `hybrid_retrieval.py` fuses dense (vector) and sparse
(BM25) result lists via alpha-weighted Reciprocal Rank Fusion, not a
linear combination of the two raw scores.

**Why:** cosine similarity (vector) and BM25 scores live on
incomparable scales that shift with corpus/query, so a weighted blend of
raw scores needs constant re-normalization to stay meaningful. RRF only
needs each list's *rank*, which sidesteps that problem entirely. `alpha`
still lets us tune dense-vs-sparse influence (`alpha=1.0` -> vector-only,
`alpha=0.0` -> BM25-only) without touching score normalization.

## 3. SQL ambiguity detection: keyword heuristic, not an LLM clarification call

**Decision:** `detect_ambiguity()` in `src/agents/sql_agent.py` flags
vague superlatives ("top", "best", "most", ...) that lack either a
concrete row count ("top 10") or a named metric/column to rank by, using
plain regex against the question text and the introspected schema
column names -- no LLM call.

**Alternatives considered:** asking an LLM "is this question ambiguous"
before generating SQL. Rejected for now: it doubles LLM latency and cost
on *every* SQL query to catch a fairly narrow failure mode (missing
LIMIT/ORDER-BY target), and a heuristic pre-filter is cheap enough to run
unconditionally.

**Why this matters:** without it, "show top customers" silently becomes
some arbitrary `ORDER BY ... LIMIT 5` the LLM invented -- a
plausible-looking but unverifiable answer. Surfacing
`{"needs_clarification": true, "clarification_question": ...}` and
skipping generation entirely is safer than guessing.

**Trade-off accepted:** heuristics both under- and over-trigger. "Total
sales by region" (no superlative) sails through even though "total"
could theoretically be ambiguous too; conversely a query that names a
column containing one of the trigger words in an unrelated sense would
false-positive. Same category of trade-off already accepted for
prompt-injection detection in `guardrails.py`. `sql.check_ambiguity:
false` in `config.yaml` (or `SQLAgent(check_ambiguity=False)`) opts out
per-deployment if the false-positive rate is too high for a given
schema/workload.

## 4. AST-based SQL validation, not regex blocklisting

**Decision (pre-existing, restated):** `validate_sql()` parses generated
SQL with `sqlglot` into a real AST and walks every node for forbidden
expression types (`Insert`, `Update`, `Delete`, `Drop`, ...), rather than
regex-matching for keywords like `DROP`/`DELETE`.

**Why it still matters here:** the new SQL AST guardrail pass-rate
metric (`eval/metrics.py::sql_ast_guardrail_pass_rate`) exists precisely
to make this guarantee measurable in CI, including the adversarial case
that motivated the AST approach in the first place -- a write hidden
inside a CTE (`WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`),
which a naive regex blocklist on the outermost statement would miss.

## 5. Ollama backend selection is explicit, never auto-detected from API-key presence

**Decision:** `src/llm_factory.py::build_llm()` picks Anthropic vs.
Ollama purely from an explicit `backend` parameter (or the `LLM_BACKEND`
env var / `agents.llm_backend` config key), defaulting to `"anthropic"`.
It does **not** silently switch to Ollama just because
`ANTHROPIC_API_KEY` happens to be unset.

**Alternatives considered:** auto-fallback to Ollama whenever no
Anthropic key is present, marketed as "API key fallback." Rejected:
that makes a request's behavior (and its answer's provenance and
quality) depend on ambient environment state the caller can't see at the
call site, and it makes tests that expect an Anthropic-shaped mock
silently start hitting a different code path in an environment that
happens to lack the key. Explicit configuration means "which model
answered this" is always inferrable from `config.yaml`/env, not from
whichever secret happened to be present at runtime.

**Why offer it at all:** fully offline/air-gapped deployments and local
development without burning API credits are real use cases; Ollama
covers both without changing any agent-level code (`SQLAgent`,
`AgentGraph`'s router classifier) since both call the same factory.

**Known gap:** `RAGAgent`'s answer-generation LLM is still wired as
`llm=None` by default in `main.py` (pre-existing behavior, not changed
by this work) -- so today only the SQL agent and router classifier
actually benefit from the Ollama backend in the deployed app. Wiring
`RAGAgent`'s generation step through the same factory is a natural
follow-up, tracked here rather than done silently as a side effect of
this change.

## 6. New eval metrics are reported, not gated, by default

**Decision:** retrieval precision/recall, router tool-selection
accuracy, and SQL AST guardrail pass rate (`eval/metrics.py`) are
computed and written into `eval/results_v3.json`, but **not** added to
`eval/check_gates.py`'s default `GATES` dict that blocks CI merges.

**Why:** `check_gates.py` treats a missing section as `MISSING` (hard
fail), not `SKIP` -- unlike the RAG/SQL/doc sections, these new metrics
don't have production ground-truth data wired up yet (retrieval
precision/recall needs a `relevant_chunk_ids`-labeled testset that
`eval/generate_testset.py` doesn't produce yet), so gating on them today
would either block every merge on an unpopulated metric or require
gate-shaped placeholder data. Reporting them first lets the numbers
stabilize before anyone decides on a threshold. The SQL guardrail pass
rate is deterministic and could reasonably graduate to gated status once
its case set is reviewed; the retrieval metric needs the labeled-testset
follow-up first.

## 7. Redis for session/cache state, with an explicit in-memory fallback

**Decision (pre-existing, restated for context):** `ResponseCache` and
the LangGraph checkpointer prefer Redis but degrade to in-memory
LRU/`MemorySaver` if Redis is unreachable, rather than failing the
request. Same "degrade, don't crash" philosophy applied throughout this
change set (reranker cascade, LLM backend, ambiguity check being
opt-out-able).
