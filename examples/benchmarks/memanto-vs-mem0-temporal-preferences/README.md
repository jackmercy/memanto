# Memanto vs Mem0 Temporal Preference Benchmark

This benchmark is a reproducible evaluation harness for issue
[#639](https://github.com/moorcheh-ai/memanto/issues/639). It stress-tests the
core memory trade-off from the bounty prompt: retrieval accuracy vs. resource
footprint when user preferences change over time.

The fixture models an executive assistant that must remember current technical
preferences while ignoring stale choices from earlier sessions. The same records
and queries are run through Memanto and a competitor adapter, then scored with a
golden dataset.

## What It Measures

- Retrieval accuracy against expected current facts
- Stale-memory leakage when old preferences contradict current ones
- Estimated token footprint for ingested and retrieved memory context
- Per-query latency and p95 latency
- A machine-readable CSV plus a Markdown summary report

## Quick Dry Run

The dry run needs no API keys. It validates the dataset, scoring, and report
format using deterministic local adapters.

```bash
cd examples/benchmarks/memanto-vs-mem0-temporal-preferences
python benchmark.py --mode dry-run --output-dir results
```

Outputs:

```text
results/query_results.csv
results/summary.md
```

## Live Run

Install dependencies from the repository root:

```bash
pip install -e .
pip install -r examples/benchmarks/memanto-vs-mem0-temporal-preferences/requirements.txt
```

Set environment variables:

```bash
set MOORCHEH_API_KEY=...
set MEM0_API_KEY=...
```

Run Memanto against Mem0:

```bash
python examples/benchmarks/memanto-vs-mem0-temporal-preferences/benchmark.py ^
  --mode live ^
  --competitor mem0 ^
  --output-dir examples/benchmarks/memanto-vs-mem0-temporal-preferences/results-live
```

If you only want to validate Memanto without a Mem0 account, use:

```bash
python examples/benchmarks/memanto-vs-mem0-temporal-preferences/benchmark.py ^
  --mode live ^
  --competitor local-window
```

`local-window` is a transparent baseline that retrieves from the full local
memory log. It is useful for debugging but should not be presented as a
dedicated memory-framework comparison.

## Dataset Design

The dataset in `data/persona_timeline.json` has three parts:

- `memories`: timestamped facts, preferences, decisions, and contradictions
- `queries`: evaluation prompts with expected current facts and stale facts
- `metadata`: reproducibility notes, environment assumptions, and scoring scope

Each query is scored as:

- `1.0`: expected current fact found and stale fact absent
- `0.5`: expected fact found but stale fact also appears in retrieved context
- `0.0`: expected fact missing

This intentionally penalizes context bloat: retrieving both old and new facts
may let an LLM reason its way to the answer, but it increases token footprint
and ambiguity.

## Reproducibility Notes

- Default retrieval limit: `5`
- Token estimate: `ceil(characters / 4)`, applied consistently to all providers
- Latency: measured around each provider's ingest and search calls
- The dry run is not an official benchmark result. It exists so reviewers can
  run the suite without private API keys.
- Live results should include the generated `summary.md` in the PR description.

## Suggested PR Checklist

- Include the dry-run output in the PR.
- If live keys are available, include the live `summary.md` metrics.
- State the exact Python version, operating system, and command line used.
- Link any social write-up required by the bounty.
