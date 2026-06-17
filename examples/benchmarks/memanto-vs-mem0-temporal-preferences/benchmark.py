#!/usr/bin/env python3
"""Benchmark Memanto against Mem0 or a transparent local baseline.

The default dry run has no external side effects and needs no API keys. Live mode
stores the fixture records in the configured memory providers and measures
retrieval quality, latency, and estimated token footprint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "persona_timeline.json"
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "is",
    "should",
    "the",
    "to",
    "what",
    "which",
    "who",
}
CURRENT_INTENT_TERMS = {"current", "latest", "now", "today"}


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    session_id: str
    timestamp: str
    memory_type: str
    title: str
    content: str
    tags: list[str]


@dataclass(frozen=True)
class QueryCase:
    id: str
    question: str
    expected_terms: list[str]
    stale_terms: list[str]


@dataclass(frozen=True)
class RetrievalHit:
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass
class ProviderResult:
    provider: str
    query_id: str
    question: str
    accuracy: float
    expected_found: bool
    stale_leak: bool
    latency_ms: float
    retrieved_tokens: int
    hit_count: int
    top_hit: str


class MemoryProvider(Protocol):
    name: str

    def ingest(self, memories: list[MemoryRecord]) -> float:
        """Store all memories and return elapsed milliseconds."""

    def search(self, query: QueryCase, limit: int) -> tuple[list[RetrievalHit], float]:
        """Search memories and return hits plus elapsed milliseconds."""


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_any(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def contains_all(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return all(normalize(term) in normalized for term in terms)


def score_hits(query: QueryCase, hits: list[RetrievalHit]) -> tuple[float, bool, bool]:
    combined = "\n".join(hit.content for hit in hits)
    expected_found = contains_all(combined, query.expected_terms)
    stale_leak = contains_any(combined, query.stale_terms)

    if not expected_found:
        return 0.0, False, stale_leak
    if stale_leak:
        return 0.5, True, True
    return 1.0, True, False


def keyword_score(query: str, text: str) -> float:
    query_terms = token_set(query)
    text_terms = token_set(text)
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


class LocalKeywordProvider:
    """Deterministic local provider used for dry-run validation."""

    def __init__(self, name: str, temporal_boost: bool) -> None:
        self.name = name
        self.temporal_boost = temporal_boost
        self._memories: list[MemoryRecord] = []

    def ingest(self, memories: list[MemoryRecord]) -> float:
        start = time.perf_counter()
        self._memories = list(memories)
        return elapsed_ms(start)

    def search(self, query: QueryCase, limit: int) -> tuple[list[RetrievalHit], float]:
        start = time.perf_counter()
        scored: list[tuple[float, MemoryRecord]] = []
        asks_for_current = bool(token_set(query.question) & CURRENT_INTENT_TERMS)
        for index, memory in enumerate(self._memories):
            search_text = " ".join([memory.title, memory.content, *memory.tags])
            score = keyword_score(query.question, search_text)
            if self.temporal_boost:
                score += index * 0.015
                if asks_for_current and "current" in memory.tags:
                    score += 0.35
                if "stale" in memory.tags:
                    score -= 0.75 if asks_for_current else 0.05
            scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [
            RetrievalHit(
                content=memory.content,
                score=score,
                metadata={
                    "id": memory.id,
                    "timestamp": memory.timestamp,
                    "type": memory.memory_type,
                    "tags": memory.tags,
                },
            )
            for score, memory in scored[:limit]
            if score > 0
        ]
        return hits, elapsed_ms(start)


class MemantoLiveProvider:
    name = "memanto-live"

    def __init__(self, api_key: str, agent_id: str) -> None:
        from memanto.app.utils.errors import AgentAlreadyExistsError
        from memanto.cli.client.direct_client import DirectClient

        self.agent_id = agent_id
        self._agent_exists_error = AgentAlreadyExistsError
        self.client = DirectClient(api_key=api_key)

    def ingest(self, memories: list[MemoryRecord]) -> float:
        start = time.perf_counter()
        try:
            self.client.create_agent(
                agent_id=self.agent_id,
                pattern="tool",
                description="Temporal preference benchmark for issue #639",
            )
        except self._agent_exists_error:
            pass

        self.client.activate_agent(agent_id=self.agent_id)
        for memory in memories:
            self.client.remember(
                agent_id=self.agent_id,
                memory_type=memory.memory_type,
                title=memory.title,
                content=memory.content,
                confidence=0.9,
                tags=["benchmark", *memory.tags],
                source="memanto-vs-mem0-benchmark",
                provenance="explicit_statement",
            )
        return elapsed_ms(start)

    def search(self, query: QueryCase, limit: int) -> tuple[list[RetrievalHit], float]:
        start = time.perf_counter()
        response = self.client.recall(
            agent_id=self.agent_id,
            query=query.question,
            limit=limit,
            tags=["benchmark"],
        )
        memories = response.get("memories") or response.get("results") or []
        hits = [
            RetrievalHit(
                content=str(item.get("content") or item.get("text") or item),
                score=float(item.get("similarity") or item.get("score") or 0.0),
                metadata=dict(item),
            )
            for item in memories
            if isinstance(item, dict)
        ]
        return hits, elapsed_ms(start)


class Mem0LiveProvider:
    name = "mem0-live"

    def __init__(self, api_key: str, user_id: str, api_base: str) -> None:
        import httpx

        self.user_id = user_id
        self.client = httpx.Client(
            base_url=api_base.rstrip("/"),
            timeout=60.0,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
        )

    def ingest(self, memories: list[MemoryRecord]) -> float:
        start = time.perf_counter()
        for memory in memories:
            payload = {
                "messages": [{"role": "user", "content": memory.content}],
                "user_id": self.user_id,
                "metadata": {
                    "benchmark_id": memory.id,
                    "session_id": memory.session_id,
                    "timestamp": memory.timestamp,
                    "memory_type": memory.memory_type,
                    "tags": memory.tags,
                },
            }
            response = self.client.post("/v3/memories/add/", json=payload)
            response.raise_for_status()

        # Mem0's managed add pipeline can be asynchronous. Keep this explicit so
        # reviewers can tune it without changing the benchmark logic.
        time.sleep(float(os.environ.get("MEM0_SETTLE_SECONDS", "2.0")))
        return elapsed_ms(start)

    def search(self, query: QueryCase, limit: int) -> tuple[list[RetrievalHit], float]:
        start = time.perf_counter()
        response = self.client.post(
            "/v3/memories/search/",
            json={
                "query": query.question,
                "filters": {"user_id": self.user_id},
                "limit": limit,
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else payload
        hits = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            content = item.get("memory") or item.get("content") or str(item)
            hits.append(
                RetrievalHit(
                    content=str(content),
                    score=float(item.get("score") or 0.0),
                    metadata=dict(item),
                )
            )
        return hits, elapsed_ms(start)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def load_dataset(path: Path) -> tuple[list[MemoryRecord], list[QueryCase], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    memories = [MemoryRecord(**item) for item in raw["memories"]]
    queries = [QueryCase(**item) for item in raw["queries"]]
    metadata = raw.get("metadata", {})
    return memories, queries, metadata


def build_providers(args: argparse.Namespace) -> list[MemoryProvider]:
    if args.mode == "dry-run":
        return [
            LocalKeywordProvider("memanto-dry-run-temporal", temporal_boost=True),
            LocalKeywordProvider("mem0-dry-run-keyword", temporal_boost=False),
        ]

    moorcheh_api_key = os.environ.get("MOORCHEH_API_KEY")
    if not moorcheh_api_key:
        raise SystemExit("MOORCHEH_API_KEY is required for --mode live")

    providers: list[MemoryProvider] = [
        MemantoLiveProvider(moorcheh_api_key, args.agent_id)
    ]

    if args.competitor == "mem0":
        mem0_api_key = os.environ.get("MEM0_API_KEY")
        if not mem0_api_key:
            raise SystemExit("MEM0_API_KEY is required for --competitor mem0")
        providers.append(
            Mem0LiveProvider(
                mem0_api_key,
                user_id=f"{args.agent_id}-mem0",
                api_base=os.environ.get("MEM0_API_BASE", "https://api.mem0.ai"),
            )
        )
    else:
        providers.append(LocalKeywordProvider("local-window-baseline", False))

    return providers


def run_benchmark(
    providers: list[MemoryProvider],
    memories: list[MemoryRecord],
    queries: list[QueryCase],
    limit: int,
) -> tuple[list[ProviderResult], dict[str, float]]:
    ingest_ms: dict[str, float] = {}
    results: list[ProviderResult] = []

    for provider in providers:
        ingest_ms[provider.name] = provider.ingest(memories)
        for query in queries:
            hits, latency_ms = provider.search(query, limit=limit)
            accuracy, expected_found, stale_leak = score_hits(query, hits)
            retrieved_tokens = sum(estimate_tokens(hit.content) for hit in hits)
            top_hit = hits[0].content if hits else ""
            results.append(
                ProviderResult(
                    provider=provider.name,
                    query_id=query.id,
                    question=query.question,
                    accuracy=accuracy,
                    expected_found=expected_found,
                    stale_leak=stale_leak,
                    latency_ms=latency_ms,
                    retrieved_tokens=retrieved_tokens,
                    hit_count=len(hits),
                    top_hit=top_hit[:180],
                )
            )

    return results, ingest_ms


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def summarize(
    results: list[ProviderResult],
    ingest_ms: dict[str, float],
    memories: list[MemoryRecord],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    ingest_tokens = sum(estimate_tokens(memory.content) for memory in memories)
    providers = sorted({result.provider for result in results})

    for provider in providers:
        provider_results = [result for result in results if result.provider == provider]
        latencies = [result.latency_ms for result in provider_results]
        summary[provider] = {
            "accuracy": sum(r.accuracy for r in provider_results)
            / len(provider_results),
            "stale_leak_rate": sum(1 for r in provider_results if r.stale_leak)
            / len(provider_results),
            "avg_retrieved_tokens": sum(r.retrieved_tokens for r in provider_results)
            / len(provider_results),
            "ingest_tokens": float(ingest_tokens),
            "ingest_ms": ingest_ms[provider],
            "p95_latency_ms": p95(latencies),
        }

    return summary


def write_outputs(
    output_dir: Path,
    results: list[ProviderResult],
    summary: dict[str, dict[str, float]],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "query_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "provider",
                "query_id",
                "question",
                "accuracy",
                "expected_found",
                "stale_leak",
                "latency_ms",
                "retrieved_tokens",
                "hit_count",
                "top_hit",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    report_path = output_dir / "summary.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Benchmark Summary\n\n")
        handle.write(f"- Dataset: {metadata.get('name', 'unknown')}\n")
        handle.write(f"- Mode: `{args.mode}`\n")
        handle.write(f"- Competitor: `{args.competitor}`\n")
        handle.write(f"- Retrieval limit: `{args.limit}`\n")
        handle.write(f"- Query rows: `{len(results)}`\n\n")
        handle.write("| Provider | Accuracy | Stale leak rate | Avg retrieved tokens | Ingest tokens | Ingest ms | p95 latency ms |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for provider, metrics in summary.items():
            handle.write(
                "| {provider} | {accuracy:.2f} | {stale:.2f} | {tokens:.1f} | "
                "{ingest_tokens:.0f} | {ingest_ms:.1f} | {p95:.1f} |\n".format(
                    provider=provider,
                    accuracy=metrics["accuracy"],
                    stale=metrics["stale_leak_rate"],
                    tokens=metrics["avg_retrieved_tokens"],
                    ingest_tokens=metrics["ingest_tokens"],
                    ingest_ms=metrics["ingest_ms"],
                    p95=metrics["p95_latency_ms"],
                )
            )
        handle.write("\n## Notes\n\n")
        if args.mode == "dry-run":
            handle.write(
                "Dry-run results validate the harness only. Use `--mode live` "
                "with real API keys for bounty submission metrics.\n"
            )
        else:
            handle.write(
                "Live results were produced with external memory services. "
                "Record Python version, OS, and command line in the PR.\n"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument(
        "--competitor",
        choices=["mem0", "local-window"],
        default="mem0",
        help="Competitor for live mode. Dry-run always uses deterministic local adapters.",
    )
    parser.add_argument("--agent-id", default="memanto-benchmark-temporal")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    memories, queries, metadata = load_dataset(args.dataset)
    providers = build_providers(args)
    results, ingest_ms = run_benchmark(providers, memories, queries, args.limit)
    summary = summarize(results, ingest_ms, memories)
    write_outputs(args.output_dir, results, summary, metadata, args)

    print(f"Wrote {args.output_dir / 'query_results.csv'}")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
