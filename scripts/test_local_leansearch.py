#!/usr/bin/env python3
"""Test script for local leansearch - demonstrates no rate limiting.

This script:
1. Indexes a Lean project's declarations
2. Runs many rapid searches to prove there's no rate limiting
3. Shows detailed stats and timing

Usage:
    # Test with the test_project
    python scripts/test_local_leansearch.py

    # Test with a custom project
    python scripts/test_local_leansearch.py /path/to/lean/project

    # Force reindex
    python scripts/test_local_leansearch.py --reindex

    # Use OpenAI embeddings (better quality)
    LEAN_EMBEDDING_PROVIDER=openai OPENAI_API_KEY=... python scripts/test_local_leansearch.py
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lean_lsp_mcp.leansearch import (
    LeanSearchManager,
    check_leansearch_available,
    get_cache_dir,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def progress_callback(message: str, current: int, total: int) -> None:
    """Print progress updates."""
    if total > 0:
        pct = (current / total) * 100
        print(f"\r  [{pct:5.1f}%] {message}", end="", flush=True)
    else:
        print(f"\r  {message}", end="", flush=True)


def format_result(result: dict, idx: int) -> str:
    """Format a search result for display."""
    lines = [f"  {idx}. {result['name']}"]
    if result.get("kind"):
        lines[0] += f" ({result['kind']})"
    if result.get("signature"):
        sig = result["signature"][:80]
        if len(result["signature"]) > 80:
            sig += "..."
        lines.append(f"      {sig}")
    if result.get("distance") is not None:
        lines.append(f"      distance: {result['distance']:.4f}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Test local leansearch")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help="Path to Lean project (default: tests/test_project)",
    )
    parser.add_argument(
        "--reindex", "-r", action="store_true", help="Force reindex even if cached"
    )
    parser.add_argument(
        "--queries",
        "-n",
        type=int,
        default=20,
        help="Number of rapid queries to test (default: 20)",
    )
    parser.add_argument(
        "--json", "-j", action="store_true", help="Output results as JSON"
    )
    args = parser.parse_args()

    # Check dependencies
    available, msg = check_leansearch_available()
    if not available:
        print(f"Error: {msg}")
        sys.exit(1)

    # Determine project path
    if args.project_path:
        project_root = Path(args.project_path).resolve()
    else:
        # Use test_project
        project_root = Path(__file__).parent.parent / "tests" / "test_project"
        if not project_root.exists():
            print(f"Default test project not found at {project_root}")
            print("Please specify a Lean project path.")
            sys.exit(1)

    project_root = project_root.resolve()
    print(f"\n{'='*60}")
    print("LOCAL LEANSEARCH TEST")
    print(f"{'='*60}")
    print(f"Project: {project_root}")
    print(f"Cache dir: {get_cache_dir()}")

    # Get embedding provider from env
    provider = os.environ.get("LEAN_EMBEDDING_PROVIDER", "default")
    model = os.environ.get("LEAN_EMBEDDING_MODEL")
    print(f"Embedding provider: {provider}" + (f" ({model})" if model else ""))

    # Create manager
    manager = LeanSearchManager(
        project_root=project_root,
        embedding_provider=provider,
        embedding_model=model,
    )

    # Index project
    print(f"\n--- Indexing ---")
    start_time = time.monotonic()

    if args.reindex:
        print("Forcing reindex...")
        count = manager.reindex(progress_callback=progress_callback)
    else:
        count = manager.index_project(progress_callback=progress_callback)

    print()  # Newline after progress

    elapsed = time.monotonic() - start_time
    print(f"Indexed {count} declarations in {elapsed:.2f}s")

    if manager.stats:
        stats = manager.stats
        print(f"\nIndex Statistics:")
        print(f"  Total files: {stats.total_files}")
        print(f"  Total declarations: {stats.total_declarations}")
        if stats.declarations_by_kind:
            print(f"  By kind:")
            for kind, cnt in sorted(
                stats.declarations_by_kind.items(), key=lambda x: -x[1]
            ):
                print(f"    {kind}: {cnt}")

    if count == 0:
        print("\nNo declarations indexed. Make sure the project has Lean files.")
        sys.exit(1)

    # Test searches
    print(f"\n--- Search Tests ---")

    test_queries = [
        "addition commutative",
        "natural number",
        "list append",
        "theorem proof",
        "function composition",
        "group homomorphism",
        "ring multiplication",
        "set intersection",
        "finite type",
        "category functor",
    ]

    # Run a few example searches
    print("\nExample searches:")
    for query in test_queries[:3]:
        print(f"\n  Query: '{query}'")
        results = manager.search(query, num_results=3)
        if results:
            for idx, r in enumerate(results, 1):
                print(format_result(r, idx))
        else:
            print("    No results")

    # Rate limit test - rapid fire queries
    print(f"\n--- Rate Limit Test ({args.queries} rapid queries) ---")
    print("Remote leansearch.net: 3 requests per 30 seconds")
    print("Local leansearch: UNLIMITED!")

    rapid_start = time.monotonic()
    query_times = []

    for i in range(args.queries):
        query = test_queries[i % len(test_queries)]
        q_start = time.monotonic()
        results = manager.search(query, num_results=5)
        q_elapsed = time.monotonic() - q_start
        query_times.append(q_elapsed)

        # Show progress
        print(f"\r  Query {i+1}/{args.queries}: {q_elapsed*1000:.1f}ms", end="")

    print()  # Newline

    rapid_elapsed = time.monotonic() - rapid_start
    avg_time = sum(query_times) / len(query_times)
    min_time = min(query_times)
    max_time = max(query_times)

    print(f"\n  Results:")
    print(f"    Total time: {rapid_elapsed:.2f}s for {args.queries} queries")
    print(f"    Avg query time: {avg_time*1000:.1f}ms")
    print(f"    Min query time: {min_time*1000:.1f}ms")
    print(f"    Max query time: {max_time*1000:.1f}ms")
    print(f"    Queries/second: {args.queries/rapid_elapsed:.1f}")

    # Compare with remote rate limit
    print(f"\n  Comparison with remote API:")
    print(f"    Remote: 3 req/30s = 0.1 req/s = {30/3:.1f}s per query avg")
    print(f"    Local:  {args.queries/rapid_elapsed:.1f} req/s = {avg_time*1000:.1f}ms per query avg")
    print(f"    Speedup: {(30/3) / avg_time:.0f}x faster!")

    # JSON output if requested
    if args.json:
        print(f"\n--- JSON Output ---")
        output = {
            "project": str(project_root),
            "index": {
                "total_declarations": count,
                "total_files": manager.stats.total_files if manager.stats else 0,
                "index_time_seconds": elapsed,
                "declarations_by_kind": (
                    manager.stats.declarations_by_kind if manager.stats else {}
                ),
            },
            "rate_test": {
                "num_queries": args.queries,
                "total_seconds": rapid_elapsed,
                "avg_ms": avg_time * 1000,
                "min_ms": min_time * 1000,
                "max_ms": max_time * 1000,
                "queries_per_second": args.queries / rapid_elapsed,
            },
        }
        print(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print("SUCCESS! Local leansearch is working with NO RATE LIMITS!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
