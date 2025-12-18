#!/usr/bin/env python3
"""Test leansearch on a real Lean project."""

import asyncio
import time
from pathlib import Path

from lean_lsp_mcp.leansearch import LeanSearchManager, check_leansearch_available

# Test project
PROJECT_PATH = Path("/Users/alokbeniwal/beads-lean4")

def test_availability():
    """Check if dependencies are available."""
    print("=" * 60)
    print("1. CHECKING AVAILABILITY")
    print("=" * 60)
    available, msg = check_leansearch_available()
    print(f"Available: {available}")
    if msg:
        print(f"Message: {msg}")
    return available

def test_indexing(manager: LeanSearchManager):
    """Test indexing performance."""
    print("\n" + "=" * 60)
    print("2. TESTING INDEXING")
    print("=" * 60)

    start = time.time()
    count = manager.index_project(force=True)
    elapsed = time.time() - start

    print(f"Indexed {count} declarations in {elapsed:.2f}s")
    print(f"Rate: {count/elapsed:.1f} declarations/second")

    if manager.stats:
        print(f"\nStats:")
        print(f"  Total files: {manager.stats.total_files}")
        print(f"  By kind: {manager.stats.declarations_by_kind}")
        print(f"  Files added: {manager.stats.files_added}")

    return count, elapsed

def test_incremental_indexing(manager: LeanSearchManager):
    """Test incremental indexing (should be fast)."""
    print("\n" + "=" * 60)
    print("3. TESTING INCREMENTAL INDEXING")
    print("=" * 60)

    start = time.time()
    count = manager.index_project(force=False)
    elapsed = time.time() - start

    print(f"Incremental check took {elapsed:.3f}s")
    print(f"Files unchanged: {manager.stats.files_unchanged if manager.stats else 'N/A'}")

    return elapsed

def test_search_quality(manager: LeanSearchManager):
    """Test search quality with various queries."""
    print("\n" + "=" * 60)
    print("4. TESTING SEARCH QUALITY")
    print("=" * 60)

    queries = [
        # Natural language
        "parse command line arguments",
        "read file contents",
        "json serialization",
        # Identifier-like
        "Issue",
        "parseArgs",
        "toJson",
        # Type signatures
        "String -> IO",
        "List -> Option",
    ]

    results_summary = []

    for query in queries:
        print(f"\nQuery: '{query}'")
        start = time.time()
        results = manager.search(query, num_results=5)
        elapsed = time.time() - start

        print(f"  Time: {elapsed*1000:.1f}ms, Found: {len(results)}")
        for i, r in enumerate(results[:3]):
            name = r.get('name', 'N/A')
            kind = r.get('kind', '?')
            score = r.get('hybrid_score', r.get('distance', 0))
            print(f"  {i+1}. [{kind}] {name} (score: {score:.3f})")

        results_summary.append({
            'query': query,
            'time_ms': elapsed * 1000,
            'count': len(results),
        })

    avg_time = sum(r['time_ms'] for r in results_summary) / len(results_summary)
    print(f"\nAverage search time: {avg_time:.1f}ms")

    return results_summary

def test_goal_search(manager: LeanSearchManager):
    """Test goal-based search."""
    print("\n" + "=" * 60)
    print("5. TESTING GOAL-BASED SEARCH")
    print("=" * 60)

    # Simulated goal states
    goals = [
        "⊢ List.length (List.map f xs) = List.length xs",
        "h : x ∈ xs\n⊢ Option.isSome (List.find? (· == x) xs) = true",
        "⊢ String.length s > 0 → s ≠ \"\"",
    ]

    for goal in goals:
        print(f"\nGoal: {goal[:60]}...")
        start = time.time()
        results = manager.search_by_goal(goal, num_results=5)
        elapsed = time.time() - start

        print(f"  Time: {elapsed*1000:.1f}ms, Found: {len(results)}")
        for i, r in enumerate(results[:3]):
            name = r.get('name', 'N/A')
            print(f"  {i+1}. {name}")

def test_index_size():
    """Check index file sizes."""
    print("\n" + "=" * 60)
    print("6. INDEX SIZE")
    print("=" * 60)

    from lean_lsp_mcp.leansearch.indexer import get_cache_dir
    cache_dir = get_cache_dir()

    if cache_dir.exists():
        total_size = 0
        for f in cache_dir.rglob("*"):
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                print(f"  {f.relative_to(cache_dir)}: {size/1024:.1f} KB")
        print(f"  TOTAL: {total_size/1024:.1f} KB ({total_size/1024/1024:.2f} MB)")

async def main():
    print("LEANSEARCH REAL-WORLD TEST")
    print(f"Project: {PROJECT_PATH}")
    print()

    if not test_availability():
        print("Dependencies not available!")
        return

    # Initialize manager
    manager = LeanSearchManager(project_root=PROJECT_PATH)

    try:
        # Run tests
        count, index_time = test_indexing(manager)

        if count > 0:
            incr_time = test_incremental_indexing(manager)
            test_search_quality(manager)
            test_goal_search(manager)
            test_index_size()

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"  Declarations indexed: {count}")
            print(f"  Full index time: {index_time:.2f}s")
            print(f"  Incremental check: {incr_time*1000:.1f}ms")
        else:
            print("\nNo declarations found - check project path")
    finally:
        manager.close()

if __name__ == "__main__":
    asyncio.run(main())
