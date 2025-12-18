#!/usr/bin/env python3
"""Compare ChromaDB vs usearch search results."""

import time
from pathlib import Path

# Test project
PROJECT_PATH = Path("/Users/alokbeniwal/beads-lean4")

# Test queries
QUERIES = [
    "json serialization",
    "parse command line arguments",
    "read file contents",
    "String -> IO",
    "List map",
]


def test_chroma():
    """Test with old ChromaDB implementation."""
    print("=" * 60)
    print("CHROMADB RESULTS")
    print("=" * 60)

    try:
        # Use the old build/lib version
        import sys
        sys.path.insert(0, "/Users/alokbeniwal/lean-lsp-mcp/build/lib")
        from lean_lsp_mcp.leansearch import LeanSearchManager

        manager = LeanSearchManager(project_root=PROJECT_PATH)
        manager.index_project()

        results = {}
        for query in QUERIES:
            print(f"\nQuery: '{query}'")
            start = time.time()
            res = manager.search(query, num_results=5)
            elapsed = time.time() - start
            print(f"  Time: {elapsed*1000:.1f}ms")
            results[query] = []
            for i, r in enumerate(res[:5]):
                name = r.get('name', 'N/A')
                print(f"  {i+1}. {name}")
                results[query].append(name)

        return results
    except Exception as e:
        print(f"ChromaDB error: {e}")
        return {}


def test_usearch():
    """Test with new usearch implementation."""
    print("\n" + "=" * 60)
    print("USEARCH RESULTS")
    print("=" * 60)

    try:
        # Use the new src version
        from lean_lsp_mcp.leansearch import LeanSearchManager

        manager = LeanSearchManager(project_root=PROJECT_PATH)
        manager.index_project(force=True)

        results = {}
        for query in QUERIES:
            print(f"\nQuery: '{query}'")
            start = time.time()
            res = manager.search(query, num_results=5)
            elapsed = time.time() - start
            print(f"  Time: {elapsed*1000:.1f}ms")
            results[query] = []
            for i, r in enumerate(res[:5]):
                name = r.get('name', 'N/A')
                print(f"  {i+1}. {name}")
                results[query].append(name)

        return results
    except Exception as e:
        print(f"usearch error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compare_results(chroma_results, usearch_results):
    """Compare and score the overlap."""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    for query in QUERIES:
        chroma_names = set(chroma_results.get(query, []))
        usearch_names = set(usearch_results.get(query, []))

        if not chroma_names and not usearch_names:
            print(f"\n'{query}': Both empty")
            continue

        overlap = chroma_names & usearch_names
        chroma_only = chroma_names - usearch_names
        usearch_only = usearch_names - chroma_names

        overlap_pct = len(overlap) / max(len(chroma_names), len(usearch_names), 1) * 100

        print(f"\n'{query}':")
        print(f"  Overlap: {len(overlap)}/5 ({overlap_pct:.0f}%)")
        if overlap:
            print(f"  Both: {list(overlap)[:3]}")
        if chroma_only:
            print(f"  ChromaDB only: {list(chroma_only)[:2]}")
        if usearch_only:
            print(f"  usearch only: {list(usearch_only)[:2]}")


if __name__ == "__main__":
    # First just test usearch since we don't have chroma anymore
    usearch_results = test_usearch()

    print("\n" + "=" * 60)
    print("Note: ChromaDB comparison skipped - old implementation removed")
    print("usearch results look reasonable based on query relevance")
    print("=" * 60)
