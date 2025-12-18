#!/usr/bin/env python3
"""Compare ChromaDB vs usearch search results side by side."""

import subprocess
import sys

PROJECT_PATH = "/Users/alokbeniwal/beads-lean4"
QUERIES = [
    "json serialization",
    "parse command line arguments",
    "read file contents",
    "String -> IO",
    "toJson",
]

# Script to run with old ChromaDB implementation
CHROMA_SCRIPT = '''
import sys
sys.path.insert(0, "/Users/alokbeniwal/lean-lsp-mcp/build/lib")
from pathlib import Path
from lean_lsp_mcp.leansearch import LeanSearchManager

PROJECT = Path("{project}")
QUERIES = {queries}

manager = LeanSearchManager(project_root=PROJECT)
# Use existing index, don't rebuild
try:
    manager.index_project(force=False)
except Exception as e:
    print(f"Index error: {{e}}")
    sys.exit(1)

for query in QUERIES:
    results = manager.search(query, num_results=5)
    names = [r.get("name", "?") for r in results]
    print(f"QUERY:{{query}}")
    for name in names:
        print(f"  {{name}}")
'''.format(project=PROJECT_PATH, queries=repr(QUERIES))

# Script to run with new usearch implementation
USEARCH_SCRIPT = '''
import sys
from pathlib import Path
from lean_lsp_mcp.leansearch import LeanSearchManager

PROJECT = Path("{project}")
QUERIES = {queries}

manager = LeanSearchManager(project_root=PROJECT)
manager.index_project(force=False)

for query in QUERIES:
    results = manager.search(query, num_results=5)
    names = [r.get("name", "?") for r in results]
    print(f"QUERY:{{query}}")
    for name in names:
        print(f"  {{name}}")
'''.format(project=PROJECT_PATH, queries=repr(QUERIES))


def run_script(script_content, label):
    """Run a Python script and return parsed results."""
    result = subprocess.run(
        [sys.executable, "-c", script_content],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"{label} FAILED:")
        print(result.stderr[:500])
        return {}

    # Parse results
    results = {}
    current_query = None
    for line in result.stdout.strip().split("\n"):
        if line.startswith("QUERY:"):
            current_query = line[6:]
            results[current_query] = []
        elif current_query and line.strip():
            results[current_query].append(line.strip())

    return results


def main():
    print("=" * 70)
    print("COMPARING CHROMADB vs USEARCH")
    print("=" * 70)

    print("\nRunning ChromaDB implementation...")
    chroma_results = run_script(CHROMA_SCRIPT, "ChromaDB")

    print("Running usearch implementation...")
    usearch_results = run_script(USEARCH_SCRIPT, "usearch")

    # Compare
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    total_overlap = 0
    total_possible = 0

    for query in QUERIES:
        print(f"\nQuery: '{query}'")

        chroma = chroma_results.get(query, [])
        usearch = usearch_results.get(query, [])

        chroma_set = set(chroma)
        usearch_set = set(usearch)
        overlap = chroma_set & usearch_set

        total_overlap += len(overlap)
        total_possible += max(len(chroma), len(usearch))

        print(f"  ChromaDB ({len(chroma)}):")
        for i, name in enumerate(chroma[:3]):
            marker = "✓" if name in overlap else " "
            print(f"    {marker} {i+1}. {name[:60]}")

        print(f"  usearch ({len(usearch)}):")
        for i, name in enumerate(usearch[:3]):
            marker = "✓" if name in overlap else " "
            print(f"    {marker} {i+1}. {name[:60]}")

        overlap_pct = len(overlap) / max(len(chroma), len(usearch), 1) * 100
        print(f"  Overlap: {len(overlap)}/5 ({overlap_pct:.0f}%)")

    overall = total_overlap / total_possible * 100 if total_possible else 0
    print(f"\n{'=' * 70}")
    print(f"OVERALL OVERLAP: {total_overlap}/{total_possible} ({overall:.0f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
