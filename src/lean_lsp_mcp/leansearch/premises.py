"""Premise-based search for finding relevant lemmas.

Uses the dependency graph from lean-training-data to find lemmas
that are likely relevant to a given goal state.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import numpy as np

from .models import PremiseGraph
from .indexer import LeanSearchIndex

logger = logging.getLogger(__name__)


class LocalPremiseSearch:
    """Search for relevant lemmas using premise graph and semantic search.

    Combines two strategies:
    1. Reverse dependency lookup: Find what uses the constants in the goal
    2. Semantic search: Find semantically similar declarations

    This provides local, rate-limit-free premise suggestions similar to
    lean_state_search and lean_hammer_premise but using local data.
    """

    def __init__(self, graph: PremiseGraph, index: LeanSearchIndex):
        """Initialize with premise graph and search index.

        Args:
            graph: PremiseGraph built from lean-training-data premises
            index: LeanSearchIndex for semantic search
        """
        self.graph = graph
        self.index = index

    def search_by_goal(
        self,
        goal_state: str,
        embed_fn: Callable[[list[str]], np.ndarray],
        num_results: int = 10,
        semantic_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find lemmas relevant to closing a goal.

        Args:
            goal_state: The goal state text (e.g., "h : n < m\n⊢ n + 1 < m + 1")
            embed_fn: Function to compute embeddings
            num_results: Number of results to return
            semantic_weight: Weight for semantic vs graph results (0-1)

        Returns:
            List of relevant declarations with scores
        """
        # 1. Extract constants from goal
        constants = self._extract_constants(goal_state)
        logger.debug(f"Extracted constants: {constants}")

        # 2. Find declarations that use these constants (reverse lookup)
        graph_candidates: dict[str, float] = {}
        for const in constants:
            users = self.graph.reverse.get(const, set())
            for user in users:
                # Score by how many goal constants this declaration uses
                graph_candidates[user] = graph_candidates.get(user, 0) + 1

        # Normalize graph scores
        if graph_candidates:
            max_score = max(graph_candidates.values())
            graph_candidates = {k: v / max_score for k, v in graph_candidates.items()}

        # 3. Get semantic matches
        semantic_results = self.index.search_by_text(
            goal_state, embed_fn, k=num_results * 3, kind="theorem"
        )

        # 4. Combine scores
        combined: dict[str, dict[str, Any]] = {}

        # Add graph candidates
        for name, score in graph_candidates.items():
            combined[name] = {
                "name": name,
                "graph_score": score,
                "semantic_score": 0.0,
            }

        # Add semantic results
        for r in semantic_results:
            name = r["name"]
            if name in combined:
                combined[name]["semantic_score"] = r.get("hybrid_score", 0.5)
                combined[name].update(
                    {k: v for k, v in r.items() if k not in combined[name]}
                )
            else:
                combined[name] = {
                    **r,
                    "graph_score": 0.0,
                    "semantic_score": r.get("hybrid_score", 0.5),
                }

        # Compute final scores
        for item in combined.values():
            graph_score = item.get("graph_score", 0.0)
            semantic_score = item.get("semantic_score", 0.0)
            item["score"] = (
                (1 - semantic_weight) * graph_score + semantic_weight * semantic_score
            )

        # Sort and return
        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return results[:num_results]

    def _extract_constants(self, goal: str) -> set[str]:
        """Extract referenced constants from goal state.

        Looks for identifiers that look like Lean constants:
        - Capitalized identifiers (e.g., Nat.add, List.map)
        - Fully qualified names with dots

        Args:
            goal: Goal state text

        Returns:
            Set of constant names
        """
        # Pattern for qualified names like Nat.add_comm, List.map
        qualified = re.findall(r"\b([A-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)\b", goal)

        # Pattern for single capitalized names that might be types/constants
        single = re.findall(r"\b([A-Z][a-zA-Z0-9_]*)\b", goal)

        # Filter out common type parameters and keywords
        excluded = {
            "Type",
            "Prop",
            "Sort",
            "True",
            "False",
            "And",
            "Or",
            "Not",
            "Iff",
            "Exists",
            "Forall",
        }

        constants = set()
        for name in qualified:
            if name not in excluded and not name.startswith("_"):
                constants.add(name)

        for name in single:
            if name not in excluded and len(name) > 2 and not name.startswith("_"):
                constants.add(name)

        return constants

    def get_explicit_premises(self, decl_name: str) -> list[str]:
        """Get explicit premises (marked with *) for a declaration.

        These are the key lemmas that a proof directly invokes.
        """
        edges = self.graph.adjacency.get(decl_name, [])
        return [e.target for e in edges if e.is_explicit]

    def get_simp_premises(self, decl_name: str) -> list[str]:
        """Get simp premises (marked with s) for a declaration.

        These are lemmas used by simp in the proof.
        """
        edges = self.graph.adjacency.get(decl_name, [])
        return [e.target for e in edges if e.is_simp]

    def get_similar_proofs(
        self,
        decl_name: str,
        num_results: int = 5,
    ) -> list[str]:
        """Find declarations with similar premise sets.

        Useful for finding proofs that use similar techniques.
        """
        target_premises = set(e.target for e in self.graph.adjacency.get(decl_name, []))
        if not target_premises:
            return []

        # Find declarations with overlapping premises
        scores: dict[str, float] = {}
        for name, edges in self.graph.adjacency.items():
            if name == decl_name:
                continue
            other_premises = set(e.target for e in edges)
            overlap = len(target_premises & other_premises)
            if overlap > 0:
                # Jaccard similarity
                union = len(target_premises | other_premises)
                scores[name] = overlap / union if union > 0 else 0

        sorted_names = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_names[:num_results]
