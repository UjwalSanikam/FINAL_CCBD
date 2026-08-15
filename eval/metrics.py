"""
metrics.py — Evaluation Metrics for ChainCheck
================================================
Recall@K and audit-trail coverage — the two core research metrics.

Recall@K: for a given query, did the correct source node appear in
          the top-K retrieved results?

AuditTrailCoverage: what fraction of generated questions have a
                    non-empty audit_trail field?
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(
    ranked_results: list[list[str]],
    expected_sources: list[str],
    k: int,
) -> float:
    """
    Fraction of queries where the expected source appears in top-K results.

    Parameters
    ----------
    ranked_results    : list of ranked source-ID lists, one per query
    expected_sources  : the single correct source ID for each query
    k                 : cutoff rank
    """
    if not ranked_results:
        return 0.0
    hits = sum(
        1 for results, expected in zip(ranked_results, expected_sources)
        if expected in results[:k]
    )
    return hits / len(ranked_results)


def recall_at_ks(
    ranked_results: list[list[str]],
    expected_sources: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Return Recall@K for each k in ks."""
    return {
        f"recall_at_{k}": recall_at_k(ranked_results, expected_sources, k)
        for k in ks
    }


def audit_trail_coverage(questions: Sequence[dict]) -> float:
    """
    Fraction of questions that have a non-empty audit_trail.
    A question without an audit trail cannot be verified by a lawyer.
    """
    if not questions:
        return 0.0
    covered = sum(1 for q in questions if q.get("audit_trail"))
    return covered / len(questions)
