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

# ── New: claim/dependency/relationship-grounded metrics ──────────────────────
# Replaces reliance on ground_truth_overlap (pure free-text keyword overlap
# against ANY ground-truth question, with no check that the claim,
# dependency, or patent were actually the right ones). These metrics key off
# the same structured fields the pipeline already produces: raw_provenance's
# claim_text, reasoning_path, and relationship.

GT_RISK_TYPE_TO_RELATIONSHIP = {
    "Proprietary Claim Mismatch": "CONTRADICTS",
    "IP Overlap": "PATENT_OVERLAP",
    "Commercial License": "LICENSE",
    "Academic Prior Art": "PRIOR_ART",
    "Architectural Dependency": "ARCHITECTURAL_DEPENDENCY",
}

_SIGNIFICANT_WORD_MIN_LEN = 4


def _significant_words(text: str) -> set[str]:
    return {
        w.strip(".,;:!?\"'()")
        for w in (text or "").lower().split()
        if len(w.strip(".,;:!?\"'()")) >= _SIGNIFICANT_WORD_MIN_LEN
    }


def _find_best_matching_question(gt_entry: dict, questions: Sequence[dict]) -> dict | None:
    """
    Find the generated question that corresponds to this ground-truth entry.

    A match requires BOTH:
      (a) the question's claim text substantially overlaps the ground-truth
          claim (>=2 shared significant words), AND
      (b) the ground-truth's expected_dependency literally appears somewhere
          in the question's reasoning_path.

    Claim-text similarity alone is not sufficient — two different claims in
    the same pitch deck can share vocabulary ("proprietary", "in-house").
    The dependency must actually be present, or this isn't the same
    underlying finding, just similar-sounding marketing language.
    """
    gt_claim = gt_entry.get("claim", "")
    expected_dep = (gt_entry.get("expected_dependency") or "").lower()
    gt_words = _significant_words(gt_claim)

    best = None
    best_overlap = 0
    for q in questions:
        prov = q.get("raw_provenance", {})
        claim_text = prov.get("claim_text", "") or prov.get("claim_id", "")
        path = [str(n).lower() for n in prov.get("reasoning_path", [])]

        dep_present = bool(expected_dep) and any(expected_dep in n for n in path)
        if not dep_present:
            continue

        overlap = len(gt_words & _significant_words(claim_text))
        if overlap > best_overlap:
            best_overlap = overlap
            best = q

    return best if best_overlap >= 2 else None


def chain_recall(questions: Sequence[dict], ground_truth_entries: Sequence[dict]) -> dict:
    """
    Fraction of ground-truth (claim, expected_dependency) chains that the
    pipeline actually discovered — i.e. some generated question's
    reasoning_path traces back to a matching claim AND passes through the
    expected dependency node. This is the primary "did we find the hidden
    dependency" metric; ground_truth_overlap (question-text keyword overlap)
    never checked this.
    """
    if not ground_truth_entries:
        return {"chain_recall": 0.0, "matched": 0, "total": 0, "unmatched_ids": []}

    matched = 0
    unmatched_ids = []
    for gt in ground_truth_entries:
        if _find_best_matching_question(gt, questions) is not None:
            matched += 1
        else:
            unmatched_ids.append(gt.get("id", "?"))

    return {
        "chain_recall": round(matched / len(ground_truth_entries), 3),
        "matched": matched,
        "total": len(ground_truth_entries),
        "unmatched_ids": unmatched_ids,
    }


def evidence_precision(questions: Sequence[dict], graph: dict) -> dict:
    """
    Fraction of generated questions whose ENTIRE reasoning_path is verifiably
    present in the fused knowledge graph: every node exists, and every
    consecutive pair of nodes is connected by a real edge. Catches stale or
    fabricated paths that made it into question output — this is a
    regression check, not just a scoring metric.
    """
    node_ids = {n.get("id") for n in graph.get("nodes", [])}
    edge_pairs = set()
    for e in graph.get("edges", graph.get("links", [])):
        edge_pairs.add((e.get("source"), e.get("target")))

    if not questions:
        return {"evidence_precision": 0.0, "verified": 0, "total": 0}

    verified = 0
    unverified_ids = []
    for q in questions:
        path = q.get("raw_provenance", {}).get("reasoning_path", [])
        if not path:
            unverified_ids.append(q.get("chain_id", "?"))
            continue
        nodes_ok = all(n in node_ids for n in path)
        edges_ok = all(
            (path[i], path[i + 1]) in edge_pairs or (path[i + 1], path[i]) in edge_pairs
            for i in range(len(path) - 1)
        )
        if nodes_ok and edges_ok:
            verified += 1
        else:
            unverified_ids.append(q.get("chain_id", "?"))

    return {
        "evidence_precision": round(verified / len(questions), 3),
        "verified": verified,
        "total": len(questions),
        "unverified_ids": unverified_ids,
    }


def question_faithfulness(questions: Sequence[dict]) -> dict:
    """
    Fraction of the FINAL question set that passes the same grounding checks
    used as a generation-time filter (validate_question_grounding,
    question_has_fabricated_patent), reported here as a corpus-level
    regression metric. This catches drift even when every individual
    question's provider_used field looks fine (e.g. a future change to the
    validator's own logic that silently weakens it).
    """
    try:
        from generation.question_gen import (
            validate_question_grounding,
            question_has_fabricated_patent,
        )
    except ImportError:
        return {"question_faithfulness": None, "note": "question_gen module not importable"}

    if not questions:
        return {"question_faithfulness": 0.0, "faithful": 0, "total": 0}

    faithful = 0
    unfaithful_ids = []
    for q in questions:
        text = q.get("question", "")
        prov = q.get("raw_provenance", {})
        ok = validate_question_grounding(text, prov) and not question_has_fabricated_patent(text, prov)
        if ok:
            faithful += 1
        else:
            unfaithful_ids.append(q.get("chain_id", "?"))

    return {
        "question_faithfulness": round(faithful / len(questions), 3),
        "faithful": faithful,
        "total": len(questions),
        "unfaithful_ids": unfaithful_ids,
    }


def risk_classification_accuracy(questions: Sequence[dict], ground_truth_entries: Sequence[dict]) -> dict:
    """
    Among ground-truth chains that WERE recalled (see chain_recall), what
    fraction were assigned the correct relationship type? Ground-truth
    risk_type strings are mapped to the canonical relationship taxonomy via
    GT_RISK_TYPE_TO_RELATIONSHIP — this reuses the same categories
    ground_truth.json already labels each entry with, so no manual
    re-annotation of existing entries is required.
    """
    if not ground_truth_entries:
        return {"risk_classification_accuracy": 0.0, "correct": 0, "total_recalled": 0}

    correct = 0
    total_recalled = 0
    mismatches = []
    for gt in ground_truth_entries:
        hit = _find_best_matching_question(gt, questions)
        if hit is None:
            continue
        total_recalled += 1
        expected_relationship = GT_RISK_TYPE_TO_RELATIONSHIP.get(gt.get("risk_type", ""))
        actual_relationship = hit.get("raw_provenance", {}).get("relationship")
        if expected_relationship and actual_relationship == expected_relationship:
            correct += 1
        else:
            mismatches.append({
                "gt_id": gt.get("id", "?"),
                "expected": expected_relationship,
                "actual": actual_relationship,
            })

    return {
        "risk_classification_accuracy": round(correct / total_recalled, 3) if total_recalled else 0.0,
        "correct": correct,
        "total_recalled": total_recalled,
        "mismatches": mismatches,
    }