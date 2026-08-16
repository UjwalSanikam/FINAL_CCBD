"""
eval_runner.py — ChainCheck Evaluation Harness
===============================================
Computes two metrics:
  1. Recall@K          — does the retriever surface the right source node?
  2. Audit Coverage    — what % of questions have an audit trail?

Also does a keyword-overlap score against ground_truth.json when provided.

Usage
-----
  # Questions evaluation only
  python eval/eval_runner.py --questions data/processed/questions.json

  # Full retrieval + question eval
  python eval/eval_runner.py \\
      --eval-queries eval/eval_queries.json \\
      --questions    data/processed/questions.json \\
      --ground-truth eval/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    audit_trail_coverage,
    recall_at_ks,
    chain_recall,
    evidence_precision,
    question_faithfulness,
    risk_classification_accuracy,
)


def evaluate_retrieval(
    eval_queries_path: Path,
    data_dir: Path,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Evaluate Retriever Recall@K against labeled query/source ground truth."""
    from resolvers.retriever import Retriever

    eval_queries = json.loads(eval_queries_path.read_text(encoding="utf-8"))
    retriever = Retriever(
        index_path=data_dir / "vector.index",
        metadata_path=data_dir / "vector_metadata.json",
    )
    ranked_results = [
        retriever.retrieve(item["query"], top_k=max(ks))
        for item in eval_queries
    ]
    expected_sources = [item["expected_source"] for item in eval_queries]
    return recall_at_ks(ranked_results, expected_sources, ks=ks)


def _auto_detect_startup_id(processed_dir: Path, known_ids: set[str]) -> str | None:
    """
    Infer which startup this run belongs to from the *_parsed.json filename
    the whitepaper parser wrote (e.g. halcyon_ai_pitch_deck_parsed.json),
    matched against the startup_id values actually present in the
    ground-truth file. Falls back to None if nothing matches, which the
    caller treats as "evaluate against everything" with an explicit warning.
    """
    parsed_files = sorted(processed_dir.glob("*_parsed.json"))
    if not parsed_files:
        return None
    stem = parsed_files[0].stem.lower()
    for sid in known_ids:
        if sid and sid.lower() in stem:
            return sid
    return None


def evaluate_questions(
    questions_path: Path,
    ground_truth_path: Path | None = None,
    graph_path: Path | None = None,
    startup_id: str | None = None,
) -> dict:
    """
    Evaluate generated question artifacts. Returns the structural metrics
    (total count, audit-trail presence, question faithfulness, evidence
    precision) unconditionally, and the ground-truth-dependent metrics
    (chain recall, risk-classification accuracy) only when a ground-truth
    file is supplied.
    """
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])

    results: dict = {
        "total_questions": len(questions),
        "audit_trail_coverage": audit_trail_coverage(questions),
        "question_faithfulness": question_faithfulness(questions),
    }

    if graph_path and graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        results["evidence_precision"] = evidence_precision(questions, graph)
    else:
        results["evidence_precision"] = {"note": f"graph not found at {graph_path}"}

    if ground_truth_path and ground_truth_path.exists():
        gt_data = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        all_entries = gt_data.get("questions", [])

        resolved_startup_id = startup_id or _auto_detect_startup_id(
            questions_path.parent, {e.get("startup_id") for e in all_entries if e.get("startup_id")}
        )

        if resolved_startup_id:
            gt_entries = [e for e in all_entries if e.get("startup_id") == resolved_startup_id]
        else:
            gt_entries = all_entries
            print(
                "WARNING: could not determine startup_id for this run — scoring against "
                f"ALL {len(all_entries)} ground-truth entries across every startup. "
                "Pass --startup-id explicitly to scope this correctly.",
                file=sys.stderr,
            )

        results["startup_id_used"] = resolved_startup_id or "ALL (unfiltered)"
        results["chain_recall"] = chain_recall(questions, gt_entries)
        results["risk_classification_accuracy"] = risk_classification_accuracy(questions, gt_entries)
        # Retained for continuity with earlier reported numbers ONLY.
        # This is free-text keyword overlap against any ground-truth
        # question and does not verify claim/dependency correctness —
        # do not use as a primary quality signal. See chain_recall instead.
        results["legacy_lexical_overlap"] = ground_truth_overlap(questions_path, ground_truth_path)

    return results


def ground_truth_overlap(questions_path: Path, gt_path: Path) -> float:
    """
    Keyword overlap between generated questions and ground truth.
    A generated question 'matches' a GT question if they share ≥2 words
    of length ≥4. Returns the fraction of generated questions that match.
    """
    gen_data = json.loads(questions_path.read_text(encoding="utf-8"))
    gt_data  = json.loads(gt_path.read_text(encoding="utf-8"))

    gen_questions = [
        (q.get("question") or q.get("generated_question") or "").lower()
        for q in gen_data.get("questions", [])
    ]
    gt_questions = [
        q["question"].lower()
        for q in gt_data.get("questions", [])
        if "question" in q
    ]

    if not gen_questions or not gt_questions:
        return 0.0

    overlap_count = 0
    for gen_q in gen_questions:
        gen_words = {w for w in gen_q.split() if len(w) >= 4}
        for gt_q in gt_questions:
            gt_words = {w for w in gt_q.split() if len(w) >= 4}
            if len(gen_words & gt_words) >= 2:
                overlap_count += 1
                break

    return round(overlap_count / len(gen_questions), 3)


def print_results(results: dict) -> None:
    print(f"\n{'═'*60}")
    print("  CHAINCHECK EVAL RESULTS")
    print(f"{'═'*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.3f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                fmt = f"{vv:.3f}" if isinstance(vv, float) else str(vv)
                print(f"    {kk:<33} {fmt}")
        else:
            print(f"  {k:<35} {v}")
    print(f"{'═'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChainCheck evaluation runner")
    parser.add_argument("--data-dir",     default="data/processed")
    parser.add_argument("--eval-queries", default=None,
                        help="Path to eval_queries.json for Recall@K")
    parser.add_argument("--questions",    default=None,
                        help="Path to generated questions.json")
    parser.add_argument("--ground-truth", default="eval/ground_truth.json")
    parser.add_argument("--startup-id",   default=None,
                        help="Which startup's ground-truth entries to score against "
                             "(e.g. 'vaultchain', 'halcyon'). Auto-detected from the "
                             "*_parsed.json filename if omitted.")
    parser.add_argument("--graph",        default=None,
                        help="Path to fused_knowledge_graph.json (default: <data-dir>/fused_knowledge_graph.json)")
    parser.add_argument("--output",       default="data/processed/eval_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results: dict = {}

    if args.eval_queries:
        results["retrieval"] = evaluate_retrieval(Path(args.eval_queries), data_dir)

    if args.questions:
        q_path = Path(args.questions)
        gt_path = Path(args.ground_truth)
        graph_path = Path(args.graph) if args.graph else data_dir / "fused_knowledge_graph.json"
        results.update(evaluate_questions(
            q_path, ground_truth_path=gt_path, graph_path=graph_path, startup_id=args.startup_id,
        ))

    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
