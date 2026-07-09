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

from eval.metrics import audit_trail_coverage, recall_at_ks


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


def evaluate_questions(questions_path: Path) -> dict:
    """Evaluate generated question artifacts for audit trail coverage."""
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    return {
        "total_questions": len(questions),
        "audit_trail_coverage": audit_trail_coverage(questions),
    }


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
    parser.add_argument("--output",       default="data/processed/eval_results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results: dict = {}

    if args.eval_queries:
        results["retrieval"] = evaluate_retrieval(Path(args.eval_queries), data_dir)

    if args.questions:
        q_path = Path(args.questions)
        results.update(evaluate_questions(q_path))

        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            results["ground_truth_overlap"] = ground_truth_overlap(q_path, gt_path)

    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
