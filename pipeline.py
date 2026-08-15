"""
pipeline.py — ChainCheck End-to-End Orchestrator
==================================================
VC Hidden Dependency Interrogator | Multi-Hop Reasoning Pipeline

14 stages in sequence. Every stage is independently re-runnable
via --start-stage so you never have to redo expensive earlier stages
during development.

Stages
------
  1   whitepaper_parser      → pitch deck technical claims
  2   github_parser          → codebase dependency map + import graph
  3   patent_parser          → patent triples + USPTO metadata
  4   license_parser         → open-source license scan
  5   knowledge_fusion       → FAISS semantic KG (PRIMARY graph — your novelty)
  6   entity_resolver        → cross-domain entity matching (precision pass)
  7   kg_builder             → typed KG from entity matches
  8   path_reasoner          → BFS multi-hop chains + dynamic evidence
  9   contradiction_detector → proprietary claim mismatches
  10  risk_analyzer          → unified risk scores (legal + license + patent)
  11  question_gen           → LLM adversarial questions (primary)
  12  mhqg_engine            → template fallback questions (no LLM needed)
  13  explainability_engine  → SHA-256 audit trail
  14  eval_harness           → Recall@K + audit-trail coverage metrics

Usage
-----
  # Full run
  python pipeline.py --pitch data/raw/startup.pdf \\
                     --repo  data/raw/repo/ \\
                     --patents data/raw/patents/

  # Resume from stage 8 (skip extraction already done)
  python pipeline.py --start-stage 8

  # No LLM call — use template fallback only
  python pipeline.py --dry-run

  # Choose LLM provider
  python pipeline.py --provider anthropic    # needs ANTHROPIC_API_KEY
  python pipeline.py --provider ollama       # needs local Ollama daemon
"""

import json
import logging
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED    = DATA_DIR / "processed"
SRC          = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC))


# ── Stage timer (vaibhava's implementation — best of the three) ───────────────

class StageTimer:
    """Tracks latency per pipeline stage. Slowest stage = Phase 3 tuning target."""

    def __init__(self):
        self.times: dict[str, float] = {}
        self._start: Optional[float] = None
        self._stage: Optional[str]   = None

    def start(self, stage: str):
        self._stage = stage
        self._start = time.perf_counter()
        logger.info("━━━ STAGE: %s ━━━", stage.upper())

    def stop(self):
        if self._stage and self._start:
            elapsed = time.perf_counter() - self._start
            self.times[self._stage] = round(elapsed, 2)
            logger.info("✓ %s completed in %.2fs", self._stage, elapsed)

    def report(self) -> dict:
        total = sum(self.times.values())
        return {
            "stage_times_seconds": self.times,
            "total_seconds": round(total, 2),
            "slowest_stage": max(self.times, key=self.times.get) if self.times else None,
        }

    def print_report(self):
        print(f"\n{'═'*60}")
        print("  PIPELINE TIMING REPORT  (Phase 3 optimization map)")
        print(f"{'═'*60}")
        total = sum(self.times.values())
        for stage, t in self.times.items():
            pct = (t / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {stage:<28} {t:>6.2f}s  {bar} {pct:.0f}%")
        print(f"  {'─'*56}")
        print(f"  {'TOTAL':<28} {total:>6.2f}s")
        if self.times:
            slowest = max(self.times, key=self.times.get)
            print(f"\n  Bottleneck → {slowest} ({self.times[slowest]:.2f}s)  ← tune here first")
        print(f"{'═'*60}\n")


# ── Stage runners ─────────────────────────────────────────────────────────────

def stage_1_whitepaper(pitch_pdf: Path) -> bool:
    """Parse pitch deck / whitepaper → startup_parsed.json"""
    try:
        from extractors.whitepaper_parser import WhitepaperParser, write_output
        PROCESSED.mkdir(parents=True, exist_ok=True)
        parser = WhitepaperParser(pitch_pdf)
        result = parser.parse()
        write_output(result, PROCESSED)
        logger.info(
            "Whitepaper: %d claims, %d entities",
            result.statistics["technical_claims_extracted"],
            result.statistics["unique_entities_found"],
        )
        return True
    except Exception as e:
        logger.error("Stage 1 failed: %s", e, exc_info=True)
        return False


def stage_2_github(repo_path: Path) -> bool:
    """Parse codebase → dependency_map.json + codebase_knowledge.json"""
    try:
        from extractors.github_parser import (
            build_dependency_map, build_import_graph,
            build_codebase_knowledge, graph_to_serializable,
        )
        PROCESSED.mkdir(parents=True, exist_ok=True)

        dep_map = build_dependency_map(repo_path)
        (PROCESSED / "dependency_map.json").write_text(
            json.dumps(dep_map, indent=2), encoding="utf-8"
        )

        import_graph, import_analysis = build_import_graph(repo_path)
        (PROCESSED / "import_graph.json").write_text(
            json.dumps(graph_to_serializable(import_graph), indent=2), encoding="utf-8"
        )

        combined = build_codebase_knowledge(dep_map, import_analysis, import_graph)
        if combined.get("import_graph_analysis"):
            combined["import_graph_analysis"].pop("all_imports_raw", None)
        (PROCESSED / "codebase_knowledge.json").write_text(
            json.dumps(combined, indent=2), encoding="utf-8"
        )
        logger.info("Codebase: %d dependencies", dep_map["metadata"]["total_dependencies"])
        return True
    except Exception as e:
        logger.error("Stage 2 failed: %s", e, exc_info=True)
        # Write placeholder so downstream stages don't crash
        (PROCESSED / "codebase_knowledge.json").write_text(
            json.dumps({
                "dependency_map": {"all_dependencies": []},
                "import_graph_analysis": {"top_third_party_libraries": []},
            }, indent=2), encoding="utf-8"
        )
        return False


def stage_3_patents(patents_dir: Path) -> bool:
    """Extract patent triples → knowledge_base.json"""
    try:
        from extractors.patent_parser import process_directory
        PROCESSED.mkdir(parents=True, exist_ok=True)
        if not patents_dir.exists():
            logger.warning("Patents dir not found — writing empty placeholder")
            (PROCESSED / "knowledge_base.json").write_text(
                json.dumps({"metadata": {"total_triples": 0}, "triples": []}, indent=2),
                encoding="utf-8",
            )
            return True
        triples = process_directory(patents_dir, PROCESSED)
        logger.info("Patents: %d triples", len(triples))
        return True
    except Exception as e:
        logger.error("Stage 3 failed: %s", e, exc_info=True)
        (PROCESSED / "knowledge_base.json").write_text(
            json.dumps({"metadata": {"total_triples": 0}, "triples": []}, indent=2),
            encoding="utf-8",
        )
        return False


def stage_4_licenses() -> bool:
    """Scan open-source licenses → license_knowledge.json"""
    try:
        from extractors.license_parser import run_license_scan
        run_license_scan(DATA_DIR)
        return True
    except Exception as e:
        logger.error("Stage 4 failed: %s", e, exc_info=True)
        return False


def stage_5_fusion(threshold: float = 0.40) -> bool:
    """
    FAISS semantic knowledge fusion → fused_knowledge_graph.json + .graphml

    This is the PRIMARY graph builder and the core novelty of ChainCheck.
    Cross-domain semantic bridging: Claim → Dependency → Patent across
    vocabulary boundaries using sentence-transformer embeddings + FAISS.
    """
    try:
        from graph.knowledge_fusion import KnowledgeFusionPipeline
        pipeline = KnowledgeFusionPipeline(DATA_DIR, similarity_threshold=threshold)
        pipeline.fuse_knowledge_domains()
        pipeline.export_fused_graph()
        return True
    except Exception as e:
        logger.error("Stage 5 failed: %s", e, exc_info=True)
        return False


def stage_6_entity_resolver(threshold: float = 0.45) -> bool:
    """
    Cross-domain entity resolution → entity_matches.json

    Secondary precision pass on top of the FAISS fusion graph.
    Adds deterministic threshold-based matches missed by embedding similarity.
    """
    try:
        from resolvers.entity_resolver import resolve_entities
        PROCESSED.mkdir(parents=True, exist_ok=True)

        candidates = sorted(PROCESSED.glob("*_parsed.json"))
        if not candidates:
            logger.warning("No *_parsed.json found — writing empty entity_matches placeholder")
            (PROCESSED / "entity_matches.json").write_text(
                json.dumps({"metadata": {"total_matches": 0}, "matches": []}, indent=2),
                encoding="utf-8",
            )
            return True

        wp_path  = candidates[0]
        cb_path  = PROCESSED / "codebase_knowledge.json"
        pat_path = PROCESSED / "knowledge_base.json"
        out_path = PROCESSED / "entity_matches.json"

        for p in (cb_path, pat_path):
            if not p.exists():
                logger.warning("%s missing — writing empty entity_matches placeholder", p.name)
                (PROCESSED / "entity_matches.json").write_text(
                    json.dumps({"metadata": {"total_matches": 0}, "matches": []}, indent=2),
                    encoding="utf-8",
                )
                return True

        matches = resolve_entities(wp_path, cb_path, pat_path, out_path, threshold=threshold)
        logger.info("Entity resolver: %d cross-domain matches", len(matches))
        return True
    except Exception as e:
        logger.error("Stage 6 failed: %s", e, exc_info=True)
        return False


def stage_7_kg_builder() -> bool:
    """Build typed KG from entity matches → kg.json"""
    try:
        from graph.kg_builder import build_knowledge_graph, build_summary, graph_to_json
        matches_path = PROCESSED / "entity_matches.json"
        if not matches_path.exists():
            logger.warning("entity_matches.json not found — skipping")
            return True

        # Locate the parsed whitepaper JSON so Claim nodes can be enriched
        # with real sentence text instead of just their bare claim_id.
        # Same *_parsed.json auto-detection pattern used in stage_6.
        wp_candidates = sorted(PROCESSED.glob("*_parsed.json"))
        whitepaper_path = wp_candidates[0] if wp_candidates else None
        if whitepaper_path is None:
            logger.warning(
                "No *_parsed.json found — Claim nodes will fall back to bare IDs "
                "(claim_text will not be populated with real sentences)."
            )

        G = build_knowledge_graph(matches_path, whitepaper_path)
        summary = build_summary(G)
        (PROCESSED / "kg.json").write_text(
            json.dumps(graph_to_json(G), indent=2), encoding="utf-8"
        )
        (PROCESSED / "kg_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info("KG: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return True
    except Exception as e:
        logger.error("Stage 7 failed: %s", e, exc_info=True)
        return False


def stage_8_path_reasoner(max_hops: int = 3, threshold: float = 0.28) -> bool:
    """
    Multi-hop BFS + dynamic evidence chains
    → hop_chains.json + structured_evidence.json

    Reads fused_knowledge_graph.json (Stage 5 output) preferentially,
    falls back to kg.json (Stage 7 output) if fused graph is absent.
    """
    try:
        from reasoning.path_reasoner import load_graph, reason, save_chains, DynamicPathReasoner

        # Prefer the richer fused graph; fall back to kg
        kg_path = PROCESSED / "fused_knowledge_graph.json"
        if not kg_path.exists():
            kg_path = PROCESSED / "kg.json"
        if not kg_path.exists():
            logger.warning("No graph file found for path reasoner — skipping")
            return True

        G = load_graph(kg_path)
        chains = reason(G, max_hops=max_hops, chain_threshold=threshold)
        save_chains(chains, PROCESSED / "hop_chains.json")
        logger.info("Hop chains: %d total", len(chains))

        # Dynamic evidence (all_simple_paths to risk nodes)
        dr = DynamicPathReasoner(DATA_DIR)
        dr.export_evidence()
        return True
    except Exception as e:
        logger.error("Stage 8 failed: %s", e, exc_info=True)
        return False


def stage_9_contradiction() -> bool:
    """Detect proprietary claim mismatches → contradiction_evidence.json"""
    try:
        from reasoning.contradiction_detector import ProprietaryContradictionDetector
        detector = ProprietaryContradictionDetector(DATA_DIR)
        contradictions = detector.detect_proprietary_mismatches()
        logger.info("Contradictions: %d found", len(contradictions))
        return True
    except Exception as e:
        logger.error("Stage 9 failed: %s", e, exc_info=True)
        return False


def stage_10_risk_analyzer() -> bool:
    """
    Unified risk taxonomy + confidence scores → vc_risk_report.json

    Consolidates legal_risk (patent status/jurisdiction/assignee),
    license_risk (copyleft / commercial-use restrictions),
    and semantic IP overlap into a single scored report.
    """
    try:
        from scoring.risk_analyzer import RiskAnalyzer
        analyzer = RiskAnalyzer(DATA_DIR)
        risks = analyzer.analyze_evidence()
        logger.info("Risk analyzer: %d actionable risks", len(risks))
        return True
    except Exception as e:
        logger.error("Stage 10 failed: %s", e, exc_info=True)
        return False


def stage_11_question_gen(
    provider: str = "anthropic",
    model: str = "",
    dry_run: bool = False,
    max_questions: Optional[int] = None,
) -> bool:
    """
    LLM adversarial question generation → questions.json

    Supports --provider anthropic (ANTHROPIC_API_KEY) or ollama (local daemon).
    Falls back to template questions automatically if the LLM call fails.
    """
    try:
        from generation.question_gen import generate_questions, save_questions

        # Prefer structured_evidence (richer provenance); fall back to hop_chains
        chains_path = PROCESSED / "structured_evidence.json"
        if not chains_path.exists():
            chains_path = PROCESSED / "hop_chains.json"
        if not chains_path.exists():
            logger.warning("No chain file found — skipping LLM question gen")
            return True

        questions = generate_questions(
            chains_path=chains_path,
            output_path=PROCESSED / "questions.json",
            provider=provider,
            model=model,
            dry_run=dry_run,
            max_questions=max_questions,
        )
        save_questions(questions, PROCESSED / "questions.json")
        logger.info("LLM question gen: %d questions", len(questions))
        return True
    except Exception as e:
        logger.error("Stage 11 failed: %s", e, exc_info=True)
        return False


def stage_12_mhqg() -> bool:
    """Template-based MHQG fallback → due_diligence_questions.json"""
    try:
        from reasoning.mhqg_engine import MHQGEngine
        engine = MHQGEngine(DATA_DIR)
        engine.run()
        return True
    except Exception as e:
        logger.error("Stage 12 failed: %s", e, exc_info=True)
        return False


def stage_13_explainability() -> bool:
    """Build SHA-256 audit trail → audited_vc_report.json"""
    try:
        from audit.explainability_engine import EvidenceAuditLayer
        # Prefer LLM questions; fall back to MHQG template questions
        source = "questions.json"
        if not (PROCESSED / source).exists():
            source = "due_diligence_questions.json"
        engine = EvidenceAuditLayer(DATA_DIR)
        engine.build_audit_trail(source_filename=source)
        return True
    except Exception as e:
        logger.error("Stage 13 failed: %s", e, exc_info=True)
        return False


def stage_14_eval(ground_truth_path: Optional[Path] = None) -> bool:
    """Recall@K + audit-trail coverage → eval_results.json"""
    try:
        from eval.eval_runner import evaluate_questions
        questions_path = PROCESSED / "questions.json"
        if not questions_path.exists():
            questions_path = PROCESSED / "due_diligence_questions.json"
        if not questions_path.exists():
            logger.warning("No questions file found — skipping eval")
            return True

        results = evaluate_questions(questions_path)

        if ground_truth_path and ground_truth_path.exists():
            gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
            gt_questions = [q["question"].lower() for q in gt.get("questions", [])]
            gen_data = json.loads(questions_path.read_text(encoding="utf-8"))
            gen_questions = [
                (q.get("question") or q.get("generated_question") or "").lower()
                for q in gen_data.get("questions", [])
            ]
            # Keyword overlap: ≥2 shared words of length ≥4
            overlap_count = 0
            for gen_q in gen_questions:
                gen_words = {w for w in gen_q.split() if len(w) >= 4}
                for gt_q in gt_questions:
                    gt_words = {w for w in gt_q.split() if len(w) >= 4}
                    if len(gen_words & gt_words) >= 2:
                        overlap_count += 1
                        break
            results["ground_truth_overlap"] = round(
                overlap_count / max(len(gen_questions), 1), 3
            )

        (PROCESSED / "eval_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        _print_eval(results)
        return True
    except Exception as e:
        logger.error("Stage 14 failed: %s", e, exc_info=True)
        return False


# ── Summary printers ──────────────────────────────────────────────────────────

def _print_eval(results: dict) -> None:
    print(f"\n{'═'*60}")
    print("  EVAL RESULTS")
    print(f"{'═'*60}")
    print(f"  Total questions      : {results.get('total_questions', 0)}")
    print(f"  Audit trail coverage : {results.get('audit_trail_coverage', 0):.2%}")
    if "ground_truth_overlap" in results:
        print(f"  Ground truth overlap : {results['ground_truth_overlap']:.2%}")
    print(f"{'═'*60}\n")


def _print_pipeline_summary(results: dict[str, bool], elapsed: float) -> None:
    print(f"\n{'═'*65}")
    print("  CHAINCHECK PIPELINE SUMMARY")
    print(f"{'═'*65}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    print(f"  Stages completed : {passed}/{total}")
    print(f"  Total runtime    : {elapsed:.1f}s")
    print()
    for stage_name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"    {status}  {stage_name}")
    print(f"\n  Key outputs (data/processed/):")
    key_files = [
        "fused_knowledge_graph.json",
        "hop_chains.json",
        "structured_evidence.json",
        "contradiction_evidence.json",
        "vc_risk_report.json",
        "questions.json",
        "due_diligence_questions.json",
        "audited_vc_report.json",
        "eval_results.json",
    ]
    for fname in key_files:
        fpath = PROCESSED / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    ✓  {fname:<42} ({size_kb:.1f} KB)")
        else:
            print(f"    -  {fname:<42} (not generated)")
    print(f"{'═'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChainCheck — VC Technical Due Diligence Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input paths
    parser.add_argument("--pitch",   default=None,
                        help="Path to startup pitch deck / whitepaper PDF")
    parser.add_argument("--repo",    default=".",
                        help="Path to startup GitHub repo root")
    parser.add_argument("--patents", default="data/raw/patents/",
                        help="Path to directory of .txt patent files")
    # Stage control
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Resume from this stage number (1-14)")
    parser.add_argument("--end-stage",   type=int, default=14,
                        help="Stop after this stage number")
    # Tuning
    parser.add_argument("--fusion-threshold",   type=float, default=0.40,
                        help="FAISS cosine threshold for semantic bridge edges")
    parser.add_argument("--resolver-threshold", type=float, default=0.45,
                        help="Cosine threshold for entity resolver (secondary pass)")
    parser.add_argument("--max-hops",           type=int,   default=3,
                        help="Max BFS depth for path reasoner")
    parser.add_argument("--max-questions",       type=int,   default=None,
                        help="Limit LLM question generation count")
    # LLM
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic",
                        help="LLM provider for Stage 11 question generation")
    parser.add_argument("--model", default="",
                        help="Override default model for the selected provider, e.g. llama3 or mistral for Ollama")
    # Flags
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all LLM calls; use template questions only")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to eval/ground_truth.json for overlap scoring")
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    timer   = StageTimer()
    results: dict[str, bool] = {}

    gt_path = Path(args.ground_truth) if args.ground_truth else (
        PROJECT_ROOT / "eval" / "ground_truth.json"
    )

    stages = {
        1:  ("whitepaper_parser",
             lambda: stage_1_whitepaper(Path(args.pitch)) if args.pitch
                     else (logger.warning("--pitch not set, skipping Stage 1") or True)),
        2:  ("github_parser",
             lambda: stage_2_github(Path(args.repo))),
        3:  ("patent_parser",
             lambda: stage_3_patents(Path(args.patents))),
        4:  ("license_parser",
             stage_4_licenses),
        5:  ("knowledge_fusion",
             lambda: stage_5_fusion(args.fusion_threshold)),
        6:  ("entity_resolver",
             lambda: stage_6_entity_resolver(args.resolver_threshold)),
        7:  ("kg_builder",
             stage_7_kg_builder),
        8:  ("path_reasoner",
             lambda: stage_8_path_reasoner(args.max_hops)),
        9:  ("contradiction_detector",
             stage_9_contradiction),
        10: ("risk_analyzer",
             stage_10_risk_analyzer),
        11: ("question_gen_llm",
             lambda: stage_11_question_gen(args.provider, args.model, args.dry_run, args.max_questions)),
        12: ("mhqg_template_fallback",
             stage_12_mhqg),
        13: ("explainability_audit",
             stage_13_explainability),
        14: ("eval_harness",
             lambda: stage_14_eval(gt_path if gt_path.exists() else None)),
    }

    start_time = time.time()

    for stage_num in range(args.start_stage, args.end_stage + 1):
        if stage_num not in stages:
            continue
        name, fn = stages[stage_num]
        timer.start(name)
        ok = fn()
        timer.stop()
        results[f"Stage {stage_num:02d}: {name}"] = ok
        if not ok and stage_num <= 5:
            # Hard stop only on critical early stages
            logger.error(
                "Critical stage %d failed. Fix and re-run with --start-stage %d",
                stage_num, stage_num,
            )
            break

    elapsed = time.time() - start_time
    timer.print_report()
    _print_pipeline_summary(results, elapsed)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
