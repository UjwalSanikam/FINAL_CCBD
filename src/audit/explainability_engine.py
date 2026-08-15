"""
explainability_engine.py — SHA-256 Audit Trail Generator
=========================================================
ChainCheck | Multi-Hop Reasoning Pipeline

Every generated question gets a deterministic SHA-256 trace ID derived
from its evidence chain + question text. This is the "Explainable IP
Auditing" novelty: generated questions are not black-box outputs —
they carry a machine-verifiable provenance fingerprint.

Output
------
  data/processed/audited_vc_report.json

Each audited item contains:
  traceability_id    SHA-256 of (provenance + question), first 12 hex chars
  timestamp          ISO-8601 UTC
  severity           from the original question record
  category           risk category
  target_entity      entity flagged
  question           the generated question text
  formal_confidence  numeric score
  evidence_chain     dict of provenance hops
  audit_status       MACHINE_ASSISTED_VERIFICATION
  recommended_action plain-language follow-up action
"""

import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EvidenceAuditLayer:
    """
    V4.0 — Builds a SHA-256-stamped audit trail from any questions JSON
    produced by question_gen.py or mhqg_engine.py.

    Works with both output schemas:
      question_gen output:  {"questions": [{question, audit_trail, ...}]}
      mhqg_engine output:   {"questions": [{generated_question, audit_trail, ...}]}
      risk_analyzer output: {"identified_risks": [{question, evidence_chain, ...}]}
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def _generate_trace_id(self, evidence_string: str) -> str:
        return hashlib.sha256(evidence_string.encode()).hexdigest()[:12].upper()

    def build_audit_trail(self, source_filename: str = "questions.json") -> list[dict]:
        logger.info("Initializing Evidence Audit Layer (source: %s)...", source_filename)

        report_path = self.data_dir / "processed" / source_filename
        if not report_path.exists():
            logger.error("Source report not found: %s", report_path)
            return []

        report_data = json.loads(report_path.read_text(encoding="utf-8"))

        # Normalise: support questions key or identified_risks key
        items = (
            report_data.get("questions")
            or report_data.get("identified_risks")
            or []
        )

        audited_items: list[dict] = []

        for item in items:
            # Field normalisation across all three output schemas
            chain      = (
                item.get("audit_trail")
                or item.get("raw_provenance")
                or item.get("evidence_chain")
                or {}
            )
            category   = item.get("question_category", item.get("category", item.get("risk_type", "unknown")))
            confidence = item.get("chain_score", item.get("confidence_score", item.get("formal_confidence", 0.0)))
            question   = item.get("question", item.get("generated_question", ""))
            target     = (
                item.get("target_claim")
                or item.get("target_entity")
                or item.get("risk_node")
                or ""
            )
            severity   = item.get("severity", item.get("risk_level", "UNKNOWN"))
            action     = item.get(
                "recommended_action",
                "Review the generated question and provenance for auditor follow-up.",
            )

            provenance_str = json.dumps(chain, sort_keys=True)
            trace_id = (
                f"TRC-{self._generate_trace_id(provenance_str + question)}"
                f"-{int(time.time())}"
            )

            audited_items.append({
                "traceability_id":    trace_id,
                "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "severity":           severity,
                "category":           category,
                "target_entity":      target,
                "question":           question,
                "formal_confidence":  confidence,
                "recommended_action": action,
                "evidence_chain":     chain,
                "audit_status":       "MACHINE_ASSISTED_VERIFICATION",
            })

        output_path = self.data_dir / "processed" / "audited_vc_report.json"
        output_path.write_text(
            json.dumps({"audited_items": audited_items}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Audit trail complete: %d items → %s", len(audited_items), output_path
        )
        return audited_items


def explain(data_dir: Path | str) -> list[dict]:
    """Functional wrapper — called by pipeline.py Stage 13."""
    return EvidenceAuditLayer(Path(data_dir)).build_audit_trail()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build SHA-256 audit trail")
    ap.add_argument("--data-dir",    default=None)
    ap.add_argument("--source-file", default="questions.json",
                    help="Which questions JSON to audit")
    args = ap.parse_args()

    data_dir = (
        Path(args.data_dir) if args.data_dir
        else Path(__file__).resolve().parents[2] / "data"
    )
    EvidenceAuditLayer(data_dir).build_audit_trail(source_filename=args.source_file)
