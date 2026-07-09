"""
test_pipeline.py — Unit and integration tests for ChainCheck
=============================================================
Run with:  pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ── Shared / Schema ───────────────────────────────────────────────────────────

class TestSchema:
    def test_node_types_present(self):
        from shared.schema import NODE_TYPES
        assert "CLAIM" in NODE_TYPES
        assert "PATENT" in NODE_TYPES
        assert "LIBRARY" in NODE_TYPES

    def test_output_files_have_all_stages(self):
        from shared.schema import OUTPUT_FILES
        required = [
            "fused_graph", "hop_chains", "structured_evidence",
            "vc_risk_report", "questions", "audited_vc_report",
        ]
        for key in required:
            assert key in OUTPUT_FILES, f"Missing output file key: {key}"


class TestBuzzwords:
    def test_military_grade_normalized(self):
        from shared.buzzwords import normalize_buzzwords
        result = normalize_buzzwords("military-grade encryption")
        assert "AES-256" in result

    def test_hyper_fast_normalized(self):
        from shared.buzzwords import normalize_buzzwords
        result = normalize_buzzwords("hyper-fast data processing")
        assert "high-throughput" in result

    def test_no_false_replacement(self):
        from shared.buzzwords import normalize_buzzwords
        text = "PostgreSQL database with standard indexing"
        assert normalize_buzzwords(text) == text  # no buzzwords → unchanged


# ── Scoring / Risk Analyzer ───────────────────────────────────────────────────

class TestConfidenceEngine:
    def setup_method(self):
        from scoring.risk_analyzer import ConfidenceEngine
        self.engine = ConfidenceEngine(base_decay_rate=0.90)

    def test_commercial_license_critical(self):
        conf, severity = self.engine.compute("Commercial License", path_length=1)
        assert severity == "CRITICAL"
        assert conf >= 0.80

    def test_long_path_reduces_confidence(self):
        conf_short, _ = self.engine.compute("IP Overlap", 1, base_similarity=0.85)
        conf_long,  _ = self.engine.compute("IP Overlap", 4, base_similarity=0.85)
        assert conf_long < conf_short

    def test_proprietary_mismatch_high_confidence(self):
        conf, severity = self.engine.compute("Proprietary Claim Mismatch", 1)
        assert conf >= 0.90
        assert severity in ("CRITICAL", "HIGH")


class TestLegalRiskAnalyzer:
    def setup_method(self):
        from scoring.risk_analyzer import LegalRiskAnalyzer
        self.analyzer = LegalRiskAnalyzer()

    def test_active_us_patent_high_risk(self):
        result = self.analyzer.analyze({
            "status": "ACTIVE",
            "license_type": "Commercial Restricted",
            "jurisdiction": "US",
            "assignee": "CompetitorX",
        })
        assert result["risk_level"] == "HIGH"
        assert result["legal_risk_score"] >= 0.70

    def test_expired_patent_low_risk(self):
        result = self.analyzer.analyze({
            "status": "EXPIRED",
            "license_type": "MIT",
            "jurisdiction": "US",
            "assignee": "UNKNOWN",
        })
        assert result["risk_level"] == "LOW"
        assert result["legal_risk_score"] == 0.0

    def test_permissive_license_low_risk(self):
        result = self.analyzer.analyze({
            "status": "ACTIVE",
            "license_type": "apache-2.0",
            "jurisdiction": "US",
            "assignee": "OpenSourceOrg",
        })
        assert result["risk_level"] == "LOW"

    def test_pending_patent_low_risk(self):
        result = self.analyzer.analyze({
            "status": "PENDING",
            "license_type": "Commercial",
            "jurisdiction": "US",
            "assignee": "CompetitorX",
        })
        assert result["risk_level"] == "LOW"


class TestLicenseClassifier:
    def test_gpl_is_copyleft(self):
        from scoring.risk_analyzer import classify_license
        cls, _ = classify_license("gpl-3.0")
        assert cls == "COPYLEFT"

    def test_mit_is_permissive(self):
        from scoring.risk_analyzer import classify_license
        cls, _ = classify_license("MIT")
        assert cls == "PERMISSIVE"

    def test_commercial_restricted(self):
        from scoring.risk_analyzer import classify_license
        cls, _ = classify_license("Commercial Restricted")
        assert cls == "COMMERCIAL_RESTRICTED"


class TestRiskAnalyzerIntegration:
    def test_analyzes_contradiction_evidence(self, tmp_path):
        from scoring.risk_analyzer import RiskAnalyzer

        processed = tmp_path / "processed"
        processed.mkdir()

        # Write minimal contradiction evidence
        (processed / "contradiction_evidence.json").write_text(json.dumps({
            "contradictions": [{
                "risk_type": "Proprietary Claim Mismatch",
                "severity": "HIGH",
                "claim_id": "Claim_0",
                "claim_text": "We use in-house auth with no open-source dependencies",
                "contradictory_module": "auth0",
                "confidence_score": 0.99,
                "recommended_action": "Explain discrepancy.",
            }]
        }), encoding="utf-8")

        risks = RiskAnalyzer(tmp_path).analyze_evidence()
        assert len(risks) >= 1
        assert any(r["category"] == "Proprietary Claim Mismatch" for r in risks)

        # Check output file was written
        out = processed / "vc_risk_report.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["metadata"]["total_risks"] >= 1


# ── Audit / Explainability ────────────────────────────────────────────────────

class TestExplainabilityEngine:
    def test_trace_id_format(self, tmp_path):
        from audit.explainability_engine import EvidenceAuditLayer

        processed = tmp_path / "processed"
        processed.mkdir()

        # Write a minimal questions.json
        (processed / "questions.json").write_text(json.dumps({
            "questions": [{
                "chain_id": "chain_0001",
                "question": "Why does your codebase use auth0?",
                "question_category": "licence_conflict",
                "audit_trail": {"hop_1": "Claim_0", "hop_2": ["auth0"], "hop_3": "gpl-3.0"},
                "confidence_score": 0.85,
                "severity": "CRITICAL",
                "recommended_action": "Review open-source dependency.",
            }]
        }), encoding="utf-8")

        engine = EvidenceAuditLayer(tmp_path)
        items  = engine.build_audit_trail()

        assert len(items) == 1
        assert items[0]["traceability_id"].startswith("TRC-")
        assert items[0]["audit_status"] == "MACHINE_ASSISTED_VERIFICATION"
        assert len(items[0]["traceability_id"]) > 10

    def test_trace_id_deterministic_for_same_evidence(self, tmp_path):
        """Same provenance + question → same first 12 hex chars (before timestamp)."""
        from audit.explainability_engine import EvidenceAuditLayer
        engine = EvidenceAuditLayer(tmp_path)
        id1 = engine._generate_trace_id("test evidence string")
        id2 = engine._generate_trace_id("test evidence string")
        assert id1 == id2

    def test_different_evidence_gives_different_trace(self, tmp_path):
        from audit.explainability_engine import EvidenceAuditLayer
        engine = EvidenceAuditLayer(tmp_path)
        id1 = engine._generate_trace_id("chain A")
        id2 = engine._generate_trace_id("chain B")
        assert id1 != id2


# ── Eval / Metrics ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_recall_at_1_perfect(self):
        from eval.metrics import recall_at_k
        results   = [["A", "B", "C"], ["D", "E"]]
        expected  = ["A", "D"]
        assert recall_at_k(results, expected, k=1) == 1.0

    def test_recall_at_1_miss(self):
        from eval.metrics import recall_at_k
        results   = [["B", "A", "C"]]
        expected  = ["A"]
        assert recall_at_k(results, expected, k=1) == 0.0

    def test_recall_at_3_hit(self):
        from eval.metrics import recall_at_k
        results   = [["B", "C", "A"]]
        expected  = ["A"]
        assert recall_at_k(results, expected, k=3) == 1.0

    def test_audit_trail_coverage_full(self):
        from eval.metrics import audit_trail_coverage
        qs = [{"audit_trail": {"hop_1": "x"}}, {"audit_trail": {"hop_1": "y"}}]
        assert audit_trail_coverage(qs) == 1.0

    def test_audit_trail_coverage_partial(self):
        from eval.metrics import audit_trail_coverage
        qs = [{"audit_trail": {"hop_1": "x"}}, {"audit_trail": {}}]
        # Empty dict is falsy
        assert audit_trail_coverage(qs) == 0.5

    def test_audit_trail_coverage_empty_list(self):
        from eval.metrics import audit_trail_coverage
        assert audit_trail_coverage([]) == 0.0

    def test_recall_at_ks_returns_all(self):
        from eval.metrics import recall_at_ks
        results  = [["A", "B", "C"]]
        expected = ["A"]
        r = recall_at_ks(results, expected, ks=(1, 3, 5))
        assert "recall_at_1" in r
        assert "recall_at_3" in r
        assert "recall_at_5" in r


# ── Ground Truth JSON ─────────────────────────────────────────────────────────

class TestGroundTruth:
    def test_ground_truth_has_entries(self):
        gt_path = ROOT / "eval" / "ground_truth.json"
        assert gt_path.exists(), "eval/ground_truth.json is missing"
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        questions = data.get("questions", [])
        assert len(questions) >= 10, (
            f"Ground truth has only {len(questions)} entries — need at least 10"
        )

    def test_ground_truth_entries_have_required_fields(self):
        gt_path = ROOT / "eval" / "ground_truth.json"
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        required_fields = ["id", "claim", "expected_dependency", "question", "severity"]
        for q in data.get("questions", []):
            for field in required_fields:
                assert field in q, f"Ground truth entry {q.get('id','?')} missing '{field}'"


# ── Contradiction Detector ────────────────────────────────────────────────────

class TestContradictionDetector:
    def test_detects_crypto_mismatch(self, tmp_path):
        import networkx as nx
        from reasoning.contradiction_detector import ProprietaryContradictionDetector

        processed = tmp_path / "processed"
        processed.mkdir()

        G = nx.DiGraph()
        G.add_node("Claim_0",
                   label="Marketing_Claim",
                   node_type="Claim",
                   full_text="We use proprietary military-grade custom encryption")
        G.add_node("pycryptodome",
                   label="Software_Dependency",
                   node_type="Library")
        G.add_edge("Claim_0", "pycryptodome", edge_type="IMPLEMENTED_BY")

        kg_data = {
            "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
        }
        (processed / "kg.json").write_text(json.dumps(kg_data), encoding="utf-8")

        detector = ProprietaryContradictionDetector(tmp_path)
        contradictions = detector.detect_proprietary_mismatches()
        assert len(contradictions) >= 1
        assert any(c["contradictory_module"] == "pycryptodome" for c in contradictions)


# ── fixtures/crypto_auth.py (the synthetic VC trap) ──────────────────────────

class TestSyntheticVCTrap:
    """Verify the fixture file exists and contains the intentional trap import."""

    def test_fixture_file_exists(self):
        fixture = ROOT / "tests" / "fixtures" / "crypto_auth.py"
        assert fixture.exists(), (
            "tests/fixtures/crypto_auth.py is missing. "
            "This is the synthetic VC-trap file used to test contradiction detection."
        )

    def test_fixture_contains_trap_import(self):
        fixture = ROOT / "tests" / "fixtures" / "crypto_auth.py"
        content = fixture.read_text()
        assert "copyleft_crypto_engine" in content, (
            "Synthetic trap import 'copyleft_crypto_engine' not found in fixture"
        )
