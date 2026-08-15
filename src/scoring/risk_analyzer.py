"""
risk_analyzer.py — Unified VC Risk Taxonomy
=============================================
ChainCheck | Multi-Hop Reasoning Pipeline

Consolidates three previously separate files into one authoritative scorer:
  - ConfidenceEngine   (path-length decay + category weights)
  - LegalRiskAnalyzer  (patent status / jurisdiction / assignee scoring)
  - RiskAnalyzer       (evidence aggregation → vc_risk_report.json)

Sub-scores in every risk record
--------------------------------
  legal_risk_score    0–1   patent status, jurisdiction, assignee weight
  license_risk        str   COPYLEFT | COMMERCIAL_RESTRICTED | PERMISSIVE
  semantic_similarity 0–1   FAISS cosine from knowledge_fusion.py
  confidence_score    0–1   final decayed score (chain length penalty applied)
  severity            str   CRITICAL | HIGH | MODERATE | LOW

Output
------
  data/processed/vc_risk_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Confidence Engine ─────────────────────────────────────────────────────────

class ConfidenceEngine:
    """
    Centralized mathematical engine for VC risk confidence scoring.

    Applies category-specific base scores and a geometric decay penalty
    proportional to chain length so long, indirect paths don't dominate.
    """

    # Base scores by risk category
    _BASE = {
        "Commercial License":          1.00,  # deterministic — no ambiguity
        "Proprietary Claim Mismatch":  0.95,  # near-certain based on keyword match
        "IP Overlap":                  None,  # uses raw FAISS similarity as base
    }
    _DEFAULT_BASE = 0.50

    # Severity thresholds by category
    _SEVERITY_RULES: list[tuple[str, float, str]] = [
        ("Commercial License", 0.80, "CRITICAL"),
        ("any",                0.85, "CRITICAL"),
        ("any",                0.65, "HIGH"),
        ("any",                0.40, "MODERATE"),
        ("any",                0.00, "LOW"),
    ]

    def __init__(self, base_decay_rate: float = 0.90) -> None:
        self.decay_rate = base_decay_rate

    def compute(
        self,
        category: str,
        path_length: int,
        base_similarity: float = 1.0,
    ) -> tuple[float, str]:
        """
        Return (final_confidence, severity_label).

        base_similarity is the FAISS cosine score from knowledge_fusion;
        only used for IP Overlap chains.
        """
        base = self._BASE.get(category)
        if base is None:
            base = base_similarity  # IP Overlap: let FAISS score drive the base

        if base is NotImplemented or base is None:
            base = self._DEFAULT_BASE

        penalty_hops = max(0, path_length - 1)
        final = round(base * (self.decay_rate ** penalty_hops), 3)
        severity = self._map_severity(final, category)
        return final, severity

    def _map_severity(self, confidence: float, category: str) -> str:
        if category == "Commercial License" and confidence >= 0.80:
            return "CRITICAL"
        if confidence >= 0.85:
            return "CRITICAL"
        if confidence >= 0.65:
            return "HIGH"
        if confidence >= 0.40:
            return "MODERATE"
        return "LOW"


# ── Legal Risk Analyzer ───────────────────────────────────────────────────────

class LegalRiskAnalyzer:
    """
    Score a patent metadata dict for IP infringement risk.

    Weighted sub-factors:
      Active status     0.40
      Commercial use    0.30
      US jurisdiction   0.20
      Named assignee    0.10
    """

    _W_ACTIVE       = 0.40
    _W_COMMERCIAL   = 0.30
    _W_JURISDICTION = 0.20
    _W_ASSIGNEE     = 0.10

    _RESTRICTED = {"commercial restricted", "commercial", "proprietary", "all rights reserved"}
    _PERMISSIVE = {"apache-2.0", "mit", "bsd", "bsd-2", "bsd-3", "public domain", "cc0", "lgpl"}

    def analyze(self, patent_data: dict) -> dict:
        """
        Returns:
          legal_risk_score  float   0–1
          risk_level        str     HIGH | MEDIUM | LOW
          reasons           list    human-readable contributing factors
          overlap_signals   list    legal claims from patent doc
        """
        status       = (patent_data.get("status") or "UNKNOWN").upper()
        license_type = (patent_data.get("license_type") or "").lower()
        jurisdiction = (patent_data.get("jurisdiction") or "").upper()
        assignee     = patent_data.get("assignee") or "UNKNOWN"
        legal_claims = patent_data.get("legal_claims") or []

        # Short-circuit for non-enforceable patents
        if status in ("EXPIRED", "PENDING"):
            return {
                "legal_risk_score": 0.0,
                "risk_level": "LOW",
                "reasons": [f"Patent is {status} — no enforceable claims"],
                "overlap_signals": [],
            }
        if patent_data.get("legal_risk_flag") is False:
            return {
                "legal_risk_score": 0.0,
                "risk_level": "LOW",
                "reasons": ["Flagged non-risky by patent parser"],
                "overlap_signals": [],
            }
        if any(p in license_type for p in self._PERMISSIVE):
            return {
                "legal_risk_score": 0.0,
                "risk_level": "LOW",
                "reasons": [f"Permissive license: {patent_data.get('license_type')}"],
                "overlap_signals": [],
            }

        score   = 0.0
        reasons = []

        if status == "ACTIVE":
            score   += self._W_ACTIVE
            reasons.append("Patent ACTIVE — claims are enforceable")

        if any(r in license_type for r in self._RESTRICTED):
            score   += self._W_COMMERCIAL
            reasons.append(f"Restricted license: '{patent_data.get('license_type')}'")

        if jurisdiction == "US":
            score   += self._W_JURISDICTION
            reasons.append("US jurisdiction — strong ITC/federal court enforcement")
        elif jurisdiction in ("EU", "GB", "UK", "DE", "FR"):
            score   += self._W_JURISDICTION * 0.6
            reasons.append(f"{jurisdiction} — moderate enforcement risk")

        if assignee.upper() not in ("UNKNOWN", "", "N/A"):
            score   += self._W_ASSIGNEE
            reasons.append(f"Held by named entity: '{assignee}'")

        score = round(min(score, 1.0), 2)
        level = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.40 else "LOW"

        return {
            "legal_risk_score": score,
            "risk_level": level,
            "reasons": reasons,
            "overlap_signals": legal_claims,
        }


# ── License Risk Classifier ───────────────────────────────────────────────────

_COPYLEFT_LICENSES = {
    "gpl", "gpl-2.0", "gpl-3.0", "agpl", "agpl-3.0",
    "lgpl", "lgpl-2.1", "eupl", "osl", "cddl", "mpl", "mpl-2.0",
    "copyleft",
}
_RESTRICTED_LICENSES = {
    "commercial", "proprietary", "all rights reserved",
    "commercial restricted", "non-commercial",
}
_PERMISSIVE_LICENSES = {
    "mit", "apache", "apache-2.0", "bsd", "bsd-2", "bsd-3",
    "isc", "cc0", "public domain", "wtfpl", "unlicense",
}


def classify_license(license_text: str) -> tuple[str, str]:
    """
    Returns (risk_class, description).
    risk_class: 'COPYLEFT' | 'COMMERCIAL_RESTRICTED' | 'PERMISSIVE' | 'UNKNOWN'
    """
    lt = (license_text or "").lower()
    if any(k in lt for k in _COPYLEFT_LICENSES):
        return "COPYLEFT", "Copyleft — may impose viral open-source obligations on commercial products"
    if any(k in lt for k in _RESTRICTED_LICENSES):
        return "COMMERCIAL_RESTRICTED", "Commercial use restricted — requires paid license"
    if any(k in lt for k in _PERMISSIVE_LICENSES):
        return "PERMISSIVE", "Permissive license — minimal commercial restrictions"
    return "UNKNOWN", "License not recognized — manual review required"


# ── Main RiskAnalyzer ─────────────────────────────────────────────────────────

class RiskAnalyzer:
    """
    Aggregates structured_evidence.json + contradiction_evidence.json
    into a single unified vc_risk_report.json with sub-scored risk records.

    Each record contains:
      - severity            CRITICAL | HIGH | MODERATE | LOW
      - category            type of risk
      - target_entity       the node flagged
      - confidence_score    final decayed confidence (0–1)
      - legal_risk_score    LegalRiskAnalyzer sub-score (if patent metadata available)
      - license_risk        license classification string
      - semantic_similarity raw FAISS cosine from fusion graph
      - evidence_chain      human-readable hop path
      - recommended_action  VC due-diligence action
    """

    def __init__(self, data_dir: Path):
        self.data_dir   = data_dir
        self.confidence = ConfidenceEngine()
        self.legal      = LegalRiskAnalyzer()

    def _enrich_with_legal(self, risk_node: str, category: str) -> dict:
        """
        Attempt to load patent metadata for risk_node from knowledge_base.json
        and compute a legal sub-score. Returns empty dict if metadata missing.
        """
        if category != "IP Overlap":
            return {}
        kb_path = self.data_dir / "processed" / "knowledge_base.json"
        if not kb_path.exists():
            return {}
        try:
            kb = json.loads(kb_path.read_text(encoding="utf-8"))
            for triple in kb.get("triples", []):
                if risk_node in (triple.get("head", ""), triple.get("tail", "")):
                    legal_result = self.legal.analyze({
                        "status":       triple.get("status", "UNKNOWN"),
                        "license_type": triple.get("license_type", ""),
                        "jurisdiction": triple.get("jurisdiction", ""),
                        "assignee":     triple.get("assignee", "UNKNOWN"),
                        "legal_risk_flag": triple.get("legal_risk_flag"),
                    })
                    return {
                        "legal_risk_score": legal_result["legal_risk_score"],
                        "legal_reasons":    legal_result["reasons"],
                    }
        except Exception:
            pass
        return {}

    def analyze_evidence(self) -> list[dict]:
        logger.info("Running unified risk analysis...")
        risks: list[dict] = []

        # ── Source 1: Structured evidence from path_reasoner ─────────────────
        ev_path = self.data_dir / "processed" / "structured_evidence.json"
        if ev_path.exists():
            data = json.loads(ev_path.read_text(encoding="utf-8"))
            for ev in data.get("evidence_objects", []):
                category     = ev["risk_type"]
                path_length  = ev["path_length"]
                raw_sim      = ev.get("confidence_score", 1.0)
                risk_node    = ev["risk_node"]
                path         = ev["reasoning_path"]

                confidence, severity = self.confidence.compute(category, path_length, raw_sim)

                # License classification from node text
                lic_class, lic_desc = classify_license(risk_node)

                legal_extra = self._enrich_with_legal(risk_node, category)

                action = "Standard review."
                if category == "Commercial License" and severity == "CRITICAL":
                    action = "Immediate legal review: codebase may face copyleft obligations."
                elif category == "IP Overlap" and severity in ("HIGH", "MODERATE"):
                    action = "Freedom-to-operate analysis required."

                if severity in ("CRITICAL", "HIGH", "MODERATE"):
                    risks.append({
                        "severity":            severity,
                        "category":            category,
                        "target_entity":       risk_node,
                        "confidence_score":    confidence,
                        "semantic_similarity": raw_sim,
                        "license_risk":        lic_class,
                        "license_description": lic_desc,
                        "recommended_action":  action,
                        "evidence_chain":      " → ".join(path),
                        **legal_extra,
                    })

        # ── Source 2: Contradiction evidence ─────────────────────────────────
        contra_path = self.data_dir / "processed" / "contradiction_evidence.json"
        if contra_path.exists():
            data = json.loads(contra_path.read_text(encoding="utf-8"))
            for contra in data.get("contradictions", []):
                confidence, severity = self.confidence.compute("Proprietary Claim Mismatch", 1)
                risks.append({
                    "severity":           severity,
                    "category":           "Proprietary Claim Mismatch",
                    "target_entity":      contra["contradictory_module"],
                    "confidence_score":   confidence,
                    "semantic_similarity": 1.0,
                    "license_risk":       "N/A",
                    "license_description": "Direct keyword match — no embedding needed",
                    "recommended_action": (
                        "Demand technical explanation: marketing claims proprietary "
                        f"{contra.get('risk_type','component')} but codebase imports "
                        f"'{contra['contradictory_module']}'."
                    ),
                    "evidence_chain": (
                        f"Marketing: '{str(contra.get('claim_text',''))[:60]}...' "
                        f"→ Codebase: '{contra['contradictory_module']}'"
                    ),
                    "claim_text": contra.get("claim_text", ""),
                })

        # Sort: CRITICAL first, then by confidence descending
        _sev_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        risks.sort(key=lambda r: (_sev_order.get(r["severity"], 9), -r["confidence_score"]))

        out_path = self.data_dir / "processed" / "vc_risk_report.json"
        out_path.write_text(
            json.dumps({
                "metadata": {
                    "total_risks": len(risks),
                    "by_severity": {
                        s: sum(1 for r in risks if r["severity"] == s)
                        for s in ("CRITICAL", "HIGH", "MODERATE", "LOW")
                    },
                },
                "identified_risks": risks,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Risk report: %d risks → %s", len(risks), out_path)
        return risks


# ── Functional wrappers ───────────────────────────────────────────────────────

def score_confidence(
    category: str,
    path_length: int,
    base_similarity: float = 1.0,
    decay_rate: float = 0.90,
) -> dict:
    confidence, severity = ConfidenceEngine(decay_rate).compute(
        category, path_length, base_similarity
    )
    return {"confidence_score": confidence, "severity": severity}


def analyze_legal_risk(patent_data: dict) -> dict:
    return LegalRiskAnalyzer().analyze(patent_data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ChainCheck unified risk analyzer")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = (
        Path(args.data_dir) if args.data_dir
        else Path(__file__).resolve().parents[2] / "data"
    )
    RiskAnalyzer(data_dir).analyze_evidence()
