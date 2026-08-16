"""Detect contradictions between proprietary claims and open-source usage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTRADICTION_TAXONOMY = {
    "auth": {
        "marketing_triggers": [
            "proprietary authentication",
            "no open-source auth",
            "custom identity",
            "in-house auth",
        ],
        "prohibited_imports": ["auth0", "okta", "flask_login", "passport", "jwt"],
    },
    "crypto": {
        "marketing_triggers": [
            # Removed bare "proprietary" — it matched ANY proprietary claim
            # in the pitch (e.g. "proprietary ranking technology"), not just
            # crypto claims, firing false positives whenever the repo
            # imported hashlib/cryptography for something unrelated (session
            # tokens, password hashing, TLS) that has nothing to do with the
            # claim being checked. Triggers now require crypto-specific
            # proprietary language, not a general presence of "proprietary."
            "proprietary encryption",
            "proprietary cryptography",
            "proprietary crypto",
            "military-grade",
            "custom encryption",
            "in-house crypto",
            "in-house encryption",
            "secret lock",
        ],
        "prohibited_imports": [
            "hashlib",
            "cryptography",
            "pycryptodome",
            "copyleft_crypto_engine",
        ],
    },
    "database": {
        "marketing_triggers": [
            "custom ledger",
            "proprietary database",
            "in-house storage",
        ],
        "prohibited_imports": ["sqlite3", "pymongo", "sqlalchemy", "redis"],
    },
}


class ProprietaryContradictionDetector:
    """Find marketing claims contradicted by dependency evidence."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.G = nx.DiGraph()
        self.load_graph()

    def load_graph(self) -> None:
        # Prefer the rich fused graph (91 codebase dependency nodes, full
        # claim text) over kg.json, which is built only from entity_resolver's
        # sparse cross-domain matches and rarely contains enough Library
        # nodes to catch real proprietary-vs-open-source contradictions.
        graph_path = self.data_dir / "processed" / "fused_knowledge_graph.json"
        if not graph_path.exists():
            graph_path = self.data_dir / "processed" / "kg.json"
        if not graph_path.exists():
            logger.error("No KG found at fused_knowledge_graph.json or kg.json.")
            return

        data = json.loads(graph_path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            self.G.add_node(node["id"], **node)
        for edge in data.get("edges", data.get("links", [])):
            self.G.add_edge(edge["source"], edge["target"], **edge)

    def detect_proprietary_mismatches(self) -> list[dict]:
        logger.info("Initializing proprietary contradiction engine...")
        claims = [
            n for n, d in self.G.nodes(data=True)
            if d.get("label") == "Marketing_Claim" or d.get("node_type") == "Claim"
        ]
        code_modules = [
            n for n, d in self.G.nodes(data=True)
            if d.get("label") in {"Code_Module", "Software_Dependency"}
            or d.get("node_type") == "Library"
        ]
        discovered_contradictions: list[dict] = []

        for claim in claims:
            attrs = self.G.nodes[claim]
            claim_text = (
                attrs.get("full_text")
                or attrs.get("text")
                or attrs.get("label")
                or claim
            ).lower()

            for category, taxonomy in CONTRADICTION_TAXONOMY.items():
                if not any(trigger in claim_text for trigger in taxonomy["marketing_triggers"]):
                    continue

                for module in code_modules:
                    if any(prohibited in module.lower() for prohibited in taxonomy["prohibited_imports"]):
                        logger.warning(
                            "Contradiction found: claimed proprietary %s but used %s.",
                            category,
                            module,
                        )
                        discovered_contradictions.append({
                            "risk_type": "Proprietary Claim Mismatch",
                            "severity": "MEDIUM",  # topic-adjacency, not a verified functional match — HIGH overstated confidence
                            "claim_id": claim,
                            "claim_text": attrs.get("full_text") or attrs.get("text") or claim,
                            "contradictory_module": module,
                            "confidence_score": 0.6,  # keyword co-occurrence, not confirmed functional overlap
                            "recommended_action": (
                                f"Ask whether {module} is used to implement the specific "
                                f"{category} functionality described as proprietary, or "
                                "for an unrelated purpose elsewhere in the codebase."
                            ),
                        })

        output_path = self.data_dir / "processed" / "contradiction_evidence.json"
        output_path.write_text(
            json.dumps({"contradictions": discovered_contradictions}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Exported %d proprietary contradictions to %s",
            len(discovered_contradictions),
            output_path,
        )
        return discovered_contradictions


def detect_contradictions(data_dir: Path | str) -> list[dict]:
    """Functional wrapper for contradiction detection."""
    return ProprietaryContradictionDetector(Path(data_dir)).detect_proprietary_mismatches()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    detector = ProprietaryContradictionDetector(project_root / "data")
    detector.detect_proprietary_mismatches()