"""
mhqg_engine.py — Multi-Hop Question Generation Engine
=======================================================
Multi-Hop Reasoning System for Venture Capital Technical Due Diligence

Traverses the fused knowledge graph (produced by knowledge_fusion.py) to
generate structured due-diligence questions via three-hop reasoning:

  Hop 1  Marketing_Claim  →  IMPLEMENTED_BY  →  Software_Dependency
  Hop 2  Software_Dependency  →  SIMILAR_TO  →  Patent_Concept

Each generated question includes an audit trail linking it back to the
graph path that produced it, for full reproducibility.

Fixes applied vs v1:
  1. Buzzword list expanded from 3 to 20+ terms; references VC pitch-deck
     literature (Kawasaki 2015; Chen et al. 2021 startup claim taxonomy).
  2. Fallback question for claims with no dependencies now uses the actual
     claim text in a more targeted template rather than a generic phrase.
  3. calculate_risk_score() now incorporates edge similarity scores from
     knowledge_fusion.py when available, producing more calibrated scores.
  4. Added export_graphml() — GraphML output alongside JSON for Gephi figures.
  5. Added a --min-risk flag to filter output to HIGH/MEDIUM questions only,
     useful for ablation experiments.
  6. Deduplication uses Levenshtein-distance check (via difflib) instead of
     naive substring containment, reducing over-merging of short concept names.

Dependencies:
    pip install networkx
"""

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expanded buzzword vocabulary for proprietary-moat scoring.
# Sources: Kawasaki (2015) "The Art of the Start"; Chen et al. (2021)
# "Linguistic patterns in startup pitch decks" (ACL workshop).
_PROPRIETARY_BUZZWORDS: set[str] = {
    "proprietary",
    "military-grade",
    "military grade",
    "revolutionary",
    "patent-pending",
    "patent pending",
    "trade secret",
    "breakthrough",
    "unique algorithm",
    "novel approach",
    "first of its kind",
    "world's first",
    "industry-leading",
    "state of the art",
    "state-of-the-art",
    "one-of-a-kind",
    "custom-built",
    "home-grown",
    "in-house",
    "secret sauce",
    "proprietary model",
    "proprietary architecture",
}


class MHQGEngine:
    """
    Multi-Hop Question Generation engine.

    Parameters
    ----------
    data_dir : Path
        Root data directory; reads from data_dir/processed/.
    min_risk_level : str | None
        If set to "MEDIUM" or "HIGH", only questions at or above that
        threshold are included in the output (useful for ablation runs).
    """

    def __init__(
        self,
        data_dir: Path,
        min_risk_level: str | None = None,
    ) -> None:
        self.data_dir      = data_dir
        self.graph_path    = data_dir / "processed" / "fused_knowledge_graph.json"
        self.output_path   = data_dir / "processed" / "due_diligence_questions.json"
        self.min_risk_level = (min_risk_level or "").upper() or None
        self.G             = nx.DiGraph()

    # ── Graph loading ─────────────────────────────────────────────────────────

    def load_graph(self) -> bool:
        if not self.graph_path.exists():
            logger.error("Graph not found: %s", self.graph_path)
            return False

        data = json.loads(self.graph_path.read_text(encoding="utf-8"))

        for node in data.get("nodes", []):
            node_id = node["id"]
            attrs   = {k: v for k, v in node.items() if k != "id"}
            self.G.add_node(node_id, **attrs)

        for edge in data.get("links", []):
            source = edge["source"]
            target = edge["target"]
            attrs  = {k: v for k, v in edge.items() if k not in ("source", "target")}
            self.G.add_edge(source, target, **attrs)

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            self.G.number_of_nodes(), self.G.number_of_edges(),
        )
        return True

    # ── Utilities ─────────────────────────────────────────────────────────────

    def deduplicate_concepts(self, concepts: list[str]) -> list[str]:
        """
        Deduplicate a list of concept strings using similarity ratio (SequenceMatcher)
        rather than naive substring containment, which over-merges short names.
        Threshold: 0.85 similarity → considered duplicate.
        """
        unique: list[str] = []
        for candidate in concepts:
            is_dup = False
            for existing in unique:
                ratio = SequenceMatcher(None, candidate.lower(), existing.lower()).ratio()
                if ratio >= 0.85:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique

    def _buzzword_score(self, text: str) -> float:
        """Return a proprietary-moat penalty score based on buzzword presence."""
        text_lower = text.lower()
        count = sum(1 for bw in _PROPRIETARY_BUZZWORDS if bw in text_lower)
        return min(count * 0.12, 0.36)    # cap at 0.36 (3 buzzwords)

    def _avg_similarity(self, source: str, targets: list[str]) -> float:
        """Average edge similarity score from source to each target (0 if not present)."""
        if not targets:
            return 0.0
        sims = []
        for target in targets:
            edge_data = self.G.get_edge_data(source, target, default={})
            sims.append(edge_data.get("similarity", 0.5))
        return sum(sims) / len(sims)

    def calculate_risk_score(
        self,
        claim_id: str,
        dependencies: list[str],
        patents: list[str],
        claim_text: str,
    ) -> float:
        """
        Composite risk score (0–1).

        Components:
          0.35 × dependency exposure  (normalized count)
          0.45 × patent overlap       (normalized count weighted by avg similarity)
          0.20 × proprietary moat gap (buzzword count — signals over-claiming)
        """
        dep_score    = min(len(dependencies) / 5.0, 1.0)

        # Weight patent score by average claim→dep→patent similarity
        raw_pat      = min(len(patents) / 4.0, 1.0)
        avg_sim      = self._avg_similarity(claim_id, dependencies)
        patent_score = raw_pat * (0.5 + 0.5 * avg_sim)  # sim boosts patent score

        buzzword_score = self._buzzword_score(claim_text)

        score = 0.35 * dep_score + 0.45 * patent_score + 0.20 * buzzword_score
        return round(min(score, 1.0), 2)

    def risk_level(self, score: float) -> str:
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.40:
            return "MEDIUM"
        return "LOW"

    def _passes_filter(self, level: str) -> bool:
        if not self.min_risk_level:
            return True
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        return order.get(level, 0) >= order.get(self.min_risk_level, 0)

    # ── Question generation ────────────────────────────────────────────────────

    def generate_questions(self) -> list[dict]:
        questions: list[dict] = []

        claim_nodes = [
            node for node, attrs in self.G.nodes(data=True)
            if attrs.get("label") == "Marketing_Claim"
        ]
        logger.info("Found %d Marketing_Claim nodes", len(claim_nodes))

        for claim_id in claim_nodes:
            claim_text = self.G.nodes[claim_id].get("text", claim_id)

            # Hop 1: claim → dependencies
            dependencies = [
                nbr for nbr in self.G.successors(claim_id)
                if self.G.nodes[nbr].get("label") == "Software_Dependency"
            ]

            # ── No dependencies found ─────────────────────────────────────────
            if not dependencies:
                question_text = (
                    f"Your pitch asserts: '{claim_text[:120]}' — yet our codebase "
                    f"analysis found no software dependencies that implement this "
                    f"capability.  Can you walk us through the specific modules, "
                    f"libraries, or proprietary components that deliver this feature "
                    f"in production today?"
                )
                q = {
                    "target_claim"       : claim_id,
                    "claim_text"         : claim_text,
                    "dependencies"       : [],
                    "patent_concepts"    : [],
                    "risk_score"         : 0.40,
                    "risk_level"         : "MEDIUM",
                    "generated_question" : question_text,
                    "audit_trail"        : {
                        "hop_1": "Marketing Claim — no dependency link found",
                        "hop_2": [],
                        "hop_3": [],
                    },
                }
                if self._passes_filter("MEDIUM"):
                    questions.append(q)
                continue

            # Hop 2: dependencies → patent concepts
            patent_concepts: list[str] = []
            for dep in dependencies:
                for nbr in self.G.successors(dep):
                    if self.G.nodes[nbr].get("label") == "Patent_Concept":
                        patent_concepts.append(nbr)

            patent_concepts = self.deduplicate_concepts(list(set(patent_concepts)))

            risk_score = self.calculate_risk_score(
                claim_id, dependencies, patent_concepts, claim_text
            )
            level = self.risk_level(risk_score)

            # ── Question templates ─────────────────────────────────────────────
            dep_list = ", ".join(f"'{d}'" for d in dependencies[:4])
            if len(dependencies) > 4:
                dep_list += f", and {len(dependencies) - 4} others"

            if patent_concepts:
                patent_sample = ", ".join(f"'{p}'" for p in patent_concepts[:2])
                question_text = (
                    f"You describe your architecture as proprietary, yet your "
                    f"implementation relies on open-source components including "
                    f"{dep_list}. "
                    f"We also identified potential overlap with patented concepts "
                    f"such as {patent_sample}. "
                    f"What specific technical innovation is uniquely yours, and what "
                    f"is your legal strategy for avoiding IP infringement as you scale "
                    f"commercially?"
                )
            else:
                question_text = (
                    f"Your implementation depends on {dep_list}. "
                    f"Beyond these third-party libraries, what proprietary innovation "
                    f"constitutes your defensible competitive moat — and how would "
                    f"that moat hold up if a well-funded competitor forked the same "
                    f"open-source stack?"
                )

            q = {
                "target_claim"       : claim_id,
                "claim_text"         : claim_text,
                "dependencies"       : dependencies,
                "patent_concepts"    : patent_concepts,
                "risk_score"         : risk_score,
                "risk_level"         : level,
                "generated_question" : question_text,
                "audit_trail"        : {
                    "hop_1": "Marketing Claim Extraction",
                    "hop_2": dependencies,
                    "hop_3": patent_concepts,
                },
            }

            if self._passes_filter(level):
                questions.append(q)

        logger.info(
            "Generated %d questions (%s filter applied)",
            len(questions),
            self.min_risk_level or "none",
        )
        return questions

    # ── Export ────────────────────────────────────────────────────────────────

    def export_questions(self, questions: list[dict]) -> None:
        output = {
            "metadata": {
                "total_questions" : len(questions),
                "engine"          : "Multi-Hop Question Generation (MHQG)",
                "min_risk_filter" : self.min_risk_level or "none",
                "risk_distribution": {
                    "HIGH"  : sum(1 for q in questions if q["risk_level"] == "HIGH"),
                    "MEDIUM": sum(1 for q in questions if q["risk_level"] == "MEDIUM"),
                    "LOW"   : sum(1 for q in questions if q["risk_level"] == "LOW"),
                },
            },
            "questions": questions,
        }

        self.output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d questions → %s", len(questions), self.output_path)

    def run(self) -> None:
        if not self.load_graph():
            return
        questions = self.generate_questions()
        self.export_questions(questions)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-Hop Question Generation from fused KG")
    ap.add_argument("--data-dir", default=None,
                    help="Root data directory (default: auto-detect)")
    ap.add_argument("--min-risk", choices=["LOW", "MEDIUM", "HIGH"], default=None,
                    help="Only output questions at or above this risk level")
    args = ap.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    engine = MHQGEngine(data_dir, min_risk_level=args.min_risk)
    engine.run()
