"""
path_reasoner.py — Multi-Hop Graph Reasoner
===========================================
Multi-Hop Reasoning System for VC Technical Due Diligence

Walks the knowledge graph (kg.json) starting from every Claim node,
using BFS to discover chains up to MAX_HOPS deep. Each chain represents
a reasoning path: Claim → Library → Patent → LicenceType.

Scores each chain by multiplying edge weights (cosine scores).
Filters low-confidence chains before passing to generation/question_gen.py.

Output:
    data/processed/hop_chains.json

Usage:
    python src/reasoning/path_reasoner.py
    python src/reasoning/path_reasoner.py --kg data/processed/kg.json --max-hops 3
"""

import json
import logging
import argparse
import itertools
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_HOPS        = 3      # maximum chain length (Claim → Lib → Patent = 2 hops)
CHAIN_THRESHOLD = 0.45   # minimum product-of-edge-weights to keep a chain
TOP_K_PER_CLAIM = 5      # max chains to keep per starting Claim node
TOP_K_GLOBAL    = 50     # max chains total passed to question_gen


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class HopChain:
    """A single scored multi-hop reasoning path through the KG."""
    chain_id: str
    start_node: str              # always a Claim node
    path_nodes: list[str]        # ordered list of node IDs
    path_node_types: list[str]   # node_type for each node in path
    path_edges: list[str]        # edge_type for each edge in path
    chain_score: float           # product of edge weights
    hop_count: int
    has_licence_conflict: bool   # True if path touches a high-risk LicenceType
    has_patent_node: bool        # True if path crosses a Patent node
    provenance: dict             # full metadata for audit trail


# ── Graph loader ──────────────────────────────────────────────────────────────

def load_graph(kg_path: Path) -> nx.DiGraph:
    """Load kg.json back into a NetworkX DiGraph."""
    data = json.loads(kg_path.read_text(encoding="utf-8"))
    G = nx.DiGraph()

    for node in data.get("nodes", []):
        node_id = node.pop("id")
        G.add_node(node_id, **node)

    for edge in data.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)

    logger.info(
        "Loaded KG: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ── BFS traversal ─────────────────────────────────────────────────────────────

def bfs_hop_chains(
    G: nx.DiGraph,
    start_node: str,
    max_hops: int = MAX_HOPS,
) -> list[list[str]]:
    """
    BFS from start_node up to max_hops depth.
    Returns all simple paths (no cycles) as lists of node IDs.
    Only follows outgoing edges.
    """
    # Queue items: (current_node, path_so_far, visited_set)
    queue   = deque([(start_node, [start_node], {start_node})])
    paths   = []

    while queue:
        current, path, visited = queue.popleft()

        # Record this path if it's longer than just the start node
        if len(path) > 1:
            paths.append(path[:])

        # Stop expanding if we've hit the hop limit
        if len(path) - 1 >= max_hops:
            continue

        for neighbor in G.successors(current):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor], visited | {neighbor}))

    return paths


LENGTH_PENALTY_FACTOR = 0.92   # multiplicative penalty applied per hop beyond 1

def score_chain(G: nx.DiGraph, path: list[str]) -> float:
   
    score = 1.0
    for i in range(len(path) - 1):
        u, v   = path[i], path[i + 1]
        weight = G[u][v].get("weight", 0.5)
        score *= weight
    hop_count = len(path) - 1
    geo_mean  = score ** (1.0 / max(hop_count, 1))
    length_penalty = LENGTH_PENALTY_FACTOR ** max(hop_count - 1, 0)
    return round(geo_mean * length_penalty, 4)


def build_provenance(G: nx.DiGraph, path: list[str]) -> dict:
    """
    Build the audit trail for a hop chain.
    Records every node's metadata and every edge's type + weight.
    This is the object that proves how the question was generated.
    """
    node_details = []
    for node_id in path:
        node_data = dict(G.nodes[node_id])
        node_details.append({
            "node_id":   node_id,
            "node_type": node_data.get("node_type"),
            "label":     node_data.get("label", node_id),
            "metadata":  {k: v for k, v in node_data.items()
                          if k not in ("node_type", "label")
                          and isinstance(v, (str, int, float, bool, type(None)))},
        })

    edge_details = []
    for i in range(len(path) - 1):
        u, v       = path[i], path[i + 1]
        edge_data  = dict(G[u][v])
        edge_details.append({
            "from":      u,
            "to":        v,
            "edge_type": edge_data.get("edge_type"),
            "weight":    edge_data.get("weight"),
            "entity_a":  edge_data.get("entity_text_a"),
            "entity_b":  edge_data.get("entity_text_b"),
        })

    return {
        "nodes": node_details,
        "edges": edge_details,
        "hop_count": len(path) - 1,
    }


# ── Chain classifier ──────────────────────────────────────────────────────────

def classify_chain(G: nx.DiGraph, path: list[str]) -> tuple[bool, bool]:
    """
    Returns (has_licence_conflict, has_patent_node).
    Used to prioritize chains most relevant for VC due diligence questions.
    """
    has_conflict = False
    has_patent   = False

    for node_id in path:
        node_data = G.nodes[node_id]
        nt = node_data.get("node_type", "")
        if nt == "Patent":
            has_patent = True
        if nt == "LicenceType" and node_data.get("risk") == "high":
            has_conflict = True
        if nt == "Library" and node_data.get("licence_risk") == "high":
            has_conflict = True

    return has_conflict, has_patent


# ── Main reasoner ─────────────────────────────────────────────────────────────

def reason(
    G: nx.DiGraph,
    max_hops: int = MAX_HOPS,
    chain_threshold: float = CHAIN_THRESHOLD,
    top_k_per_claim: int = TOP_K_PER_CLAIM,
    top_k_global: int = TOP_K_GLOBAL,
) -> list[HopChain]:
    """
    Run multi-hop reasoning over the full KG.
    Starts BFS from every Claim node, scores all paths, filters and ranks.
    """
    claim_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "Claim"
    ]
    logger.info("Starting BFS from %d Claim nodes", len(claim_nodes))

    all_chains: list[HopChain] = []
    chain_counter = 0

    for claim_node in claim_nodes:
        paths = bfs_hop_chains(G, claim_node, max_hops=max_hops)
        scored: list[tuple[float, list[str]]] = []

        for path in paths:
            score = score_chain(G, path)
            if score >= chain_threshold:
                scored.append((score, path))

        # Keep top-K chains per claim, sorted by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k_per_claim]

        for score, path in scored:
            chain_counter += 1
            node_types = [G.nodes[n].get("node_type", "unknown") for n in path]
            edge_types = [
                G[path[i]][path[i + 1]].get("edge_type", "unknown")
                for i in range(len(path) - 1)
            ]
            has_conflict, has_patent = classify_chain(G, path)
            provenance = build_provenance(G, path)

            all_chains.append(HopChain(
                chain_id=f"chain_{chain_counter:04d}",
                start_node=claim_node,
                path_nodes=path,
                path_node_types=node_types,
                path_edges=edge_types,
                chain_score=score,
                hop_count=len(path) - 1,
                has_licence_conflict=has_conflict,
                has_patent_node=has_patent,
                provenance=provenance,
            ))

    # Global sort: prioritize high-risk licence conflicts + patent nodes first
    all_chains.sort(
        key=lambda c: (
            c.has_licence_conflict,
            c.has_patent_node,
            c.chain_score,
        ),
        reverse=True,
    )
    all_chains = all_chains[:top_k_global]

    logger.info(
        "Hop reasoning complete: %d chains kept (threshold=%.2f)",
        len(all_chains), chain_threshold
    )
    return all_chains


# ── Output ────────────────────────────────────────────────────────────────────

def save_chains(chains: list[HopChain], output_path: Path) -> None:
    output = {
        "metadata": {
            "total_chains": len(chains),
            "chains_with_licence_conflict": sum(1 for c in chains if c.has_licence_conflict),
            "chains_with_patent_node":      sum(1 for c in chains if c.has_patent_node),
            "avg_chain_score": round(
                sum(c.chain_score for c in chains) / max(len(chains), 1), 4
            ),
            "hop_distribution": {
                str(h): sum(1 for c in chains if c.hop_count == h)
                for h in sorted(set(c.hop_count for c in chains))
            },
        },
        "chains": [asdict(c) for c in chains],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Saved %d hop chains → %s", len(chains), output_path)


def print_summary(chains: list[HopChain]) -> None:
    print(f"\n{'═'*70}")
    print("  HOP CHAINS SUMMARY")
    print(f"{'═'*70}")
    print(f"  Total chains       : {len(chains)}")
    print(f"  Licence conflicts  : {sum(1 for c in chains if c.has_licence_conflict)}")
    print(f"  Patent crossings   : {sum(1 for c in chains if c.has_patent_node)}")

    print(f"\n  Top 5 chains:")
    for chain in chains[:5]:
        path_str = " → ".join(
            f"{n}({t})" for n, t in
            zip(chain.path_nodes, chain.path_node_types)
        )
        flags = []
        if chain.has_licence_conflict:
            flags.append("LICENCE⚠")
        if chain.has_patent_node:
            flags.append("PATENT")
        flag_str = " ".join(flags)
        print(f"\n  [{chain.chain_id}] score={chain.chain_score}  {flag_str}")
        print(f"     {path_str[:90]}")
        print(f"     edges: {' → '.join(chain.path_edges)}")
    print(f"{'═'*70}\n")


class DynamicPathReasoner:
    """
    Ujwal-style dynamic traversal over a fused graph.

    This class exports ``structured_evidence.json`` for downstream scoring. It
    supports both this repo's ``kg.json`` node-link shape and the older
    ``fused_knowledge_graph.json`` shape.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.G = nx.DiGraph()
        self.load_graph()

    def _resolve_graph_path(self) -> Path:
        kg_path = self.data_dir / "processed" / "kg.json"
        fused_path = self.data_dir / "processed" / "fused_knowledge_graph.json"
        return kg_path if kg_path.exists() else fused_path

    def load_graph(self) -> None:
        graph_path = self._resolve_graph_path()
        if not graph_path.exists():
            logger.error("Graph not found. Expected kg.json or fused_knowledge_graph.json.")
            return

        data = json.loads(graph_path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            self.G.add_node(node["id"], **node)
        for edge in data.get("edges", data.get("links", [])):
            self.G.add_edge(edge["source"], edge["target"], **edge)

        logger.info(
            "Loaded graph for dynamic traversal: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    def calculate_path_confidence(self, path: list[str]) -> float:
        """Calculate confidence from edge weights/similarity with path decay."""
        base_confidence = 1.0
        for i in range(len(path) - 1):
            edge_data = self.G.get_edge_data(path[i], path[i + 1], default={})
            sim_score = edge_data.get("similarity", edge_data.get("weight", 0.95))
            base_confidence *= sim_score

        length_penalty = 0.9 ** max(len(path) - 2, 0)
        return round(base_confidence * length_penalty, 3)

    def _node_role(self, node_id: str) -> str:
        attrs = self.G.nodes[node_id]
        return attrs.get("label") or attrs.get("node_type") or ""

    def discover_evidence_chains(self, max_depth: int = 4, per_pair_limit: int = 50) -> list[dict]:
        """
        Find simple paths from claims to patent/license risk nodes.

        Paths are sorted by confidence so downstream stages can consume the
        highest-signal evidence first.
        """
        logger.info("Discovering dynamic evidence chains (max depth=%d)", max_depth)
        claims = [
            n for n, d in self.G.nodes(data=True)
            if d.get("label") == "Marketing_Claim" or d.get("node_type") == "Claim"
        ]
        risk_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("label") in ["Patent_Concept", "OpenSource_License"]
            or d.get("node_type") in ["Patent", "LicenceType"]
        ]

        evidence_objects: list[dict] = []
        for claim in claims:
            for risk_node in risk_nodes:
                try:
                    paths = list(itertools.islice(
                        nx.all_simple_paths(
                            self.G,
                            source=claim,
                            target=risk_node,
                            cutoff=max_depth,
                        ),
                        per_pair_limit,
                    ))
                except (nx.NodeNotFound, nx.NetworkXError, nx.NetworkXNoPath):
                    continue

                for path in paths:
                    risk_role = self._node_role(risk_node)
                    if risk_role in {"Patent_Concept", "Patent"}:
                        risk_type = "IP Overlap"
                    elif risk_role in {"OpenSource_License", "LicenceType"}:
                        risk_type = "Commercial License"
                    else:
                        risk_type = risk_role or "Unknown"

                    claim_attrs = self.G.nodes[claim]
                    evidence_objects.append({
                        "claim_id": claim,
                        "claim_text": (
                            claim_attrs.get("full_text")
                            or claim_attrs.get("text")
                            or claim_attrs.get("label")
                            or claim
                        ),
                        "risk_node": risk_node,
                        "risk_type": risk_type,
                        "path_length": len(path) - 1,
                        "confidence_score": self.calculate_path_confidence(path),
                        "reasoning_path": path,
                    })

        evidence_objects.sort(key=lambda x: x["confidence_score"], reverse=True)
        return evidence_objects

    def export_evidence(self) -> list[dict]:
        evidence_chains = self.discover_evidence_chains()
        output_path = self.data_dir / "processed" / "structured_evidence.json"
        output_path.write_text(
            json.dumps({"evidence_objects": evidence_chains}, indent=2),
            encoding="utf-8",
        )
        logger.info("Exported %d structured evidence objects to %s", len(evidence_chains), output_path)
        return evidence_chains


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-hop BFS reasoner over the VC due-diligence knowledge graph"
    )
    parser.add_argument("--kg",      default="data/processed/kg.json")
    parser.add_argument("--output",  default="data/processed/hop_chains.json")
    parser.add_argument("--max-hops",    type=int,   default=MAX_HOPS)
    parser.add_argument("--threshold",   type=float, default=CHAIN_THRESHOLD)
    parser.add_argument("--top-k",       type=int,   default=TOP_K_GLOBAL)
    args = parser.parse_args()

    G      = load_graph(Path(args.kg))
    chains = reason(
        G,
        max_hops=args.max_hops,
        chain_threshold=args.threshold,
        top_k_global=args.top_k,
    )
    print_summary(chains)
    save_chains(chains, Path(args.output))


if __name__ == "__main__":
    main()


