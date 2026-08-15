"""
kg_builder.py — Knowledge Graph Builder
========================================
Multi-Hop Reasoning System for VC Technical Due Diligence

Takes entity_matches.json (from entity_resolver.py) and builds a typed
directed NetworkX graph where nodes are entities and edges are semantic
relationships inferred from their domain context.

Node types:   Claim | Library | Patent | LicenceType
Edge types:   implements · cites · conflicts_with · licenced_under · similar_to

Output:
    data/processed/kg.json          (node-link format for path_reasoner)
    data/processed/kg_summary.json  (human-readable stats)

Usage:
    python src/graph/kg_builder.py
    python src/graph/kg_builder.py --matches data/processed/entity_matches.json
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.schema import (
    EDGE_LICENSED_UNDER,
    EDGE_POTENTIALLY_IMPLEMENTED_BY,
    EDGE_REQUIRES_IP_REVIEW,
    EDGE_SUPPORTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Licence risk classification ───────────────────────────────────────────────
# Used to add LicenceType nodes and flag commercial-use conflicts for VC queries

LICENCE_RISK = {
    # High risk — commercial use restricted
    "gpl-2.0":       "high",
    "gpl-3.0":       "high",
    "agpl-3.0":      "high",
    "lgpl-2.1":      "medium",
    "lgpl-3.0":      "medium",
    "cc-by-nc":      "high",
    "cc-by-nc-sa":   "high",
    "sspl":          "high",
    "busl":          "high",
    # Low risk — permissive
    "mit":           "low",
    "apache-2.0":    "low",
    "bsd-2-clause":  "low",
    "bsd-3-clause":  "low",
    "isc":           "low",
    "mpl-2.0":       "low",
    "cc0-1.0":       "low",
    "unlicense":     "low",
}

COMMON_LIBRARY_LICENCES = {
    # Maps library name → known SPDX licence identifier
    # Used when licence metadata is absent from github_parser output
    "torch":               "bsd-3-clause",
    "tensorflow":          "apache-2.0",
    "transformers":        "apache-2.0",
    "sentence-transformers": "apache-2.0",
    "faiss":               "mit",
    "networkx":            "bsd-3-clause",
    "spacy":               "mit",
    "langchain":           "mit",
    "openai":              "mit",
    "anthropic":           "mit",
    "fastapi":             "mit",
    "flask":               "bsd-3-clause",
    "django":              "bsd-3-clause",
    "numpy":               "bsd-3-clause",
    "pandas":              "bsd-3-clause",
    "scikit-learn":        "bsd-3-clause",
    "chromadb":            "apache-2.0",
    "pinecone":            "apache-2.0",
    "redis":               "bsd-3-clause",
    "neo4j":               "gpl-3.0",      # ← flagged as high risk
    "py2neo":              "apache-2.0",
    "pydantic":            "mit",
    "sqlalchemy":          "mit",
    "boto3":               "apache-2.0",
    "pytest":              "mit",
    "requests":            "apache-2.0",
    "aiohttp":             "apache-2.0",
    "grpcio":              "apache-2.0",
    "protobuf":            "bsd-3-clause",
    "libp2p":              "mit",
    "cryptography":        "apache-2.0",
    "pycryptodome":        "bsd-2-clause",
}


# ── Edge type inference ───────────────────────────────────────────────────────

def infer_edge_type(domain_a: str, domain_b: str, score: float) -> str:
    """
    Infer a semantic edge label from the two domains being connected.
    This is the typed relationship that path_reasoner will traverse.
    """
    pair = tuple(sorted([domain_a, domain_b]))
    if pair == ("codebase", "whitepaper"):
        return EDGE_POTENTIALLY_IMPLEMENTED_BY
    if pair == ("patent", "whitepaper"):
        return EDGE_REQUIRES_IP_REVIEW
    if pair == ("codebase", "patent"):
        if score >= 0.88:
            return EDGE_REQUIRES_IP_REVIEW
        return EDGE_SUPPORTS
    return EDGE_SUPPORTS


# ── Node builders ─────────────────────────────────────────────────────────────

def _load_claim_text_lookup(whitepaper_path: Path | None) -> dict:
    """
    Build a {claim_id: claim_text} lookup from the parsed whitepaper JSON
    (e.g. data/processed/<name>_parsed.json). Tries several plausible field
    names since the exact schema key isn't guaranteed, and fails soft —
    returning an empty dict — if the file is missing or unreadable, so a
    missing whitepaper file never crashes graph construction.
    """
    lookup: dict = {}
    if not whitepaper_path or not whitepaper_path.exists():
        return lookup

    try:
        data = json.loads(whitepaper_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read whitepaper file %s for claim text: %s", whitepaper_path, e)
        return lookup

    claims = data.get("technical_claims", data.get("claims", data.get("marketing_claims", [])))
    for i, c in enumerate(claims, start=1):
        cid = c.get("claim_id") or c.get("id") or f"claim_{i:04d}"
        text = (
            c.get("sentence")
            or c.get("text")
            or c.get("claim_text")
            or c.get("full_text")
            or c.get("content")
            or c.get("statement")
        )
        if text:
            lookup[cid] = text.strip()

    if not lookup:
        logger.warning(
            "Whitepaper file %s loaded but no claim text fields recognized — "
            "Claim nodes will fall back to their bare IDs.", whitepaper_path
        )
    else:
        logger.info("Loaded %d claim texts from %s", len(lookup), whitepaper_path)

    return lookup


def _add_claim_node(G: nx.DiGraph, source_id: str, meta: dict, source_file: str,
                     claim_text_lookup: dict | None = None):
    if not G.has_node(source_id):
        # Real sentence text, in priority order: an explicit lookup built from
        # the parsed whitepaper JSON, then whatever entity_resolver happened
        # to carry forward in meta, then finally the bare ID as a last resort
        # (so downstream code always has *something* rather than a KeyError).
        claim_text_lookup = claim_text_lookup or {}
        real_text = (
            claim_text_lookup.get(source_id)
            or meta.get("text")
            or meta.get("claim_text")
            or meta.get("full_text")
            or source_id
        )
        G.add_node(source_id,
            node_type="Claim",
            label=source_id,
            full_text=real_text,
            claim_type=meta.get("claim_type", "general"),
            confidence=meta.get("confidence", 0.0),
            page=meta.get("page"),
            source_file=source_file,
        )


def _add_library_node(G: nx.DiGraph, name: str, meta: dict, source_file: str):
    if not G.has_node(name):
        licence = COMMON_LIBRARY_LICENCES.get(name.lower(), "unknown")
        risk    = LICENCE_RISK.get(licence, "unknown")
        G.add_node(name,
            node_type="Library",
            label=name,
            category=meta.get("category", "Utility / Other"),
            ecosystem=meta.get("ecosystem", "unknown"),
            licence=licence,
            licence_risk=risk,
            source_file=source_file,
        )
        # Add LicenceType node and edge if licence is known
        if licence != "unknown":
            _add_licence_node(G, licence)
            if not G.has_edge(name, licence):
                G.add_edge(name, licence,
                    edge_type=EDGE_LICENSED_UNDER,
                    weight=1.0,
                    risk=risk,
                )


def _add_patent_node(G: nx.DiGraph, patent_id: str, meta: dict, source_file: str):
    node_id = f"patent:{patent_id}"
    new_context = {
        "relationship": meta.get("relationship", ""),
        "tail_entity": meta.get("tail") or meta.get("head", ""),
        "source_sentence": meta.get("source_sentence", "")[:200],
    }
    if not G.has_node(node_id):
        # Prefer a real sentence from the patent over the bare ID, so
        # downstream question generation has genuine content to reason
        # from instead of hallucinating a title/claim that doesn't exist.
        real_text = new_context["source_sentence"] or meta.get("title") or patent_id
        G.add_node(node_id,
            node_type="Patent",
            label=patent_id,
            full_text=real_text,
            relationship=new_context["relationship"],
            tail_entity=new_context["tail_entity"],
            source_sentence=new_context["source_sentence"],
            source_file=source_file,
            matched_contexts=[new_context],
        )
    else:
        G.nodes[node_id]["matched_contexts"].append(new_context)
    return node_id


def _add_licence_node(G: nx.DiGraph, licence: str):
    if not G.has_node(licence):
        risk = LICENCE_RISK.get(licence, "unknown")
        G.add_node(licence,
            node_type="LicenceType",
            label=licence,
            risk=risk,
            commercial_use_restricted=(risk == "high"),
        )


# ── Core graph builder ────────────────────────────────────────────────────────

def build_knowledge_graph(matches_path: Path, whitepaper_path: Path | None = None) -> nx.DiGraph:
    """
    Main function. Reads entity_matches.json and builds the KG.

    For each match:
      1. Add node A (typed by domain)
      2. Add node B (typed by domain)
      3. Add directed edge A → B with inferred type + cosine score as weight
      4. For Library nodes: add LicenceType node + LICENSED_UNDER edge

    If whitepaper_path is given, Claim nodes are enriched with their real
    sentence text (see _load_claim_text_lookup) instead of just their ID.
    """
    data = json.loads(matches_path.read_text(encoding="utf-8"))
    matches = data.get("matches", [])
    logger.info("Loading %d matches from %s", len(matches), matches_path)

    claim_text_lookup = _load_claim_text_lookup(whitepaper_path)

    G = nx.DiGraph()
    edge_type_counts: dict[str, int] = defaultdict(int)
    skipped = 0

    for m in matches:
        domain_a    = m.get("domain_a", "")
        domain_b    = m.get("domain_b", "")
        entity_a    = m.get("entity_a", "").strip()
        entity_b    = m.get("entity_b", "").strip()
        source_id_a = m.get("source_id_a", "")
        source_id_b = m.get("source_id_b", "")
        score       = float(m.get("cosine_score", 0.0))
        prov        = m.get("provenance", {})
        meta_a      = prov.get("entity_a_meta", {})
        meta_b      = prov.get("entity_b_meta", {})
        file_a      = prov.get("entity_a_file", "")
        file_b      = prov.get("entity_b_file", "")

        if not entity_a or not entity_b:
            skipped += 1
            continue

        # ── Add nodes by domain ──────────────────────────────────────────
        node_id_a = None
        node_id_b = None

        if domain_a == "whitepaper":
            node_id_a = source_id_a   # e.g. "claim_0001"
            _add_claim_node(G, node_id_a, meta_a, file_a, claim_text_lookup)
        elif domain_a == "codebase":
            node_id_a = entity_a      # library name
            _add_library_node(G, node_id_a, meta_a, file_a)
        elif domain_a == "patent":
            node_id_a = _add_patent_node(G, source_id_a, meta_a, file_a)

        if domain_b == "whitepaper":
            node_id_b = source_id_b
            _add_claim_node(G, node_id_b, meta_b, file_b, claim_text_lookup)
        elif domain_b == "codebase":
            node_id_b = entity_b
            _add_library_node(G, node_id_b, meta_b, file_b)
        elif domain_b == "patent":
            node_id_b = _add_patent_node(G, source_id_b, meta_b, file_b)

        if node_id_a is None or node_id_b is None:
            skipped += 1
            continue

        # ── Reject unverified claim↔patent shortcuts ─────────────────────
        # A claim and a patent being semantically similar is a candidate
        # signal, not proof of a relationship (see knowledge_fusion.py
        # improvement notes). Without codebase/library evidence in between,
        # this pair does NOT get a direct graph edge — only a real
        # Claim → Library → Patent chain should let path_reasoner connect
        # a marketing claim to a patent.
        pair = tuple(sorted([domain_a, domain_b]))
        if pair == ("patent", "whitepaper"):
            skipped += 1
            continue

        # ── Add typed edge ───────────────────────────────────────────────
        edge_type = infer_edge_type(domain_a, domain_b, score)

        
        if G.has_edge(node_id_a, node_id_b):
            edge_data = G[node_id_a][node_id_b]
            # Always append this match to the full contributing-evidence list
            edge_data.setdefault("all_matches", []).append({
                "entity_text_a": entity_a,
                "entity_text_b": entity_b,
                "cosine_score": score,
                "normalized_a": prov.get("normalized_a"),
                "normalized_b": prov.get("normalized_b"),
            })
            # Only promote this match to "primary" if it's the new best score
            if score > edge_data["weight"]:
                edge_data["weight"] = score
                edge_data["entity_text_a"] = entity_a
                edge_data["entity_text_b"] = entity_b
                edge_data["provenance"] = {
                    "normalized_a": prov.get("normalized_a"),
                    "normalized_b": prov.get("normalized_b"),
                    "cosine_score": score,
                }
        else:
            G.add_edge(node_id_a, node_id_b,
                edge_type=edge_type,
                weight=score,
                entity_text_a=entity_a,
                entity_text_b=entity_b,
                provenance={
                    "normalized_a": prov.get("normalized_a"),
                    "normalized_b": prov.get("normalized_b"),
                    "cosine_score": score,
                },
                all_matches=[{
                    "entity_text_a": entity_a,
                    "entity_text_b": entity_b,
                    "cosine_score": score,
                    "normalized_a": prov.get("normalized_a"),
                    "normalized_b": prov.get("normalized_b"),
                }],
            )
            edge_type_counts[edge_type] += 1

    logger.info(
        "KG built: %d nodes, %d edges, %d skipped",
        G.number_of_nodes(), G.number_of_edges(), skipped
    )
    return G


# ── Serialization ─────────────────────────────────────────────────────────────

def graph_to_json(G: nx.DiGraph) -> dict:
    """Convert to JSON-serializable node-link format."""
    return {
        "directed": True,
        "nodes": [
            {"id": n, **{k: v for k, v in d.items()
                         if isinstance(v, (str, int, float, bool, type(None), list, dict))}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v,
             **{k: v2 for k, v2 in d.items()
                if isinstance(v2, (str, int, float, bool, type(None), list, dict))}}
            for u, v, d in G.edges(data=True)
        ],
    }


def build_summary(G: nx.DiGraph) -> dict:
    """Build human-readable summary for inspection."""
    node_types: dict[str, int] = defaultdict(int)
    edge_types: dict[str, int] = defaultdict(int)
    high_risk_libs = []

    for n, d in G.nodes(data=True):
        node_types[d.get("node_type", "unknown")] += 1
        if d.get("node_type") == "Library" and d.get("licence_risk") == "high":
            high_risk_libs.append({
                "library": n,
                "licence": d.get("licence"),
                "category": d.get("category"),
            })

    for _, _, d in G.edges(data=True):
        edge_types[d.get("edge_type", "unknown")] += 1

    # Find highest-degree nodes (most connected = most relevant to hops)
    in_deg  = sorted(G.in_degree(),  key=lambda x: x[1], reverse=True)[:10]
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "high_risk_libraries": high_risk_libs,
        "hub_nodes_by_in_degree":  [{"node": n, "in_degree": d}  for n, d in in_deg],
        "hub_nodes_by_out_degree": [{"node": n, "out_degree": d} for n, d in out_deg],
        "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        "weakly_connected_components": nx.number_weakly_connected_components(G) if G.number_of_nodes() > 0 else 0,
    }


def print_summary(summary: dict) -> None:
    print(f"\n{'═'*68}")
    print("  KNOWLEDGE GRAPH SUMMARY")
    print(f"{'═'*68}")
    print(f"  Nodes : {summary['total_nodes']}   Edges : {summary['total_edges']}")
    print(f"  Weakly connected components : {summary['weakly_connected_components']}")

    print(f"\n  Node types:")
    for nt, count in summary["node_types"].items():
        print(f"     {nt:<20} {count:>4}")

    print(f"\n  Edge types:")
    for et, count in summary["edge_types"].items():
        print(f"     {et:<25} {count:>4}")

    if summary["high_risk_libraries"]:
        print(f"\n  ⚠️  High-risk licence libraries (commercial-use restricted):")
        for lib in summary["high_risk_libraries"]:
            print(f"     {lib['library']:<25} {lib['licence']}")

    print(f"\n  Top hub nodes (in-degree):")
    for h in summary["hub_nodes_by_in_degree"][:5]:
        print(f"     {h['node']:<40} ← {h['in_degree']} edges")
    print(f"{'═'*68}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build typed knowledge graph from entity_matches.json"
    )
    parser.add_argument("--matches", "-m",
        default="data/processed/entity_matches.json",
        help="Path to entity_matches.json (entity_resolver output)"
    )
    parser.add_argument("--whitepaper", "-w",
        default=None,
        help="Path to the parsed whitepaper JSON (e.g. data/processed/<name>_parsed.json). "
             "If omitted, the first *_parsed.json found in the output directory is used."
    )
    parser.add_argument("--output", "-o",
        default="data/processed/",
        help="Output directory for kg.json and kg_summary.json"
    )
    args = parser.parse_args()

    matches_path = Path(args.matches)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    whitepaper_path = Path(args.whitepaper) if args.whitepaper else None
    if whitepaper_path is None:
        candidates = sorted(output_dir.glob("*_parsed.json"))
        if candidates:
            whitepaper_path = candidates[0]
            logger.info("Auto-detected whitepaper file: %s", whitepaper_path)
        else:
            logger.warning(
                "No *_parsed.json found in %s — Claim nodes will fall back to bare IDs. "
                "Pass --whitepaper explicitly if your parsed file lives elsewhere.",
                output_dir,
            )

    G       = build_knowledge_graph(matches_path, whitepaper_path)
    summary = build_summary(G)
    print_summary(summary)

    kg_path = output_dir / "kg.json"
    kg_path.write_text(
        json.dumps(graph_to_json(G), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Saved knowledge graph → %s", kg_path)

    summary_path = output_dir / "kg_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info("Saved KG summary → %s", summary_path)


if __name__ == "__main__":
    main()