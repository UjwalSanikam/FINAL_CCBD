"""
knowledge_fusion.py — Cross-Domain Knowledge Graph Builder
============================================================
Multi-Hop Reasoning System for Venture Capital Technical Due Diligence

MERGED VERSION — combines:
  • Samhitha's KnowledgeFusionPipeline  (semantic cosine bridging, clean loaders)
  • Ujwal's   SemanticFusionPipeline    (FAISS-accelerated bridging, Track 4 licenses,
                                         import-graph ingestion, IndexFlatIP cosine)

Fuses four heterogeneous knowledge tracks into a single directed graph:
  Track 1  Marketing claims     (whitepaper_parser.py  → *_parsed.json)
  Track 2  Codebase modules     (github_parser.py      → codebase_knowledge.json)
  Track 3  Patent concepts      (patent_parser.py      → knowledge_base.json)
  Track 4  Open-source licenses (license_parser.py     → license_knowledge.json)

Every node carries BOTH attribute keys so all downstream consumers work
without changes regardless of which schema they expect:
  • `label`     — Samhitha / mhqg_engine  schema  ("Marketing_Claim" etc.)
  • `node_type` — Vaibhav  / hop_reasoner schema  ("Claim" etc.)

Bridge layers
  Layer 1  Claim → Dependency/Module   IMPLEMENTED_BY      (semantic FAISS)
  Layer 2  Dependency/Module → Patent  SIMILAR_TO          (semantic FAISS)
  Layer 3  Module → Patent             REQUIRES_IP_REVIEW  (Ujwal's clean-path variant)

Key improvements over both originals:
  1. Dual-label on every node (Option C decision) — all downstream files
     (mhqg_engine, hop_reasoner, contradiction_detector) work without edits.
  2. FAISS IndexFlatIP (inner-product on L2-normalised vectors == cosine)
     replaces the O(n²) numpy loop — significantly faster on large graphs.
  3. Ujwal's Track 4 license ingestion added as an optional fourth data source.
  4. Ujwal's import-graph loader kept alongside Samhitha's dependency-map
     loader — both are ingested and deduplicated so no data is lost.
  5. Pitch-deck loader is now filename-agnostic (globs for *_parsed.json)
     instead of hardcoding "expense_ninja_pitch_parsed.json".
  6. EXPIRED patents are excluded from bridge layers (no false risk signals).
  7. Configurable similarity threshold via --threshold CLI flag (default 0.40).
  8. Dual export: JSON (fused_knowledge_graph.json) + GraphML (Gephi figures).
  9. Per-layer edge-count logging for debugging / ablation experiments.

Dependencies:
    pip install networkx sentence-transformers numpy faiss-cpu
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_NAME         = "all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD  = 0.40   # raised from Samhitha's 0.35; keeps precision higher
_BRIDGE_TOP_K       = 3      # how many candidates to consider per node in FAISS search
_HASH_EMBEDDING_DIM = 384    # match MiniLM dimensionality for downstream FAISS logic


class _HashingTextEncoder:
    """Deterministic local fallback when the sentence-transformer model is unavailable."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        vectors = np.zeros((len(texts), _HASH_EMBEDDING_DIM), dtype="float32")
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_]+", text.lower())
            if not tokens:
                tokens = [text.lower() or "<empty>"]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % _HASH_EMBEDDING_DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, bucket] += sign
        return vectors

# ── Dual-label map ────────────────────────────────────────────────────────────
# Every node gets BOTH label systems so no downstream file needs editing.
#
#   label (Samhitha / mhqg_engine)   node_type (Vaibhav / hop_reasoner)
#   ─────────────────────────────── ─────────────────────────────────────
#   Marketing_Claim                  Claim
#   Software_Dependency              Library
#   Patent_Concept                   Patent
#   OpenSource_License               LicenceType
#   Code_Module                      Library          (import-graph nodes)

_LABEL_TO_NODE_TYPE = {
    "Marketing_Claim":    "Claim",
    "Software_Dependency": "Library",
    "Patent_Concept":     "Patent",
    "OpenSource_License": "LicenceType",
    "Code_Module":        "Library",
}


def _make_node_attrs(label: str, **extra) -> dict:
    """
    Build a node-attribute dict that carries both label systems.
    Any extra keyword arguments are stored as-is (e.g. text, category, status).
    """
    return {
        "label":     label,
        "node_type": _LABEL_TO_NODE_TYPE.get(label, label),
        **extra,
    }


# ── Vector search helpers ─────────────────────────────────────────────────────

def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product is cosine similarity."""
    normed = embeddings.astype("float32", copy=True)
    norms = np.linalg.norm(normed, axis=1, keepdims=True)
    np.divide(normed, norms, out=normed, where=norms != 0)
    return normed


def _cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    """Fallback scalar cosine similarity (used in tests / small loops)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# ── Main class ────────────────────────────────────────────────────────────────

class KnowledgeFusionPipeline:
    """
    Builds and exports the fused, dual-labelled knowledge graph.

    Parameters
    ----------
    data_dir : Path
        Root data directory.  Processed JSON files are expected under
        data_dir/processed/.
    similarity_threshold : float
        Minimum cosine similarity to add a semantic bridge edge (0–1).
        Default 0.40 — tune lower (0.30) for denser graphs or higher (0.55)
        for precision-focused ablation runs.
    """

    def __init__(
        self,
        data_dir: Path,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.data_dir  = data_dir
        self.threshold = similarity_threshold
        self.G         = nx.DiGraph()

        self._encoder = self._load_encoder()

    def _load_encoder(self):
        """Use transformer embeddings by default; fallback to hashing when unavailable."""
        if os.getenv("CHAINCHECK_USE_HASHING", "").lower() in {"1", "true", "yes"}:
            logger.info(
                "Using deterministic local hashing embeddings because CHAINCHECK_USE_HASHING is set."
            )
            return _HashingTextEncoder()

        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers is not installed. Using deterministic local hashing embeddings."
            )
            return _HashingTextEncoder()

        try:
            logger.info("Loading cached sentence-transformer model: %s", _MODEL_NAME)
            return SentenceTransformer(_MODEL_NAME, local_files_only=True)
        except TypeError:
            try:
                logger.info("Loading sentence-transformer model: %s", _MODEL_NAME)
                return SentenceTransformer(_MODEL_NAME)
            except Exception as e:
                logger.warning(
                    "Sentence-transformer unavailable (%s). Using deterministic local hashing embeddings.",
                    e,
                )
                return _HashingTextEncoder()
        except Exception as e:
            logger.warning(
                "Cached sentence-transformer unavailable (%s). Using deterministic local hashing embeddings.",
                e,
            )
            return _HashingTextEncoder()

    # =========================================================================
    # DATA LOADERS  (one method per track)
    # =========================================================================

    # ── Track 1: Marketing claims from pitch deck ─────────────────────────────

    def load_pitch_claims(self) -> None:
        """
        Filename-agnostic loader: picks up the first *_parsed.json file found
        in data/processed/ so it works with any startup name, not just
        expense_ninja.  Falls back gracefully if no file exists.
        """
        processed_dir = self.data_dir / "processed"
        candidates    = sorted(processed_dir.glob("*_parsed.json"))

        if not candidates:
            logger.warning("No *_parsed.json found in %s — skipping Track 1", processed_dir)
            return

        claims_path = candidates[0]
        logger.info("Loading pitch deck claims from: %s", claims_path.name)

        pitch_data = json.loads(claims_path.read_text(encoding="utf-8"))

        # Support both key names used across the three parsers
        claims = pitch_data.get("technical_claims", pitch_data.get("claims", []))

        loaded = 0
        for claim in claims:
            # Support both claim_id (Samhitha) and auto-generated index
            claim_id   = claim.get("claim_id", f"Claim_{loaded}")
            claim_text = claim.get("sentence", str(claim))

            self.G.add_node(
                claim_id,
                **_make_node_attrs(
                    "Marketing_Claim",
                    text      = claim_text,
                    full_text = claim_text,        # kept for contradiction_detector
                    claim_type= claim.get("claim_type", "general"),
                    confidence= claim.get("confidence"),
                    page      = claim.get("page"),
                ),
            )
            loaded += 1

        logger.info("Track 1: loaded %d marketing claims", loaded)

    # ── Track 2: Codebase — dependency map + import graph ─────────────────────

    def load_dependencies(self) -> None:
        """
        Loads BOTH outputs from github_parser.py:
          • dependency_map.json   → flat list of library dependencies (Samhitha)
          • codebase_knowledge.json → import graph with modules and edges (Ujwal)
        Deduplicates by node ID so no double-counting occurs.
        """
        processed_dir = self.data_dir / "processed"
        loaded        = 0

        # ── Flat dependency list (Samhitha's output format) ───────────────────
        dep_path = processed_dir / "dependency_map.json"
        if dep_path.exists():
            dep_data = json.loads(dep_path.read_text(encoding="utf-8"))
            for dep in dep_data.get("all_dependencies", []):
                name = dep["name"]
                if name not in self.G:
                    self.G.add_node(
                        name,
                        **_make_node_attrs(
                            "Software_Dependency",
                            category  = dep.get("category", ""),
                            ecosystem = dep.get("ecosystem", ""),
                        ),
                    )
                    loaded += 1
        else:
            logger.warning("dependency_map.json not found — skipping flat dep list")

        # ── Import graph (Ujwal's codebase_knowledge.json) ────────────────────
        code_path = processed_dir / "codebase_knowledge.json"
        if code_path.exists():
            code_data = json.loads(code_path.read_text(encoding="utf-8"))
            ig = code_data.get("import_graph_structure", {})

            for node in ig.get("nodes", []):
                node_id = node["id"]
                if node_id not in self.G:
                    self.G.add_node(
                        node_id,
                        **_make_node_attrs(
                            "Code_Module",
                            module_type = node.get("type", ""),
                        ),
                    )
                    loaded += 1

            # Import edges between modules (structural codebase knowledge)
            for edge in ig.get("edges", []):
                self.G.add_edge(
                    edge["source"],
                    edge["target"],
                    relationship = "IMPORTS",
                    edge_type    = "IMPORTS",
                )
        else:
            logger.warning("codebase_knowledge.json not found — skipping import graph")

        logger.info("Track 2: loaded %d codebase nodes", loaded)

    # ── Track 3: Patent concepts ──────────────────────────────────────────────

    def load_patents(self) -> None:
        """
        Loads patent triples from patent_parser.py output.
        EXPIRED patents are loaded as nodes but flagged so bridge layer 2
        skips them (no false risk signals on dead IP).
        """
        patent_path = self.data_dir / "processed" / "knowledge_base.json"
        if not patent_path.exists():
            logger.warning("knowledge_base.json not found — skipping Track 3")
            return

        patent_data = json.loads(patent_path.read_text(encoding="utf-8"))
        triples     = patent_data.get("triples", [])
        loaded      = 0

        for triple in triples:
            head   = triple["head"]
            tail   = triple["tail"]
            status = triple.get("status", "UNKNOWN")

            # legal_risk_flag: True = active/risky, False = expired/safe
            legal_risk = triple.get("legal_risk_flag", status != "EXPIRED")

            for node_id in (head, tail):
                if node_id not in self.G:
                    self.G.add_node(
                        node_id,
                        **_make_node_attrs(
                            "Patent_Concept",
                            patent_id      = triple.get("patent_id", ""),
                            assignee       = triple.get("assignee", ""),
                            jurisdiction   = triple.get("jurisdiction", ""),
                            status         = status,
                            legal_risk_flag= legal_risk,
                            # hop_reasoner uses 'risk' field
                            risk           = "high" if legal_risk else "low",
                        ),
                    )
                    loaded += 1

            # Intra-patent structural edge
            self.G.add_edge(
                head, tail,
                relationship = triple["relationship"],
                edge_type    = triple["relationship"],
                patent_id    = triple.get("patent_id", ""),
            )

        logger.info("Track 3: loaded %d patent concept nodes", loaded)

    # ── Track 4: Open-source license intelligence (Ujwal) ────────────────────

    def load_licenses(self) -> None:
        """
        Optional Track 4 — loads license_parser.py output.
        Adds OpenSource_License / LicenceType nodes so hop_reasoner can flag
        commercial-use violations as a separate risk signal.
        Skips silently if license_knowledge.json does not exist.
        """
        license_path = self.data_dir / "processed" / "license_knowledge.json"
        if not license_path.exists():
            logger.info("license_knowledge.json not found — Track 4 skipped (optional)")
            return

        license_data = json.loads(license_path.read_text(encoding="utf-8"))
        loaded       = 0

        for entry in license_data.get("licenses", []):
            lic_node = entry["license"]
            module   = entry["module"]

            if lic_node not in self.G:
                self.G.add_node(
                    lic_node,
                    **_make_node_attrs(
                        "OpenSource_License",
                        # hop_reasoner uses 'risk' field on LicenceType nodes
                        risk            = entry.get("risk", "low"),
                        commercial_use  = entry.get("commercial_use", True),
                        copyleft        = entry.get("copyleft", False),
                    ),
                )
                loaded += 1

            self.G.add_edge(
                module, lic_node,
                relationship = entry.get("relationship", "LICENCED_UNDER"),
                edge_type    = "LICENCED_UNDER",
                weight       = 1.0,    # deterministic match — full weight
            )

        logger.info("Track 4: loaded %d license nodes", loaded)

    # =========================================================================
    # SEMANTIC BRIDGE LAYERS  (FAISS-accelerated)
    # =========================================================================

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into float32 embeddings."""
        vecs = self._encoder.encode(texts, batch_size=64, show_progress_bar=False)
        return np.array(vecs, dtype="float32")

    def _get_nodes_by_label(self, *labels: str) -> list[tuple[str, dict]]:
        """Return all (node_id, attrs) pairs whose label is in labels."""
        return [
            (n, d) for n, d in self.G.nodes(data=True)
            if d.get("label") in labels
        ]

    def _faiss_bridge(
        self,
        source_nodes : list[tuple[str, dict]],
        source_texts : list[str],
        target_nodes : list[tuple[str, dict]],
        target_texts : list[str],
        relationship : str,
        edge_type    : str,
        layer_name   : str,
    ) -> int:
        """
        Generic FAISS-accelerated bridge builder.

        For every source node, finds the top-K most similar target nodes
        and adds a directed edge if cosine similarity ≥ self.threshold.

        Returns the number of edges added.
        """
        if not source_nodes or not target_nodes:
            logger.warning("Bridge %s skipped — empty source or target set", layer_name)
            return 0

        src_embs = self._encode_texts(source_texts)
        tgt_embs = self._encode_texts(target_texts)

        src_normed = _normalize_rows(src_embs)
        tgt_normed = _normalize_rows(tgt_embs)

        k = min(_BRIDGE_TOP_K, len(target_nodes))
        sim_matrix = src_normed @ tgt_normed.T
        idx = np.argsort(-sim_matrix, axis=1)[:, :k]
        scores = np.take_along_axis(sim_matrix, idx, axis=1)

        edges_added = 0
        src_ids     = [n for n, _ in source_nodes]
        tgt_ids     = [n for n, _ in target_nodes]

        for i, src_id in enumerate(src_ids):
            for j in range(k):
                score = float(scores[i][j])
                if score >= self.threshold:
                    tgt_id = tgt_ids[idx[i][j]]
                    self.G.add_edge(
                        src_id, tgt_id,
                        relationship = relationship,
                        edge_type    = edge_type,
                        similarity   = round(score, 4),
                        weight       = round(score, 4),  # hop_reasoner uses 'weight'
                    )
                    edges_added += 1
                    logger.debug(
                        "  [%s] %.3f  %s  →  %s",
                        layer_name, score, src_id[:40], tgt_id[:40],
                    )

        logger.info(
            "Bridge %s: %d %s edges added (threshold=%.2f)",
            layer_name, edges_added, relationship, self.threshold,
        )
        return edges_added

    # ── Bridge layer 1: Marketing Claims → Software Dependencies ─────────────

    def bridge_claims_to_dependencies(self) -> None:
        """
        IMPLEMENTED_BY edges:  Marketing_Claim → Software_Dependency | Code_Module

        A claim about "AES-256 data protection" should link to "pycryptodome"
        even though neither string literally contains the other — handled via
        semantic similarity rather than keyword matching.
        """
        claim_nodes = self._get_nodes_by_label("Marketing_Claim")
        dep_nodes   = self._get_nodes_by_label("Software_Dependency", "Code_Module")

        # Encode claims by full text; encode deps by "name category" string
        claim_texts = [d.get("text", n) for n, d in claim_nodes]
        dep_texts   = [
            f"{n} {d.get('category', '')} {d.get('module_type', '')}".strip()
            for n, d in dep_nodes
        ]

        self._faiss_bridge(
            source_nodes = claim_nodes,
            source_texts = claim_texts,
            target_nodes = dep_nodes,
            target_texts = dep_texts,
            relationship = "IMPLEMENTED_BY",
            edge_type    = "IMPLEMENTED_BY",
            layer_name   = "Layer1 (Claim→Dep)",
        )

    # ── Bridge layer 2: Software Dependencies → Patent Concepts ──────────────

    def bridge_dependencies_to_patents(self) -> None:
        """
        SIMILAR_TO edges:  Software_Dependency | Code_Module → Patent_Concept

        Only bridges to ACTIVE patents (legal_risk_flag=True) to avoid
        false risk signals from expired IP.
        """
        dep_nodes = self._get_nodes_by_label("Software_Dependency", "Code_Module")

        # Skip EXPIRED patents — important for legal accuracy
        patent_nodes = [
            (n, d) for n, d in self.G.nodes(data=True)
            if d.get("label") == "Patent_Concept"
            and d.get("legal_risk_flag", True)
        ]

        dep_texts = [
            f"{n} {d.get('category', '')}".strip()
            for n, d in dep_nodes
        ]
        patent_texts = [n for n, _ in patent_nodes]   # node ID is the concept text

        self._faiss_bridge(
            source_nodes = dep_nodes,
            source_texts = dep_texts,
            target_nodes = patent_nodes,
            target_texts = patent_texts,
            relationship = "SIMILAR_TO",
            edge_type    = "SIMILAR_TO",
            layer_name   = "Layer2 (Dep→Patent)",
        )

    # ── Bridge layer 3: Code Modules → Patent Concepts (Ujwal's variant) ─────

    def bridge_modules_to_patents(self) -> None:
        """
        REQUIRES_IP_REVIEW edges:  Code_Module → Patent_Concept

        Ujwal's original bridge — links raw import-graph module paths
        (e.g. "src.crypto_auth") directly to patent concepts.
        Path separators are replaced with spaces before embedding so the
        NLP model reads them as natural language ("src crypto auth").
        """
        module_nodes = self._get_nodes_by_label("Code_Module")

        patent_nodes = [
            (n, d) for n, d in self.G.nodes(data=True)
            if d.get("label") == "Patent_Concept"
            and d.get("legal_risk_flag", True)
        ]

        if not module_nodes or not patent_nodes:
            logger.info("Bridge Layer3 skipped — no Code_Module or Patent nodes")
            return

        # Clean module paths so the NLP model reads them as words
        module_texts = [
            n.replace(".", " ").replace("_", " ").replace("/", " ")
            for n, _ in module_nodes
        ]
        patent_texts = [n for n, _ in patent_nodes]

        self._faiss_bridge(
            source_nodes = module_nodes,
            source_texts = module_texts,
            target_nodes = patent_nodes,
            target_texts = patent_texts,
            relationship = "REQUIRES_IP_REVIEW",
            edge_type    = "REQUIRES_IP_REVIEW",
            layer_name   = "Layer3 (Module→Patent)",
        )

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================

    def fuse_knowledge_domains(self) -> nx.DiGraph:
        """
        Run the full fusion pipeline:
          1. Load all four data tracks
          2. Run three semantic bridge layers
          3. Return the completed graph
        """
        logger.info(
            "═══ Starting Knowledge Fusion (threshold=%.2f) ═══",
            self.threshold,
        )

        # ── Stage 1: load nodes ───────────────────────────────────────────────
        self.load_pitch_claims()       # Track 1
        self.load_dependencies()       # Track 2
        self.load_patents()            # Track 3
        self.load_licenses()           # Track 4 (optional — skips if file missing)

        logger.info(
            "Nodes loaded → %d total  (%d claims, %d deps/modules, %d patents, %d licenses)",
            self.G.number_of_nodes(),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Marketing_Claim"),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("label") in ("Software_Dependency", "Code_Module")),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Patent_Concept"),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "OpenSource_License"),
        )

        # ── Stage 2: semantic bridges ─────────────────────────────────────────
        self.bridge_claims_to_dependencies()    # Layer 1
        self.bridge_dependencies_to_patents()   # Layer 2
        self.bridge_modules_to_patents()        # Layer 3 (Ujwal's variant)

        logger.info(
            "═══ Fusion complete → %d nodes, %d edges ═══",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )
        return self.G

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_fused_graph(self) -> None:
        """
        Dual export:
          • fused_knowledge_graph.json  — consumed by mhqg_engine.py,
                                          contradiction_detector.py,
                                          hop_reasoner.py
          • fused_knowledge_graph.graphml — Gephi / paper visualisation
        """
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # ── JSON ──────────────────────────────────────────────────────────────
        json_path  = processed_dir / "fused_knowledge_graph.json"
        graph_data = {
            "metadata": {
                "total_nodes"  : self.G.number_of_nodes(),
                "total_edges"  : self.G.number_of_edges(),
                "threshold"    : self.threshold,
                "node_type_counts": {
                    "Marketing_Claim"   : sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Marketing_Claim"),
                    "Software_Dependency": sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Software_Dependency"),
                    "Code_Module"       : sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Code_Module"),
                    "Patent_Concept"    : sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "Patent_Concept"),
                    "OpenSource_License": sum(1 for _, d in self.G.nodes(data=True) if d.get("label") == "OpenSource_License"),
                },
            },
            # "nodes" key: each node carries both label + node_type
            "nodes": [{"id": n, **d} for n, d in self.G.nodes(data=True)],
            # "links" key: used by mhqg_engine.py and contradiction_detector.py
            "links": [
                {"source": u, "target": v, **d}
                for u, v, d in self.G.edges(data=True)
            ],
            # "edges" key: alias of links — used by hop_reasoner.py (Vaibhav)
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.G.edges(data=True)
            ],
        }
        json_path.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved fused graph (JSON)    → %s", json_path)

        # ── GraphML ───────────────────────────────────────────────────────────
        graphml_path = processed_dir / "fused_knowledge_graph.graphml"
        # NetworkX cannot serialise None values to GraphML — replace with empty string
        G_clean = self.G.copy()
        for _, attrs in G_clean.nodes(data=True):
            for k, v in list(attrs.items()):
                if v is None:
                    attrs[k] = ""
        for _, _, attrs in G_clean.edges(data=True):
            for k, v in list(attrs.items()):
                if v is None:
                    attrs[k] = ""

        nx.write_graphml(G_clean, str(graphml_path))
        logger.info("Saved fused graph (GraphML) → %s", graphml_path)

        # ── Summary ───────────────────────────────────────────────────────────
        self._print_summary()

    def _print_summary(self) -> None:
        edge_types: dict[str, int] = {}
        for _, _, d in self.G.edges(data=True):
            et = d.get("relationship", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        print(f"\n{'═'*60}")
        print("  KNOWLEDGE FUSION SUMMARY")
        print(f"{'═'*60}")
        print(f"  Total nodes : {self.G.number_of_nodes()}")
        print(f"  Total edges : {self.G.number_of_edges()}")
        print(f"\n  Edge breakdown (top 20):")
        for etype, count in sorted(edge_types.items(), key=lambda x: -x[1])[:20]:
            print(f"    {etype:<30} {count:>5}")
        remaining = max(len(edge_types) - 20, 0)
        if remaining:
            print(f"    ... {remaining} additional edge types omitted")
        print(f"{'═'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Fuse whitepaper, codebase, patent, and license KGs into one graph"
    )
    ap.add_argument(
        "--data-dir", default=None,
        help="Root data directory (default: auto-detected from script location)",
    )
    ap.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for bridge edges (default: {_DEFAULT_THRESHOLD})",
    )
    args = ap.parse_args()

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(__file__).resolve().parents[2] / "data"
    )

    pipeline = KnowledgeFusionPipeline(data_dir, similarity_threshold=args.threshold)
    pipeline.fuse_knowledge_domains()
    pipeline.export_fused_graph()
