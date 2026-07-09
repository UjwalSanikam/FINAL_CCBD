"""
github_parser.py — Track 2: Codebase Structure Extractor
=========================================================
Multi-Hop Reasoning System for Venture Capital Technical Due Diligence

Two complementary analysis modes in one script:

MODE A — Dependency Mapper
  Parses requirements.txt / package.json / Pipfile / pyproject.toml to build
  a structured map of external library dependencies.
  Output: dependency_map.json + dependency_graph.json

MODE B — Import Graph Analyzer (NetworkX)
  Walks Python source files, extracts all `import` and `from X import Y`
  statements, and builds a directed module-dependency graph using NetworkX.
  Identifies: hub modules, orphan files, circular dependency risks, and
  third-party vs. internal library usage.
  Output: import_graph.json + import_graph_edges.json

Both modes run simultaneously by default and their outputs are combined into
a unified codebase_knowledge.json for downstream KG ingestion.

Usage:
    python src/extractors/github_parser.py
    python src/extractors/github_parser.py --repo /path/to/repo --output data/processed/
    python src/extractors/github_parser.py --mode deps     # Mode A only
    python src/extractors/github_parser.py --mode imports  # Mode B only
"""

import os
import re
import ast
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODE A — DEPENDENCY MAPPER
# Parses package manifest files to extract library names, versions, ecosystems.
# ─────────────────────────────────────────────────────────────────────────────

# Canonical categories for common ML/backend libraries (VC due-diligence relevant)
_LIBRARY_CATEGORIES = {
    # ML / AI
    "torch": "Deep Learning", "tensorflow": "Deep Learning", "keras": "Deep Learning",
    "jax": "Deep Learning", "flax": "Deep Learning", "paddle": "Deep Learning",
    "transformers": "NLP / LLM", "sentence-transformers": "NLP / LLM",
    "spacy": "NLP / LLM", "nltk": "NLP / LLM", "gensim": "NLP / LLM",
    "openai": "NLP / LLM", "langchain": "NLP / LLM", "llama-index": "NLP / LLM",
    "anthropic": "NLP / LLM",
    "scikit-learn": "ML / Classical", "xgboost": "ML / Classical",
    "lightgbm": "ML / Classical", "catboost": "ML / Classical",
    "numpy": "Data Science", "pandas": "Data Science", "scipy": "Data Science",
    "matplotlib": "Visualization", "plotly": "Visualization", "seaborn": "Visualization",
    "networkx": "Graph / KG", "neo4j": "Graph / KG", "py2neo": "Graph / KG",
    "rdflib": "Graph / KG", "pykg2vec": "Graph / KG",
    # Backend / API
    "fastapi": "API / Backend", "flask": "API / Backend", "django": "API / Backend",
    "uvicorn": "API / Backend", "gunicorn": "API / Backend", "starlette": "API / Backend",
    "pydantic": "Data Validation", "sqlalchemy": "Database", "alembic": "Database",
    "redis": "Database", "pymongo": "Database", "psycopg2": "Database",
    # Vector DB / RAG
    "chromadb": "Vector DB / RAG", "pinecone": "Vector DB / RAG",
    "faiss": "Vector DB / RAG", "qdrant": "Vector DB / RAG",
    "weaviate": "Vector DB / RAG", "milvus": "Vector DB / RAG",
    # DevOps / Cloud
    "boto3": "Cloud / AWS", "google-cloud": "Cloud / GCP", "azure": "Cloud / Azure",
    "docker": "DevOps", "kubernetes": "DevOps",
    # Testing
    "pytest": "Testing", "unittest": "Testing", "hypothesis": "Testing",
}


def _categorize_library(name: str) -> str:
    """Return a VC-relevant category for a library name."""
    name_lower = name.lower().replace("_", "-")
    for key, cat in _LIBRARY_CATEGORIES.items():
        if name_lower == key or name_lower.startswith(key + "-"):
            return cat
    return "Utility / Other"


def parse_requirements_txt(filepath: Path) -> list[dict]:
    """
    Parse requirements.txt / requirements-dev.txt / requirements-test.txt.
    Handles: ==, >=, <=, ~=, !=, extras [brackets], and comment lines.
    """
    deps = []
    version_pattern = re.compile(
        r"^([A-Za-z0-9_\-\.]+)"          # package name
        r"(?:\[[\w,]+\])?"                # optional extras [e.g., pydantic[email]]
        r"\s*([><=!~]{0,3}[0-9\.\*,\s><=!~]*)?",  # version specifier
        re.IGNORECASE
    )
    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = version_pattern.match(line)
        if m:
            name = m.group(1).strip()
            version = (m.group(2) or "").strip() or "any"
            deps.append({
                "name": name,
                "version_spec": version,
                "ecosystem": "PyPI",
                "category": _categorize_library(name),
                "source_file": str(filepath),
            })
    logger.info("Parsed %d deps from %s", len(deps), filepath.name)
    return deps


def parse_package_json(filepath: Path) -> list[dict]:
    """
    Parse Node.js package.json.
    Extracts dependencies, devDependencies, peerDependencies.
    """
    data = json.loads(filepath.read_text(encoding="utf-8"))
    deps = []
    for dep_type in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        for name, version in data.get(dep_type, {}).items():
            deps.append({
                "name": name,
                "version_spec": version,
                "ecosystem": "npm",
                "dep_type": dep_type,
                "category": _categorize_library(name),
                "source_file": str(filepath),
            })
    logger.info("Parsed %d deps from %s", len(deps), filepath.name)
    return deps


def parse_pyproject_toml(filepath: Path) -> list[dict]:
    """
    Parse pyproject.toml (PEP 517/518).
    Handles both [project].dependencies and [tool.poetry.dependencies].
    Falls back to regex if tomllib unavailable.
    """
    raw = filepath.read_text(encoding="utf-8")
    deps = []
    dep_pattern = re.compile(
        r'"([A-Za-z0-9_\-\.]+)\s*([><=!~][^"]*)?"|'
        r"'([A-Za-z0-9_\-\.]+)\s*([><=!~][^']*)?'",
        re.IGNORECASE,
    )
    for m in dep_pattern.finditer(raw):
        name = (m.group(1) or m.group(3) or "").strip()
        version = (m.group(2) or m.group(4) or "").strip() or "any"
        if name:
            deps.append({
                "name": name,
                "version_spec": version,
                "ecosystem": "PyPI",
                "category": _categorize_library(name),
                "source_file": str(filepath),
            })
    logger.info("Parsed %d deps from %s", len(deps), filepath.name)
    return deps


def build_dependency_map(repo_root: Path) -> dict:
    """
    MODE A main function.
    Scans repo root for known manifest files and aggregates all dependencies
    into a structured map with ecosystem/category breakdown.
    """
    all_deps: list[dict] = []
    manifest_found = False

    manifest_parsers = {
        "requirements.txt": parse_requirements_txt,
        "requirements-dev.txt": parse_requirements_txt,
        "requirements-test.txt": parse_requirements_txt,
        "package.json": parse_package_json,
        "pyproject.toml": parse_pyproject_toml,
    }

    for filename, parser in manifest_parsers.items():
        path = repo_root / filename
        if path.exists():
            manifest_found = True
            try:
                deps = parser(path)
                all_deps.extend(deps)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", filename, e)

    # Recursive search for requirements files in subdirs
    for req_file in repo_root.rglob("requirements*.txt"):
        if req_file.name not in manifest_parsers:
            try:
                all_deps.extend(parse_requirements_txt(req_file))
                manifest_found = True
            except Exception as e:
                logger.warning("Failed to parse %s: %s", req_file, e)

    if not manifest_found:
        logger.warning("No manifest files (requirements.txt, package.json, etc.) found in %s", repo_root)

    # Build category summary (VC-relevant: what tech stack is being used?)
    by_category: dict[str, list[str]] = defaultdict(list)
    by_ecosystem: dict[str, int] = defaultdict(int)

    for dep in all_deps:
        by_category[dep["category"]].append(dep["name"])
        by_ecosystem[dep["ecosystem"]] += 1

    dependency_map = {
        "metadata": {
            "total_dependencies": len(all_deps),
            "ecosystems": dict(by_ecosystem),
            "category_counts": {k: len(v) for k, v in by_category.items()},
        },
        "by_category": {k: sorted(set(v)) for k, v in by_category.items()},
        "all_dependencies": all_deps,
    }

    logger.info(
        "Dependency map: %d total deps across %d categories",
        len(all_deps), len(by_category)
    )
    return dependency_map


# ─────────────────────────────────────────────────────────────────────────────
# MODE B — IMPORT GRAPH ANALYZER (NetworkX)
# Walks Python source, extracts imports, builds directed dependency graph.
# ─────────────────────────────────────────────────────────────────────────────

# Python stdlib module names (partial list — common ones)
_STDLIB_MODULES = {
    "os", "sys", "re", "io", "json", "ast", "math", "time", "datetime",
    "pathlib", "typing", "collections", "itertools", "functools", "operator",
    "logging", "argparse", "subprocess", "threading", "multiprocessing",
    "socket", "http", "urllib", "email", "html", "xml", "csv", "hashlib",
    "base64", "uuid", "copy", "pickle", "struct", "random", "string",
    "textwrap", "shutil", "glob", "fnmatch", "tempfile", "unittest",
    "contextlib", "abc", "dataclasses", "enum", "warnings", "traceback",
    "inspect", "importlib", "platform", "gc", "weakref", "heapq", "bisect",
}

_SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
              ".tox", "dist", "build", ".eggs", "site-packages", ".pytest_cache"}


def _get_root_module(import_name: str) -> str:
    """Return the top-level package name (e.g., 'sklearn' from 'sklearn.linear_model')."""
    return import_name.split(".")[0]


def extract_imports_from_file(filepath: Path) -> list[dict]:
    """
    Parse a Python file and extract all import statements using AST.
    Returns list of import records with module name, alias, what's imported.
    Uses AST for correctness — no regex guessing.
    """
    imports = []
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        logger.debug("Syntax error in %s: %s", filepath, e)
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "root_module": _get_root_module(alias.name),
                    "alias": alias.asname,
                    "names": [],
                    "line": node.lineno,
                    "file": str(filepath),
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append({
                "type": "from_import",
                "module": module,
                "root_module": _get_root_module(module) if module else "",
                "alias": None,
                "names": names,
                "line": node.lineno,
                "level": node.level,  # relative import depth
                "file": str(filepath),
            })

    return imports


def classify_import(module_name: str, known_internal_modules: set[str]) -> str:
    """Classify an import as 'stdlib', 'internal', or 'third_party'."""
    root = _get_root_module(module_name)
    if root in _STDLIB_MODULES:
        return "stdlib"
    if root in known_internal_modules:
        return "internal"
    return "third_party"


def build_import_graph(repo_root: Path) -> tuple[nx.DiGraph, dict]:
    """
    MODE B main function.
    Walks all .py files, extracts imports, and builds a directed NetworkX graph:
      Node = Python module/file
      Edge = "A imports B"

    Returns (graph, analysis_report).
    """
    G = nx.DiGraph()
    all_imports: list[dict] = []

    # First pass: collect all internal module names (repo's own .py files)
    py_files = []
    for py_file in repo_root.rglob("*.py"):
        if any(skip in py_file.parts for skip in _SKIP_DIRS):
            continue
        py_files.append(py_file)

    # Build set of internal module names from file paths
    internal_modules: set[str] = set()
    for py_file in py_files:
        rel = py_file.relative_to(repo_root)
        # Convert path to module name: src/extractors/parser.py → src.extractors.parser
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)
        internal_modules.add(parts[0])   # top-level package, e.g. "src"
        internal_modules.add(module_name)
        # Also register the path WITHOUT the "src" prefix and its own
        # top-level segment — pipeline.py inserts src/ onto sys.path, so
        # internal imports happen as "extractors.x" not "src.extractors.x"
        if len(parts) > 1:
            internal_modules.add(".".join(parts[1:]))
            internal_modules.add(parts[1])
        G.add_node(module_name, type="internal", filepath=str(py_file))

    # Second pass: extract imports and add edges
    for py_file in py_files:
        rel = py_file.relative_to(repo_root)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        source_module = ".".join(parts)

        file_imports = extract_imports_from_file(py_file)
        all_imports.extend(file_imports)

        for imp in file_imports:
            target = imp["module"] or imp["root_module"]
            if not target:
                continue
            imp_class = classify_import(target, internal_modules)
            imp["classification"] = imp_class

            root = imp["root_module"] or target
            # Add node for the imported module if not already present
            if not G.has_node(root):
                G.add_node(root, type=imp_class)

            # Add directed edge: source_module → imported_module
            if G.has_edge(source_module, root):
                G[source_module][root]["count"] += 1
            else:
                G.add_edge(source_module, root, count=1, classification=imp_class)

    # ── Graph Analysis ─────────────────────────────────────────────────────

    internal_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "internal"]
    third_party_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "third_party"]

    # Hub modules: internal files that many other files import (high in-degree)
    in_degrees = dict(G.in_degree(internal_nodes))
    hub_modules = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    # Orphan files: internal files with no imports and no importers
    orphan_modules = [
        n for n in internal_nodes
        if G.in_degree(n) == 0 and G.out_degree(n) == 0
    ]

    # Circular dependency detection
    try:
        cycles = list(nx.simple_cycles(G.subgraph(internal_nodes)))
        cycle_count = len(cycles)
        cycle_examples = cycles[:3]  # show first 3
    except Exception:
        cycle_count = 0
        cycle_examples = []

    # Most-imported third-party libraries
    tp_in_degree = {
        n: G.in_degree(n) for n in third_party_nodes
    }
    top_third_party = sorted(tp_in_degree.items(), key=lambda x: x[1], reverse=True)[:15]

    # Graph connectivity stats
    try:
        internal_subgraph = G.subgraph(internal_nodes)
        weakly_connected = nx.number_weakly_connected_components(internal_subgraph)
    except Exception:
        weakly_connected = None

    analysis_report = {
        "metadata": {
            "total_py_files": len(py_files),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "internal_modules": len(internal_nodes),
            "third_party_libraries": len(third_party_nodes),
            "stdlib_modules_used": len(
                [n for n, d in G.nodes(data=True) if d.get("type") == "stdlib"]
            ),
            "weakly_connected_components": weakly_connected,
        },
        "hub_modules": [
            {"module": m, "imported_by_count": c} for m, c in hub_modules
        ],
        "orphan_modules": orphan_modules,
        "circular_dependency_risks": {
            "cycle_count": cycle_count,
            "examples": cycle_examples,
        },
        "top_third_party_libraries": [
            {"library": lib, "imported_by_count": c} for lib, c in top_third_party
        ],
        "all_imports_raw": all_imports,
    }

    logger.info(
        "Import graph: %d nodes, %d edges, %d cycles detected",
        G.number_of_nodes(), G.number_of_edges(), cycle_count
    )
    return G, analysis_report


def graph_to_serializable(G: nx.DiGraph) -> dict:
    """Convert NetworkX graph to JSON-serializable dict (node-link format)."""
    return {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool, type(None)))}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, **{k: v2 for k, v2 in d.items()}}
            for u, v, d in G.edges(data=True)
        ],
        "directed": True,
    }


def print_import_graph_summary(analysis: dict) -> None:
    """Pretty-print import graph analysis results."""
    meta = analysis["metadata"]
    print(f"\n{'═'*70}")
    print("  IMPORT GRAPH ANALYSIS — Codebase Structure")
    print(f"{'═'*70}")
    print(f"  Python files analysed : {meta['total_py_files']}")
    print(f"  Internal modules      : {meta['internal_modules']}")
    print(f"  Third-party libraries : {meta['third_party_libraries']}")
    print(f"  Graph edges (imports) : {meta['total_edges']}")
    print(f"  Circular dep risks    : {analysis['circular_dependency_risks']['cycle_count']}")

    print(f"\n  🔴 Hub Modules (most imported internally):")
    for h in analysis["hub_modules"][:5]:
        print(f"     {h['module']:<40} ← imported by {h['imported_by_count']} modules")

    print(f"\n  📦 Top Third-Party Libraries:")
    for lib in analysis["top_third_party_libraries"][:8]:
        print(f"     {lib['library']:<30} ← used in {lib['imported_by_count']} files")

    if analysis["orphan_modules"]:
        print(f"\n  ⚠️  Orphan modules (no imports, not imported): {len(analysis['orphan_modules'])}")
        for m in analysis["orphan_modules"][:3]:
            print(f"     {m}")

    if analysis["circular_dependency_risks"]["cycle_count"] > 0:
        print(f"\n  🔁 Circular dependency examples:")
        for cycle in analysis["circular_dependency_risks"]["examples"]:
            print(f"     {' → '.join(cycle)} → ...")
    print(f"{'═'*70}\n")


def print_dependency_map_summary(dep_map: dict) -> None:
    """Pretty-print dependency map results."""
    meta = dep_map["metadata"]
    print(f"\n{'═'*70}")
    print("  DEPENDENCY MAP — External Libraries")
    print(f"{'═'*70}")
    print(f"  Total dependencies : {meta['total_dependencies']}")
    print(f"  Ecosystems         : {meta['ecosystems']}")
    print(f"\n  By Category:")
    for cat, count in sorted(meta["category_counts"].items(), key=lambda x: x[1], reverse=True):
        libs = dep_map["by_category"][cat]
        print(f"     {cat:<25} ({count:>3} libs)  {', '.join(libs[:4])}{'...' if len(libs)>4 else ''}")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED OUTPUT — Unified codebase knowledge base for KG ingestion
# ─────────────────────────────────────────────────────────────────────────────

def build_codebase_knowledge(
    dep_map: Optional[dict],
    import_analysis: Optional[dict],
    import_graph: Optional[nx.DiGraph],
) -> dict:
    """
    Merge Mode A + Mode B outputs into a unified codebase knowledge structure
    ready for ingestion into the VC due-diligence Knowledge Graph.
    """
    knowledge = {
        "metadata": {
            "extractor": "github_parser.py",
            "tracks": ["dependency_mapping", "import_graph_analysis"],
            "description": (
                "Structured codebase knowledge for VC Technical Due Diligence KG. "
                "Captures external library fingerprint and internal module topology."
            ),
        },
        "dependency_map": dep_map,
        "import_graph_analysis": import_analysis,
        "import_graph_structure": graph_to_serializable(import_graph) if import_graph else None,
    }
    return knowledge


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "GitHub/Codebase Parser for VC Technical Due Diligence.\n"
            "Mode A: Dependency map from requirements.txt / package.json\n"
            "Mode B: Import graph analysis with NetworkX\n"
            "Default: both modes run simultaneously."
        )
    )
    parser.add_argument(
        "--repo", "-r",
        default=".",
        help="Root directory of the repository to analyse."
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/",
        help="Output directory for JSON results."
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["deps", "imports", "both"],
        default="both",
        help="'deps' = dependency map only, 'imports' = import graph only, 'both' = default."
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dep_map = None
    import_analysis = None
    import_graph = None

    # ── MODE A: Dependency Mapper ─────────────────────────────────────────
    if args.mode in ("deps", "both"):
        logger.info("=== MODE A: Dependency Mapper ===")
        dep_map = build_dependency_map(repo_root)
        print_dependency_map_summary(dep_map)
        dep_out = output_dir / "dependency_map.json"
        dep_out.write_text(json.dumps(dep_map, indent=2), encoding="utf-8")
        logger.info("Saved dependency map → %s", dep_out)

    # ── MODE B: Import Graph Analyzer ────────────────────────────────────
    if args.mode in ("imports", "both"):
        logger.info("=== MODE B: Import Graph Analyzer (NetworkX) ===")
        import_graph, import_analysis = build_import_graph(repo_root)
        print_import_graph_summary(import_analysis)
        graph_out = output_dir / "import_graph.json"
        graph_serializable = graph_to_serializable(import_graph)
        graph_out.write_text(json.dumps(graph_serializable, indent=2), encoding="utf-8")
        logger.info("Saved import graph → %s", graph_out)
        analysis_out = output_dir / "import_graph_analysis.json"
        # Remove raw imports from analysis to keep file clean (they're in the graph)
        analysis_clean = {k: v for k, v in import_analysis.items() if k != "all_imports_raw"}
        analysis_out.write_text(json.dumps(analysis_clean, indent=2), encoding="utf-8")
        logger.info("Saved import analysis → %s", analysis_out)

    # ── COMBINED OUTPUT ───────────────────────────────────────────────────
    if args.mode == "both":
        combined = build_codebase_knowledge(dep_map, import_analysis, import_graph)
        combined_out = output_dir / "codebase_knowledge.json"
        # Slim version without all_imports_raw for readability
        if combined.get("import_graph_analysis"):
            combined["import_graph_analysis"] = {
                k: v for k, v in combined["import_graph_analysis"].items()
                if k != "all_imports_raw"
            }
        combined_out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        logger.info("Saved unified codebase knowledge → %s", combined_out)

    logger.info("github_parser.py complete. Outputs in: %s", output_dir)


if __name__ == "__main__":
    main()

