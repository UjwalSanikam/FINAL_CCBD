import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# A simulated Open-Source License Database for your Big Data pipeline
LICENSE_DB = {
    "os": "PSF-2.0 (Permissive - Safe)",
    "json": "PSF-2.0 (Permissive - Safe)",
    "networkx": "BSD-3-Clause (Permissive - Safe)",
    "pydantic": "MIT (Permissive - Safe)",
    "hashlib": "PSF-2.0 (Permissive - Safe)",
    "copyleft_crypto_engine": "GPL-3.0 (Strict Copyleft - 🚨 HIGH VC RISK!)"
}

def run_license_scan(data_dir: Path):
    logger.info("Starting Track 4: Open-Source License Intelligence Scan...")
    
    code_path = data_dir / "processed" / "codebase_knowledge.json"
    if not code_path.exists():
        logger.error("codebase_knowledge.json not found! Run github_parser.py first.")
        return

    code_data = json.loads(code_path.read_text(encoding="utf-8"))
    nodes = code_data.get("import_graph_structure", {}).get("nodes", [])

    license_triples = []
    
    for node in nodes:
        module_name = node["id"]
        # We only scan external third-party libraries for licenses, not internal code
        if node.get("type") == "internal":
            continue 

        # Lookup the license. If we don't know it, default to MIT
        license_name = LICENSE_DB.get(module_name, "UNKNOWN (Requires Manual Review ⚠️)")
        
        license_triples.append({
            "module": module_name,
            "relationship": "LICENSED_UNDER",
            "license": license_name
        })
        logger.info(f"Scanned Dependency: '{module_name}' -> Licensed under: {license_name}")

    output_path = data_dir / "processed" / "license_knowledge.json"
    output_path.write_text(json.dumps({"licenses": license_triples}, indent=2), encoding="utf-8")
    logger.info(f"Track 4 License Data saved to → {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
