"""
schema.py — Canonical constants for the ChainCheck pipeline.

Every module that references node types, edge types, or output filenames
must import from here instead of using string literals scattered across
files. This is what stops the three repos from disagreeing on what a
node_type is called.
"""

# ── Node types ─────────────────────────────────────────────────────────────────
# Dual-label system: every node carries BOTH keys so all consumers work
# regardless of which schema they expect (vaibhava / Ujwal / Samhitha).
#
#   label (knowledge_fusion schema)      node_type (hop_reasoner schema)
#   ────────────────────────────────     ─────────────────────────────────
#   Marketing_Claim                      Claim
#   Software_Dependency                  Library
#   Patent_Concept                       Patent
#   OpenSource_License                   LicenceType
#   Code_Module                          Library

NODE_TYPES = {
    "CLAIM":               "Claim",
    "LIBRARY":             "Library",
    "PATENT":              "Patent",
    "LICENCE_TYPE":        "LicenceType",
    "STARTUP":             "Startup",
    "MARKETING_CLAIM":     "Marketing_Claim",
    "PATENT_CONCEPT":      "Patent_Concept",
    "CODE_MODULE":         "Code_Module",
    "OPENSOURCE_LICENSE":  "OpenSource_License",
    "SOFTWARE_DEPENDENCY": "Software_Dependency",
}

# ── Edge types ─────────────────────────────────────────────────────────────────

EDGE_TYPES = {
    "IMPLEMENTED_BY":              "IMPLEMENTED_BY",       # Claim → Dependency (FAISS bridge)
    "SIMILAR_TO":                  "SIMILAR_TO",           # Dependency → Patent (FAISS bridge)
    "REQUIRES_IP_REVIEW":          "REQUIRES_IP_REVIEW",   # Module → Patent (Ujwal's bridge)
    "IMPLEMENTS":                  "implements",
    "CITES":                       "cites",
    "CONFLICTS_WITH":              "conflicts_with",
    "LICENCED_UNDER":              "LICENCED_UNDER",
    "LICENSED_UNDER":              "LICENCED_UNDER",       # alias — normalised to LICENCED_UNDER
    "IMPORTS":                     "IMPORTS",
    "SUPPORTS":                    "supports",
    "ASSERTS":                     "ASSERTS",
    "POTENTIALLY_IMPLEMENTED_BY":  "POTENTIALLY_IMPLEMENTED_BY",
}

# ── Canonical output filenames ─────────────────────────────────────────────────
# Every stage MUST write to exactly these filenames so downstream stages
# can find them without configuration.

OUTPUT_FILES = {
    "whitepaper":               "startup_parsed.json",
    "dependency_map":           "dependency_map.json",
    "import_graph":             "import_graph.json",
    "codebase":                 "codebase_knowledge.json",
    "patents":                  "knowledge_base.json",
    "licenses":                 "license_knowledge.json",
    "fused_graph":              "fused_knowledge_graph.json",
    "fused_graphml":            "fused_knowledge_graph.graphml",
    "entity_matches":           "entity_matches.json",
    "kg":                       "kg.json",
    "kg_summary":               "kg_summary.json",
    "hop_chains":               "hop_chains.json",
    "structured_evidence":      "structured_evidence.json",
    "contradiction_evidence":   "contradiction_evidence.json",
    "vc_risk_report":           "vc_risk_report.json",
    "questions":                "questions.json",              # LLM output
    "mhqg_questions":           "due_diligence_questions.json",  # template fallback
    "audited_vc_report":        "audited_vc_report.json",
    "eval_results":             "eval_results.json",
    "pipeline_timing":          "pipeline_timing.json",
}

# ── Direct constant aliases ────────────────────────────────────────────────────
# Keep these so existing code that does `from shared.schema import NODE_CLAIM`
# doesn't break.

NODE_CLAIM          = NODE_TYPES["CLAIM"]
NODE_LIBRARY        = NODE_TYPES["LIBRARY"]
NODE_PATENT         = NODE_TYPES["PATENT"]
NODE_LICENCE_TYPE   = NODE_TYPES["LICENCE_TYPE"]
NODE_STARTUP        = NODE_TYPES["STARTUP"]

EDGE_IMPLEMENTED_BY             = EDGE_TYPES["IMPLEMENTED_BY"]
EDGE_SIMILAR_TO                 = EDGE_TYPES["SIMILAR_TO"]
EDGE_REQUIRES_IP_REVIEW         = EDGE_TYPES["REQUIRES_IP_REVIEW"]
EDGE_LICENCED_UNDER             = EDGE_TYPES["LICENCED_UNDER"]
EDGE_LICENSED_UNDER             = EDGE_TYPES["LICENCED_UNDER"]
EDGE_IMPLEMENTS                 = EDGE_TYPES["IMPLEMENTS"]
EDGE_CITES                      = EDGE_TYPES["CITES"]
EDGE_CONFLICTS_WITH             = EDGE_TYPES["CONFLICTS_WITH"]
EDGE_IMPORTS                    = EDGE_TYPES["IMPORTS"]
EDGE_SUPPORTS                   = EDGE_TYPES["SUPPORTS"]
EDGE_ASSERTS                    = EDGE_TYPES["ASSERTS"]
EDGE_POTENTIALLY_IMPLEMENTED_BY = EDGE_TYPES["POTENTIALLY_IMPLEMENTED_BY"]
