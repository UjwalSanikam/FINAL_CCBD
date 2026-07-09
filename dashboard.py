"""
dashboard.py — ChainCheck VC Due Diligence Dashboard
=====================================================
Streamlit frontend that reads REAL pipeline outputs from data/processed/.

Run after the pipeline has completed at least through Stage 11:
  streamlit run dashboard.py

Displays
--------
  - Pipeline run status (which output files exist)
  - Knowledge graph stats (nodes, edges, bridge layers)
  - Risk report summary (severity breakdown + sortable table)
  - Generated due-diligence questions with audit trails
  - Contradiction evidence
  - Raw JSON explorer for any output file
"""

from pathlib import Path
import json
import streamlit as st

st.set_page_config(
    page_title="ChainCheck — VC Due Diligence",
    layout="wide",
    page_icon="🔍",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE      = Path(__file__).resolve().parent
PROCESSED = BASE / "data" / "processed"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(filename: str) -> dict | None:
    p = PROCESSED / filename
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load {filename}: {e}")
        return None

def _exists(filename: str) -> bool:
    return (PROCESSED / filename).exists()

def _severity_color(s: str) -> str:
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}.get(s, "⚪")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 ChainCheck")
st.subheader("AI-Powered VC Technical Due Diligence Dashboard")
st.markdown("---")

if not PROCESSED.exists():
    st.error(
        f"No processed data found at `{PROCESSED}`. "
        "Run `python pipeline.py --pitch your_pitch.pdf` first."
    )
    st.stop()

# ── Pipeline Status ───────────────────────────────────────────────────────────

st.header("📊 Pipeline Status")

STATUS_FILES = {
    "Whitepaper Parsed":     "startup_parsed.json",
    "Codebase Analyzed":     "codebase_knowledge.json",
    "Patents Extracted":     "knowledge_base.json",
    "Licenses Scanned":      "license_knowledge.json",
    "Knowledge Graph Fused": "fused_knowledge_graph.json",
    "Entity Matches":        "entity_matches.json",
    "Hop Chains":            "hop_chains.json",
    "Structured Evidence":   "structured_evidence.json",
    "Contradictions":        "contradiction_evidence.json",
    "Risk Report":           "vc_risk_report.json",
    "Questions Generated":   "questions.json",
    "MHQG Fallback":         "due_diligence_questions.json",
    "Audit Trail":           "audited_vc_report.json",
    "Eval Results":          "eval_results.json",
}

cols = st.columns(4)
items = list(STATUS_FILES.items())
for i, (label, fname) in enumerate(items):
    ok = _exists(fname)
    cols[i % 4].metric(label, "✅ Done" if ok else "⏳ Pending")

st.markdown("---")

# ── Fusion Graph Stats ────────────────────────────────────────────────────────

fused = _load("fused_knowledge_graph.json")
if fused:
    st.header("🕸️ Knowledge Fusion Graph")
    meta = fused.get("metadata", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Nodes",  meta.get("total_nodes", "—"))
    c2.metric("Total Edges",  meta.get("total_edges", "—"))
    nc = meta.get("node_type_counts", {})
    c3.metric("Marketing Claims",     nc.get("Marketing_Claim", "—"))
    c4.metric("Software Dependencies", nc.get("Software_Dependency", "—"))
    c5.metric("Patent Concepts",      nc.get("Patent_Concept", "—"))

    # Edge type breakdown
    edges = fused.get("edges", fused.get("links", []))
    if edges:
        edge_types: dict[str, int] = {}
        for e in edges:
            et = e.get("relationship", e.get("edge_type", "unknown"))
            edge_types[et] = edge_types.get(et, 0) + 1
        st.subheader("Bridge Layers (edge types)")
        edge_cols = st.columns(len(edge_types))
        for i, (etype, count) in enumerate(
            sorted(edge_types.items(), key=lambda x: -x[1])
        ):
            edge_cols[i % len(edge_cols)].metric(etype, count)

    st.markdown(f"*Fusion threshold: `{meta.get('threshold', '?')}`*")
    st.markdown("---")

# ── Risk Report ───────────────────────────────────────────────────────────────

risk_data = _load("vc_risk_report.json")
if risk_data:
    st.header("⚠️ Risk Report")
    meta = risk_data.get("metadata", {})
    by_sev = meta.get("by_severity", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Risks",  meta.get("total_risks", 0))
    c2.metric("🔴 Critical",  by_sev.get("CRITICAL", 0))
    c3.metric("🟠 High",      by_sev.get("HIGH", 0))
    c4.metric("🟡 Moderate",  by_sev.get("MODERATE", 0))
    c5.metric("🟢 Low",       by_sev.get("LOW", 0))

    risks = risk_data.get("identified_risks", [])
    if risks:
        sev_filter = st.multiselect(
            "Filter by severity",
            ["CRITICAL", "HIGH", "MODERATE", "LOW"],
            default=["CRITICAL", "HIGH"],
        )
        filtered = [r for r in risks if r.get("severity") in sev_filter]

        for r in filtered:
            sev  = r.get("severity", "?")
            icon = _severity_color(sev)
            with st.expander(
                f"{icon} [{sev}] {r.get('category','?')} — `{r.get('target_entity','?')[:60]}`"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{r.get('confidence_score', 0):.3f}")
                col2.metric("License Risk", r.get("license_risk", "—"))
                col3.metric("Semantic Similarity",
                            f"{r.get('semantic_similarity', 0):.3f}")

                st.markdown(f"**Evidence chain:**  \n`{r.get('evidence_chain', '—')}`")
                st.markdown(f"**Recommended action:** {r.get('recommended_action','—')}")

                if "legal_risk_score" in r:
                    st.markdown(f"**Legal risk score:** {r['legal_risk_score']}")
                    if r.get("legal_reasons"):
                        st.markdown("Legal reasons: " + "; ".join(r["legal_reasons"]))
    st.markdown("---")

# ── Questions ─────────────────────────────────────────────────────────────────

q_data = _load("questions.json") or _load("due_diligence_questions.json")
if q_data:
    questions = q_data.get("questions", [])
    st.header(f"❓ Generated Due-Diligence Questions ({len(questions)})")

    q_meta = q_data.get("metadata", {})
    if q_meta:
        qcols = st.columns(4)
        qcols[0].metric("Total", q_meta.get("total_questions", len(questions)))
        qcols[1].metric("Licence Flags", q_meta.get("with_licence_conflict", "—"))
        qcols[2].metric("Patent Flags",  q_meta.get("with_patent_node", "—"))
        cat_dict = q_meta.get("by_category", q_meta.get("risk_distribution", {}))
        if cat_dict:
            qcols[3].metric("Categories", len(cat_dict))

    cat_filter = st.multiselect(
        "Filter by category",
        list({
            q.get("question_category", q.get("risk_level", "?"))
            for q in questions
        }),
    )

    for i, q in enumerate(questions, 1):
        cat = q.get("question_category", q.get("risk_level", "?"))
        if cat_filter and cat not in cat_filter:
            continue

        qtext = q.get("question") or q.get("generated_question") or ""
        score = q.get("chain_score", q.get("risk_score", 0.0))
        lic   = "🔒" if q.get("has_licence_conflict") else ""
        pat   = "📄" if q.get("has_patent_node") else ""

        with st.expander(f"Q{i} {lic}{pat} [{cat.upper()}] score={score:.3f}"):
            st.markdown(f"### {qtext}")

            at = q.get("audit_trail", {})
            if at:
                st.markdown("**Audit trail:**")
                hop1 = at.get("hop_1", "")
                hop2 = at.get("hop_2", "")
                hop3 = at.get("hop_3", "")
                if hop1: st.markdown(f"  → Hop 1: `{str(hop1)[:100]}`")
                if hop2: st.markdown(f"  → Hop 2: `{str(hop2)[:100]}`")
                if hop3: st.markdown(f"  → Hop 3: `{str(hop3)[:100]}`")
                if at.get("evidence_source"):
                    st.caption(f"Source: {at['evidence_source']}")

            provider = q.get("provider_used", "")
            if provider:
                st.caption(f"Generated by: {provider}")
    st.markdown("---")

# ── Contradictions ────────────────────────────────────────────────────────────

contra_data = _load("contradiction_evidence.json")
if contra_data:
    contradictions = contra_data.get("contradictions", [])
    if contradictions:
        st.header(f"⚔️ Proprietary Claim Contradictions ({len(contradictions)})")
        for c in contradictions:
            with st.expander(
                f"🔴 {c.get('risk_type','?')} — `{c.get('contradictory_module','?')}`"
            ):
                st.markdown(f"**Claim:** {c.get('claim_text','—')[:200]}")
                st.markdown(f"**Contradictory module:** `{c.get('contradictory_module','?')}`")
                st.markdown(f"**Recommended action:** {c.get('recommended_action','—')}")
                st.metric("Confidence", f"{c.get('confidence_score', 0):.3f}")
        st.markdown("---")

# ── Audit Trail ───────────────────────────────────────────────────────────────

audit_data = _load("audited_vc_report.json")
if audit_data:
    audited = audit_data.get("audited_items", [])
    st.header(f"🔐 SHA-256 Audit Trail ({len(audited)} items)")
    for item in audited[:10]:
        sev  = item.get("severity", "?")
        icon = _severity_color(sev)
        tid  = item.get("traceability_id", "?")
        cat  = item.get("category", "?")
        with st.expander(f"{icon} `{tid}` — {cat}"):
            st.markdown(f"**Question:** {item.get('question','—')}")
            st.markdown(f"**Formal confidence:** {item.get('formal_confidence', 0):.3f}")
            st.markdown(f"**Recommended action:** {item.get('recommended_action','—')}")
            st.caption(
                f"Timestamp: {item.get('timestamp','?')}  |  "
                f"Status: {item.get('audit_status','?')}"
            )
    if len(audited) > 10:
        st.caption(f"Showing first 10 of {len(audited)} items.")
    st.markdown("---")

# ── Eval Results ──────────────────────────────────────────────────────────────

eval_data = _load("eval_results.json")
if eval_data:
    st.header("📈 Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Questions",     eval_data.get("total_questions", "—"))
    c2.metric(
        "Audit Trail Coverage",
        f"{eval_data.get('audit_trail_coverage', 0):.2%}"
    )
    if "ground_truth_overlap" in eval_data:
        c3.metric(
            "Ground Truth Overlap",
            f"{eval_data['ground_truth_overlap']:.2%}"
        )
    st.markdown("---")

# ── Raw JSON Explorer ─────────────────────────────────────────────────────────

st.header("🗂️ Raw Output Explorer")
available = [f.name for f in PROCESSED.iterdir() if f.suffix == ".json"]
selected = st.selectbox("Select output file", sorted(available))
if selected:
    raw = _load(selected)
    if raw is not None:
        st.json(raw, expanded=False)

st.caption(
    "ChainCheck | Multi-Hop Reasoning for VC Technical Due Diligence  "
    "| All data from `data/processed/`"
)
