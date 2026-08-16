"""
question_gen.py — Adversarial LLM Question Generator
=====================================================
ChainCheck | Multi-Hop Reasoning Pipeline

Takes structured_evidence.json or hop_chains.json and calls an LLM to
generate one pointed, adversarial due-diligence question per chain.

Supports two providers via --provider flag:
  anthropic  — Uses claude-sonnet-4-6 (set ANTHROPIC_API_KEY)
  ollama     — Uses a local Ollama model (default: llama3; run `ollama pull llama3`)

Every generated question is returned with its full provenance audit trail
so the explainability engine can stamp a SHA-256 trace ID.

Output
------
  data/processed/questions.json

Usage
-----
  export ANTHROPIC_API_KEY=your_key_here
  python src/generation/question_gen.py --provider anthropic

  # No API key — local Ollama
  python src/generation/question_gen.py --provider ollama --model llama3

  # Dry-run: print prompts, no API call
  python src/generation/question_gen.py --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import time
import argparse
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_MODEL   = "claude-sonnet-4-6"
OLLAMA_MODEL      = "llama3"
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT_S  = float(os.environ.get("OLLAMA_TIMEOUT_S", "20"))
MAX_TOKENS        = 512
RETRY_ATTEMPTS    = 3
RETRY_DELAY_S     = 2.0

SYSTEM_PROMPT = """You are an adversarial technical due-diligence analyst for a top-tier
venture capital firm. Your job is to generate exactly ONE sharp, targeted question
for investors to ask startup founders.

Rules:
- The question must follow directly from the evidence chain provided.
- Never ask a question you could answer yourself with generic knowledge.
- Use technical vocabulary appropriate to the domain.
- Maximum 3 sentences. Do not add preamble, explanation, or lists.
- Output only the question, nothing else."""


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DueDiligenceQuestion:
    chain_id:          str
    question:          str
    question_category: str
    has_licence_conflict: bool
    has_patent_node:   bool
    chain_score:       float
    audit_trail:       dict
    raw_provenance:    dict
    provider_used:     str


class OllamaModelNotFound(RuntimeError):
    """Raised when Ollama is reachable but the requested local model is absent."""


# ── LLM providers ─────────────────────────────────────────────────────────────

def _call_anthropic(prompt: str) -> str:
    """Call Anthropic API. Raises on failure."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _ollama_request(path: str, payload: Optional[dict] = None) -> dict:
    """Call Ollama's HTTP API using only the standard library."""
    url = f"{OLLAMA_HOST}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama API error from {url}: HTTP {e.code} {body.strip()}"
        ) from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        raise ConnectionError(
            f"Ollama daemon is not reachable at {OLLAMA_HOST}. "
            "Start it with `ollama serve` or set OLLAMA_HOST."
        ) from reason
    except (TimeoutError, socket.timeout) as e:
        raise TimeoutError(
            f"Ollama request to {OLLAMA_HOST} timed out after {OLLAMA_TIMEOUT_S:.1f}s"
        ) from e


def _check_ollama_available(model: str) -> None:
    """Fail fast with a useful message before per-question generation starts."""
    data = _ollama_request("/api/tags")
    available_names = {
        item.get("name", "")
        for item in data.get("models", [])
        if item.get("name")
    }
    available_bases = {name.split(":", 1)[0] for name in available_names}
    requested_base = model.split(":", 1)[0]
    if available_bases and requested_base not in available_bases:
        installed = ", ".join(sorted(available_names)) or "none"
        raise OllamaModelNotFound(
            f"Ollama is reachable at {OLLAMA_HOST}, but model '{model}' is not installed. "
            f"Installed models: {installed}. Run `ollama pull {model}` or pass "
            "`--model <installed-model>`."
        )


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama daemon. Raises clear connection/model errors."""
    response = _ollama_request(
        "/api/chat",
        {
            "model": model,
            "stream": False,
            "options": {"num_predict": MAX_TOKENS},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    )
    try:
        return response["message"]["content"].strip()
    except KeyError as e:
        raise RuntimeError(f"Unexpected Ollama response from {OLLAMA_HOST}: {response}") from e


def _call_llm(prompt: str, provider: str, model: str) -> str:
    """Route to the correct LLM provider with retries."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            if provider == "anthropic":
                return _call_anthropic(prompt)
            elif provider == "ollama":
                return _call_ollama(prompt, model=model)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"All {RETRY_ATTEMPTS} LLM attempts failed")


# ── Prompt builder ────────────────────────────────────────────────────────────

_RELATIONSHIP_INSTRUCTIONS = {
    "CONTRADICTS": (
        "This is a CONTRADICTION. The pitch deck explicitly denies reliance on "
        "third-party/open-source technology, but the evidence shows exactly that "
        "dependency in use. Ask the founder to reconcile the specific denial with "
        "the specific dependency found — name both directly."
    ),
    "PRIOR_ART": (
        "This is an ACADEMIC PRIOR ART risk. The pitch claims in-house/original "
        "development, but the evidence shows a dependency that implements a "
        "previously published technique. Ask whether this prior art was disclosed "
        "or considered, not whether infringement occurred."
    ),
    "PATENT_OVERLAP": (
        "This is a PATENT OVERLAP risk. Ask a claim-element-specific question about "
        "how the implementation compares to the patent's actual claims, not a vague "
        "'does this infringe' question."
    ),
    "LICENSE": (
        "This is a LICENSE risk. Ask specifically about license obligations of the "
        "named dependency and how they interact with the startup's IP ownership claims."
    ),
    "DEPENDENCY": (
        "This is a general dependency-risk finding. Ask what the dependency is used "
        "for and what the commercialization/maintenance implications are."
    ),
    "ARCHITECTURAL_DEPENDENCY": (
        "This is an ARCHITECTURAL DEPENDENCY, not a confirmed patent risk. The "
        "path reaches a patent-flagged node only via a weak semantic bridge from "
        "a real code dependency/module — not through direct patent claim evidence. "
        "Ask what role this dependency actually plays in the architecture and "
        "whether the pitch's proprietary/in-house framing is accurate for this "
        "component. Do NOT phrase this as a patent-infringement question — no "
        "genuine patent-claim evidence supports that framing here. "
        "MANDATORY: the question MUST explicitly name, verbatim, both the "
        "intermediate node and the terminal node from the reasoning path below "
        "— do not paraphrase them as 'external services' or 'this dependency'. "
        "Name them exactly as given."
    ),
    "PATENT_RELEVANT": (
        "This is only WEAK PATENT RELEVANCE — a semantic-similarity match to "
        "patent language, with no dependency routing and no verified patent "
        "edge. Do NOT frame this as overlap or infringement. Ask, at most, "
        "whether the team is aware of this patent and has reviewed it — "
        "phrased as pure awareness-checking, not a risk allegation."
    ),
}

def _build_prompt_from_structured_evidence(ev: dict) -> str:
    """Build a prompt from structured_evidence.json format.

    Deliberately uses ONLY ev['claim_text'] — the single primary claim
    chosen by consolidate_by_dependency — never _merged_claim_texts.
    Question-generation context must be scoped strictly to this one
    reasoning path; merged claims exist for audit/classification only.
    """
    claim  = f'"{ev.get("claim_text", ev.get("claim_id", ""))}"'
    path   = ev.get("reasoning_path", [])
    relationship_kind = ev.get("_relationship") or classify_relationship(ev)
    rtype  = _RELATIONSHIP_TO_CATEGORY[relationship_kind]
    score  = ev.get("confidence_score", 0.0)
    risk_n = ev.get("risk_node", "")
    relationship = ev.get("relationship", "inferred")
    instruction = _RELATIONSHIP_INSTRUCTIONS[relationship_kind]

    path_str = " → ".join(path) if path else "N/A"

    if relationship == "verified":
        evidence_strength = (
            "VERIFIED — this path is backed entirely by structural facts "
            "(confirmed code imports and/or confirmed license bindings), not by "
            "semantic guesswork."
        )
    else:
        evidence_strength = (
            "INFERRED — at least one step in this path is an embedding-similarity "
            "match, not a confirmed relationship. The connection may be coincidental "
            "term overlap rather than a real functional or legal link."
        )

    required_terms = ""
    if relationship_kind == "ARCHITECTURAL_DEPENDENCY" and len(path) >= 3:
        required_terms = (
            f"\nRequired terms — your question MUST contain both of these "
            f"exact strings: \"{path[1]}\" and \"{path[-1]}\""
        )

    return f"""EVIDENCE CHAIN
--------------
Claim from pitch deck: {claim}
Risk type: {rtype}
Relationship: {relationship_kind} — {instruction}
Evidence strength: {evidence_strength}
Confidence score: {score:.3f}
Reasoning path: {path_str}
Risk node flagged: {risk_n}{required_terms}
Generate exactly one due-diligence question an investor should ask the founder
about this evidence chain, following the relationship-specific instruction above.

If evidence strength is VERIFIED, ask a direct, pointed question about the
implications of this confirmed fact.

If evidence strength is INFERRED, do NOT assume the connection is real or that
infringement/overlap has occurred. Phrase the question to first ask the founder
to confirm or clarify whether a genuine relationship exists, before asking about
its implications."""


def _build_prompt_from_hop_chain(chain: dict) -> str:
    """Build a prompt from hop_chains.json format."""
    start     = chain.get("start_node", "")
    nodes     = chain.get("path_nodes", [])
    edges     = chain.get("path_edges", [])
    score     = chain.get("chain_score", 0.0)
    prov      = chain.get("provenance", {})
    node_data = prov.get("nodes", [])

    path_parts = []
    for i, (n, e) in enumerate(zip(nodes, edges + [""])):
        ntype = chain.get("path_node_types", [""])[i] if i < len(chain.get("path_node_types", [])) else ""
        meta = next((nd.get("metadata", {}) for nd in node_data if nd.get("node_id") == n), {})
        text = meta.get("text") or meta.get("full_text") or n
        path_parts.append(f"[{ntype}] {text[:80]}")
        if e:
            path_parts.append(f"  —[{e}]→")

    path_str   = "\n".join(path_parts)
    lic_flag   = chain.get("has_licence_conflict", False)
    pat_flag   = chain.get("has_patent_node", False)
    risk_flags = []
    if lic_flag:
        risk_flags.append("LICENSE CONFLICT")
    if pat_flag:
        risk_flags.append("PATENT OVERLAP")
    flags_str  = ", ".join(risk_flags) or "General IP concern"

    return f"""MULTI-HOP EVIDENCE CHAIN
------------------------
Chain score: {score:.4f}
Risk flags: {flags_str}
Starting node: {start}

Reasoning path:
{path_str}

Generate exactly one adversarial due-diligence question an investor should ask
the founder about this evidence chain."""


# ── Template fallback ─────────────────────────────────────────────────────────

def _clean_snippet(text: str, limit: int = 140) -> str:
    """Return a compact, word-boundary-safe snippet for deterministic questions."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{trimmed}..."


def _article_for(text: str) -> str:
    return "an" if str(text).strip().lower()[:1] in {"a", "e", "i", "o", "u"} else "a"


def _template_question(chain: dict, source_format: str) -> str:
    """Deterministic, strictly evidence-scoped fallback question.

    Each branch names only entities that appear in this chain's own
    reasoning_path / claim_text / risk_node — never merged-claim content,
    never patents/licenses unless the relationship type actually supports it.
    """
    if source_format != "structured_evidence":
        # hop_chain format — unchanged
        start    = chain.get("start_node", "your claimed technology")
        nodes    = chain.get("path_nodes", [])
        has_pat  = chain.get("has_patent_node", False)
        has_lic  = chain.get("has_licence_conflict", False)
        if has_lic and has_pat:
            suffix = "given both open-source license obligations and active patent overlap?"
        elif has_lic:
            suffix = "given the open-source license restrictions we identified?"
        elif has_pat:
            suffix = "given the active patent we found in this dependency chain?"
        else:
            suffix = "and how does this third-party dependency chain affect your IP defensibility?"
        mid = f"'{nodes[1]}'" if len(nodes) > 1 else "external libraries"
        return (
            f"Your architecture references '{start}', which relies on {mid}. "
            f"What is your commercialization strategy {suffix}"
        )

    claim = _clean_snippet(chain.get("claim_text", chain.get("claim_id", "unknown claim")))
    node = chain.get("risk_node", "an external component")
    path = chain.get("reasoning_path", []) or []
    relationship = chain.get("_relationship") or chain.get("relationship") or classify_relationship(chain)

    if relationship == "CONTRADICTS":
        return (
            f'The pitch states "{claim}", while repository evidence shows \'{node}\' '
            f"in use. Can you clarify whether {node} is used in the production "
            f"implementation, and which components remain independently developed?"
        )

    if relationship == "PRIOR_ART":
        return (
            f'The pitch states "{claim}", while the repository shows a dependency on '
            f"'{node}', which implements a previously published technique. Was this "
            f"prior art disclosed or considered as part of your IP strategy?"
        )

    if relationship == "ARCHITECTURAL_DEPENDENCY" and len(path) >= 3:
        mid = path[1]
        return (
            f'The pitch states "{claim}", while the repository shows \'{mid}\' being '
            f"used within '{node}'. What role does {mid} play in the production "
            f"architecture, and which parts of the system were independently implemented?"
        )

    if relationship == "PATENT_OVERLAP":
        return (
            f'The pitch states "{claim}". Which specific elements of your implementation '
            f"were compared against the claims of '{node}' during your freedom-to-operate "
            f"review?"
        )

    if relationship == "PATENT_RELEVANT":
        return (
            f"Our evidence graph flags a weak semantic similarity between your pitch "
            f'claim "{claim}" and the language of \'{node}\'. Is your team aware of this '
            f"reference, and has it been reviewed?"
        )

    if relationship == "LICENSE":
        return (
            f'The pitch states "{claim}". What license obligations apply to \'{node}\', '
            f"and how do they interact with your IP ownership claims?"
        )

    # DEPENDENCY — default
    return (
        f'The pitch states "{claim}", while repository evidence identifies \'{node}\' as a '
        f"dependency. Which components rely on {node}, and what functionality remains "
        f"proprietary?"
    )

# ── Relationship classifier ───────────────────────────────────────────────────
# Distinguishes WHY a claim connects to evidence, not just WHAT node type it
# touches. Without this, every claim→patent path becomes "IP Overlap" even
# when the real story is a contradiction (claim denies X, code does X) or
# prior art (claim says "in-house", dependency is a published technique).

_CONTRADICTION_CLAIM_PHRASES = [
    "not a wrapper", "no third-party", "not third-party",
    "built and own", "own outright", "no open-source",
    "not open-source", "without relying on", "independently",
]
_ORIGINALITY_CLAIM_PHRASES = [
    "in-house", "built in house", "owned in-house", "in house",
    "years of r&d", "in-house r&d", "designed and implemented in-house",
]
_KNOWN_ACADEMIC_DEPENDENCIES = {
    "colbert-ai", "colbert", "rank-bm25", "sentence-transformers", "transformers",
}

_KNOWN_ACADEMIC_DEPENDENCIES = {
    "colbert-ai", "colbert", "rank-bm25", "sentence-transformers", "transformers",
}

# Stdlib modules are never a meaningful dependency risk — they're always
# present, never third-party, and never something a claim could contradict.
try:
    import sys as _sys
    _STDLIB_MODULE_NAMES = set(_sys.stdlib_module_names)
except AttributeError:
    # Python <3.10 fallback: small manual list covering what shows up in
    # import-graph parsing for typical repos.
    _STDLIB_MODULE_NAMES = {
        "typing", "abc", "collections", "functools", "dataclasses", "enum",
        "itertools", "json", "os", "sys", "re", "io", "pathlib", "logging",
        "datetime", "uuid", "base64", "socket", "threading", "copy", "types",
        "operator", "string", "math", "random", "argparse", "traceback",
        "warnings", "inspect", "contextlib", "asyncio", "shutil", "subprocess",
    }


def is_low_information_node(risk_node: str) -> bool:
    """
    True for terminal nodes that carry no due-diligence signal: bare stdlib
    module names, or unresolved-license placeholders. These should never
    become the subject of a question regardless of which claim they attach to.
    """
    node = str(risk_node or "").strip().lower()
    if node in _STDLIB_MODULE_NAMES:
        return True
    if node.startswith("unknown"):
        return True
    return False


def classify_relationship(ev: dict) -> str:
    """
    Returns one of: CONTRADICTS, PRIOR_ART, PATENT_OVERLAP, LICENSE, DEPENDENCY.
    Heuristic, text-based — operates on claim_text + reasoning_path node names
    since structured_evidence chains carry no explicit relationship label yet.
    """
    claim_text = (ev.get("claim_text") or "").lower()
    path = ev.get("reasoning_path", []) or []
    path_lower = [str(p).lower() for p in path]
    path_str = " ".join(path_lower)
    risk_node = str(ev.get("risk_node", "")).lower()

    # Any evidence object whose TERMINAL node is a real dependency (per
    # path_reasoner's own typing, via risk_type == "Dependency") counts as
    # touching a dependency for contradiction purposes — not just the
    # narrow academic whitelist. That whitelist is still used separately
    # for prior-art detection, since prior art specifically means "this is
    # a known published technique," which is a narrower claim than "this is
    # some dependency."
    is_real_dependency_terminal = ev.get("risk_type") == "Dependency"
    touches_known_academic_dep = any(
        dep in path_str for dep in _KNOWN_ACADEMIC_DEPENDENCIES
    )

    # Contradiction: claim explicitly denies third-party/open-source reliance,
    # but the path actually routes through (or terminates at) a real dependency.
    if (is_real_dependency_terminal or touches_known_academic_dep) and any(
        p in claim_text for p in _CONTRADICTION_CLAIM_PHRASES
    ):
        return "CONTRADICTS"

    # Prior art: claim asserts originality/in-house R&D, but the path routes
    # through a dependency known to implement a published academic technique.
    # Deliberately kept narrow (whitelist only) — "in-house" + "uses some
    # random dependency" isn't prior art, it's just a dependency question.
    if touches_known_academic_dep and any(p in claim_text for p in _ORIGINALITY_CLAIM_PHRASES):
        return "PRIOR_ART"

    # License: terminal risk node is a license identifier or license-flagged.
    # License: terminal risk node is a license identifier or license-flagged.
    if ev.get("risk_type") == "Commercial License" or risk_node in {
        "mit", "apache-2.0", "gpl-2.0", "gpl-3.0", "agpl-3.0", "bsd-2-clause",
        "bsd-3-clause", "unknown (requires manual review ⚠️)",
    }:
        return "LICENSE"

    # path_reasoner now assigns risk_type conservatively based on whether a
    # VERIFIED (non-semantic) edge actually reaches a patent node — read
    # those categories directly instead of re-deriving from node presence.
    claim_mentions_patent = any(
        p in claim_text for p in ("provisional patent", "patent filed", "patent pending", "covered by a patent")
    )
    risk_type = ev.get("risk_type")

    if risk_type == "Architectural Dependency":
        return "ARCHITECTURAL_DEPENDENCY"

    if risk_type == "Patent Relevance":
        # Semantic similarity to patent language, no dependency routing and
        # no verified patent edge — weakest possible patent-adjacent signal.
        # Never allowed to read as overlap.
        return "PATENT_RELEVANT" if not claim_mentions_patent else "PATENT_OVERLAP"

    if risk_type == "Patent Overlap" or (risk_type == "IP Overlap" and ev.get("has_patent_node")):
        # Legitimate genuine overlap: a verified edge actually reached a
        # patent node, or the older literal "IP Overlap" string is present
        # AND has_patent_node backs it up (kept for backward compatibility
        # with any cached/legacy evidence objects).
        return "PATENT_OVERLAP"

    if claim_mentions_patent and ("patent" in risk_node or risk_type in {"Patent Relevance", "Architectural Dependency"}):
        return "PATENT_OVERLAP"

    return "DEPENDENCY"


_RELATIONSHIP_TO_CATEGORY = {
    "CONTRADICTS":               "Ownership Contradiction",
    "PRIOR_ART":                 "Academic Prior Art",
    "PATENT_OVERLAP":            "IP Overlap",
    "PATENT_RELEVANT":           "Patent Relevance",
    "ARCHITECTURAL_DEPENDENCY":  "Architectural Dependency",
    "LICENSE":                   "Commercial License",
    "DEPENDENCY":                "Dependency Risk",
}

# ── Hallucination guard ───────────────────────────────────────────────────────
# Deterministically checks whether the LLM invented a patent number/identifier
# that isn't actually present anywhere in that chain's evidence. This is
# cheaper and more reliable than asking the LLM to self-check, and it never
# needs a second LLM call — a failing question just falls back to the
# template, which never fabricates identifiers in the first place.

_PATENT_ID_PATTERN = re.compile(r'\bUS\s?-?\d{6,}\w*\b', re.IGNORECASE)
_PATENT_MENTION_PATTERN = re.compile(
    r'\bpatent\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Za-z]{0,2}\d{4,}|X{3,}|XYZ)\b',
    re.IGNORECASE,
)


def _extract_real_patent_ids(ev: dict) -> set[str]:
    """Collect any real patent-ID-shaped tokens (e.g. US12511322) that
    actually appear in this chain's own evidence — claim text and
    reasoning path — so we know what the question is allowed to cite."""
    haystack = " ".join([
        str(ev.get("claim_text", "")),
        " ".join(str(n) for n in ev.get("reasoning_path", [])),
        str(ev.get("risk_node", "")),
    ])
    return {m.upper().replace(" ", "") for m in _PATENT_ID_PATTERN.findall(haystack)}


def question_has_fabricated_patent(question_text: str, ev: dict) -> bool:
    """True if the question cites a specific patent number/placeholder that
    isn't backed by any real patent ID in this chain's evidence."""
    real_ids = _extract_real_patent_ids(ev)
    for match in _PATENT_MENTION_PATTERN.finditer(question_text):
        token = match.group(1).upper()
        if token not in real_ids:
            return True
    return False

def validate_question_grounding(question_text: str, ev: dict) -> bool:
    """
    Returns False if the question raises a topic the reasoning path doesn't
    support — patents, licensing, infringement, commercialization — unless
    the chain's classified relationship actually licenses that topic.
    Also returns False for ARCHITECTURAL_DEPENDENCY chains that fail to
    name the intermediate/terminal path nodes verbatim (points 3/4).
    A failing question falls back to the template, which is grounded by
    construction and names required nodes by construction.
    """
    relationship = ev.get("_relationship") or ev.get("relationship") or classify_relationship(ev)
    text = question_text.lower()

    if relationship == "ARCHITECTURAL_DEPENDENCY":
        path = ev.get("reasoning_path", []) or []
        if len(path) >= 3:
            intermediate, terminal = path[1], path[-1]
            if intermediate.lower() not in text or terminal.lower() not in text:
                return False

    if re.search(r"\bpatent(s|ed|ing)?\b", text) and relationship not in {"PATENT_OVERLAP", "PATENT_RELEVANT"}:
        return False
    if re.search(r"\blicens(e|ing)\b", text) and relationship != "LICENSE":
        return False
    if re.search(r"\binfring\w*", text) and relationship != "PATENT_OVERLAP":
        return False
    if re.search(r"\bcommerciali[sz]", text) and relationship not in {"DEPENDENCY", "ARCHITECTURAL_DEPENDENCY"}:
        return False
    if re.search(r"\btrade secret", text) and relationship not in {"CONTRADICTS", "PRIOR_ART"}:
        return False

    return True


# ── Category classifier ───────────────────────────────────────────────────────

def _classify(chain: dict, source_format: str) -> str:
    if source_format == "structured_evidence":
        relationship = classify_relationship(chain)
        chain["_relationship"] = relationship  # stash for prompt builder / dedup
        return _RELATIONSHIP_TO_CATEGORY[relationship]
    has_lic = chain.get("has_licence_conflict", False)
    has_pat = chain.get("has_patent_node", False)
    if has_lic and has_pat:
        return "licence_and_patent"
    if has_lic:
        return "licence_conflict"
    if has_pat:
        return "patent_overlap"
    return "dependency_risk"


# ── Main generator ────────────────────────────────────────────────────────────

def generate_questions(
    chains_path: Path,
    output_path: Path,
    provider: str = "anthropic",
    model: str = "",
    dry_run: bool = False,
    max_questions: Optional[int] = None,
) -> list[DueDiligenceQuestion]:
    """
    Load chains from chains_path, generate one question per chain.
    Falls back to templates automatically if the LLM call fails.
    """
    _model = model or (OLLAMA_MODEL if provider == "ollama" else ANTHROPIC_MODEL)

    data = json.loads(chains_path.read_text(encoding="utf-8"))

    # Auto-detect format
    if "evidence_objects" in data:
        chains = data["evidence_objects"]
        source_format = "structured_evidence"
    else:
        chains = data.get("chains", [])
        source_format = "hop_chains"

    if source_format == "structured_evidence":
        # Drop low-information risk nodes (stdlib modules, unresolved licenses)
        # before anything else competes with them for a question slot.
        chains = [
            c for c in chains
            if not is_low_information_node(c.get("risk_node", ""))
        ]

        # Tag each chain with its relationship type up front so dedup can use it.
        for c in chains:
            c["_relationship"] = classify_relationship(c)

        # Merge chains that share the same terminal dependency/node across
        # different claims — one underlying risk shouldn't become N questions.
        chains = consolidate_by_dependency(chains)

        # Diversity cap: keep at most 2 questions per (claim_id, relationship)
        # family, prioritizing the highest-confidence chain in each family, so
        # one claim/dependency pair can't consume the whole question budget.
        from collections import defaultdict
        by_family: dict[tuple, list[dict]] = defaultdict(list)
        for c in sorted(chains, key=lambda x: x.get("confidence_score", 0.0), reverse=True):
            key = (c.get("claim_id"), c.get("_relationship"))
            if len(by_family[key]) < 2:
                by_family[key].append(c)

        diversified = [c for group in by_family.values() for c in group]
        diversified.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        chains = diversified

    if max_questions:
        if source_format == "structured_evidence":
            chains = select_diverse_top_n(chains, max_questions)
        else:
            chains = chains[:max_questions]

    logger.info(
        "Generating %d questions from %s via %s%s",
        len(chains), source_format, provider,
        " (DRY RUN)" if dry_run else "",
    )

    provider_available = True
    provider_error = ""
    if provider == "ollama" and not dry_run:
        try:
            _check_ollama_available(_model)
        except Exception as e:
            provider_available = False
            provider_error = str(e)
            logger.error("%s Falling back to deterministic templates for this run.", provider_error)

    questions: list[DueDiligenceQuestion] = []
    template_count = 0

    for i, chain in enumerate(chains):
        chain_id = chain.get("chain_id", f"chain_{i+1:04d}")

        if source_format == "structured_evidence":
            prompt = _build_prompt_from_structured_evidence(chain)
        else:
            prompt = _build_prompt_from_hop_chain(chain)

        if dry_run:
            logger.info("--- DRY RUN PROMPT [%s] ---\n%s\n---", chain_id, prompt)
            question_text = _template_question(chain, source_format)
            provider_used = "dry_run_template"
        elif not provider_available:
            question_text = _template_question(chain, source_format)
            provider_used = f"template_fallback_ollama_unavailable:{_model}"
            template_count += 1
        else:
            try:
                question_text = _call_llm(prompt, provider, _model)
                provider_used = f"{provider}/{_model}"
                if source_format == "structured_evidence":
                    if question_has_fabricated_patent(question_text, chain):
                        logger.warning(
                            "[%s] LLM question cited an unsupported patent reference — using template fallback",
                            chain_id,
                        )
                        question_text = _template_question(chain, source_format)
                        provider_used = f"{provider}/{_model}+hallucination_guard"
                        template_count += 1
                    elif not validate_question_grounding(question_text, chain):
                        logger.warning(
                            "[%s] LLM question raised an ungrounded topic — using template fallback",
                            chain_id,
                        )
                        question_text = _template_question(chain, source_format)
                        provider_used = f"{provider}/{_model}+grounding_guard"
                        template_count += 1
            except Exception as e:
                logger.warning("[%s] LLM failed, using template: %s", chain_id, e)
                question_text = _template_question(chain, source_format)
                provider_used = "template_fallback"
                template_count += 1

        # Build audit trail
        if source_format == "structured_evidence":
            # Canonicalize: ONE "relationship" field holding the classified
            # type (CONTRADICTS / PATENT_OVERLAP / ...), separate
            # "relationship_basis" holding verified/inferred. Do this after
            # the prompt was already built above, since the prompt builder
            # reads the pre-canonical "relationship" (verified/inferred).
            relationship_type = chain.get("_relationship") or classify_relationship(chain)
            chain["relationship_basis"] = chain.get("relationship", "inferred")
            chain["relationship"] = relationship_type
            chain.pop("_relationship", None)

            audit_trail = {
                "hop_1": f"Pitch deck claim: {chain.get('claim_text', chain.get('claim_id', ''))[:80]}",
                "hop_2": chain.get("reasoning_path", []),
                "hop_3": chain.get("risk_node", ""),
                "evidence_source": str(chains_path),
            }
            provenance = {k: v for k, v in chain.items() if k != "question"}
        else:
            prov = chain.get("provenance", {})
            node_dets = prov.get("nodes", [])
            audit_trail = {
                "hop_1": node_dets[0].get("node_id", "") if node_dets else "",
                "hop_2": [nd.get("node_id", "") for nd in node_dets[1:]],
                "hop_3": chain.get("path_nodes", [])[-1] if chain.get("path_nodes") else "",
                "evidence_source": str(chains_path),
            }
            provenance = chain.get("provenance", {})

        questions.append(DueDiligenceQuestion(
            chain_id=chain_id,
            question=question_text,
            question_category=_classify(chain, source_format),
            has_licence_conflict=chain.get("has_licence_conflict", False),
            has_patent_node=chain.get("has_patent_node", False),
            chain_score=chain.get("chain_score", chain.get("confidence_score", 0.0)),
            audit_trail=audit_trail,
            raw_provenance=provenance,
            provider_used=provider_used,
        ))

    if template_count:
        logger.warning(
            "%d/%d questions used template fallback — check LLM connectivity",
            template_count, len(questions),
        )

    return questions

# ── Cross-claim dependency deduplication ──────────────────────────────────────
# Multiple claims can independently point at the same dependency/technology
# (e.g. claim_0004, claim_0007, and claim_0008 all reference rank-bm25). Left
# alone, that produces N near-duplicate questions about one underlying risk.
# This merges same-risk_node chains into one, keeping the strongest
# relationship type present in the group and recording every contributing
# claim so the merged question can reference all of them.

_RELATIONSHIP_PRIORITY = [
    "CONTRADICTS", "PRIOR_ART", "ARCHITECTURAL_DEPENDENCY",
    "PATENT_OVERLAP", "LICENSE", "DEPENDENCY",
]


def consolidate_by_dependency(chains: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chains:
        key = str(c.get("risk_node", "")).strip().lower()
        groups[key].append(c)

    def relationship_rank(c: dict) -> int:
        rel = c.get("_relationship") or classify_relationship(c)
        c["_relationship"] = rel
        try:
            return _RELATIONSHIP_PRIORITY.index(rel)
        except ValueError:
            return len(_RELATIONSHIP_PRIORITY)

    consolidated: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            consolidated.append(group[0])
            continue

        group_sorted = sorted(group, key=lambda c: (relationship_rank(c), -c.get("confidence_score", 0.0)))
        primary = dict(group_sorted[0])
        primary["_merged_claim_ids"] = [c.get("claim_id", "") for c in group]
        primary["_merged_claim_texts"] = [c.get("claim_text", "") for c in group]
        primary["confidence_score"] = max(c.get("confidence_score", 0.0) for c in group)
        consolidated.append(primary)

    consolidated.sort(key=lambda c: c.get("confidence_score", 0.0), reverse=True)
    return consolidated

def select_diverse_top_n(chains: list[dict], n: int) -> list[dict]:
    """
    Guarantee representation across relationship types when trimming to
    max_questions. A flat confidence sort silently drops PATENT_OVERLAP/
    CONTRADICTS/PRIOR_ART chains, since they're structurally handicapped
    by hop-count length penalties relative to 1-hop Dependency chains —
    even when they're the more decision-relevant finding. Round-robin
    across relationship buckets (each internally confidence-sorted, since
    `chains` arrives pre-sorted) so every risk type gets a fair shot at
    the budget before any one type can crowd out the rest.
    """
    from collections import defaultdict, deque
    buckets: dict[str, deque] = defaultdict(deque)
    for c in chains:
        buckets[c.get("_relationship", "DEPENDENCY")].append(c)

    order = [r for r in _RELATIONSHIP_PRIORITY if r in buckets]
    order += [r for r in buckets if r not in order]

    selected: list[dict] = []
    while len(selected) < n and any(buckets.values()):
        for rel in order:
            if buckets[rel]:
                selected.append(buckets[rel].popleft())
                if len(selected) >= n:
                    break
    return selected


def _claim_summary(ev: dict) -> str:
    """Renders one claim, or — for a consolidated multi-claim chain —
    every contributing claim, so the prompt/template can reference all of
    them instead of silently dropping the merge down to just one."""
    claim_ids = ev.get("_merged_claim_ids")
    claim_texts = ev.get("_merged_claim_texts")
    if claim_ids and claim_texts and len(claim_ids) > 1:
        lines = [f'- ({cid}): "{txt}"' for cid, txt in zip(claim_ids, claim_texts)]
        return "Multiple pitch claims independently reference this same evidence:\n" + "\n".join(lines)
    return f'"{ev.get("claim_text", ev.get("claim_id", ""))}"'

def save_questions(questions: list[DueDiligenceQuestion], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "total_questions": len(questions),
            "by_category": {
                cat: sum(1 for q in questions if q.question_category == cat)
                for cat in sorted({q.question_category for q in questions})
            },
            "with_licence_conflict": sum(1 for q in questions if q.has_licence_conflict),
            "with_patent_node":      sum(1 for q in questions if q.has_patent_node),
        },
        "questions": [asdict(q) for q in questions],
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d questions → %s", len(questions), output_path)


def print_questions(questions: list[DueDiligenceQuestion]) -> None:
    print(f"\n{'═'*70}")
    print("  GENERATED DUE-DILIGENCE QUESTIONS")
    print(f"{'═'*70}")
    for i, q in enumerate(questions, 1):
        flags = []
        if q.has_licence_conflict: flags.append("LICENCE⚠")
        if q.has_patent_node:      flags.append("PATENT")
        flag_str = " ".join(flags)
        print(f"\n  Q{i}. [{q.chain_id}] {q.question_category.upper()} {flag_str}")
        print(f"  Score: {q.chain_score:.4f}  Provider: {q.provider_used}")
        print(f"  ▸ {q.question}")
        print(f"  Audit: {q.audit_trail.get('hop_1', '')[:60]} → "
              f"{str(q.audit_trail.get('hop_3', ''))[:40]}")
    print(f"{'═'*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ChainCheck adversarial question generator")
    ap.add_argument("--chains",    default=None,
                    help="Path to structured_evidence.json or hop_chains.json")
    ap.add_argument("--output",    default=None)
    ap.add_argument("--provider",  choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model",     default="",
                    help="Override default model name for the selected provider")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--max",       type=int, default=None)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    chains_path = (
        Path(args.chains) if args.chains
        else (base / "data" / "processed" / "structured_evidence.json")
    )
    if not chains_path.exists():
        chains_path = base / "data" / "processed" / "hop_chains.json"

    output_path = (
        Path(args.output) if args.output
        else (base / "data" / "processed" / "questions.json")
    )

    questions = generate_questions(
        chains_path=chains_path,
        output_path=output_path,
        provider=args.provider,
        model=args.model,
        dry_run=args.dry_run,
        max_questions=args.max,
    )
    print_questions(questions)
    save_questions(questions, output_path)
