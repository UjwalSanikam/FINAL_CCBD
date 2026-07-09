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

def _build_prompt_from_structured_evidence(ev: dict) -> str:
    """Build a prompt from structured_evidence.json format."""
    claim  = ev.get("claim_text", ev.get("claim_id", ""))
    path   = ev.get("reasoning_path", [])
    rtype  = ev.get("risk_type", "Unknown Risk")
    score  = ev.get("confidence_score", 0.0)
    risk_n = ev.get("risk_node", "")

    path_str = " → ".join(path) if path else "N/A"

    return f"""EVIDENCE CHAIN
--------------
Claim from pitch deck: "{claim}"
Risk type: {rtype}
Confidence score: {score:.3f}
Reasoning path: {path_str}
Risk node flagged: {risk_n}

Generate exactly one adversarial due-diligence question an investor should ask
the founder about this evidence chain."""


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

def _template_question(chain: dict, source_format: str) -> str:
    """Generate a deterministic fallback question when LLM is unavailable."""
    if source_format == "structured_evidence":
        claim  = chain.get("claim_text", chain.get("claim_id", "unknown claim"))[:100]
        rtype  = chain.get("risk_type", "IP risk")
        node   = chain.get("risk_node", "an external component")
        return (
            f"You claim '{claim}', yet our analysis identified a {rtype} "
            f"involving '{node}'. Can you explain your legal strategy and "
            f"how you have validated that this does not create liability at scale?"
        )
    else:
        # hop_chain format
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


# ── Category classifier ───────────────────────────────────────────────────────

def _classify(chain: dict, source_format: str) -> str:
    if source_format == "structured_evidence":
        return chain.get("risk_type", "IP Overlap")
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

    if max_questions:
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
            except Exception as e:
                logger.warning("[%s] LLM failed, using template: %s", chain_id, e)
                question_text = _template_question(chain, source_format)
                provider_used = "template_fallback"
                template_count += 1

        # Build audit trail
        if source_format == "structured_evidence":
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
            has_patent_node=chain.get("has_patent_node", chain.get("risk_type") == "IP Overlap"),
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
