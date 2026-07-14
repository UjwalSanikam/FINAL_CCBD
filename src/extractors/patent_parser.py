"""
patent_parser.py — Track 3: Patent Relation Extractor
======================================================
Multi-Hop Reasoning System for Venture Capital Technical Due Diligence

Implementation based on:
    Siddharth, L. & Luo, J. (2024). "Retrieval Augmented Generation using
    Engineering Design Knowledge." Knowledge-Based Systems, 303, 112410.
    https://doi.org/10.1016/j.knosys.2024.112410

Core methodology (Section 3 of the paper):
  - Extract noun phrases as candidate head/tail entities
  - For each (head, tail) entity pair in a sentence, extract the tokens
    between/around them as the explicit relationship
  - Output structured {head entity :: relationship :: tail entity} triples
  - Store triples in JSON for downstream RAG / Knowledge Graph construction

Usage:
    python src/extractors/patent_parser.py
    python src/extractors/patent_parser.py --input data/raw/my_patent.txt
    python src/extractors/patent_parser.py --input data/raw/ --output data/processed/
"""

import re
import json
import argparse
import logging
from pathlib import Path
from itertools import combinations
from typing import Optional

# ── Optional spaCy (falls back to rule-based NP extraction) ──────────────────
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
        _SPACY_FULL = True
    except OSError:
        # Model missing — use blank pipeline with sentencizer for sentence splitting
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
        _SPACY_FULL = False
        logging.warning(
            "en_core_web_sm not found — using blank spaCy + rule-based NP extraction. "
            "Run 'python -m spacy download en_core_web_sm' for higher accuracy."
        )
except ImportError:
    # spaCy not installed at all — fall back to regex-based sentence splitting
    _NLP = None
    _SPACY_FULL = False
    logging.warning(
        "spaCy not installed — using regex sentence splitting + rule-based NP extraction. "
        "Install spaCy and en_core_web_sm for higher accuracy."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING  (Paper §3.1 heuristics)
# ─────────────────────────────────────────────────────────────────────────────
_SKIP_SECTIONS = re.compile(
    r"^(background|prior art|brief description of|examples?|embodiments?|"
    r"preamble|title|publication date|grant date|assignee|inventors|status|"
    r"patent number|cpc classifications|number of claims)",
    re.IGNORECASE,
)
_BRACKET_NOISE = re.compile(r"[\(\[\{<][^\)\]\}>]{0,80}[\)\]\}>]")
_FIG_REF       = re.compile(r"\bFIG\.\s*\d+\w*", re.IGNORECASE)


def clean_patent_text(raw: str) -> str:
    text = _FIG_REF.sub(" FIGREF ", raw)
    text = _BRACKET_NOISE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sections(raw: str) -> dict:
    # strip the "PATENT ID: XXXX" line — it has inline content so it won't
    # match the header pattern below, and would otherwise leak into
    # PREAMBLE and get tokenized as prose
    raw = re.sub(r"^PATENT ID:.*$", "", raw, flags=re.MULTILINE)

    # Real patent files (e.g. USPTO/Google Patents style dumps) use headers
    # of several shapes, none of which is "alone on its own line":
    #   "Title: Cryptographic hash function acceleration..."   (inline content)
    #   "Claims (20)"                                           (trailing count)
    #   "What is claimed is:"                                   (USPTO boilerplate alias for Claims)
    # This pattern matches the header keyword at the start of a line,
    # optionally followed by a parenthetical count and/or a colon, and
    # captures whatever else remains on that same line as inline content.
    header_re = re.compile(
        r"^(ABSTRACT|CLAIMS?|WHAT IS CLAIMED IS|DESCRIPTION|SUMMARY|FIELD|"
        r"BACKGROUND|DRAWINGS?|EXAMPLES?|EMBODIMENTS?|"
        r"TITLE|PUBLICATION DATE|GRANT DATE|ASSIGNEE|INVENTORS|STATUS|"
        r"PATENT NUMBER|CPC CLASSIFICATIONS|NUMBER OF CLAIMS)"
        r"\s*(?:\([^)]*\))?\s*:?[ \t]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Headers that mean the same thing but appear under different names get
    # normalized to one canonical section key, so downstream code (and
    # _SKIP_SECTIONS) only needs to know about "CLAIMS", not every phrasing.
    _HEADER_ALIASES = {"WHAT IS CLAIMED IS": "CLAIMS"}

    matches = list(header_re.finditer(raw))
    sections: dict = {}

    if not matches:
        # No recognizable headers at all — keep everything so downstream
        # code still has text to work with, rather than silently dropping it.
        sections["PREAMBLE"] = raw.strip()
        return sections

    # Anything before the first recognized header is preamble text.
    if matches[0].start() > 0:
        pre = raw[: matches[0].start()].strip()
        if pre:
            sections["PREAMBLE"] = pre

    for idx, m in enumerate(matches):
        name = m.group(1).upper().rstrip(":")
        name = _HEADER_ALIASES.get(name, name)
        inline_content = m.group(2).strip() if m.group(2) else ""
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()
        content = f"{inline_content} {body}".strip() if inline_content else body

        if name in sections:
            sections[name] = f"{sections[name]} {content}".strip()
        else:
            sections[name] = content

    return sections


def get_artefact_sentences(raw: str) -> list:
    sections = split_into_sections(raw)
    kept = []
    for name, content in sections.items():
        if _SKIP_SECTIONS.match(name):
            logger.debug("Skipping section: %s", name)
            continue
        kept.append(content)

    cleaned = clean_patent_text(" ".join(kept))
    doc = _NLP(cleaned)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        wc = len(text.split())
        if 3 <= wc <= 100:   # paper §3.1: retain sentences ≤100 words
            sentences.append(text)
    logger.info("Extracted %d artefact sentences.", len(sentences))
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 2. NOUN PHRASE EXTRACTION  (Paper §3.1: entities from spaCy noun phrases)
# ─────────────────────────────────────────────────────────────────────────────
_STOP_ENTITIES = {
    "a","an","the","this","these","that","those","said","each","every","any",
    "it","they","which","who","where","when","how","what","one","two","three",
    "also","further","wherein","thereby","thereof","therefrom","herein",
}

_NP_REGEX = re.compile(
    r"\b(?:(?:a|an|the|this|these|that|those|said|each|every|any|"
    r"plurality of|multiple|various)\s+)?"
    r"(?:[A-Za-z]+-)*[A-Za-z]+"
    r"(?:\s+(?:[A-Za-z]+-)*[A-Za-z]+){0,4}\b"
)


def extract_noun_phrases(sentence: str) -> list:
    if _SPACY_FULL:
        doc = _NLP(sentence)
        nps = [
            c.text.strip() for c in doc.noun_chunks
            if c.text.strip().lower() not in _STOP_ENTITIES and len(c.text.split()) <= 6
        ]
    else:
        raw = _NP_REGEX.findall(sentence)
        seen, nps = set(), []
        for np in raw:
            np = np.strip()
            if np.lower() not in _STOP_ENTITIES and 1 <= len(np.split()) <= 6 and not np.isdigit():
                key = np.lower()
                if key not in seen:
                    seen.add(key)
                    nps.append(np)
    return nps


# ─────────────────────────────────────────────────────────────────────────────
# 3. RELATIONSHIP EXTRACTION  (Paper §3.2 — REL token identification)
#    Engineering-domain relationship verb/preposition patterns from Table 1
# ─────────────────────────────────────────────────────────────────────────────

# Compiled as a simple alternation (no multi-line string issues)
_REL_INDICATORS = (
    r"compris(?:es?|ing)|includ(?:es?|ing)|consist(?:s|ing) of|"
    r"contain(?:s|ing)|constitut(?:es?|ing)|"
    r"has|have|hav(?:e|ing)|"
    r"connected to|attached to|mounted on|coupled to|"
    r"linked to|joined to|fixed to|secured to|"
    r"is connected|is attached|is mounted|is coupled|"
    r"is linked|is joined|is fixed|is secured|"
    r"is disposed|is located|is positioned|is configured|"
    r"is adapted|is designed|is arranged|is operable|"
    r"is transmitted|is converted|is transferred|is generated|"
    r"is supplied|is controlled|is driven|is powered|"
    r"is directed to|relates to|pertains to|"
    r"extends through|extends from|extends into|extends along|extends to|"
    r"passes through|runs through|"
    r"converts|transfers|transmits|generates|supplies|"
    r"monitors|measures|detects|senses|controls|adjusts|"
    r"modifies|optimizes|"
    r"communicated via|communicated through|communicated by|"
    r"such as|including|"
    r"from|to|via|through|by|with|within|between|among|into|"
    r"of|in|on|at|for|as"
)
_REL_VERBS = re.compile(_REL_INDICATORS, re.IGNORECASE)


def extract_relationship_between(sentence: str, head: str, tail: str) -> Optional[str]:
    """
    Implements paper Figure 2 logic:
    Given sentence + marked HEAD/TAIL, extract the REL tokens between them.
    """
    s_lower = sentence.lower()
    head_idx = s_lower.find(head.lower())
    tail_idx = s_lower.find(tail.lower())

    if head_idx == -1 or tail_idx == -1:
        return None

    # Handle both orderings (paper Table 1 shows tail can precede head)
    if head_idx < tail_idx:
        between = sentence[head_idx + len(head): tail_idx].strip()
    else:
        between = sentence[tail_idx + len(tail): head_idx].strip()

    rel = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", between)
    rel = re.sub(r"\s+", " ", rel).strip()

    # If too long, find strongest indicator
    if len(rel.split()) > 8 or not rel:
        m = _REL_VERBS.search(between)
        if m:
            start = max(0, m.start() - 3)
            end   = min(len(between), m.end() + 15)
            rel   = between[start:end].strip()
            rel   = re.sub(r"^[\s,;:]+|[\s,;:]+$", "", rel).strip()
        else:
            return None

    return rel if rel and len(rel) >= 2 else None


def extract_triples_from_sentence(sentence: str) -> list:
    """Full triple extraction for one sentence — paper Figure 1 / §3.2."""
    nps = extract_noun_phrases(sentence)
    if len(nps) < 2:
        return []

    triples = []
    for head, tail in combinations(nps[:10], 2):
        if head.lower() == tail.lower():
            continue
        rel = extract_relationship_between(sentence, head, tail)
        if rel:
            triples.append({
                "head":            head.strip(),
                "relationship":    rel.strip(),
                "tail":            tail.strip(),
                "source_sentence": sentence,
            })
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATENT-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_triples_from_patent(text: str, patent_id: str = "UNKNOWN") -> list:
    sentences   = get_artefact_sentences(text)
    all_triples = []
    for sent in sentences:
        triples = extract_triples_from_sentence(sent)
        for t in triples:
            t["patent_id"] = patent_id
        all_triples.extend(triples)
    logger.info("Patent %s → %d sentences → %d triples.", patent_id, len(sentences), len(all_triples))
    return all_triples


def save_triples(triples: list, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_triples": len(triples),
                "format":        "{head entity :: relationship :: tail entity}",
                "reference":     "Siddharth & Luo (2024) Knowledge-Based Systems 303:112410",
            },
            "triples": triples,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d triples → %s", len(triples), output_path)


def process_directory(input_dir: str, output_dir: str) -> list:
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_triples = []
    for f in input_path.glob("*.txt"):
        triples = extract_triples_from_patent(f.read_text(encoding="utf-8"), patent_id=f.stem)
        all_triples.extend(triples)
        save_triples(triples, str(output_path / f"{f.stem}_triples.json"))
    save_triples(all_triples, str(output_path / "knowledge_base.json"))
    return all_triples


def print_triples_table(triples: list, max_rows: int = 25) -> None:
    print(f"\n{'─'*78}")
    print(f"  Extracted KG Triples  ({min(max_rows, len(triples))} of {len(triples)} shown)")
    print(f"  Format: {{head :: relationship :: tail}}")
    print(f"  Siddharth & Luo (2024) KBS 303:112410")
    print(f"{'─'*78}")
    for t in triples[:max_rows]:
        print(f"  {t['head']} :: {t['relationship']} :: {t['tail']}")
    print(f"{'─'*78}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Patent Relation Extractor — {head :: rel :: tail} triples (Siddharth & Luo 2024)"
    )
    parser.add_argument("--input",  "-i", default="data/raw/",       help="Patent .txt file or directory")
    parser.add_argument("--output", "-o", default="data/processed/", help="Output directory")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_file():
        triples = extract_triples_from_patent(p.read_text(encoding="utf-8"), patent_id=p.stem)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        save_triples(triples, str(out / f"{p.stem}_triples.json"))
        print_triples_table(triples)
    elif p.is_dir():
        triples = process_directory(str(p), args.output)
        print_triples_table(triples)
    else:
        logger.error("Input path '%s' does not exist.", args.input)
        raise SystemExit(1)


if __name__ == "__main__":
    main()