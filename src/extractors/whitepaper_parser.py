"""
whitepaper_parser.py
--------------------
Track 1 – Startup Claims Extractor
Multi-Hop Reasoning System for VC Technical Due Diligence

Ingests a PDF whitepaper/pitch deck from data/raw/, extracts and structures
technical claims, named entities, and feature assertions, then writes a
clean JSON file to data/processed/ for downstream Knowledge Graph fusion.

Fixes applied vs v1:
  1. OCR fallback via pytesseract + pdf2image for scanned/image-only PDFs
  2. Switched primary extractor to pdfplumber (layout=True) for multi-column support
  3. Fixed _to_serialisable double-serialization bug (was calling asdict twice)
  4. Eliminated duplicate page re-scan in statistics block
  5. Finer confidence scoring (normalized per claim-type pattern count)
  6. Extended sentence splitter to protect version numbers & citation markers
  7. Added --ocr-only flag to force OCR regardless of text layer presence
  8. Graceful import degradation: pdfplumber → pypdf → OCR, all optional

Dependencies (install what you need):
    pip install pdfplumber pypdf pytesseract pdf2image pillow pydantic
    # System: apt install tesseract-ocr poppler-utils  (for OCR fallback)

Usage:
    python src/extractors/whitepaper_parser.py --input data/raw/bitcoin.pdf
    python src/extractors/whitepaper_parser.py --input data/raw/ --batch
    python src/extractors/whitepaper_parser.py --input data/raw/scan.pdf --ocr-only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.buzzwords import BUZZWORD_MAP

# ── PDF extraction backends (graceful degradation) ────────────────────────────
_PDFPLUMBER_AVAILABLE = False
_PYPDF_AVAILABLE = False
_OCR_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    _PYPDF_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image  # noqa: F401 — confirms Pillow is present
    _OCR_AVAILABLE = True
except ImportError:
    pass

# ── Optional pydantic validation layer ────────────────────────────────────────
try:
    from pydantic import BaseModel, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TechnicalClaim:
    """A single extracted technical assertion from the whitepaper."""
    claim_id: str
    page: int
    sentence: str                       # cleaned, de-noised sentence
    claim_type: str                     # performance | architecture | security | protocol | data
    confidence: float                   # heuristic score 0-1
    buzzwords: list[str] = field(default_factory=list)
    numeric_assertions: list[str] = field(default_factory=list)   # e.g. "10,000 TPS"
    entities: list[str] = field(default_factory=list)             # ORGs, techs, protocols


@dataclass
class FeatureAssertion:
    """A product/system capability claimed by the startup."""
    assertion_id: str
    page: int
    raw_text: str
    normalized: str           # marketing → engineering label
    category: str             # scalability | security | consensus | storage | network


@dataclass
class ParsedWhitepaper:
    """Top-level output object written to JSON."""
    source_file: str
    parsed_at: str
    total_pages: int
    pdf_metadata: dict
    extraction_method: str            # "pdfplumber" | "pypdf" | "ocr"
    full_text_preview: str            # first 500 chars for sanity-check
    technical_claims: list[TechnicalClaim]
    feature_assertions: list[FeatureAssertion]
    named_entities: list[str]         # deduplicated across the doc
    statistics: dict


# ── Optional pydantic schema ──────────────────────────────────────────────────
if PYDANTIC_AVAILABLE:
    class WhitepaperOutputSchema(BaseModel):
        source_file: str
        parsed_at: str
        total_pages: int
        pdf_metadata: dict
        extraction_method: str
        full_text_preview: str
        technical_claims: list[dict]
        feature_assertions: list[dict]
        named_entities: list[str]
        statistics: dict

        @field_validator("total_pages")
        @classmethod
        def pages_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError("total_pages must be >= 1")
            return v


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
})


def _normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_LIGATURES)


def _clean_pdf_text(raw: str) -> str:
    """Remove common PDF extraction artefacts."""
    # Merge hyphenated line-breaks  e.g. "block-\nchain" → "blockchain"
    text = re.sub(r"-\n(\w)", r"\1", raw)
    # Collapse whitespace runs (preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise excessive newlines → paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lone page-number lines (integers 1-999 on their own line)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    # Remove short all-caps lines (headers/footers)
    text = re.sub(r"(?m)^[A-Z][A-Z\s\-]{2,40}$", "", text)
    text = _normalise_unicode(text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Sentence splitter resilient to abbreviations, version numbers,
    citation markers, and decimal numbers.

    FIX v2: added protection for:
      - version strings  (v1.2.3,  2.0.1, etc.)
      - citation markers ([1]. [12].)
      - decimal numbers  (3.14. The → won't split mid-number)
    """
    # Protect common abbreviations
    text = re.sub(
        r"\b(Dr|Mr|Mrs|Prof|Fig|et al|vs|i\.e|e\.g|approx|ref|No|vol|pp|ed|ch)\.",
        r"\1<DOT>", text,
    )
    # Protect version strings: v1.2 / 2.0.1 / 3.14
    text = re.sub(r"\bv?(\d+)\.(\d+)(\.\d+)*\.", r"v\1<DOT>\2\3.", text)
    # Protect citation markers: [1]. [12].
    text = re.sub(r"\[(\d+)\]\.", r"[\1]<DOT>", text)

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [
        s.replace("<DOT>", ".").strip()
        for s in sentences
        if len(s.split()) > 4
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Entity & claim extraction patterns
# ─────────────────────────────────────────────────────────────────────────────


# Technical claim type classifiers
CLAIM_PATTERNS: dict[str, list[str]] = {
    "performance": [
        r"\d[\d,\.]+\s*(tps|tx/s|transactions per second|req/s|rps|ops/s|mb/s|gb/s|ms|milliseconds?|microseconds?)",
        r"\d+[x×]\s*(faster|speedup|improvement|throughput)",
        r"latency\s+(of|under|below|less than)\s+\d",
        r"(sub[\s-]?\d+|<\s*\d+)\s*(ms|microseconds?|second)",
        r"(throughput|bandwidth|capacity)\s+of\s+\d",
    ],
    "security": [
        r"(256|128|512)[\s-]?bit\s+(encryption|key|hash)",
        r"(sha[\s-]?2|sha[\s-]?3|aes|rsa|ecdsa|ed25519)",
        r"(zero[\s-]?knowledge|zk[\s-]?(proof|snark|stark))",
        r"(byzantine|bft|fault[\s-]?toleran)",
        r"(tamper[\s-]?(proof|evident|resistant)|immutable)",
        r"(sybil|replay|51%|double[\s-]spend)\s+(attack|resistant|proof)",
    ],
    "architecture": [
        r"(microservices?|monolith|serverless|event[\s-]?driven)",
        r"(p2p|peer[\s-]to[\s-]peer|distributed|decentrali[sz]ed)",
        r"(merkle|dag|trie|patricia|radix)\s+(tree|graph|structure)",
        r"(sharding|partitioning|horizontal[\s-]?scal)",
        r"(consensus|gossip|paxos|raft|pbft|tendermint)",
        r"(layer[\s-]?[12]|l[12][\s,\)]|sidechain|rollup|plasma|state[\s-]?channel)",
    ],
    "protocol": [
        r"(tcp|udp|http[s]?|grpc|websocket|libp2p|devp2p)",
        r"(proof[\s-]of[\s-](work|stake|authority|space|history))",
        r"(evm|wasm|webassembly|bytecode|opcode)",
        r"(abi|rpc|api|sdk|cli|daemon|node)\s+(interface|endpoint|call)",
        r"(mining|validator|staking|delegation|slashing)",
    ],
    "data": [
        r"(leveldb|rocksdb|badger|sqlite|postgres|ipfs|swarm|filecoin)",
        r"(utxo|account[\s-]?model|state[\s-]?trie|receipt)",
        r"(bloom[\s-]?filter|patricia[\s-]?trie|sparse[\s-]?merkle)",
        r"(serializ|deseri|rlp|protobuf|cbor|msgpack)",
        r"(replication|replicat|durability|consistency|availability)",
    ],
}

# Simplified regex-based NER (no spaCy dependency)
ENTITY_PATTERNS = [
    r"\b(Bitcoin|Ethereum|Solana|Cardano|Polkadot|Cosmos|Avalanche|IPFS|Filecoin|"
    r"Hyperledger|EVM|Wasm|WebAssembly|Tendermint|Paxos|Raft|PBFT|"
    r"SHA[\s-]?[23]\d*|AES[\s-]?\d+|RSA[\s-]?\d+|ECDSA|Ed25519|secp256k1|"
    r"libp2p|devp2p|RLP|EIP[\s-]?\d+|BIP[\s-]?\d+|RFC[\s-]?\d+)\b",
    r"\b(Merkle|Patricia|Bloom|Trie|DAG|UTXO|PoW|PoS|PoA|BFT|ZK[\s-]?SNARK|ZK[\s-]?STARK)\b",
    r"\b[A-Z][a-z]+(?:Chain|Net|DB|VM|FS|IO|OS|AI|ML|SDK|API)\b",
]

NUMERIC_ASSERTION_RE = re.compile(
    r"\b\d[\d,\.]*\s*"
    r"(tps|tx/s|req/s|rps|ops/s|mb/s|gb/s|pb|tb|gb|ms|μs|ns|"
    r"seconds?|minutes?|hours?|days?|nodes?|peers?|validators?|"
    r"bits?|bytes?|kb|mb|gb|%|x|times?)\b",
    re.IGNORECASE,
)


def _detect_claim_type(sentence: str) -> tuple[str, float]:
    """
    Return (claim_type, confidence) for a sentence.

    FIX v2: confidence is now normalized per claim-type pattern count so that
    a type with 5 patterns doesn't trivially outscore one with 2 patterns,
    and a single match no longer yields 0.5 confidence regardless of context.
    """
    s = sentence.lower()
    best_type, best_norm_score = "general", 0.0
    for ctype, patterns in CLAIM_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, s, re.IGNORECASE))
        if matches == 0:
            continue
        # Normalize: fraction of patterns hit, capped so ≥60 % hit → high confidence
        norm = matches / len(patterns)
        if norm > best_norm_score:
            best_norm_score = norm
            best_type = ctype
    confidence = round(min(best_norm_score * 1.5, 1.0), 2)   # scale up slightly
    return best_type, confidence


def _extract_entities(sentence: str) -> list[str]:
    """Simple regex-based named-entity recognition."""
    found = []
    for pat in ENTITY_PATTERNS:
        found.extend(re.findall(pat, sentence))
    return list(dict.fromkeys(found))   # deduplicate, preserve order


def _extract_buzzwords(sentence: str) -> tuple[list[str], str]:
    """Find marketing buzzwords; return (buzzwords_found, normalized_sentence)."""
    found_buzz: list[str] = []
    normalized = sentence
    for pattern, replacement in BUZZWORD_MAP.items():
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            found_buzz.append(m.group(0).lower())
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return found_buzz, normalized.strip()


def _classify_feature_category(sentence: str) -> str:
    s = sentence.lower()
    if any(w in s for w in ["scale", "throughput", "tps", "sharding", "partition"]):
        return "scalability"
    if any(w in s for w in ["encrypt", "hash", "signature", "zero-knowledge", "bft", "tamper"]):
        return "security"
    if any(w in s for w in ["consensus", "proof-of", "validator", "mining", "stake"]):
        return "consensus"
    if any(w in s for w in ["storage", "database", "db", "ipfs", "filestore", "archive"]):
        return "storage"
    if any(w in s for w in ["network", "p2p", "peer", "gossip", "libp2p", "tcp", "udp"]):
        return "network"
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# PDF reading backends
# ─────────────────────────────────────────────────────────────────────────────

def _read_with_pdfplumber(pdf_path: Path) -> tuple[list[tuple[int, str]], dict, str]:
    """
    Primary extractor: pdfplumber with layout=True for multi-column support.
    Returns ([(page_num, text), ...], metadata, method_label).
    """
    pages_text: list[tuple[int, str]] = []
    metadata: dict = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        if pdf.metadata:
            metadata = {k: str(v) for k, v in pdf.metadata.items() if v}
        for i, page in enumerate(pdf.pages, start=1):
            try:
                # layout=True preserves column order better than default
                raw = page.extract_text(layout=True) or ""
            except Exception as e:
                log.warning("pdfplumber: page %d unreadable (%s) — skipping", i, e)
                raw = ""
            if raw.strip():
                pages_text.append((i, raw))
    return pages_text, metadata, "pdfplumber"


def _read_with_pypdf(pdf_path: Path) -> tuple[list[tuple[int, str]], dict, str]:
    """
    Fallback extractor: pypdf (no layout awareness, but widely compatible).
    """
    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        raise ValueError(f"Corrupt or malformed PDF: {e}") from e
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}") from e

    if reader.is_encrypted:
        raise ValueError("PDF is password-protected; cannot extract text.")

    metadata: dict = {}
    if reader.metadata:
        metadata = {k.lstrip("/"): str(v) for k, v in reader.metadata.items() if v}

    pages_text: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            log.warning("pypdf: page %d unreadable (%s) — skipping", i, e)
            raw = ""
        if raw.strip():
            pages_text.append((i, raw))

    return pages_text, metadata, "pypdf"


def _read_with_ocr(pdf_path: Path, dpi: int = 300) -> tuple[list[tuple[int, str]], dict, str]:
    """
    OCR fallback using pytesseract + pdf2image.
    Rasterizes each page at `dpi` then runs Tesseract.

    NEW in v2: this path is now properly wired into the pipeline so scanned
    pitch decks are processed rather than rejected with a ValueError.
    """
    if not _OCR_AVAILABLE:
        raise ValueError(
            "OCR fallback requires pytesseract, pdf2image, and Pillow.\n"
            "Install: pip install pytesseract pdf2image pillow\n"
            "System:  apt install tesseract-ocr poppler-utils"
        )

    log.info("Running Tesseract OCR at %d DPI — this may take a moment…", dpi)
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        raise ValueError(f"pdf2image could not rasterize PDF: {e}") from e

    pages_text: list[tuple[int, str]] = []
    for i, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img, lang="eng")
        except Exception as e:
            log.warning("OCR: page %d failed (%s) — skipping", i, e)
            text = ""
        if text.strip():
            pages_text.append((i, text))
            log.info("OCR page %d: %d chars extracted", i, len(text))

    return pages_text, {}, "ocr"


def _read_pdf(
    pdf_path: Path,
    force_ocr: bool = False,
) -> tuple[list[tuple[int, str]], dict, str]:
    """
    Smart PDF reader with three-tier fallback:
      1. pdfplumber  (layout-aware, handles multi-column pitch decks)
      2. pypdf       (wide compatibility fallback)
      3. Tesseract OCR (scanned / image-only PDFs)          ← NEW in v2

    Returns (pages_text, metadata, extraction_method).
    Raises ValueError only if ALL tiers fail.
    """
    if not _PDFPLUMBER_AVAILABLE and not _PYPDF_AVAILABLE:
        raise ValueError(
            "No PDF library found. Install at least one: "
            "pip install pdfplumber OR pip install pypdf"
        )

    if force_ocr:
        log.info("--ocr-only flag set; skipping text-layer extraction")
        return _read_with_ocr(pdf_path)

    # Tier 1: pdfplumber
    if _PDFPLUMBER_AVAILABLE:
        try:
            pages_text, metadata, method = _read_with_pdfplumber(pdf_path)
            if pages_text:
                log.info("Extracted text via pdfplumber from %d page(s)", len(pages_text))
                return pages_text, metadata, method
            log.info("pdfplumber yielded no text — trying pypdf")
        except Exception as e:
            log.warning("pdfplumber failed (%s) — trying pypdf", e)

    # Tier 2: pypdf
    if _PYPDF_AVAILABLE:
        try:
            pages_text, metadata, method = _read_with_pypdf(pdf_path)
            if pages_text:
                log.info("Extracted text via pypdf from %d page(s)", len(pages_text))
                return pages_text, metadata, method
            log.info("pypdf yielded no text — trying OCR")
        except ValueError:
            raise   # propagate encrypted / corrupt errors directly
        except Exception as e:
            log.warning("pypdf failed (%s) — trying OCR", e)

    # Tier 3: OCR fallback
    log.info("No text layer detected — falling back to Tesseract OCR")
    pages_text, metadata, method = _read_with_ocr(pdf_path)
    if not pages_text:
        raise ValueError(
            "All extraction methods failed (pdfplumber, pypdf, Tesseract OCR).\n"
            "The PDF may be blank, corrupted, or contain only vector graphics."
        )
    log.info("OCR extracted text from %d page(s)", len(pages_text))
    return pages_text, metadata, method


# ─────────────────────────────────────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────────────────────────────────────

class WhitepaperParser:
    """Extracts structured technical intelligence from a PDF whitepaper."""

    def __init__(self, pdf_path: Path, force_ocr: bool = False):
        self.pdf_path = pdf_path
        self.force_ocr = force_ocr

    def parse(self) -> ParsedWhitepaper:
        log.info("Parsing: %s", self.pdf_path.name)
        pages_text, metadata, extraction_method = _read_pdf(
            self.pdf_path, force_ocr=self.force_ocr
        )
        log.info("Extracted text from %d pages via %s", len(pages_text), extraction_method)

        technical_claims: list[TechnicalClaim] = []
        feature_assertions: list[FeatureAssertion] = []
        all_entities: list[str] = []
        claim_counter, assertion_counter = 0, 0
        full_text_parts: list[str] = []

        # FIX v2: track sentence count inline — eliminates the double re-scan
        # that the v1 statistics block performed.
        total_sentences = 0

        for page_num, raw_text in pages_text:
            cleaned = _clean_pdf_text(raw_text)
            full_text_parts.append(cleaned)
            sentences = _split_sentences(cleaned)
            total_sentences += len(sentences)

            for sentence in sentences:
                claim_type, confidence = _detect_claim_type(sentence)
                buzzwords, normalized = _extract_buzzwords(sentence)
                entities = _extract_entities(sentence)
                numerics = NUMERIC_ASSERTION_RE.findall(sentence)
                all_entities.extend(entities)

                has_signal = (
                    claim_type != "general"
                    or buzzwords
                    or numerics
                    or entities
                )
                if has_signal:
                    claim_counter += 1
                    technical_claims.append(TechnicalClaim(
                        claim_id=f"claim_{claim_counter:04d}",
                        page=page_num,
                        sentence=sentence,
                        claim_type=claim_type,
                        confidence=confidence,
                        buzzwords=buzzwords,
                        numeric_assertions=[
                            f"{n[0]} {n[1]}" if isinstance(n, tuple) else str(n)
                            for n in numerics
                        ],
                        entities=entities,
                    ))

                if buzzwords:
                    assertion_counter += 1
                    category = _classify_feature_category(normalized)
                    feature_assertions.append(FeatureAssertion(
                        assertion_id=f"assertion_{assertion_counter:04d}",
                        page=page_num,
                        raw_text=sentence,
                        normalized=normalized,
                        category=category,
                    ))

        full_text = "\n\n".join(full_text_parts)

        # Deduplicate entities, case-insensitive, preserve first-seen order
        seen: set[str] = set()
        unique_entities: list[str] = []
        for e in all_entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        # Build claims-by-type breakdown
        claims_by_type: dict[str, int] = {}
        for tc in technical_claims:
            claims_by_type[tc.claim_type] = claims_by_type.get(tc.claim_type, 0) + 1

        stats = {
            "pages_with_text": len(pages_text),
            "total_sentences_scanned": total_sentences,   # FIX: no second pass
            "technical_claims_extracted": len(technical_claims),
            "feature_assertions_extracted": len(feature_assertions),
            "unique_entities_found": len(unique_entities),
            "claims_by_type": claims_by_type,
            "extraction_method": extraction_method,
        }

        return ParsedWhitepaper(
            source_file=str(self.pdf_path),
            parsed_at=datetime.now(timezone.utc).isoformat(),
            total_pages=len(pages_text),
            pdf_metadata=metadata,
            extraction_method=extraction_method,
            full_text_preview=full_text[:500],
            technical_claims=technical_claims,
            feature_assertions=feature_assertions,
            named_entities=unique_entities,
            statistics=stats,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def _to_serialisable(obj: ParsedWhitepaper) -> dict:
    """
    Convert ParsedWhitepaper dataclass → plain dict for JSON serialization.

    FIX v2: v1 called asdict(obj) (which already recurses into nested
    dataclasses) AND then manually called asdict() again on each nested list,
    effectively double-serializing. Now we just call asdict() once.
    """
    return asdict(obj)


def write_output(result: ParsedWhitepaper, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_file).stem
    out_path = output_dir / f"{stem}_parsed.json"

    payload = _to_serialisable(result)

    if PYDANTIC_AVAILABLE:
        try:
            WhitepaperOutputSchema(**payload)
            log.info("Pydantic schema validation: PASSED")
        except Exception as e:
            log.warning("Pydantic schema validation warning: %s", e)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Output written → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track 1 – Whitepaper PDF parser for VC Due Diligence KG pipeline"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a single PDF file, or a directory (with --batch)",
    )
    p.add_argument(
        "--output-dir", default="data/processed",
        help="Output directory for JSON files (default: data/processed)",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Process all *.pdf files in --input directory",
    )
    p.add_argument(
        "--ocr-only", action="store_true",
        help="Skip text-layer extraction and always use Tesseract OCR "
             "(useful for fully scanned pitch decks)",
    )
    p.add_argument(
        "--ocr-dpi", type=int, default=300,
        help="DPI for OCR rasterization (default: 300; lower = faster, higher = more accurate)",
    )
    return p.parse_args()


def run(
    input_path: Path,
    output_dir: Path,
    force_ocr: bool = False,
) -> Optional[Path]:
    """Parse a single PDF. Returns output path on success, None on failure."""
    try:
        parser = WhitepaperParser(input_path, force_ocr=force_ocr)
        result = parser.parse()
        out = write_output(result, output_dir)
        log.info(
            "Done ✓  |  method=%s  claims=%d  assertions=%d  entities=%d",
            result.extraction_method,
            result.statistics["technical_claims_extracted"],
            result.statistics["feature_assertions_extracted"],
            result.statistics["unique_entities_found"],
        )
        return out
    except ValueError as e:
        log.error("Skipping %s — %s", input_path.name, e)
        return None
    except Exception as e:
        log.error("Unexpected error on %s: %s", input_path.name, e, exc_info=True)
        return None


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    force_ocr: bool = args.ocr_only

    # Patch OCR DPI if user supplied a custom value
    if args.ocr_dpi != 300 and _OCR_AVAILABLE:
        import functools
        original_ocr = _read_with_ocr
        globals()["_read_with_ocr"] = functools.partial(original_ocr, dpi=args.ocr_dpi)

    if args.batch:
        if not input_path.is_dir():
            sys.exit(f"ERROR: --batch requires --input to be a directory, got: {input_path}")
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"No PDF files found in {input_path}")
        log.info("Batch mode: found %d PDF(s)", len(pdfs))
        successes = [run(p, output_dir, force_ocr) for p in pdfs]
        ok = sum(1 for s in successes if s is not None)
        log.info("Batch complete: %d/%d succeeded", ok, len(pdfs))
    else:
        if not input_path.exists():
            sys.exit(f"ERROR: File not found: {input_path}")
        if input_path.suffix.lower() != ".pdf":
            sys.exit(f"ERROR: Expected a .pdf file, got: {input_path}")
        out = run(input_path, output_dir, force_ocr)
        if out is None:
            sys.exit(1)


if __name__ == "__main__":
    main()


