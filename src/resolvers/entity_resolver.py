"""
entity_resolver.py
------------------
Cross-Domain Entity Resolution Layer
Multi-Hop Reasoning System for VC Technical Due Diligence

Connects the three parser outputs by finding semantically equivalent
entities across domains using dense vector similarity.

Inputs:
    data/processed/<name>_parsed.json        (whitepaper_parser output)
    data/processed/codebase_knowledge.json   (github_parser output)
    data/processed/knowledge_base.json       (patent_parser output)

Output:
    data/processed/entity_matches.json

Usage:
    pip install sentence-transformers faiss-cpu
    python src/resolvers/entity_resolver.py
    python src/resolvers/entity_resolver.py --whitepaper data/processed/bitcoin_parsed.json
"""
import re
import json
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "all-MiniLM-L6-v2"   # fast, 384-dim, good cross-domain recall
SCORE_THRESH = 0.78                  # cosine similarity threshold (tune in Phase 3)
TOP_K        = 5                     # max matches per entity
MAX_REL_LEN  = 60                    # truncate long patent relationships
HASH_EMBEDDING_DIM = 384             # match MiniLM dimensionality for FAISS indexes

# Reuse the buzzword normalization from whitepaper_parser so entity strings
# are pre-normalized before embedding (reduces vocabulary mismatch).
BUZZWORD_MAP = {
    "hyper-fast":              "high-throughput",
    "data-mesh":               "distributed-data-architecture",
    "web3":                    "decentralized-web",
    "ai-powered":              "ml-augmented",
    "blockchain-enabled":      "distributed-ledger",
    "zero-knowledge":          "zk-proof",
    "quantum-resistant":       "post-quantum-cryptography",
    "infinite scal":           "horizontal-scalability",
    "trustless":               "cryptographically-verified",
    "real-time":               "low-latency",
    "lightning-fast":          "sub-second-latency",
    "military-grade":          "aes-256",
    "enterprise-grade":        "production-hardened",
    "on-chain":                "stored-in-ledger",
    "off-chain":               "off-ledger-computation",
    "layer-2":                 "l2-scaling-protocol",
    "smart-contract":          "self-executing-contract-code",
    "proof-of-work":           "pow",
    "proof-of-stake":          "pos",
    "sharding":                "horizontal-database-partitioning",
    "merkle":                  "merkle-tree-hash-structure",
}


class HashingTextEncoder:
    """Deterministic local fallback when transformer embeddings are unavailable."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        vectors = np.zeros((len(texts), HASH_EMBEDDING_DIM), dtype="float32")
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_]+", text.lower())
            if not tokens:
                tokens = [text.lower() or "<empty>"]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % HASH_EMBEDDING_DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, bucket] += sign
        return vectors


def load_embedding_model():
    """Use transformer embeddings by default; fallback to hashing when unavailable."""
    if os.getenv("CHAINCHECK_USE_HASHING", "").lower() in {"1", "true", "yes"}:
        logger.info(
            "Using deterministic local hashing embeddings because CHAINCHECK_USE_HASHING is set."
        )
        return HashingTextEncoder(), "hashing-local"

    if SentenceTransformer is None:
        logger.warning(
            "sentence-transformers is not installed. Falling back to local hashing."
        )
        return HashingTextEncoder(), "hashing-local"

    try:
        logger.info("Loading cached embedding model: %s", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True), MODEL_NAME
    except TypeError:
        try:
            logger.info("Loading embedding model: %s", MODEL_NAME)
            return SentenceTransformer(MODEL_NAME, device="cpu"), MODEL_NAME
        except Exception as e:
            logger.warning("Embedding model unavailable (%s). Falling back to local hashing.", e)
            return HashingTextEncoder(), "hashing-local"
    except Exception as e:
        logger.warning("Cached embedding model unavailable (%s). Falling back to local hashing.", e)
        return HashingTextEncoder(), "hashing-local"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EntityRecord:
    """A single entity from any source, tagged with its origin."""
    text: str            # raw entity string
    normalized: str      # cleaned + buzzword-expanded text used for embedding
    domain: str          # "whitepaper" | "codebase" | "patent"
    source_id: str       # claim_id, library name, or patent_id
    source_file: str
    extra: dict          # domain-specific metadata (claim_type, licence, etc.)


@dataclass
class EntityMatch:
    """A resolved cross-domain match between two entities."""
    entity_a: str
    domain_a: str
    source_id_a: str
    entity_b: str
    domain_b: str
    source_id_b: str
    cosine_score: float
    provenance: dict     # full source metadata for audit trail


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_entity(text: str) -> str:
    """
    Lowercase, strip punctuation, expand buzzwords.
    Shared vocabulary reduction across all three domains.
    """
    t = text.lower().strip()
    # Remove common punctuation but keep hyphens (important for tech terms)
    t = "".join(c if c.isalnum() or c in "-_ " else " " for c in t)
    t = " ".join(t.split())  # collapse whitespace
    # Expand buzzwords (shared regex map — see whitepaper_parser for matching logic)
    for pattern, canonical in BUZZWORD_MAP.items():
        t = re.sub(pattern, canonical, t, flags=re.IGNORECASE)
    return t


# ── Entity extraction from each parser's JSON ─────────────────────────────────

def extract_from_whitepaper(parsed_json: dict, source_file: str) -> list[EntityRecord]:
    records = []
    seen_texts = set()

    for claim in parsed_json.get("technical_claims", []):
        claim_type = claim.get("claim_type", "general")
        claim_id   = claim["claim_id"]
        page       = claim.get("page")
        sentence   = claim.get("sentence", "")

        # 1. Named entities (original behaviour)
        for entity in claim.get("entities", []):
            if len(entity) < 3 or entity.lower() in seen_texts:
                continue
            seen_texts.add(entity.lower())
            records.append(EntityRecord(
                text=entity,
                normalized=normalize_entity(entity),
                domain="whitepaper",
                source_id=claim_id,
                source_file=source_file,
                extra={"claim_type": claim_type, "confidence": claim.get("confidence"), "page": page}
            ))

        # 2. Buzzwords (original behaviour)
        for buzz in claim.get("buzzwords", []):
            key = buzz.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            records.append(EntityRecord(
                text=buzz,
                normalized=normalize_entity(buzz),
                domain="whitepaper",
                source_id=claim_id,
                source_file=source_file,
                extra={"claim_type": claim_type, "page": page, "is_buzzword": True}
            ))

        # 3. NEW — use the full claim sentence as an entity record
        # This captures technical content even when NER misses individual terms
        if claim_type != "general" and len(sentence) > 20:
            key = sentence[:50].lower()
            if key not in seen_texts:
                seen_texts.add(key)
                records.append(EntityRecord(
                    text=sentence[:120],
                    normalized=normalize_entity(sentence[:120]),
                    domain="whitepaper",
                    source_id=claim_id,
                    source_file=source_file,
                    extra={"claim_type": claim_type, "confidence": claim.get("confidence"),
                           "page": page, "is_full_claim": True}
                ))

        # 4. NEW — extract key noun phrases from sentences using simple chunking
        # Targets technical terms the regex NER misses
        tech_phrases = extract_tech_phrases(sentence)
        for phrase in tech_phrases:
            key = phrase.lower()
            if key in seen_texts or len(phrase) < 5:
                continue
            seen_texts.add(key)
            records.append(EntityRecord(
                text=phrase,
                normalized=normalize_entity(phrase),
                domain="whitepaper",
                source_id=claim_id,
                source_file=source_file,
                extra={"claim_type": claim_type, "page": page, "is_extracted_phrase": True}
            ))

    logger.info("Whitepaper: extracted %d entity records", len(records))
    return records


def extract_tech_phrases(sentence: str) -> list[str]:
    """
    Extract technical noun phrases missed by regex NER.
    Targets hyphenated terms, protocol names, and compound technical nouns.
    """
    phrases = []
    # Hyphenated tech terms: peer-to-peer, double-spending, proof-of-work
    hyphenated = re.findall(r'\b[a-z]+(?:-[a-z]+){1,4}\b', sentence.lower())
    phrases.extend([p for p in hyphenated if len(p) > 6])

    # Capitalised multi-word terms: Merkle Tree, Hash Function, Byzantine Fault
    capitalised = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', sentence)
    phrases.extend(capitalised)

    # Technical single words (longer ones likely to be meaningful)
    tech_singles = re.findall(
        r'\b(timestamp(?:ing)?|blockchain|cryptograph\w+|hash(?:ing)?|'
        r'consensus|decentrali[sz]\w+|Byzantine|validator|signature|'
        r'transaction|broadcast|ledger|protocol|encrypt\w+|nonce|'
        r'node(?:s)?|mining|proof|verification|distributed)\b',
        sentence, re.IGNORECASE
    )
    phrases.extend(tech_singles)

    return list(set(phrases))
def extract_from_codebase(codebase_json: dict, source_file: str) -> list[EntityRecord]:
    """
    Pull library names (with categories) from github_parser output.
    Library name + category string is embedded together for richer context.
    """
    records = []
    dep_map = codebase_json.get("dependency_map") or {}
    for dep in dep_map.get("all_dependencies", []):
        name     = dep.get("name", "")
        category = dep.get("category", "")
        ecosystem = dep.get("ecosystem", "")
        if not name:
            continue
        # Embed "libp2p peer-to-peer networking library" not just "libp2p"
        enriched = f"{name} {category.lower()} library"
        records.append(EntityRecord(
            text=name,
            normalized=normalize_entity(enriched),
            domain="codebase",
            source_id=name,
            source_file=source_file,
            extra={
                "category": category,
                "ecosystem": ecosystem,
                "version_spec": dep.get("version_spec"),
            }
        ))

    # Also pull top third-party libraries from import graph
    import_analysis = codebase_json.get("import_graph_analysis") or {}
    for lib_entry in import_analysis.get("top_third_party_libraries", []):
        lib = lib_entry.get("library", "")
        if lib and not any(r.source_id == lib for r in records):
            records.append(EntityRecord(
                text=lib,
                normalized=normalize_entity(lib),
                domain="codebase",
                source_id=lib,
                source_file=source_file,
                extra={
                    "imported_by_count": lib_entry.get("imported_by_count"),
                    "from_import_graph": True,
                }
            ))

    logger.info("Codebase: extracted %d entity records", len(records))
    return records


def extract_from_patents(patent_json: dict, source_file: str) -> list[EntityRecord]:
    """
    Pull head and tail entities from patent_parser triples.
    Embeds 'head relationship tail' as a unit for richer context.
    """
    records = []
    triples = patent_json.get("triples", [])

    if len(triples) > 1500:
        logger.info(
            "Patent triple volume is high (%d entries); embedding will be batched instead of truncating.",
            len(triples),
        )

    for triple in triples:
        head = triple.get("head", "").strip()
        tail = triple.get("tail", "").strip()
        rel  = triple.get("relationship", "").strip()[:MAX_REL_LEN]
        pid  = triple.get("patent_id", "UNKNOWN")

        if not head or len(head) < 3:
            continue

        head_context = f"{head} {rel} {tail}".strip()
        records.append(EntityRecord(
            text=head,
            normalized=normalize_entity(head_context),
            domain="patent",
            source_id=pid,
            source_file=source_file,
            extra={
                "relationship": rel,
                "tail": tail,
                "source_sentence": triple.get("source_sentence", "")[:200],
            }
        ))

        if tail and len(tail) >= 3:
            tail_context = f"{tail} {rel} {head}".strip()
            records.append(EntityRecord(
                text=tail,
                normalized=normalize_entity(tail_context),
                domain="patent",
                source_id=pid,
                source_file=source_file,
                extra={
                    "relationship": rel,
                    "head": head,
                    "source_sentence": triple.get("source_sentence", "")[:200],
                }
            ))

    logger.info("Patents: extracted %d entity records", len(records))
    return records


# ── Vector matching ───────────────────────────────────────────────────────────

def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product is cosine similarity."""
    normed = embeddings.astype("float32", copy=True)
    norms = np.linalg.norm(normed, axis=1, keepdims=True)
    np.divide(normed, norms, out=normed, where=norms != 0)
    return normed


def cross_domain_search(
    query_records: list[EntityRecord],
    query_embeddings: np.ndarray,
    target_records: list[EntityRecord],
    target_embeddings: np.ndarray,
    top_k: int = TOP_K,
    threshold: float = SCORE_THRESH,
) -> list[EntityMatch]:
    """
    For each query entity, find top-k matches in the target domain.
    Enforces cross-domain: query.domain != target.domain (checked upstream).
    """
    matches = []
    if query_embeddings.shape[0] == 0 or target_embeddings.shape[0] == 0:
        return matches

    query_normed = normalize_rows(query_embeddings)
    target_normed = normalize_rows(target_embeddings)
    k = min(top_k, len(target_records))
    sim_matrix = query_normed @ target_normed.T
    indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    scores = np.take_along_axis(sim_matrix, indices, axis=1)

    for i, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        qr = query_records[i]
        for score, idx in zip(row_scores, row_indices):
            if idx < 0 or float(score) < threshold:
                continue
            tr = target_records[idx]
            matches.append(EntityMatch(
                entity_a=qr.text,
                domain_a=qr.domain,
                source_id_a=qr.source_id,
                entity_b=tr.text,
                domain_b=tr.domain,
                source_id_b=tr.source_id,
                cosine_score=round(float(score), 4),
                provenance={
                    "entity_a_meta": qr.extra,
                    "entity_a_file": qr.source_file,
                    "entity_b_meta": tr.extra,
                    "entity_b_file": tr.source_file,
                    "normalized_a": qr.normalized,
                    "normalized_b": tr.normalized,
                }
            ))

    return matches


# ── Main resolver ─────────────────────────────────────────────────────────────

def resolve_entities(
    whitepaper_path: Path,
    codebase_path: Path,
    patent_path: Path,
    output_path: Path,
    threshold: float = SCORE_THRESH,
) -> list[EntityMatch]:
    """
    Full cross-domain entity resolution pipeline.
    Runs all three pairwise cross-domain searches:
        whitepaper ↔ codebase
        whitepaper ↔ patent
        codebase   ↔ patent
    """

    # ── 1. Load JSON outputs from the three parsers ────────────────────────
    logger.info("Loading parser outputs...")
    wp_json  = json.loads(whitepaper_path.read_text(encoding="utf-8"))
    cb_json  = json.loads(codebase_path.read_text(encoding="utf-8"))
    pat_json = json.loads(patent_path.read_text(encoding="utf-8"))

    # ── 2. Extract entity records ──────────────────────────────────────────
    wp_records  = extract_from_whitepaper(wp_json,  str(whitepaper_path))
    cb_records  = extract_from_codebase(cb_json,    str(codebase_path))
    pat_records = extract_from_patents(pat_json,    str(patent_path))

    if not wp_records:
        logger.warning("No whitepaper entities — check whitepaper_parser output")
    if not cb_records:
        logger.warning("No codebase entities — check github_parser output")
    if not pat_records:
        logger.warning("No patent entities — check patent_parser output")

    # ── 3. Embed all entity pools ──────────────────────────────────────────
    model, model_name = load_embedding_model()

    def embed(records: list[EntityRecord]) -> np.ndarray:
        texts = [r.normalized for r in records]
        chunk_size = 128
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), chunk_size):
            batch = texts[i : i + chunk_size]
            vec = model.encode(batch, batch_size=64, show_progress_bar=False)
            chunks.append(np.array(vec, dtype="float32"))
        if chunks:
            return np.vstack(chunks)
        return np.zeros((0, 384), dtype="float32")

    logger.info("Embedding whitepaper entities (%d)...", len(wp_records))
    wp_emb  = embed(wp_records)  if wp_records  else np.zeros((0, 384), dtype="float32")

    logger.info("Embedding codebase entities (%d)...",  len(cb_records))
    cb_emb  = embed(cb_records)  if cb_records  else np.zeros((0, 384), dtype="float32")

    logger.info("Embedding patent entities (%d)...",    len(pat_records))
    pat_emb = embed(pat_records) if pat_records else np.zeros((0, 384), dtype="float32")

    # ── 4. Run three cross-domain pairwise searches ────────────────────────
    all_matches: list[EntityMatch] = []

    # whitepaper → codebase
    if wp_records and cb_records:
        logger.info("Matching whitepaper → codebase...")
        m = cross_domain_search(wp_records, wp_emb, cb_records, cb_emb, threshold=threshold)
        logger.info("  Found %d matches", len(m))
        all_matches.extend(m)

    # whitepaper → patent
    if wp_records and pat_records:
        logger.info("Matching whitepaper → patent...")
        m = cross_domain_search(wp_records, wp_emb, pat_records, pat_emb, threshold=threshold)
        logger.info("  Found %d matches", len(m))
        all_matches.extend(m)

    # codebase → patent
    if cb_records and pat_records:
        logger.info("Matching codebase → patent...")
        m = cross_domain_search(cb_records, cb_emb, pat_records, pat_emb, threshold=threshold)
        logger.info("  Found %d matches", len(m))
        all_matches.extend(m)

    # ── 5. Deduplicate and sort by score ──────────────────────────────────
    # Remove symmetric duplicates (A→B same as B→A)
    seen = set()
    deduped = []
    for m in sorted(all_matches, key=lambda x: x.cosine_score, reverse=True):
        key = tuple(sorted([
            f"{m.domain_a}:{m.source_id_a}:{m.entity_a}",
            f"{m.domain_b}:{m.source_id_b}:{m.entity_b}",
        ]))
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    logger.info("Total unique cross-domain matches: %d", len(deduped))

    # ── 6. Write output ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "total_matches": len(deduped),
            "threshold_used": threshold,
            "model": model_name,
            "sources": {
                "whitepaper": str(whitepaper_path),
                "codebase":   str(codebase_path),
                "patent":     str(patent_path),
            },
            "entity_counts": {
                "whitepaper": len(wp_records),
                "codebase":   len(cb_records),
                "patent":     len(pat_records),
            }
        },
        "matches": [asdict(m) for m in deduped],
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved entity matches → %s", output_path)

    return deduped


def print_summary(matches: list[EntityMatch]) -> None:
    print(f"\n{'═'*72}")
    print("  ENTITY RESOLUTION SUMMARY")
    print(f"{'═'*72}")

    by_pair = {}
    for m in matches:
        pair = f"{m.domain_a} ↔ {m.domain_b}"
        by_pair[pair] = by_pair.get(pair, 0) + 1

    for pair, count in sorted(by_pair.items()):
        print(f"  {pair:<35} {count:>4} matches")

    print(f"\n  Top 10 highest-confidence matches:")
    print(f"  {'Score':<8} {'Domain A':<12} {'Entity A':<28} {'Entity B'}")
    print(f"  {'-'*70}")
    for m in sorted(matches, key=lambda x: x.cosine_score, reverse=True)[:10]:
        print(f"  {m.cosine_score:<8.4f} {m.domain_a:<12} {m.entity_a[:26]:<28} {m.entity_b[:24]}")
    print(f"{'═'*72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain entity resolver for VC due-diligence pipeline"
    )
    parser.add_argument("--whitepaper", "-w",
        default="data/processed/",
        help="Path to whitepaper parsed JSON (or directory to auto-detect first file)"
    )
    parser.add_argument("--codebase", "-c",
        default="data/processed/codebase_knowledge.json"
    )
    parser.add_argument("--patents", "-p",
        default="data/processed/knowledge_base.json"
    )
    parser.add_argument("--output", "-o",
        default="data/processed/entity_matches.json"
    )
    parser.add_argument("--threshold", "-t",
        type=float, default=SCORE_THRESH,
        help="Cosine similarity threshold (default: 0.78)"
    )
    args = parser.parse_args()

    # Auto-detect whitepaper JSON if a directory is given
    wp_path = Path(args.whitepaper)
    if wp_path.is_dir():
        candidates = list(wp_path.glob("*_parsed.json"))
        if not candidates:
            raise SystemExit(f"No *_parsed.json files found in {wp_path}")
        wp_path = sorted(candidates)[0]
        logger.info("Auto-detected whitepaper: %s", wp_path)

    matches = resolve_entities(
        whitepaper_path=wp_path,
        codebase_path=Path(args.codebase),
        patent_path=Path(args.patents),
        output_path=Path(args.output),
        threshold=args.threshold,
    )
    print_summary(matches)


if __name__ == "__main__":
    main()
