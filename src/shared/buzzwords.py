"""
buzzwords.py — Marketing-to-Engineering Vocabulary Pre-Filter
=============================================================
ChainCheck | Multi-Hop Reasoning Pipeline

IMPORTANT: This is a PRE-FILTER, not the primary vocabulary-mismatch
solver. The real mathematical bridging happens in knowledge_fusion.py
via FAISS cosine similarity on sentence-transformer embeddings.

This map normalizes common marketing buzzwords before embedding so
that phrases like "military-grade encryption" are replaced with
"AES-256 / FIPS-140" before the sentence-transformer encodes them.
This boosts similarity scores for the FAISS bridge layers.

Usage
-----
  from shared.buzzwords import normalize_buzzwords

  text = normalize_buzzwords("Our military-grade hyper-fast data-mesh")
  # → "Our AES-256 / FIPS-140 high-throughput distributed-data-architecture"
"""

import re

BUZZWORD_MAP: dict[str, str] = {
    # Performance marketing
    r"hyper[\s-]?fast":                  "high-throughput",
    r"lightning[\s-]?fast":              "sub-second-latency",
    r"real[\s-]?time":                   "low-latency (<100ms)",
    r"infinite scal":                    "horizontal-scalability",
    r"planet[\s-]?scale":                "globally-distributed",

    # Architecture claims
    r"data[\s-]?mesh":                   "distributed-data-architecture",
    r"web\s?3":                          "decentralized-web",
    r"blockchain[\s-]?enabled":          "distributed-ledger",
    r"layer[\s-]?2":                     "L2-scaling-protocol",
    r"smart[\s-]?contract":              "self-executing-contract-code",
    r"sharding":                         "horizontal-database-partitioning",
    r"merkle":                           "Merkle-tree-hash-structure",
    r"on[\s-]?chain":                    "stored-in-ledger",
    r"off[\s-]?chain":                   "off-ledger-computation",

    # Security / crypto claims
    r"military[\s-]?grade":              "AES-256 / FIPS-140",
    r"zero[\s-]?knowledge":              "ZK-proof",
    r"quantum[\s-]?resistant":           "post-quantum-cryptography",
    r"trustless":                        "cryptographically-verified",
    r"zero[\s-]?trust":                  "mutual-TLS-and-identity-verification",

    # Consensus
    r"consensus (mechanism|algorithm|protocol)": "distributed-agreement-protocol",
    r"proof[\s-]of[\s-]work":            "PoW",
    r"proof[\s-]of[\s-]stake":           "PoS",
    r"byzantine":                        "BFT-tolerant",

    # Integration / UX claims
    r"seamless(ly)?":                    "transparent-integration",
    r"plug[\s-]?and[\s-]?play":          "drop-in-deployment",
    r"enterprise[\s-]?grade":            "production-hardened",

    # Marketing fluff
    r"ai[\s-]?powered":                  "ML-augmented",
    r"paradigm[\s-]?shift":              "architectural-change",
    r"disruptive":                       "novel-approach",
    r"revolutionary":                    "novel-approach",
    r"next[\s-]?gen(eration)?":          "next-iteration",
    r"state[\s-]?of[\s-]?the[\s-]?art": "current-best-practice",
    r"world'?s first":                   "novel-unverified-claim",
    r"first[\s-]?of[\s-]?its[\s-]?kind": "novel-unverified-claim",
    r"home[\s-]?grown":                  "in-house-developed",
    r"in[\s-]?house":                    "internally-developed",
    r"secret sauce":                     "proprietary-algorithm",
    r"unique algorithm":                 "proprietary-algorithm",
    r"breakthrough":                     "significant-advance-unverified",
}

__all__ = ["BUZZWORD_MAP", "normalize_buzzwords"]


def normalize_buzzwords(text: str) -> str:
    """
    Replace marketing buzzwords with their engineering equivalents.
    Apply before embedding for improved FAISS cosine similarity matching.
    """
    for pattern, replacement in BUZZWORD_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
