"""
crypto_auth.py — Synthetic VC Trap Fixture
===========================================
ChainCheck | tests/fixtures/

This file is intentionally designed to trigger contradiction_detector.py.
It represents a startup that claims "proprietary military-grade encryption"
but actually imports open-source cryptographic libraries.

Used by:
  tests/test_pipeline.py → TestContradictionDetector
  eval/ → ground truth entry GT-001

DO NOT put this in src/ — it is a TEST FIXTURE, not production code.

In a real VC due diligence scenario, the contradiction_detector would find
that the startup's whitepaper says "in-house crypto, no open-source auth"
but the codebase imports the modules below.
"""

# Intentional contradiction: startup claims "proprietary encryption"
# but imports standard open-source cryptographic libraries
import hashlib
import hmac

# This import triggers the crypto contradiction rule
# in contradiction_detector.py's CONTRADICTION_TAXONOMY
from cryptography.hazmat.primitives import hashes  # noqa: F401 (synthetic import)

# This would also be flagged — a startup claiming "no third-party auth"
# but using a well-known open-source library
try:
    import copyleft_crypto_engine  # noqa: F401 — this is the primary trap import
except ImportError:
    # The import is intentionally unresolvable in real environments.
    # Its presence in the codebase is what contradiction_detector flags.
    pass


class ProprietaryCryptoEngine:
    """
    Fake class that a startup might claim is their proprietary implementation.
    In reality it wraps hashlib (open-source) — exactly the type of contradiction
    ChainCheck is designed to surface.
    """

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm

    def hash(self, data: bytes) -> bytes:
        # Claims to be proprietary — actually just stdlib hashlib
        return hashlib.new(self.algorithm, data).digest()

    def verify(self, data: bytes, expected: bytes) -> bool:
        return hmac.compare_digest(self.hash(data), expected)
