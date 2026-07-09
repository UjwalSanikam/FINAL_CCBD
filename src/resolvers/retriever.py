"""
retriever.py — Semantic Retriever
===================================
Multi-Hop Reasoning System for Venture Capital Technical Due Diligence

Wraps the FAISS vector index built by faiss_indexer.py and exposes a clean
`Retriever.retrieve(query, top_k)` interface used by llm_mhqg.py.

Fixes applied vs v1:
  1. Path mismatch corrected: v1 referenced patent_faiss.index and
     faiss_metadata.json but faiss_indexer.py writes vector.index and
     vector_metadata.json.  All paths now match faiss_indexer.py output.
  2. Refactored from a top-level script into a Retriever class so it can be
     cleanly imported by llm_mhqg.py — eliminating the duplicate inline
     retrieval logic that existed in that file.
  3. Added similarity score to results so callers can threshold on quality.
  4. Added graceful error handling: clear message if index or metadata missing.
  5. Standalone __main__ block retained for quick CLI testing.

Dependencies:
    pip install faiss-cpu sentence-transformers numpy
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic retriever backed by a FAISS flat-L2 index.

    Parameters
    ----------
    index_path : Path
        Path to the FAISS index file written by faiss_indexer.py
        (default: data/processed/vector.index).
    metadata_path : Path
        Path to the JSON metadata sidecar written by faiss_indexer.py
        (default: data/processed/vector_metadata.json).
    model_name : str
        Sentence-Transformers model name.  Must match the model used
        during indexing (default: all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        index_path: Path | str = "data/processed/vector.index",
        metadata_path: Path | str = "data/processed/vector_metadata.json",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                "Run faiss_indexer.py first to build the index."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                "Run faiss_indexer.py first to build the index."
            )

        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        logger.info("Loading FAISS index from: %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        logger.info("Loading metadata from: %s", self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Support both int and string keys (JSON always stores as string)
        self.metadata: dict[int, dict] = {int(k): v for k, v in raw.items()}

        logger.info(
            "Retriever ready — %d vectors, %d metadata records",
            self.index.ntotal,
            len(self.metadata),
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top-k most similar documents to `query`.

        Parameters
        ----------
        query : str
            Free-text query (a startup claim, a patent concept, etc.).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict is the original metadata record plus:
              - "similarity_score" : float  (lower = closer in L2 space)
              - "rank"             : int    (1-based)
        """
        if not query.strip():
            return []

        embedding = self.model.encode([query])
        embedding = np.array(embedding, dtype="float32")

        distances, indices = self.index.search(embedding, top_k)

        results: list[dict] = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1:           # FAISS returns -1 for empty slots
                continue
            record = self.metadata.get(int(idx), {}).copy()
            record["similarity_score"] = float(dist)
            record["rank"] = rank
            results.append(record)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Test the FAISS semantic retriever")
    ap.add_argument("--query", "-q", default="military grade cryptographic hashing",
                    help="Query string")
    ap.add_argument("--top-k", "-k", type=int, default=5,
                    help="Number of results")
    ap.add_argument("--index", default="data/processed/vector.index")
    ap.add_argument("--metadata", default="data/processed/vector_metadata.json")
    args = ap.parse_args()

    retriever = Retriever(
        index_path=args.index,
        metadata_path=args.metadata,
    )

    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nTop {len(results)} results for: '{args.query}'")
    print("─" * 60)
    for r in results:
        print(f"  [{r['rank']}] source={r.get('source')}  score={r['similarity_score']:.4f}")
        print(f"       text={r.get('text', '')[:120]}")
    print("─" * 60)

