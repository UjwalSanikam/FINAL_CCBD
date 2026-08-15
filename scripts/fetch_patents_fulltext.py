"""
Fetch full patent text for explicit patent numbers.

This is the small, direct counterpart to ``patent_downloader.py``: it does not
search by query, it only fetches IDs you provide and writes parser-compatible
``.txt`` files under ``data/raw/patents/``.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

try:
    from scripts.patent_downloader import (
        DELAY,
        enrich_via_patentsview,
        fetch_patent_fulltext,
        render_patent_text,
    )
except ModuleNotFoundError:
    from patent_downloader import (
        DELAY,
        enrich_via_patentsview,
        fetch_patent_fulltext,
        render_patent_text,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_filename(patent_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", patent_id.upper().replace(" ", ""))


def read_patent_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.ids:
        ids.extend(args.ids)
    if args.ids_file:
        for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)

    seen: set[str] = set()
    deduped: list[str] = []
    for patent_id in ids:
        cleaned = patent_id.strip()
        key = cleaned.upper()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def fetch_one(patent_id: str, output_dir: Path, overwrite: bool = False) -> bool:
    out_path = output_dir / f"{_safe_filename(patent_id)}.txt"
    if out_path.exists() and not overwrite and out_path.stat().st_size > 300:
        logger.info("Already exists: %s", out_path)
        return True

    logger.info("Fetching patent full text: %s", patent_id)
    data = fetch_patent_fulltext(patent_id)
    if not data:
        logger.warning("No data returned for %s", patent_id)
        return False

    data.update(enrich_via_patentsview(data.get("patent_id", patent_id)))
    rendered = render_patent_text(data)
    if len(rendered) < 300:
        logger.warning("Fetched content for %s is too thin (%d chars)", patent_id, len(rendered))
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("Saved %s (%d chars)", out_path, len(rendered))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Google Patents full text into patent_parser-compatible .txt files."
    )
    parser.add_argument("ids", nargs="*", help="Patent IDs, e.g. US11775945B2 CN112073382B")
    parser.add_argument("--ids-file", help="Text file with one patent ID per line")
    parser.add_argument("--output", "-o", default="data/raw/patents", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing patent .txt files")
    parser.add_argument("--delay", type=float, default=DELAY, help="Delay between requests")
    args = parser.parse_args()

    patent_ids = read_patent_ids(args)
    if not patent_ids:
        parser.error("provide at least one patent ID or --ids-file")

    output_dir = Path(args.output)
    success = 0
    for index, patent_id in enumerate(patent_ids):
        if fetch_one(patent_id, output_dir, overwrite=args.overwrite):
            success += 1
        if index < len(patent_ids) - 1:
            time.sleep(args.delay)

    print(f"Fetched {success}/{len(patent_ids)} patents into {output_dir}")


if __name__ == "__main__":
    main()
