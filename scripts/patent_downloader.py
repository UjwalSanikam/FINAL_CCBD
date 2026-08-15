"""
patent_downloader.py — Free Patent Full-Text Fetcher (2026 rewrite)
====================================================================
Why this exists:
  Both api.patentsview.org (legacy PatentsView API) and
  developer.uspto.gov/ibd-api (legacy USPTO Developer Hub) are DEAD.
  - api.patentsview.org was shut down 2025-05-01. Its replacement,
    search.patentsview.org, requires a free API key (key issuance is
    currently paused as of this writing).
  - developer.uspto.gov/ibd-api was replaced by api.uspto.gov (Open
    Data Portal), which also requires a free API key (USPTO.gov
    account + ID.me verification).
  - Lens.org requires a paid/registered API key — a bare POST will
    always 401.

  This script instead pulls full text (title, abstract, description,
  claims) directly from Google Patents, which is still free and
  requires no key. It parses by `itemprop` attribute (abstract/
  description/claims), which has stayed stable across Google's CSS
  redesigns — unlike guessing class names, which breaks every time
  they restyle the page.

  If you DO have a PatentsView PatentSearch API key, set the
  PATENTSVIEW_API_KEY env var and this script will use it to enrich
  results with CPC codes / assignee / inventor metadata. It's optional.

Usage:
    python patent_downloader.py --query "distributed hash table" --count 5
    python patent_downloader.py --preset blockchain --count 5
    python patent_downloader.py --ids US10820200B2 US9749140B2
    python patent_downloader.py --query "zero knowledge proof" --output data/raw/patents/
"""

import os
import re
import time
import json
import logging
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DELAY = 2.0  # be polite — Google will rate-limit / block aggressive scraping
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PATENTSVIEW_API_KEY = os.environ.get("PATENTSVIEW_API_KEY", "")

PRESETS = {
    "blockchain":          "distributed ledger cryptographic hash consensus",
    "nlp":                 "natural language processing text classification transformer",
    "vector_db":           "approximate nearest neighbor vector embedding similarity search",
    "crypto":              "zero knowledge proof cryptographic protocol verification",
    "p2p":                 "peer to peer network distributed protocol routing",
    "distributed_systems": "distributed hash table fault tolerant replication",
    "ml_inference":        "neural network inference optimization hardware accelerator",
    "fintech":             "payment processing transaction settlement distributed ledger",
}

session = requests.Session()
session.headers.update(HEADERS)


# ── Search (Google Patents' own search endpoint — same one the UI uses) ─────

def search_google_patents(query: str, count: int = 5) -> list[dict]:
    """
    Hits the same XHR endpoint patents.google.com/ uses internally for search.
    No API key needed. Returns up to `count` results with publication numbers.
    """
    url = "https://patents.google.com/xhr/query"
    params = {"url": f"q={query}", "exp": "", "content": "1"}
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("Google Patents search failed: %s", e)
        return []

    results = []
    try:
        clusters = data.get("results", {}).get("cluster", [])
        for cluster in clusters:
            for item in cluster.get("result", []):
                p = item.get("patent", {})
                pub_num = p.get("publication_number", "")
                if not pub_num:
                    continue
                results.append({
                    "patent_id": pub_num,
                    "title":     p.get("title", "").strip(),
                    "snippet":   p.get("snippet", "").strip(),
                })
                if len(results) >= count:
                    return results
    except Exception as e:
        logger.warning("Could not parse Google Patents search response: %s", e)
    return results


# ── Full text fetch from a single Google Patents page ───────────────────────

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_patent_fulltext(patent_id: str) -> dict:
    """
    Fetch title/abstract/description/claims/inventors/assignee from a
    Google Patents page, keyed off `itemprop` attributes (stable across
    Google's repeated CSS/class renames).
    """
    pid = patent_id.upper().replace(" ", "")
    if not pid.startswith("US") and not re.match(r"^[A-Z]{2}", pid):
        pid = "US" + pid

    url = f"https://patents.google.com/patent/{pid}/en"
    try:
        r = session.get(url, timeout=25)
        r.raise_for_status()
        r.encoding = "utf-8"   # Google Patents always serves utf-8; requests'
                                # auto-detection mis-guesses on CJK content
    except Exception as e:
        logger.warning("Fetch failed for %s: %s", pid, e)
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    out = {"patent_id": pid, "url": url}

    title_tag = soup.find(attrs={"itemprop": "title"})
    if title_tag:
        out["title"] = _clean(title_tag.get_text())
    elif soup.title:
        out["title"] = _clean(soup.title.get_text().split(" - Google Patents")[0])

    abstract_tag = soup.find(attrs={"itemprop": "abstract"})
    if abstract_tag:
        out["abstract"] = _clean(abstract_tag.get_text(" "))

    description_tag = soup.find(attrs={"itemprop": "description"})
    if description_tag:
        desc = _clean(description_tag.get_text(" "))
        out["description"] = desc[:20000]  # cap — descriptions can be huge

    claims_tag = soup.find(attrs={"itemprop": "claims"})
    if claims_tag:
        # Individual claims are usually <div class="claim" num="...">
        claim_divs = claims_tag.find_all(attrs={"num": True}) or claims_tag.find_all("div")
        claims_list = []
        if claim_divs:
            for i, c in enumerate(claim_divs, 1):
                t = _clean(c.get_text(" "))
                if t and len(t) > 10:
                    claims_list.append(f"{i}. {t}")
        if not claims_list:
            # fall back to the whole claims block as one chunk
            t = _clean(claims_tag.get_text(" "))
            if t:
                claims_list = [t]
        out["claims"] = "\n".join(claims_list)[:20000]

    inventors = [_clean(d.get_text()) for d in soup.find_all(attrs={"itemprop": "inventor"})]
    if inventors:
        out["inventors"] = inventors

    assignees = [_clean(d.get_text()) for d in soup.find_all(attrs={"itemprop": "assigneeCurrent"})]
    if not assignees:
        assignees = [_clean(d.get_text()) for d in soup.find_all(attrs={"itemprop": "assigneeOriginal"})]
    if assignees:
        out["assignee"] = ", ".join(dict.fromkeys(assignees))  # de-dupe, keep order

    pub_date_tag = soup.find(attrs={"itemprop": "publicationDate"})
    if pub_date_tag:
        out["publication_date"] = _clean(pub_date_tag.get_text())

    return out


# ── Optional enrichment via PatentSearch API (new PatentsView, needs key) ──

def enrich_via_patentsview(patent_id: str) -> dict:
    """
    Optional. Only runs if PATENTSVIEW_API_KEY is set in the environment.
    Adds CPC classification codes on top of what Google Patents gave us.
    Request a free key at: https://search.patentsview.org (key issuance
    may currently be paused — this is best-effort, not required).
    """
    if not PATENTSVIEW_API_KEY:
        return {}
    num = re.sub(r"^[A-Z]{2}", "", patent_id)
    num = re.sub(r"[A-Z]\d*$", "", num)
    url = "https://search.patentsview.org/api/v1/patent/"
    headers = {"X-Api-Key": PATENTSVIEW_API_KEY}
    params = {
        "q": json.dumps({"patent_id": num}),
        "f": json.dumps(["patent_id", "patent_title", "cpc_current"]),
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        patents = data.get("patents", [])
        if not patents:
            return {}
        cpcs = patents[0].get("cpc_current", [])
        cpc_ids = list({c.get("cpc_group_id", "") for c in cpcs if c.get("cpc_group_id")})
        return {"cpc_codes": cpc_ids[:10]} if cpc_ids else {}
    except Exception as e:
        logger.warning("PatentsView enrichment skipped for %s: %s", patent_id, e)
        return {}


# ── Assemble + save ──────────────────────────────────────────────────────────

def render_patent_text(data: dict) -> str:
    sections = [f"PATENT ID: {data.get('patent_id','')}"]
    if data.get("title"):
        sections.append(f"TITLE\n{data['title']}")
    if data.get("publication_date"):
        sections.append(f"PUBLICATION DATE\n{data['publication_date']}")
    if data.get("assignee"):
        sections.append(f"ASSIGNEE\n{data['assignee']}")
    if data.get("inventors"):
        sections.append(f"INVENTORS\n{', '.join(data['inventors'][:8])}")
    if data.get("cpc_codes"):
        sections.append(f"CPC CLASSIFICATIONS\n{', '.join(data['cpc_codes'])}")
    if data.get("abstract"):
        sections.append(f"ABSTRACT\n{data['abstract']}")
    if data.get("claims"):
        sections.append(f"CLAIMS\n{data['claims']}")
    if data.get("description"):
        sections.append(f"DESCRIPTION\n{data['description']}")
    return "\n\n".join(sections)


def download_patent(patent_id: str, output_dir: Path) -> bool:
    safe_id = re.sub(r"[^\w\-]", "_", patent_id)
    out_path = output_dir / f"{safe_id}.txt"
    if out_path.exists() and out_path.stat().st_size > 1500:
        logger.info("Already have %s — skipping", out_path.name)
        return True

    logger.info("Fetching: %s", patent_id)
    data = fetch_patent_fulltext(patent_id)
    if not data or not (data.get("abstract") or data.get("description")):
        logger.warning("No usable content for %s — skipping", patent_id)
        return False

    data.update(enrich_via_patentsview(patent_id))
    text = render_patent_text(data)

    if len(text) < 300:
        logger.warning("Content too thin for %s (%d chars) — skipping", patent_id, len(text))
        return False

    out_path.write_text(text, encoding="utf-8")
    logger.info("Saved %s (%d chars)", out_path.name, len(text))
    return True


def search_and_download(query: str, count: int, output_dir: Path) -> int:
    results = search_google_patents(query, count)
    if not results:
        logger.warning("No results for '%s'. Try rephrasing, or use --ids directly.", query)
        return 0

    print(f"\n  Found {len(results)} patents for: '{query}'")
    for r in results:
        print(f"  [{r['patent_id']}] {r['title'][:65]}")
    print()

    success = 0
    for r in results:
        if download_patent(r["patent_id"], output_dir):
            success += 1
        time.sleep(DELAY)
    return success


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download patent full text — Google Patents based, no API key required")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", help='Search query, e.g. "distributed hash table"')
    group.add_argument("--preset", "-p", choices=list(PRESETS.keys()), help="Use a preset query")
    group.add_argument("--ids", "-i", nargs="+", help="Specific patent IDs, e.g. US10820200B2")

    parser.add_argument("--count", "-n", type=int, default=5)
    parser.add_argument("--output", "-o", default="data/raw/patents/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output: {output_dir.resolve()}")
    if PATENTSVIEW_API_KEY:
        print("  PatentsView enrichment: ON (API key found)\n")
    else:
        print("  PatentsView enrichment: OFF (no PATENTSVIEW_API_KEY set — optional)\n")

    if args.query:
        n = search_and_download(args.query, args.count, output_dir)
        print(f"\n  Downloaded {n} patents")
    elif args.preset:
        query = PRESETS[args.preset]
        print(f"  Preset '{args.preset}': {query}")
        n = search_and_download(query, args.count, output_dir)
        print(f"\n  Downloaded {n} patents")
    elif args.ids:
        success = 0
        for pid in args.ids:
            if download_patent(pid.strip(), output_dir):
                success += 1
            time.sleep(DELAY)
        print(f"\n  Downloaded {success}/{len(args.ids)} patents")

    files = list(output_dir.glob("*.txt"))
    print(f"\n  Patents in {output_dir}/ ({len(files)} files):")
    for f in sorted(files):
        kb = f.stat().st_size / 1024
        status = "good" if kb > 3 else "thin"
        print(f"     {f.name:<35} {kb:>6.1f} KB  [{status}]")
    print()


if __name__ == "__main__":
    main()