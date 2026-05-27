"""
Collects 800 GitHub repositories across diverse engineering domains and maturity levels.

Two-phase design:
  Phase 1 — gather search items (fast, main thread)
  Phase 2 — enrich each repo with per-repo signals (ThreadPoolExecutor, 5 workers)

commit_activity endpoint is intentionally skipped; use pushed_at recency as the
activity proxy instead. Contributor count is optional with a 5-second timeout.
"""

import base64
import csv
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env")

BASE_URL = "https://api.github.com"
OUTPUT_PATH = Path("data/raw/repositories.csv")
TARGET_COUNT = 800
WORKERS = 5  # parallel enrichment threads

CSV_FIELDS = [
    "name", "full_name", "description", "stars", "forks", "watchers",
    "open_issues_count", "size", "language", "topics", "created_at",
    "updated_at", "pushed_at", "has_wiki", "has_projects", "has_downloads",
    "license", "default_branch", "contributors_count", "commits_last_year",
    "has_ci", "readme_length", "releases_count", "open_prs_count", "closed_prs_count",
]

# Diverse queries: 8 engineering domains × multiple maturity levels (star ranges)
SEARCH_QUERIES = [
    # Web — JavaScript / TypeScript
    "topic:react stars:50..500 language:javascript",
    "topic:vue stars:30..300",
    "topic:nextjs stars:30..400",
    "topic:angular stars:30..300 language:typescript",
    "topic:express language:javascript stars:10..200",
    # Web — backend frameworks
    "topic:django language:python stars:10..300",
    "topic:fastapi language:python stars:10..300",
    "topic:flask language:python stars:10..200",
    "topic:spring language:java stars:10..200",
    "topic:rails language:ruby stars:10..200",
    # Data science / ML / AI
    "topic:machine-learning language:python stars:20..400",
    "topic:deep-learning language:python stars:10..300",
    "topic:nlp language:python stars:10..200",
    "topic:computer-vision language:python stars:10..200",
    "topic:data-science language:python stars:5..150",
    # DevOps / Infrastructure
    "topic:docker stars:30..400",
    "topic:kubernetes stars:30..400",
    "topic:terraform stars:10..300",
    "topic:ansible language:python stars:10..200",
    "topic:ci-cd stars:5..200",
    # Mobile
    "topic:android language:java stars:20..300",
    "topic:android language:kotlin stars:20..300",
    "topic:ios language:swift stars:10..200",
    "topic:flutter stars:20..300",
    "topic:react-native stars:10..200",
    # Systems / CLI / Low-level
    "topic:cli language:go stars:5..200",
    "language:rust topic:library stars:10..300",
    "language:c topic:embedded stars:5..150",
    "language:cpp topic:game stars:10..200",
    "language:go topic:microservice stars:5..150",
    # Security / Cryptography
    "topic:security language:python stars:10..200",
    "topic:cryptography stars:5..150",
    "topic:penetration-testing stars:5..150",
    # Low-maturity — beginner / early-stage (few stars, recently created)
    "language:python stars:1..10 created:>2023-01-01 size:>5",
    "language:javascript stars:1..10 created:>2023-01-01 size:>5",
    "language:java stars:1..10 created:>2022-01-01 size:>5",
    "language:rust stars:1..8 created:>2022-01-01 size:>5",
    "language:go stars:1..8 created:>2022-01-01 size:>5",
    "language:typescript stars:1..10 created:>2023-01-01 size:>5",
]


# ---------------------------------------------------------------------------
# HTTP layer — thread-local sessions + rate limit handling
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _session() -> requests.Session:
    """One persistent Session per thread (connection pool reuse)."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        })
        _thread_local.session = s
    return _thread_local.session


def api_get(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = 10,
    retries: int = 3,
) -> Optional[requests.Response]:
    """
    Thread-safe GET. Sleeps only when rate-limited (403/429); returns None on
    timeout or permanent failure. Never sleeps on 202 — callers skip that signal.
    """
    for attempt in range(retries):
        try:
            resp = _session().get(url, params=params, timeout=timeout)
        except requests.Timeout:
            return None  # caller treats missing signal as None/0
        except requests.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(1 << attempt)  # 1 s, 2 s back-off
                continue
            logger.debug("Request failed: %s", exc)
            return None

        remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        reset_at = int(resp.headers.get("X-RateLimit-Reset", int(time.time()) + 60))

        if resp.status_code == 429 or (resp.status_code == 403 and remaining == 0):
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else max(reset_at - time.time() + 3, 5)
            logger.warning(
                "[%s] rate-limited — sleeping %.0fs (remaining=%d)",
                threading.current_thread().name, wait, remaining,
            )
            time.sleep(wait)
            continue

        # 202 = stats still computing; skip rather than burn time retrying
        if resp.status_code == 202:
            return None

        return resp

    return None


# ---------------------------------------------------------------------------
# Link-header helper
# ---------------------------------------------------------------------------

def _last_page(link_header: str) -> int:
    match = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header or "")
    return int(match.group(1)) if match else 1


# ---------------------------------------------------------------------------
# Per-repo signal helpers (each runs inside a worker thread)
# ---------------------------------------------------------------------------

def _contributor_count(owner: str, repo: str) -> Optional[int]:
    """Uses the Link-header trick. Short timeout so large repos don't block workers."""
    resp = api_get(
        f"{BASE_URL}/repos/{owner}/{repo}/contributors",
        {"per_page": 1, "anon": "true"},
        timeout=5,
        retries=1,
    )
    if resp is None or resp.status_code != 200:
        return None
    last = _last_page(resp.headers.get("Link", ""))
    return last if last > 1 else len(resp.json())


def _paginated_count(url: str, extra: Optional[Dict] = None) -> int:
    """Count items at paginated endpoint via Link-header last-page trick."""
    resp = api_get(url, {"per_page": 1, **(extra or {})})
    if resp is None or resp.status_code != 200:
        return 0
    last = _last_page(resp.headers.get("Link", ""))
    return last if last > 1 else len(resp.json())


def _readme_length(owner: str, repo: str) -> int:
    resp = api_get(f"{BASE_URL}/repos/{owner}/{repo}/readme")
    if resp is None or resp.status_code != 200:
        return 0
    try:
        content = resp.json().get("content", "")
        return len(base64.b64decode(content.replace("\n", "")).decode("utf-8", errors="ignore"))
    except Exception:
        return 0


def _has_ci(owner: str, repo: str) -> bool:
    resp = api_get(f"{BASE_URL}/repos/{owner}/{repo}/contents/.github/workflows")
    return resp is not None and resp.status_code == 200


# ---------------------------------------------------------------------------
# Row builder — called inside each worker thread
# ---------------------------------------------------------------------------

def build_row(item: dict) -> dict:
    full_name = item["full_name"]
    owner, repo = full_name.split("/", 1)
    base = f"{BASE_URL}/repos/{full_name}"
    license_info = item.get("license")

    return {
        "name": item["name"],
        "full_name": full_name,
        "description": (item.get("description") or "").replace("\n", " "),
        "stars": item["stargazers_count"],
        "forks": item["forks_count"],
        "watchers": item["watchers_count"],
        "open_issues_count": item["open_issues_count"],
        "size": item["size"],
        "language": item.get("language"),
        "topics": ";".join(item.get("topics") or []),
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
        "pushed_at": item["pushed_at"],
        "has_wiki": item.get("has_wiki", False),
        "has_projects": item.get("has_projects", False),
        "has_downloads": item.get("has_downloads", False),
        "license": license_info.get("name") if license_info else None,
        "default_branch": item.get("default_branch", "main"),
        "contributors_count": _contributor_count(owner, repo),
        "commits_last_year": None,  # skipped; pushed_at is the activity proxy
        "has_ci": _has_ci(owner, repo),
        "readme_length": _readme_length(owner, repo),
        "releases_count": _paginated_count(f"{base}/releases"),
        "open_prs_count": _paginated_count(f"{base}/pulls", {"state": "open"}),
        "closed_prs_count": _paginated_count(f"{base}/pulls", {"state": "closed"}),
    }


# ---------------------------------------------------------------------------
# Phase 1 — gather search items (main thread only)
# ---------------------------------------------------------------------------

def _gather_search_items() -> List[dict]:
    seen: set = set()
    items: List[dict] = []

    search_session = requests.Session()
    search_session.headers.update({
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    })

    for query in SEARCH_QUERIES:
        if len(items) >= TARGET_COUNT:
            break

        logger.info("Searching: %r  (have %d)", query, len(items))
        page = 1

        while len(items) < TARGET_COUNT:
            try:
                resp = search_session.get(
                    f"{BASE_URL}/search/repositories",
                    params={"q": query, "sort": "stars", "order": "desc",
                            "per_page": 100, "page": page},
                    timeout=15,
                )
            except requests.RequestException as exc:
                logger.warning("Search request failed: %s", exc)
                break

            remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
            reset_at = int(resp.headers.get("X-RateLimit-Reset", int(time.time()) + 60))

            if resp.status_code == 429 or (resp.status_code == 403 and remaining == 0):
                wait = max(reset_at - time.time() + 3, 5)
                logger.warning("Search rate-limited. Sleeping %.0fs", wait)
                time.sleep(wait)
                continue  # retry same page

            if resp.status_code != 200:
                logger.warning("Search %d for %r", resp.status_code, query)
                break

            batch = resp.json().get("items", [])
            if not batch:
                break

            for repo_item in batch:
                if len(items) >= TARGET_COUNT:
                    break
                fn = repo_item["full_name"]
                if fn not in seen:
                    seen.add(fn)
                    items.append(repo_item)

            if len(batch) < 100:
                break  # no more pages for this query

            page += 1
            time.sleep(1)  # search API: 30 req/min authenticated

    logger.info("Search complete: %d unique repos", len(items))
    return items[:TARGET_COUNT]


# ---------------------------------------------------------------------------
# Phase 2 — parallel enrichment + incremental CSV write (main thread writes)
# ---------------------------------------------------------------------------

def _enrich_and_save(items: List[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    failed = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="enricher") as executor:
            futures = {executor.submit(build_row, item): item["full_name"] for item in items}

            for future in as_completed(futures):
                full_name = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    logger.warning("Failed to enrich %s: %s", full_name, exc)
                    failed += 1
                    continue

                writer.writerow(row)
                f.flush()
                collected += 1

                if collected % 50 == 0:
                    logger.info(
                        "Progress: %d/%d enriched (%d failed so far)",
                        collected, len(items), failed,
                    )

    logger.info("Saved %d repos to %s  (%d failed)", collected, OUTPUT_PATH, failed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_repos() -> None:
    t0 = time.time()
    items = _gather_search_items()
    logger.info("Phase 1 done in %.0fs. Starting parallel enrichment...", time.time() - t0)
    _enrich_and_save(items)
    logger.info("Total elapsed: %.0fs", time.time() - t0)


if __name__ == "__main__":
    collect_repos()
