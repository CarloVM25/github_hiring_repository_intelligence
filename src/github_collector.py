"""

Diverse GitHub repository collector.

8 categories × 4 maturity tiers × 25 repos = 800 total.

Maturity tiers (star count as proxy for project reach):
  intern     1 –    50 stars   learning / beginner project
  junior    50 –   500 stars   active personal or small-team project
  senior   500 –  5000 stars   adopted community library or tool
  lead        >  5000 stars    ecosystem-defining, widely-used project

Resume-safe design:
  - Loads data/raw/repositories.csv on startup; skips already-enriched repos.
  - Appends new rows rather than overwriting, so interruptions lose at most one
    in-flight row.
  - Checkpoints (flush + log) every CHECKPOINT_EVERY completed rows.
  - Global rate-limit state shared across threads prevents redundant hammering.
  - CI detection inferred from topics first; API call only when topics are silent.
  - Contributor count is optional (5 s timeout, 1 retry); stored as None on failure.

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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

TIER_SIZE = 25          # repos per (category, tier) slot → 8 × 4 × 25 = 800
WORKERS = 5             # parallel enrichment threads
CHECKPOINT_EVERY = 50   # log progress and confirm flush every N rows

TIERS = ["intern", "junior", "senior", "lead"]

TIER_SORT: Dict[str, Tuple[str, str]] = {
    "intern": ("updated", "desc"),  # recently-active beginners
    "junior": ("updated", "desc"),
    "senior": ("stars",   "desc"),  # most-adopted community libs
    "lead":   ("stars",   "desc"),
}

CSV_FIELDS = [
    "category", "tier",

TARGET_COUNT = 800
WORKERS = 5  # parallel enrichment threads

CSV_FIELDS = [

    "name", "full_name", "description", "stars", "forks", "watchers",
    "open_issues_count", "size", "language", "topics", "created_at",
    "updated_at", "pushed_at", "has_wiki", "has_projects", "has_downloads",
    "license", "default_branch", "contributors_count", "commits_last_year",
    "has_ci", "readme_length", "releases_count", "open_prs_count", "closed_prs_count",
]


# Topics that conclusively indicate CI is present — avoids one API call per repo
_CI_TOPICS = frozenset({
    "github-actions", "ci-cd", "travis", "travis-ci",
    "circleci", "actions", "jenkins", "gitlab-ci",
})

PLAN: Dict[str, Dict[str, List[str]]] = {

    # ── Python Data Science ──────────────────────────────────────────────────
    "python_datascience": {
        "intern": [
            "language:python topic:data-science stars:1..50",
            "language:python topic:pandas stars:1..50",
            "language:python topic:jupyter stars:1..50",
            "language:python topic:matplotlib stars:1..50",
            "language:python topic:visualization stars:1..50",
        ],
        "junior": [
            "language:python topic:data-science stars:50..500",
            "language:python topic:pandas stars:50..500",
            "language:python topic:jupyter stars:50..500",
        ],
        "senior": [
            "language:python topic:data-science stars:500..5000",
            "language:python topic:pandas stars:500..5000",
            "language:python topic:numpy stars:500..5000",
        ],
        "lead": [
            "language:python topic:data-science stars:>5000",
            "language:python topic:pandas stars:>5000",
            "language:python topic:numpy stars:>5000",
            "language:python topic:scipy stars:>5000",
        ],
    },

    # ── JavaScript Web ───────────────────────────────────────────────────────
    "javascript_web": {
        "intern": [
            "language:javascript topic:react stars:1..50",
            "language:javascript topic:frontend stars:1..50",
            "language:javascript topic:web stars:1..50",
            "language:javascript topic:vue stars:1..50",
        ],
        "junior": [
            "language:javascript topic:react stars:50..500",
            "language:javascript topic:vue stars:50..500",
            "language:javascript topic:frontend stars:50..500",
        ],
        "senior": [
            "language:javascript topic:react stars:500..5000",
            "language:javascript topic:vue stars:500..5000",
            "language:javascript topic:nextjs stars:500..5000",
        ],
        "lead": [
            "language:javascript topic:react stars:>5000",
            "language:javascript topic:framework stars:>5000",
            "language:javascript topic:nodejs stars:>5000",
            "language:javascript stars:>5000",
        ],
    },

    # ── Rust Systems ─────────────────────────────────────────────────────────
    "rust_systems": {
        "intern": [
            "language:rust stars:1..50 size:>10",
            "language:rust topic:cli stars:1..50",
            "language:rust topic:library stars:1..50",
            "language:rust topic:tool stars:1..50",
        ],
        "junior": [
            "language:rust stars:50..500",
            "language:rust topic:cli stars:50..500",
            "language:rust topic:systems stars:50..500",
        ],
        "senior": [
            "language:rust stars:500..5000",
            "language:rust topic:async stars:500..5000",
            "language:rust topic:networking stars:500..5000",
        ],
        "lead": [
            "language:rust stars:>5000",
        ],
    },

    # ── Go Backend ───────────────────────────────────────────────────────────
    "go_backend": {
        "intern": [
            "language:go topic:api stars:1..50",
            "language:go topic:backend stars:1..50",
            "language:go topic:cli stars:1..50",
            "language:go stars:1..50 size:>10",
        ],
        "junior": [
            "language:go topic:api stars:50..500",
            "language:go topic:microservice stars:50..500",
            "language:go topic:backend stars:50..500",
            "language:go stars:50..500",
        ],
        "senior": [
            "language:go topic:api stars:500..5000",
            "language:go topic:backend stars:500..5000",
            "language:go stars:500..5000",
        ],
        "lead": [
            "language:go stars:>5000",
        ],
    },

    # ── Java Enterprise ──────────────────────────────────────────────────────
    "java_enterprise": {
        "intern": [
            "language:java topic:spring stars:1..50",
            "language:java topic:springboot stars:1..50",
            "language:java topic:backend stars:1..50",
            "language:java stars:1..50 size:>10",
        ],
        "junior": [
            "language:java topic:spring stars:50..500",
            "language:java topic:springboot stars:50..500",
            "language:java topic:backend stars:50..500",
            "language:java stars:50..500",
        ],
        "senior": [
            "language:java topic:spring stars:500..5000",
            "language:java topic:springboot stars:500..5000",
            "language:java stars:500..5000",
        ],
        "lead": [
            "language:java stars:>5000",
            "language:java topic:spring stars:>5000",
        ],
    },

    # ── Python ML / AI ───────────────────────────────────────────────────────
    "python_ml_ai": {
        "intern": [
            "language:python topic:machine-learning stars:1..50",
            "language:python topic:deep-learning stars:1..50",
            "language:python topic:nlp stars:1..50",
            "language:python topic:neural-network stars:1..50",
        ],
        "junior": [
            "language:python topic:machine-learning stars:50..500",
            "language:python topic:deep-learning stars:50..500",
            "language:python topic:nlp stars:50..500",
        ],
        "senior": [
            "language:python topic:machine-learning stars:500..5000",
            "language:python topic:deep-learning stars:500..5000",
            "language:python topic:computer-vision stars:500..5000",
        ],
        "lead": [
            "language:python topic:machine-learning stars:>5000",
            "language:python topic:deep-learning stars:>5000",
            "language:python topic:pytorch stars:>5000",
            "language:python topic:tensorflow stars:>5000",
        ],
    },

    # ── DevOps / Infrastructure ──────────────────────────────────────────────
    "devops_infra": {
        "intern": [
            "topic:docker stars:1..50",
            "topic:devops stars:1..50",
            "topic:ansible stars:1..50",
            "topic:ci-cd stars:1..50",
        ],
        "junior": [
            "topic:docker stars:50..500",
            "topic:kubernetes stars:50..500",
            "topic:devops stars:50..500",
            "topic:ansible stars:50..500",
        ],
        "senior": [
            "topic:docker stars:500..5000",
            "topic:kubernetes stars:500..5000",
            "topic:terraform stars:500..5000",
        ],
        "lead": [
            "topic:kubernetes stars:>5000",
            "topic:docker stars:>5000",
            "topic:terraform stars:>5000",
        ],
    },

    # ── Mobile (Android / iOS / Flutter) ────────────────────────────────────
    "mobile": {
        "intern": [
            "topic:android stars:1..50",
            "topic:flutter stars:1..50",
            "topic:ios stars:1..50",
            "topic:react-native stars:1..50",
        ],
        "junior": [
            "topic:android stars:50..500",
            "topic:flutter stars:50..500",
            "topic:ios stars:50..500",
        ],
        "senior": [
            "topic:android stars:500..5000",
            "topic:flutter stars:500..5000",
            "topic:ios stars:500..5000",
        ],
        "lead": [
            "topic:flutter stars:>5000",
            "topic:android stars:>5000",
            "topic:ios stars:>5000",
            "topic:react-native stars:>5000",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Global rate-limit state (shared across threads, intentionally racy — benign)
# ─────────────────────────────────────────────────────────────────────────────

_rl_reset_at: float = 0.0   # Unix timestamp when the rate limit resets


def _rl_wait() -> None:
    """Block the calling thread until the rate limit window has passed."""
    wait = _rl_reset_at - time.time()
    if wait > 0:
        logger.info("[%s] pre-flight RL wait %.0fs", threading.current_thread().name, wait)
        time.sleep(wait + 1)


def _rl_mark(reset_at: float) -> None:
    global _rl_reset_at
    if reset_at > _rl_reset_at:
        _rl_reset_at = reset_at


# ─────────────────────────────────────────────────────────────────────────────
# HTTP layer — thread-local sessions
# ─────────────────────────────────────────────────────────────────────────────

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

    Thread-safe GET with rate-limit coordination.
    Checks the shared _rl_reset_at before each attempt so threads that are
    not yet rate-limited still back off when another thread has been told to wait.
    """
    for attempt in range(retries):
        _rl_wait()

        try:
            resp = _session().get(url, params=params, timeout=timeout)
        except requests.Timeout:
            return None
        except requests.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(1 << attempt)
                continue
            logger.debug("Request error: %s", exc)

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

                "[%s] rate-limited — marking reset at +%.0fs (remaining=%d)",
                threading.current_thread().name, wait, remaining,
            )
            _rl_mark(time.time() + wait)
            time.sleep(wait)
            continue

        if resp.status_code == 202:
            return None  # stats computing — skip

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



# ─────────────────────────────────────────────────────────────────────────────
# Resume: load existing CSV to discover already-enriched repos
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing() -> Tuple[Set[str], Dict]:
    """
    Read the existing CSV (if any) and return:
      already_enriched  — set of full_names already written
      slot_counts       — {(category, tier): count} from existing rows

    Malformed rows (full_name != 'owner/repo') are silently skipped.
    """
    if not OUTPUT_PATH.exists():
        return set(), defaultdict(int)

    already_enriched: Set[str] = set()
    slot_counts: Dict = defaultdict(int)

    try:
        with open(OUTPUT_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fn = row.get("full_name", "")
                if fn.count("/") != 1:       # skip malformed rows
                    continue
                cat  = row.get("category", "")
                tier = row.get("tier", "")
                if not cat or not tier:
                    continue
                already_enriched.add(fn)
                slot_counts[(cat, tier)] += 1
    except Exception as exc:
        logger.warning("Could not read existing CSV (%s) — starting fresh", exc)
        return set(), defaultdict(int)

    logger.info(
        "Resume: found %d existing repos across %d slots",
        len(already_enriched), len(slot_counts),
    )
    for cat in PLAN:
        parts = "  ".join(
            f"{t}={slot_counts.get((cat, t), 0)}/{TIER_SIZE}" for t in TIERS
        )
        logger.info("  %-24s  %s", cat, parts)

    return already_enriched, slot_counts


# ─────────────────────────────────────────────────────────────────────────────
# Per-repo enrichment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _contributor_count(owner: str, repo: str) -> Optional[int]:
    """Optional: 5 s timeout, 1 retry. Returns None rather than blocking."""
    resp = api_get(
        f"{BASE_URL}/repos/{owner}/{repo}/contributors",
        {"per_page": 1, "anon": "true"},
        timeout=5, retries=1,

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



def _has_ci(owner: str, repo: str, topics: Set[str]) -> bool:
    """
    CI detection with two-tier cost:
      1. (free)  Check topics for explicit CI signals — saves one API call when present.
      2. (1 call) Fall back to checking .github/workflows via the Contents API.
    """
    if topics & _CI_TOPICS:
        return True

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

    topics_list = item.get("topics") or []
    topics_set = {t.lower() for t in topics_list}

    return {
        "category":    item["_category"],
        "tier":        item["_tier"],
        "name":        item["name"],
        "full_name":   full_name,
        "description": (item.get("description") or "").replace("\n", " "),
        "stars":       item["stargazers_count"],
        "forks":       item["forks_count"],
        "watchers":    item["watchers_count"],
        "open_issues_count": item["open_issues_count"],
        "size":        item["size"],
        "language":    item.get("language"),
        "topics":      ";".join(topics_list),
        "created_at":  item["created_at"],
        "updated_at":  item["updated_at"],
        "pushed_at":   item["pushed_at"],
        "has_wiki":       item.get("has_wiki", False),
        "has_projects":   item.get("has_projects", False),
        "has_downloads":  item.get("has_downloads", False),
        "license":        license_info.get("name") if license_info else None,
        "default_branch": item.get("default_branch", "main"),
        "contributors_count": _contributor_count(owner, repo),
        "commits_last_year":  None,          # endpoint skipped; use pushed_at
        "has_ci":         _has_ci(owner, repo, topics_set),
        "readme_length":  _readme_length(owner, repo),


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



# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: structured search
# ─────────────────────────────────────────────────────────────────────────────

def _search_one(session: requests.Session, query: str, sort: str, order: str) -> List[dict]:
    while True:
        _rl_wait()
        try:
            resp = session.get(
                f"{BASE_URL}/search/repositories",
                params={"q": query, "sort": sort, "order": order, "per_page": 100},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.warning("Search failed: %s", exc)
            return []

        remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        reset_at = int(resp.headers.get("X-RateLimit-Reset", int(time.time()) + 60))

        if resp.status_code == 429 or (resp.status_code == 403 and remaining == 0):
            wait = max(reset_at - time.time() + 3, 5)
            logger.warning("Search rate-limited. Sleeping %.0fs", wait)
            _rl_mark(time.time() + wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            logger.warning("Search HTTP %d: %r", resp.status_code, query)
            return []

        return resp.json().get("items", [])


def _gather_search_items(
    already_enriched: Set[str],
    slot_counts: Dict,
) -> List[dict]:
    """
    Only collects repos for slots that still need repos.
    already_enriched pre-seeds the global-dedup set so repos already in the
    CSV are never re-queued for enrichment.
    """
    seen: Set[str] = set(already_enriched)
    local_counts: Dict = defaultdict(int, slot_counts)
    new_items: List[dict] = []

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


    total_slots = len(PLAN) * len(TIERS)
    slot_n = 0

    for cat, tier_map in PLAN.items():
        for tier in TIERS:
            slot_n += 1
            slot_key = (cat, tier)

            if local_counts[slot_key] >= TIER_SIZE:
                logger.info("[%2d/%d] %-24s %-8s  COMPLETE — skipping", slot_n, total_slots, cat, tier)
                continue

            sort, order = TIER_SORT[tier]

            for query in tier_map[tier]:
                need = TIER_SIZE - local_counts[slot_key]
                if need <= 0:
                    break

                logger.info(
                    "[%2d/%d] %-24s %-8s  have=%d  need=%d  %r",
                    slot_n, total_slots, cat, tier,
                    local_counts[slot_key], need, query,
                )

                for item in _search_one(search_session, query, sort, order):
                    if local_counts[slot_key] >= TIER_SIZE:
                        break
                    fn = item["full_name"]
                    if fn in seen:
                        continue
                    seen.add(fn)
                    item["_category"] = cat
                    item["_tier"] = tier
                    new_items.append(item)
                    local_counts[slot_key] += 1

                time.sleep(2)   # stay well under 30 search req/min

            got = local_counts[slot_key]
            if got < TIER_SIZE:
                logger.warning("Slot %s/%s under-filled: %d/%d", cat, tier, got, TIER_SIZE)

    logger.info(
        "Search complete: %d new repos to enrich  (%d already done)",
        len(new_items), len(already_enriched),
    )
    return new_items


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: parallel enrichment + append-safe CSV write
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_and_save(items: List[dict]) -> None:
    if not items:
        logger.info("Nothing to enrich — dataset is already complete.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append if file exists with content; write fresh otherwise
    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0
    mode = "a" if file_exists else "w"
    write_header = not file_exists

    logger.info(
        "Opening %s in %r mode to write %d rows",
        OUTPUT_PATH, mode, len(items),
    )

    collected = 0
    failed = 0

    with open(OUTPUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

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

                    logger.warning("Failed %s: %s", full_name, exc)

                    logger.warning("Failed to enrich %s: %s", full_name, exc)

                    failed += 1
                    continue

                writer.writerow(row)

                collected += 1

                # Checkpoint: explicit flush + progress log every N rows
                if collected % CHECKPOINT_EVERY == 0:
                    f.flush()
                    logger.info(
                        "Checkpoint: %d/%d enriched  (%d failed)  — flushed to disk",
                        collected, len(items), failed,
                    )

        f.flush()   # final flush for the tail < CHECKPOINT_EVERY rows

    logger.info(
        "Done: %d new rows appended to %s  (%d failed)",
        collected, OUTPUT_PATH, failed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def collect_repos() -> None:
    t0 = time.time()

    already_enriched, slot_counts = _load_existing()

    new_items = _gather_search_items(already_enriched, slot_counts)

    if not new_items:
        logger.info("Dataset already complete (%d repos). Nothing to do.", len(already_enriched))
        return

    # Log what we're about to collect
    new_counts: Dict = defaultdict(lambda: defaultdict(int))
    for it in new_items:
        new_counts[it["_category"]][it["_tier"]] += 1

    header = f"{'category':<24}  " + "  ".join(f"{t:<8}" for t in TIERS) + "  new"
    logger.info("New repos to enrich:\n  %s", header)
    for cat in PLAN:
        row_str = "  ".join(f"{new_counts[cat][t]:<8}" for t in TIERS)
        total_new = sum(new_counts[cat][t] for t in TIERS)
        if total_new:
            logger.info("  %-24s  %s  %d", cat, row_str, total_new)

    logger.info("Starting parallel enrichment with %d workers...", WORKERS)
    _enrich_and_save(new_items)

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
