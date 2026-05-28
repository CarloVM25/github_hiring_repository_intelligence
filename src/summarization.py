"""
Summarization pipeline: converts raw repo features into natural language text
suitable for LLM-based weak supervision labeling.

Steps:
  1. Load data/processed/repositories_clean.csv  (has readme_category, has_ci,
     category/tier, and ISO date strings)
  2. Load data/raw/repositories.csv              (has unscaled counts: stars,
     forks, contributors_count, open_issues_count, releases_count)
  3. Merge on full_name; compute repo_age_days and days_since_push from ISO dates
  4. Build text_summary for each repo using a fixed natural-language template
  5. Add search_category column (mirrors category)
  6. Save to data/processed/repositories_with_summaries.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEAN_PATH = Path("data/processed/repositories_clean.csv")
RAW_PATH = Path("data/raw/repositories.csv")
OUT_PATH = Path("data/processed/repositories_with_summaries.csv")

# Columns pulled from the clean CSV (categorical / already-processed)
CLEAN_COLS = [
    "full_name", "name", "language", "description",
    "category", "tier", "readme_category", "has_ci",
    "topics", "created_at", "pushed_at", "is_active",
]

# Columns pulled from the raw CSV (unscaled counts)
RAW_COLS = [
    "full_name",
    "stars", "forks", "contributors_count",
    "open_issues_count", "releases_count",
]


def _load() -> pd.DataFrame:
    for path in (CLEAN_PATH, RAW_PATH):
        if not path.exists():
            logger.error("Required file not found: %s", path)
            sys.exit(1)

    clean = pd.read_csv(CLEAN_PATH, usecols=CLEAN_COLS, dtype=str)
    raw = pd.read_csv(RAW_PATH, usecols=RAW_COLS, dtype=str)

    df = clean.merge(raw, on="full_name", how="left")
    logger.info("Merged %d repos (%d clean, %d raw)", len(df), len(clean), len(raw))
    return df


def _compute_dates(df: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    for col in ("created_at", "pushed_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["repo_age_days"] = (now - df["created_at"]).dt.days.fillna(0).astype(int)
    df["days_since_push"] = (now - df["pushed_at"]).dt.days.fillna(0).astype(int)
    return df


def _coerce_counts(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = [
        "stars", "forks", "contributors_count",
        "open_issues_count", "releases_count",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _build_summary(row: pd.Series) -> str:
    name = row["name"] or row["full_name"]
    language = row["language"] if row["language"] not in ("", "Unknown", None) else "unknown language"
    stars = int(row["stars"])
    forks = int(row["forks"])
    contributors = int(row["contributors_count"])
    age = int(row["repo_age_days"])
    since_push = int(row["days_since_push"])
    ci_phrase = "has" if str(row["has_ci"]) in ("1", "1.0", "True", "true") else "does not have"
    readme = str(row["readme_category"]) if row["readme_category"] not in ("", "nan", None) else "short"
    releases = int(row["releases_count"])
    issues = int(row["open_issues_count"])

    raw_topics = str(row["topics"]) if row["topics"] not in ("", "nan", None) else ""
    topics = raw_topics.replace("|", ", ").strip(", ") if raw_topics else "none"

    return (
        f"Repository {name} is a {language} project with {stars:,} stars and {forks:,} forks. "
        f"It has {contributors} contributors and was created {age} days ago. "
        f"Last activity was {since_push} days ago. "
        f"It {ci_phrase} CI/CD workflows. "
        f"The README is {readme}. "
        f"It has {releases} releases and {issues} open issues. "
        f"Topics: {topics}."
    )


def summarize() -> pd.DataFrame:
    df = _load()
    df = _compute_dates(df)
    df = _coerce_counts(df)

    logger.info("Generating text summaries...")
    df["text_summary"] = df.apply(_build_summary, axis=1)
    df["search_category"] = df["category"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info("Saved -> %s  (%d rows, %d cols)", OUT_PATH, *df.shape)

    # Spot-check
    sample = df[["full_name", "text_summary"]].head(3)
    for _, row in sample.iterrows():
        print(f"\n[{row['full_name']}]\n{row['text_summary']}")

    return df


if __name__ == "__main__":
    summarize()
