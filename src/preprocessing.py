"""
Preprocessing pipeline for the GitHub repository dataset.

Steps:
  1. Load  data/raw/repositories.csv
  2. Parse ISO dates → repo_age_days, days_since_push, days_since_update
  3. Clean missing values and coerce types
  4. Engineer features:
       is_active            pushed within the last 90 days
       readme_category      short / medium / long bucketed on character count
       stars_log            log1p(stars) to reduce right skew
       forks_per_star       forks / (stars + 1)
       issues_per_contributor  open_issues / (contributors + 1)
  5. StandardScaler-normalise all continuous numeric columns
  6. Save data/processed/repositories_clean.csv
  7. Print dataset summary
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/repositories.csv")
CLEAN_PATH = Path("data/processed/repositories_clean.csv")

# Columns whose values come from the GitHub API as "True"/"False" strings
BOOL_COLS = ["has_wiki", "has_projects", "has_downloads", "has_ci"]

# Raw count/size columns from the collector
COUNT_COLS = [
    "stars", "forks", "watchers", "open_issues_count", "size",
    "contributors_count", "commits_last_year", "readme_length",
    "releases_count", "open_prs_count", "closed_prs_count",
]

DATE_COLS = ["created_at", "updated_at", "pushed_at"]

# Continuous columns to StandardScale (excludes binary flags and categoricals)
SCALE_COLS = [
    "stars", "forks", "watchers", "open_issues_count", "size",
    "contributors_count", "readme_length",
    "releases_count", "open_prs_count", "closed_prs_count",
    "repo_age_days", "days_since_push", "days_since_update",
    "stars_log", "forks_per_star", "issues_per_contributor",
]
# commits_last_year is intentionally omitted: the collector skipped that
# endpoint so every value is 0 (constant column → StandardScaler produces NaN).


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    if not RAW_PATH.exists():
        logger.error("Raw data not found at %s — run github_collector.py first.", RAW_PATH)
        sys.exit(1)
    df = pd.read_csv(RAW_PATH, dtype=str)  # read everything as str; we coerce below
    logger.info("Loaded %d repos, %d columns from %s", len(df), len(df.columns), RAW_PATH)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["repo_age_days"] = (now - df["created_at"]).dt.days.astype(float)
    df["days_since_push"] = (now - df["pushed_at"]).dt.days.astype(float)
    df["days_since_update"] = (now - df["updated_at"]).dt.days.astype(float)

    # Keep ISO strings in the output for traceability
    for col in DATE_COLS:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # --- Boolean columns (GitHub API serialises them as "True"/"False") ---
    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0,
                "1": 1, "0": 0, "": 0}
    for col in BOOL_COLS:
        df[col] = df[col].map(bool_map).fillna(0).astype(int)

    # --- Numeric columns: coerce, fill with 0 ---
    for col in COUNT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- Computed age columns: fill NaN with column median ---
    for col in ["repo_age_days", "days_since_push", "days_since_update"]:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # --- Categorical / text ---
    df["description"] = df["description"].fillna("").str.strip()
    df["language"] = df["language"].fillna("Unknown").replace("", "Unknown")
    df["license"] = df["license"].fillna("None").replace("", "None")
    df["topics"] = df["topics"].fillna("")

    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Activity flag — pushed in the last 90 days
    df["is_active"] = (df["days_since_push"] <= 90).astype(int)

    # README depth proxy
    bins = [0, 500, 2_000, float("inf")]
    labels = ["short", "medium", "long"]
    df["readme_category"] = (
        pd.cut(df["readme_length"], bins=bins, labels=labels, right=False)
        .astype(str)
        .replace("nan", "short")   # 0-length readme → short
    )

    # Log-transform stars to compress the long tail
    df["stars_log"] = np.log1p(df["stars"])

    # Normalised engagement ratios (+1 to avoid division by zero)
    df["forks_per_star"] = df["forks"] / (df["stars"] + 1)
    df["issues_per_contributor"] = df["open_issues_count"] / (df["contributors_count"] + 1)

    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    # Drop any constant columns from the scaling list to prevent NaN from std=0
    non_constant = [c for c in SCALE_COLS if df[c].std(ddof=0) > 0]
    skipped = set(SCALE_COLS) - set(non_constant)
    if skipped:
        logger.info("Skipping normalisation for constant columns: %s", sorted(skipped))

    scaler = StandardScaler()
    df[non_constant] = scaler.fit_transform(df[non_constant])

    return df


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  Repos:       {len(df):,}")
    print(f"  Columns:     {len(df.columns)}")

    print(f"\n  Active repos (pushed <=90d): "
          f"{df['is_active'].sum():>4}  ({df['is_active'].mean() * 100:.1f}%)")
    print(f"  Has CI (.github/workflows):  "
          f"{df['has_ci'].sum():>4}  ({df['has_ci'].mean() * 100:.1f}%)")
    print(f"  Has Wiki:                    "
          f"{df['has_wiki'].sum():>4}  ({df['has_wiki'].mean() * 100:.1f}%)")
    print(f"  Has Releases:                "
          f"{(df['releases_count'] > 0).sum():>4}")

    print("\n  README categories:")
    for cat, n in df["readme_category"].value_counts().items():
        print(f"    {cat:<8} {n:>4}  ({n / len(df) * 100:.1f}%)")

    print("\n  Top 10 languages:")
    for lang, n in df["language"].value_counts().head(10).items():
        print(f"    {lang:<20} {n:>4}")

    print("\n  Stars (raw) percentiles:")
    pcts = df["stars"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0])
    for q, v in pcts.items():
        print(f"    p{int(q*100):<3}  {v:>8.0f}")

    print("\n  Key engineered feature stats (pre-scaling):")
    stat_cols = [
        "stars_log", "forks_per_star", "issues_per_contributor",
        "repo_age_days", "days_since_push",
    ]
    print(df[stat_cols].describe().round(3).to_string(
        index=True,
        col_space=22,
    ))

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n  Remaining missing values:")
        print(missing.to_string())
    else:
        print("\n  No missing values remaining.")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess() -> pd.DataFrame:
    df = _load()
    df = _parse_dates(df)
    df = _clean(df)
    df = _engineer(df)

    # Print summary against interpretable pre-scale values, then normalise
    _print_summary(df)

    df = _normalise(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info("Saved cleaned data -> %s  (%d rows, %d cols)", CLEAN_PATH, *df.shape)

    return df


if __name__ == "__main__":
    preprocess()
