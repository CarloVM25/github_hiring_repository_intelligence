"""
LLM labeling pipeline.

Uses Groq API (llama-3.3-70b-versatile) to assign each repository one of six
maturity labels: intern, junior, senior, lead, template, low_value.

Steps:
  1. Load data/processed/repositories_with_summaries.csv
  2. Resume: if data/labeled/repositories_labeled.csv already exists, skip
     repos that have a valid label so the run can be restarted safely
  3. For each unlabeled repo, send text_summary to Groq; validate the response
     is one of the six labels; retry up to 3 times on API errors or bad output
  4. Checkpoint: flush to disk every BATCH_SIZE rows
  5. Save final result to data/labeled/repositories_labeled.csv
  6. Print label distribution
"""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUMMARIES_PATH = Path("data/processed/repositories_with_summaries.csv")
OUT_PATH = Path("data/labeled/repositories_labeled.csv")

MODEL = "llama-3.1-8b-instant"
VALID_LABELS = {"intern", "junior", "senior", "lead", "template", "low_value"}
BATCH_SIZE = 50
REQUEST_DELAY = 3.0   # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5.0     # seconds to wait after a failed call before retrying

SYSTEM_PROMPT = (
    "You are an expert engineering recruiter evaluating GitHub repositories. "
    "Classify the repository into exactly one category based on its signals. "
    "Categories: "
    "intern (simple personal project, 1-2 contributors, few stars, no CI, minimal releases), "
    "junior (small team project, some structure, limited CI, moderate activity), "
    "senior (well-structured project, active CI/CD, multiple contributors, regular releases, "
    "good documentation), "
    "lead (highly complex system, large contributor base, extensive CI/CD, many releases, "
    "industry-standard practices), "
    "template (boilerplate, starter kit, or example repository), "
    "low_value (abandoned, empty, or minimal effort repository). "
    "Respond with ONLY the category label, nothing else."
)


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def _make_client() -> Groq:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        logger.error("GROQ_API_KEY not found in environment / .env file.")
        sys.exit(1)
    return Groq(api_key=key)


def _call_groq(client: Groq, text_summary: str) -> str | None:
    """Return a validated label string, or None on unrecoverable failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text_summary},
                ],
                temperature=0,
                max_tokens=16,
            )
            raw = response.choices[0].message.content.strip().lower()
            # Strip punctuation the model occasionally appends
            label = raw.rstrip(".,;:!?")
            if label in VALID_LABELS:
                return label
            # Model returned something not in VALID_LABELS
            logger.warning("Attempt %d: unexpected label %r — retrying", attempt, raw)
        except Exception as exc:
            logger.warning("Attempt %d: API error: %s — retrying in %.0fs",
                           attempt, exc, RETRY_DELAY)
            time.sleep(RETRY_DELAY)
            continue

        time.sleep(RETRY_DELAY)

    logger.error("All %d attempts failed for summary: %.80s…", MAX_RETRIES, text_summary)
    return None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _load_existing_labels() -> dict[str, str]:
    """Return {full_name: label} for rows already successfully labeled."""
    if not OUT_PATH.exists():
        return {}
    try:
        existing = pd.read_csv(OUT_PATH, usecols=["full_name", "label"], dtype=str)
        valid_rows = existing[existing["label"].notna() & (existing["label"] != "")]
        result = dict(zip(valid_rows["full_name"], valid_rows["label"]))
        logger.info("Resume: found %d already-labeled repos", len(result))
        return result
    except Exception as exc:
        logger.warning("Could not read existing labels (%s) — starting fresh", exc)
        return {}


# ---------------------------------------------------------------------------
# Checkpoint write
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def label_repos() -> pd.DataFrame:
    if not SUMMARIES_PATH.exists():
        logger.error("Summaries file not found: %s — run summarization.py first.",
                     SUMMARIES_PATH)
        sys.exit(1)

    df = pd.read_csv(SUMMARIES_PATH, dtype=str)
    logger.info("Loaded %d repos from %s", len(df), SUMMARIES_PATH)

    existing = _load_existing_labels()

    # Seed the label column from any prior run
    if "label" not in df.columns:
        df["label"] = ""
    df.loc[df["full_name"].isin(existing), "label"] = \
        df.loc[df["full_name"].isin(existing), "full_name"].map(existing)

    todo = df[~df["full_name"].isin(existing)].index.tolist()
    logger.info("%d repos need labeling", len(todo))

    if not todo:
        logger.info("Nothing to do — all repos already labeled.")
        _print_distribution(df)
        return df

    client = _make_client()
    labeled_count = 0
    failed_count = 0

    for i, idx in enumerate(todo, start=1):
        summary = df.at[idx, "text_summary"]
        full_name = df.at[idx, "full_name"]

        label = _call_groq(client, summary)
        if label:
            df.at[idx, "label"] = label
            labeled_count += 1
        else:
            df.at[idx, "label"] = "low_value"   # safe fallback for failed rows
            failed_count += 1
            logger.error("[%d/%d] FAILED  %s  → defaulted to low_value",
                         i, len(todo), full_name)

        if i % BATCH_SIZE == 0 or i == len(todo):
            _save(df)
            logger.info("Checkpoint [%d/%d]  labeled=%d  failed=%d",
                        i, len(todo), labeled_count, failed_count)

        if i < len(todo):                   # no delay after the very last call
            time.sleep(REQUEST_DELAY)

    _save(df)
    logger.info("Done: %d labeled, %d failed  →  %s", labeled_count, failed_count, OUT_PATH)
    _print_distribution(df)
    return df


# ---------------------------------------------------------------------------
# Distribution printer
# ---------------------------------------------------------------------------

def _print_distribution(df: pd.DataFrame) -> None:
    sep = "=" * 50
    print(f"\n{sep}")
    print("  LABEL DISTRIBUTION")
    print(sep)

    total = len(df)
    counts = df["label"].value_counts()
    for label in sorted(VALID_LABELS):
        n = counts.get(label, 0)
        bar = "#" * int(n / total * 40)
        print(f"  {label:<12}  {n:>4}  ({n / total * 100:5.1f}%)  {bar}")

    unlabeled = (df["label"] == "") | (~df["label"].isin(VALID_LABELS))
    if unlabeled.any():
        print(f"\n  unlabeled / invalid: {unlabeled.sum()}")

    print(f"\n  Total: {total}")
    print(f"{sep}\n")

    # Per search-category breakdown
    if "search_category" in df.columns:
        print("  BY SEARCH CATEGORY")
        print(sep)
        pivot = (
            df.groupby(["search_category", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=sorted(VALID_LABELS), fill_value=0)
        )
        print(pivot.to_string())
        print(f"{sep}\n")


if __name__ == "__main__":
    label_repos()
