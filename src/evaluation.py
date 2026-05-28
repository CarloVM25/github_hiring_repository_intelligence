"""
Evaluation module — compares DistilBERT against a numeric baseline.

Steps:
  1. Load output/metrics/evaluation_results.json  (BERT metrics from Colab)
  2. Load data/splits/train.csv + test.csv
  3. Train a baseline logistic regression using log1p(stars) and
     log1p(contributors) parsed from the text_summary column
  4. Print a per-class classification report for both models
  5. Analyse errors: BERT via precision/recall gap; baseline via confusion matrix
  6. Print a side-by-side summary table
  7. Save output/metrics/model_comparison.csv
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

METRICS_PATH   = Path("output/metrics")
SPLITS_PATH    = Path("data/splits")
EVAL_JSON      = METRICS_PATH / "evaluation_results.json"
COMPARISON_CSV = METRICS_PATH / "model_comparison.csv"

LABEL_NAMES = ["intern", "junior", "senior", "lead", "template", "low_value"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}

_RE_STARS    = re.compile(r"with ([\d,]+) stars")
_RE_CONTRIBS = re.compile(r"has ([\d,]+) contributors")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_split(name: str) -> pd.DataFrame:
    path = SPLITS_PATH / f"{name}.csv"
    if not path.exists():
        logger.error("Split not found: %s — run preprocessing.py first.", path)
        sys.exit(1)
    df = pd.read_csv(path).dropna(subset=["label", "text_summary"])
    df = df[df["label"].isin(LABEL_NAMES)].reset_index(drop=True)
    logger.info("Loaded %s split: %d rows", name, len(df))
    return df


def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Parse stars and contributors_count from text_summary strings.
    Returns an (n, 2) array of [log1p(stars), log1p(contributors)].
    Both features are log-transformed to compress the heavy right tail.
    """
    def _parse(text: str) -> tuple[float, float]:
        m_s = _RE_STARS.search(text)
        m_c = _RE_CONTRIBS.search(text)
        stars    = float(m_s.group(1).replace(",", "")) if m_s else 0.0
        contribs = float(m_c.group(1).replace(",", "")) if m_c else 0.0
        return np.log1p(stars), np.log1p(contribs)

    rows = [_parse(str(t)) for t in df["text_summary"]]
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# BERT metrics
# ---------------------------------------------------------------------------

def _load_bert_metrics() -> dict:
    if not EVAL_JSON.exists():
        logger.error(
            "BERT metrics not found at %s.\n"
            "Run notebooks/bert_training.ipynb on Colab first,\n"
            "then copy evaluation_results.json to %s.",
            EVAL_JSON, METRICS_PATH,
        )
        sys.exit(1)
    with open(EVAL_JSON) as f:
        data = json.load(f)
    logger.info("BERT metrics loaded from %s", EVAL_JSON)
    return data


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def _run_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Logistic regression on two engineered numeric features:
      - log1p(stars)
      - log1p(contributors_count)
    Both parsed from the text_summary column so no external join is needed.
    """
    X_train = _extract_features(train_df)
    y_train = train_df["label"].map(LABEL2ID).values

    X_test  = _extract_features(test_df)
    y_test  = test_df["label"].map(LABEL2ID).values

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    return {
        "y_true":    y_test.tolist(),
        "y_pred":    y_pred.tolist(),
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred,    average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred,        average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def _print_bert_report(bert: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  BERT (DistilBERT) -- PER-CLASS METRICS")
    print("  source: evaluation_results.json")
    print(sep)
    print(f"  {'category':<14}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>8}")
    print("  " + "-" * 54)
    for label in LABEL_NAMES:
        pc = bert["per_class"].get(label, {})
        p  = pc.get("precision", 0.0)
        r  = pc.get("recall",    0.0)
        f  = pc.get("f1",        0.0)
        s  = pc.get("support",   0)
        print(f"  {label:<14}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}  {s:>8}")
    print("  " + "-" * 54)
    ov = bert["overall"]
    print(f"  {'weighted avg':<14}  {ov['precision_weighted']:>10.4f}"
          f"  {ov['recall_weighted']:>8.4f}  {ov['f1_weighted']:>8.4f}")
    print(f"\n  Overall accuracy: {ov['accuracy']:.4f}")
    print(f"{sep}\n")


def _print_baseline_report(baseline: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  BASELINE -- LOGISTIC REGRESSION")
    print("  features: log1p(stars), log1p(contributors)")
    print(sep)
    print(classification_report(
        baseline["y_true"],
        baseline["y_pred"],
        target_names=LABEL_NAMES,
        labels=list(range(len(LABEL_NAMES))),
        digits=4,
        zero_division=0,
    ))
    print(f"  Overall accuracy: {baseline['accuracy']:.4f}")
    print(f"{sep}\n")


def _print_error_analysis(bert: dict, baseline: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  ERROR ANALYSIS")
    print(sep)

    # ── BERT: precision vs recall gap reveals confusion direction ──────────
    # P > R  ->  model is conservative; actual instances leak into other classes
    # P < R  ->  model over-fires; other classes bleed into this bucket
    print("\n  BERT -- confusion signals (precision / recall gap):")
    print(f"  {'category':<14}  {'precision':>10}  {'recall':>8}  {'gap (P-R)':>10}  direction")
    print("  " + "-" * 64)

    diffs = [
        (bert["per_class"].get(l, {}).get("precision", 0.0)
         - bert["per_class"].get(l, {}).get("recall",    0.0),
         l,
         bert["per_class"].get(l, {}).get("precision", 0.0),
         bert["per_class"].get(l, {}).get("recall",    0.0),
         )
        for l in LABEL_NAMES
    ]
    for gap, label, p, r in sorted(diffs):          # most negative first
        if gap < -0.05:
            note = f"over-predicted  (other classes absorbed into {label})"
        elif gap > 0.05:
            note = f"under-predicted (true {label} leaked into other classes)"
        else:
            note = "balanced"
        sign = "+" if gap >= 0 else ""
        print(f"  {label:<14}  {p:>10.4f}  {r:>8.4f}  {sign}{gap:>9.4f}  {note}")

    # ── Baseline: real confusion matrix ───────────────────────────────────
    cm = confusion_matrix(
        baseline["y_true"], baseline["y_pred"],
        labels=list(range(len(LABEL_NAMES))),
    )
    col_w = 11
    print(f"\n\n  BASELINE -- confusion matrix (rows = true, cols = predicted):\n")
    print(" " * 16 + "".join(f"{n:>{col_w}}" for n in LABEL_NAMES))
    for i, label in enumerate(LABEL_NAMES):
        cells = []
        for j in range(len(LABEL_NAMES)):
            val = cm[i, j]
            cells.append(f"{val:>{col_w}}")
        print(f"  {label:<14}" + "".join(cells))

    # Top confused pairs (off-diagonal, most frequent first)
    pairs = sorted(
        [
            (cm[i, j], LABEL_NAMES[i], LABEL_NAMES[j])
            for i in range(len(LABEL_NAMES))
            for j in range(len(LABEL_NAMES))
            if i != j and cm[i, j] > 0
        ],
        reverse=True,
    )
    print("\n\n  Top confused pairs (baseline, off-diagonal):")
    for count, true_l, pred_l in pairs[:6]:
        bar = "#" * count
        print(f"  true={true_l:<12}  predicted={pred_l:<12}  n={count:>3}  {bar}")

    print(f"\n{sep}\n")


def _print_comparison(bert: dict, baseline: dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  MODEL COMPARISON SUMMARY")
    print(sep)

    bert_vals = {
        "accuracy":  bert["overall"]["accuracy"],
        "precision": bert["overall"]["precision_weighted"],
        "recall":    bert["overall"]["recall_weighted"],
        "f1":        bert["overall"]["f1_weighted"],
    }

    print(f"\n  {'metric':<14}  {'baseline (LR)':>16}  {'DistilBERT':>14}  {'delta':>10}  note")
    print("  " + "-" * 64)
    for m in ("accuracy", "precision", "recall", "f1"):
        b     = baseline[m]
        d     = bert_vals[m]
        delta = d - b
        sign  = "+" if delta >= 0 else ""
        note  = "BERT better" if delta >= 0.02 else ("LR better" if delta <= -0.02 else "similar")
        print(f"  {m:<14}  {b:>16.4f}  {d:>14.4f}  {sign}{delta:>9.4f}  {note}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_comparison(bert: dict, baseline: dict) -> None:
    bert_overall = {
        "accuracy":  bert["overall"]["accuracy"],
        "precision": bert["overall"]["precision_weighted"],
        "recall":    bert["overall"]["recall_weighted"],
        "f1":        bert["overall"]["f1_weighted"],
    }

    rows = []

    for model_name, vals in [
        ("baseline_logistic_regression", baseline),
        ("distilbert-base-uncased",      bert_overall),
    ]:
        rows.append({
            "model":     model_name,
            "scope":     "overall",
            "category":  "all",
            "accuracy":  round(vals["accuracy"],  4),
            "precision": round(vals["precision"], 4),
            "recall":    round(vals["recall"],    4),
            "f1":        round(vals["f1"],        4),
        })

    # Per-class — BERT (from JSON)
    for label in LABEL_NAMES:
        pc = bert["per_class"].get(label, {})
        rows.append({
            "model":     "distilbert-base-uncased",
            "scope":     "per_class",
            "category":  label,
            "accuracy":  None,
            "precision": round(pc.get("precision", 0.0), 4),
            "recall":    round(pc.get("recall",    0.0), 4),
            "f1":        round(pc.get("f1",        0.0), 4),
        })

    # Per-class — baseline (computed from raw predictions)
    y_true = baseline["y_true"]
    y_pred = baseline["y_pred"]
    ids    = list(range(len(LABEL_NAMES)))
    prec   = precision_score(y_true, y_pred, average=None, labels=ids, zero_division=0)
    rec    = recall_score(   y_true, y_pred, average=None, labels=ids, zero_division=0)
    f1s    = f1_score(       y_true, y_pred, average=None, labels=ids, zero_division=0)
    for i, label in enumerate(LABEL_NAMES):
        rows.append({
            "model":     "baseline_logistic_regression",
            "scope":     "per_class",
            "category":  label,
            "accuracy":  None,
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1s[i]),  4),
        })

    df = pd.DataFrame(rows, columns=["model", "scope", "category",
                                     "accuracy", "precision", "recall", "f1"])
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARISON_CSV, index=False)
    logger.info("Saved -> %s  (%d rows)", COMPARISON_CSV, len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate() -> None:
    train_df = _load_split("train")
    test_df  = _load_split("test")

    bert     = _load_bert_metrics()
    baseline = _run_baseline(train_df, test_df)
    logger.info(
        "Baseline  accuracy=%.4f  f1=%.4f",
        baseline["accuracy"], baseline["f1"],
    )

    _print_bert_report(bert)
    _print_baseline_report(baseline)
    _print_error_analysis(bert, baseline)
    _print_comparison(bert, baseline)
    _save_comparison(bert, baseline)


if __name__ == "__main__":
    evaluate()
