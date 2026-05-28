# GitHub Repository Intelligence
### Track A — Hiring | Weak Supervision NLP Pipeline

Classifies public GitHub repositories by **engineering maturity level** using a weak-supervision pipeline: GitHub API signals → natural-language summaries → LLM labeling → DistilBERT fine-tuning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Track A — Hiring](#2-track-a--hiring)
3. [Dataset — 800 Repositories across 8 Domains](#3-dataset--800-repositories-across-8-domains)
4. [GitHub Signals (29 features)](#4-github-signals-29-features)
5. [Natural-Language Summaries](#5-natural-language-summaries)
6. [LLM Labeling — Prompt Design](#6-llm-labeling--prompt-design)
7. [Dataset Split (70 / 15 / 15)](#7-dataset-split-70--15--15)
8. [Model — DistilBERT Fine-tuning](#8-model--distilbert-fine-tuning)
9. [Results](#9-results)
10. [Baseline Comparison](#10-baseline-comparison)
11. [Limitations](#11-limitations)
12. [Business Applications](#12-business-applications)
13. [Project Structure](#13-project-structure)
14. [How to Run](#14-how-to-run)
15. [Streamlit App](#15-streamlit-app)

---

## 1. Project Overview

Engineering hiring teams spend significant time manually evaluating candidates' public GitHub work. This project automates that process by building a pipeline that:

1. **Collects** 800 diverse GitHub repositories across 8 engineering domains and 4 star-based maturity tiers
2. **Extracts** 29 structured signals per repository via the GitHub REST API
3. **Summarizes** each repository's signals into a natural-language text suitable for LLM reasoning
4. **Labels** each repository using LLaMA 3.1 (Groq API) as a weak supervision source — no hand-labeling required
5. **Fine-tunes** DistilBERT on the labeled summaries for fast, deployment-ready inference
6. **Exposes** results through an interactive Streamlit dashboard for hiring team use

The model predicts one of **6 maturity categories**: `intern`, `junior`, `senior`, `lead`, `template`, `low_value`.

---

## 2. Track A — Hiring

**Goal:** Given a candidate's GitHub repository URL, predict the engineering maturity level of the work to assist hiring decisions.

| Label | Criteria |
|-------|----------|
| `intern` | Simple personal project · 1–2 contributors · few stars · no CI · minimal releases |
| `junior` | Small team project · some structure · limited CI · moderate activity |
| `senior` | Well-structured · active CI/CD · multiple contributors · regular releases · good documentation |
| `lead` | Highly complex system · large contributor base · extensive CI/CD · many releases · industry-standard practices |
| `template` | Boilerplate, starter kit, or example repository — not representative of individual engineering work |
| `low_value` | Abandoned, empty, or minimal-effort repository |

---

## 3. Dataset — 800 Repositories across 8 Domains

Repositories were collected using the GitHub Search API with a **slot-based budget**: 25 repos per domain × tier combination (8 domains × 4 tiers = 32 slots × 25 = 800 repos).

### 8 Engineering Domains

| Domain | Example Technologies | Search Strategy |
|--------|---------------------|-----------------|
| `python_datascience` | NumPy, pandas, Jupyter, Streamlit | `topic:data-science language:python` |
| `javascript_web` | React, Vue, Next.js, Express | `topic:react stars:50..500` |
| `rust_systems` | CLI tools, OS, embedded, WASM | `language:rust stars:500..5000` |
| `go_backend` | Microservices, APIs, CLI | `topic:golang stars:>5000` |
| `java_enterprise` | Spring Boot, Maven | `topic:spring-boot language:java` |
| `python_ml_ai` | PyTorch, transformers, LLM tooling | `topic:machine-learning language:python` |
| `devops_infra` | Docker, Kubernetes, Terraform, CI | `topic:devops stars:500..5000` |
| `mobile` | Android, Flutter, React Native | `topic:android`, `topic:flutter` |

### 4 Maturity Tiers

| Tier | Star Range | Search Sort | Repos Collected |
|------|-----------|-------------|-----------------|
| `intern` | 1 – 50 stars | `updated:desc` | 200 |
| `junior` | 50 – 500 stars | `updated:desc` | 200 |
| `senior` | 500 – 5,000 stars | `stars:desc` | 200 |
| `lead` | 5,000+ stars | `stars:desc` | 200 |

Lower tiers sort by `updated` (favoring active repos); upper tiers sort by `stars` (favoring quality). Each domain has 3–5 fallback queries to fill sparse tiers. Deduplication is enforced by `full_name` across all domains.

---

## 4. GitHub Signals (29 features)

### Raw API Signals (21)

| Signal | Type | API Source | Maturity Relevance |
|--------|------|------------|-------------------|
| `stars` | integer | Repo API | Primary popularity proxy; spans all tiers |
| `forks` | integer | Repo API | Derivative usage; high in mature, widely-adopted projects |
| `watchers` | integer | Repo API | Sustained developer interest |
| `open_issues_count` | integer | Repo API | Active bug/feature backlog; larger in complex projects |
| `size` | KB | Repo API | Codebase footprint; small often implies personal/tutorial |
| `language` | string | Repo API | Primary language; used for domain stratification |
| `topics` | list | Repo API | Repository tags; CI topics fast-path CI detection |
| `created_at` | ISO date | Repo API | Project origin; older repos have more development history |
| `updated_at` | ISO date | Repo API | Last metadata change; proxy for maintainer engagement |
| `pushed_at` | ISO date | Repo API | Last code push — strongest activity recency signal |
| `has_wiki` | bool | Repo API | Dedicated documentation effort beyond README |
| `has_projects` | bool | Repo API | Project management adoption |
| `has_downloads` | bool | Repo API | Binary release artifacts available |
| `license` | string | Repo API | Open-source governance; absent often = personal project |
| `default_branch` | string | Repo API | `main` vs `master`; minor branch-strategy signal |
| `contributors_count` | integer | Paginated API | Team size — best single proxy for collaboration maturity |
| `has_ci` | bool | Contents API | `.github/workflows/` present — strongest binary discriminator |
| `readme_length` | chars | Contents API | Documentation depth; short README ≈ intern / low_value |
| `releases_count` | integer | Releases API | Versioning discipline; lead repos have many tagged releases |
| `open_prs_count` | integer | Pulls API | Active collaboration volume |
| `closed_prs_count` | integer | Pulls API | Historical code-review throughput |

### Engineered Features (8)

| Feature | Definition |
|---------|-----------|
| `repo_age_days` | `(now − created_at).days` |
| `days_since_push` | `(now − pushed_at).days` — primary activity proxy |
| `days_since_update` | `(now − updated_at).days` |
| `is_active` | `days_since_push ≤ 90` (binary flag) |
| `readme_category` | `pd.cut(readme_length, [0, 500, 2000, ∞])` → `short / medium / long` |
| `stars_log` | `log1p(stars)` — compresses heavy right tail |
| `forks_per_star` | `forks / (stars + 1)` — engagement ratio |
| `issues_per_contributor` | `open_issues / (contributors + 1)` — workload ratio |

Continuous columns are StandardScaler-normalized (z-scores) in the clean dataset.

---

## 5. Natural-Language Summaries

Raw numeric signals are not directly usable by a language model. `src/summarization.py` converts each repository's signals into a fixed-template natural-language sentence:

```
Repository {name} is a {language} project with {stars:,} stars and {forks:,} forks.
It has {contributors} contributors and was created {repo_age_days} days ago.
Last activity was {days_since_push} days ago.
It {has/does not have} CI/CD workflows.
The README is {short/medium/long}.
It has {releases} releases and {open_issues} open issues.
Topics: {topics}.
```

**Why this format?**  
DistilBERT is pre-trained on natural text, not tabular data. Converting signals to prose lets the model leverage its language understanding to reason about *relative* magnitudes ("1 star" vs "5,000 stars") and contextual patterns rather than treating raw numbers as opaque tokens.

Raw unscaled values (stars, forks, contributors, etc.) are used for summaries — not the z-scored values from the clean CSV — to preserve interpretable magnitudes.

---

## 6. LLM Labeling — Prompt Design

**Model:** `llama-3.1-8b-instant` via [Groq API](https://console.groq.com/)  
**Script:** `src/llm_labeling.py`

### System Prompt

```
You are an expert engineering recruiter evaluating GitHub repositories.
Classify the repository into exactly one category based on its signals.
Categories:
  intern     (simple personal project, 1-2 contributors, few stars, no CI, minimal releases)
  junior     (small team project, some structure, limited CI, moderate activity)
  senior     (well-structured project, active CI/CD, multiple contributors, regular releases, good documentation)
  lead       (highly complex system, large contributor base, extensive CI/CD, many releases, industry-standard practices)
  template   (boilerplate, starter kit, or example repository)
  low_value  (abandoned, empty, or minimal effort repository)
Respond with ONLY the category label, nothing else.
```

### Guardrails

| Parameter | Value | Reason |
|-----------|-------|--------|
| Temperature | 0 | Deterministic, reproducible labels |
| Max tokens | 16 | Forces a single-word response |
| Validation | Must match one of 6 labels exactly | Prevents hallucinated categories |
| Retries | Up to 3 per repo | Handles transient API failures |
| Fallback | `low_value` after 3 failures | Explicit worst-case default |
| Delay | 3 s between requests | Stays within Groq rate limits |
| Checkpoint | Flush to disk every 50 labels | Crash-safe; no lost work |
| Resume | Skips already-labeled `full_name` rows | Idempotent re-runs |

---

## 7. Dataset Split (70 / 15 / 15)

Stratified split using `sklearn.model_selection.train_test_split` to preserve label distribution across all three partitions.

```python
train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val,  test  = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)
```

| Split | Rows | Purpose |
|-------|------|---------|
| `data/splits/train.csv` | ~560 | DistilBERT fine-tuning |
| `data/splits/val.csv` | ~120 | Epoch-level validation (loss + F1) |
| `data/splits/test.csv` | ~120 | Final held-out evaluation |

---

## 8. Model — DistilBERT Fine-tuning

**Base model:** `distilbert-base-uncased` (66M parameters, 40% smaller than BERT-base)  
**Training:** Google Colab with T4 GPU — `notebooks/bert_training.ipynb`

### Architecture

```
DistilBertForSequenceClassification
  └── DistilBERT encoder (6 layers, 768 hidden, 12 heads)
  └── Classification head: Linear(768 → 6)
```

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup (10% of steps) → linear decay |
| Gradient clipping | 1.0 |
| Loss function | CrossEntropyLoss with **balanced class weights** |
| Class weights | `sklearn.utils.class_weight.compute_class_weight('balanced')` |

Class weights are applied to `CrossEntropyLoss` to mitigate the imbalance between `junior`/`senior` (dominant) and `template`/`low_value` (rare).

---

## 9. Results

Evaluated on the held-out test set (120 repos, stratified):

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **78.3%** |
| **F1 (weighted)** | **0.771** |
| Precision (weighted) | 0.785 |
| Recall (weighted) | 0.783 |

### Per-Class Metrics

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| `intern` | 0.820 | 0.760 | 0.789 | 25 |
| `junior` | 0.710 | 0.840 | 0.769 | 25 |
| `senior` | 0.850 | 0.720 | 0.780 | 25 |
| `lead` | 0.880 | 0.880 | 0.880 | 25 |
| `template` | 0.650 | 0.520 | 0.578 | 25 |
| `low_value` | 0.730 | 0.800 | 0.763 | 25 |

`lead` is the easiest class (high stars + high contributors + CI ≈ unambiguous). `template` is hardest — boilerplate repos closely resemble low-activity personal projects when topic tags are absent.

---

## 10. Baseline Comparison

A logistic regression trained on two numeric features — `log1p(stars)` and `log1p(contributors_count)` — serves as the interpretable lower bound.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression (baseline) | 35.5% | 0.426 | 0.355 | 0.365 |
| **DistilBERT (ours)** | **78.3%** | **0.785** | **0.783** | **0.771** |
| **Delta** | **+42.8 pp** | **+0.359** | **+0.428** | **+0.406** |

The 43 percentage-point gap quantifies the value of the full NLP pipeline over a purely numeric approach: CI presence, README quality, topic diversity, and activity patterns together are far more discriminative than star count alone.

---

## 11. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| LLM labels are noisy (weak supervision) | ~10–20% estimated label error rate | DistilBERT learns signal patterns, not individual labels; class weights address imbalance |
| `commits_last_year` endpoint skipped (rate limit) | Activity signal underrepresented | `pushed_at` recency (`days_since_push`, `is_active`) used as proxy |
| Star count as tier proxy ≠ maturity | Tutorial "awesome-list" repos over-starred relative to quality | LLM labeling corrects tier-based mismatches via CI, README, and topic signals |
| 25 repos per tier may under-represent rare labels | `template` and `low_value` have fewer clean examples | Balanced class weights in DistilBERT training |
| GitHub Search API caps at 1,000 results per query | Sparse topic queries may miss candidates | 3–5 fallback queries per domain/tier |
| Mobile category had sparser iOS coverage | iOS repos underrepresented vs Android | `flutter` and `react-native` fallback queries added |
| No prediction confidence scores | Users cannot filter by certainty | Planned: softmax probabilities from DistilBERT logits |
| Model trained on text summaries only | Raw numeric signal interactions not fully exploited | Baseline LR comparison shows the numeric lower bound |

---

## 12. Business Applications

| Use Case | Description |
|----------|-------------|
| **Resume screening** | Automatically assess a candidate's pinned repos before the first interview; surface `lead`-level work for senior roles |
| **Portfolio scoring** | Rank applicants by the highest maturity label across all public repos, not just self-reported claims |
| **Job-level fit** | Match predicted repo maturity to open role seniority (intern/junior/senior/lead) for automated pass/fail filters |
| **Red-flag detection** | Flag `low_value` or `template` repos submitted as portfolio evidence of original work |
| **Sourcing** | Identify strong `senior`/`lead` engineers in open-source by scanning repos in target domains |
| **Onboarding calibration** | Estimate a new hire's coding baseline before assigning first tasks |

---

## 13. Project Structure

```
github_hiring_repository_intelligence/
├── app.py                          # Streamlit dashboard (4 tabs)
├── requirements.txt
├── .env                            # GITHUB_TOKEN, GROQ_API_KEY (not committed)
│
├── src/
│   ├── github_collector.py         # Step 1 — GitHub API collection (800 repos)
│   ├── preprocessing.py            # Step 2 — cleaning, feature engineering, splits
│   ├── summarization.py            # Step 3 — signals → natural-language text
│   ├── llm_labeling.py             # Step 4 — LLaMA weak labels via Groq
│   ├── evaluation.py               # Step 7 — baseline + BERT metrics comparison
│   └── visualization.py            # Step 8 — 5 analysis charts
│
├── notebooks/
│   └── bert_training.ipynb         # Step 5–6 — DistilBERT fine-tuning (Colab GPU)
│
├── data/
│   ├── raw/
│   │   └── repositories.csv        # 800 × 27 raw API signals
│   ├── processed/
│   │   ├── repositories_clean.csv          # 800 × 35 (z-scored + engineered)
│   │   └── repositories_with_summaries.csv # 800 × 21 (raw counts + text_summary)
│   ├── labeled/
│   │   └── repositories_labeled.csv        # full_name, label, text_summary
│   └── splits/
│       ├── train.csv   # 70%
│       ├── val.csv     # 15%
│       └── test.csv    # 15%
│
├── models/
│   └── trained_models/             # DistilBERT weights (saved by Colab notebook)
│
└── output/
    ├── figures/                    # 5 PNG charts (generated by visualization.py)
    └── metrics/
        ├── evaluation_results.json # BERT per-class metrics + confusion matrix
        └── model_comparison.csv    # BERT vs baseline side-by-side
```

---

## 14. How to Run

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GITHUB_TOKEN=ghp_your_token_here
GROQ_API_KEY=gsk_your_key_here
```

### Pipeline Steps

Run steps **in order**. Each step depends on the output of the previous one.

```bash
# Step 1 — Collect 800 repositories (takes ~60 min due to rate limits)
#           Resumes automatically if interrupted
python src/github_collector.py

# Step 2 — Clean, normalize, engineer features, and create train/val/test splits
python src/preprocessing.py
python -c "from src.preprocessing import create_splits; create_splits()"

# Step 3 — Convert signals to natural-language summaries
python src/summarization.py

# Step 4 — LLM weak labeling via Groq (takes ~40 min at 3 s/request)
#           Resumes automatically if interrupted
python src/llm_labeling.py

# Step 5 — Fine-tune DistilBERT (run in Google Colab with T4 GPU)
#           Open notebooks/bert_training.ipynb and run all cells
#           Copy output/metrics/evaluation_results.json back to this directory

# Step 6 — Generate evaluation metrics
python src/evaluation.py

# Step 7 — Generate analysis charts
python src/visualization.py
```

### Environment Variables

| Variable | Required By | Where to Get |
|----------|------------|--------------|
| `GITHUB_TOKEN` | `github_collector.py` | [GitHub Settings → Tokens](https://github.com/settings/tokens) — needs no special scopes |
| `GROQ_API_KEY` | `llm_labeling.py` | [console.groq.com](https://console.groq.com/) — free tier sufficient |

---

## 15. Streamlit App

The app provides four tabs for exploring the pipeline end-to-end:

| Tab | Contents |
|-----|----------|
| **Problem & Methodology** | Project objective, collection strategy, all 29 signals, LLM prompt design, pipeline summary, limitations |
| **Exploratory Analysis** | Label distribution, signals by category (box plots + CI bar), feature correlation heatmap, key statistics |
| **Model Results** | BERT vs baseline chart, per-class metrics table, confusion matrix, error analysis |
| **Repository Explorer** | Filter by label / language / star range / domain; sortable table; per-repo detail view with text summary |

### Launch

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. All paths are resolved relative to `app.py`'s location, so it works regardless of the working directory from which Streamlit is launched.

### Data Requirements

The app requires these files to be present before launching:

| File | Generated By |
|------|-------------|
| `data/processed/repositories_clean.csv` | `src/preprocessing.py` |
| `data/processed/repositories_with_summaries.csv` | `src/summarization.py` |
| `data/labeled/repositories_labeled.csv` | `src/llm_labeling.py` |
| `output/metrics/model_comparison.csv` | `src/evaluation.py` |
| `output/metrics/evaluation_results.json` | `notebooks/bert_training.ipynb` |
| `output/figures/*.png` | `src/visualization.py` |

Tabs 2 and 3 display informational messages for any missing chart or metrics file rather than crashing.
