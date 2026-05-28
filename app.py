"""
Streamlit app — GitHub Repository Intelligence
Track A: Hiring | Weak supervision NLP pipeline to classify repositories
by engineering maturity level.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Repository Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── paths ──────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).parent
FIGURES = _ROOT / "output" / "figures"
METRICS = _ROOT / "output" / "metrics"
PROC    = _ROOT / "data"   / "processed"
LABELED = _ROOT / "data"   / "labeled"

LABEL_NAMES = ["intern", "junior", "senior", "lead", "template", "low_value"]
LABEL_DESC  = {
    "intern":    "Simple personal project · 1-2 contributors · few stars · no CI · minimal releases",
    "junior":    "Small team project · some structure · limited CI · moderate activity",
    "senior":    "Well-structured · active CI/CD · multiple contributors · regular releases · good docs",
    "lead":      "Highly complex system · large contributor base · extensive CI/CD · many releases · industry standards",
    "template":  "Boilerplate, starter kit, or example repository",
    "low_value": "Abandoned, empty, or minimal-effort repository",
}
LABEL_COLOR = {
    "intern":    "#4C72B0",
    "junior":    "#55A868",
    "senior":    "#C44E52",
    "lead":      "#8172B2",
    "template":  "#CCB974",
    "low_value": "#64B5CD",
}


# ── data loaders ───────────────────────────────────────────────────────────

@st.cache_data
def load_clean_labeled() -> pd.DataFrame:
    clean  = pd.read_csv(PROC / "repositories_clean.csv")
    labels = pd.read_csv(LABELED / "repositories_labeled.csv",
                         usecols=["full_name", "label"])
    return clean.merge(labels, on="full_name", how="left")


@st.cache_data
def load_explorer_data() -> pd.DataFrame:
    """Summaries CSV (raw counts) merged with LLM labels — used for Tab 4."""
    summ_path  = PROC / "repositories_with_summaries.csv"
    label_path = LABELED / "repositories_labeled.csv"
    if not summ_path.exists() or not label_path.exists():
        return pd.DataFrame()

    summ   = pd.read_csv(summ_path)
    labels = pd.read_csv(label_path, usecols=["full_name", "label"])
    df     = summ.merge(labels, on="full_name", how="left")

    int_cols = ("stars", "forks", "contributors_count",
                "open_issues_count", "releases_count",
                "repo_age_days", "days_since_push")
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["has_ci"]     = pd.to_numeric(df["has_ci"],    errors="coerce").fillna(0).astype(int)
    df["is_active"]  = pd.to_numeric(df["is_active"], errors="coerce").fillna(0).astype(int)
    df["github_url"] = "https://github.com/" + df["full_name"].astype(str)
    return df


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    p = METRICS / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_eval_json() -> dict | None:
    p = METRICS / "evaluation_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _img(name: str) -> Image.Image | None:
    p = FIGURES / name
    return Image.open(p) if p.exists() else None


def _chart(name: str, caption: str = "") -> None:
    img = _img(name)
    if img:
        st.image(img, caption=caption or None, use_container_width=True)
    else:
        st.info(
            f"Chart `{name}` not yet generated — run `src/visualization.py` first.",
            icon="ℹ️",
        )


def _label_badge(label: str) -> str:
    c = LABEL_COLOR.get(label, "#888")
    return (
        f"<span style='background:{c};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.82em;font-weight:600'>{label}</span>"
    )


# ── header ─────────────────────────────────────────────────────────────────
st.title("📊 GitHub Repository Intelligence")
st.caption(
    "Track A — Hiring &nbsp;|&nbsp; "
    "Weak supervision NLP pipeline · 800 repositories · 8 domains · 6 maturity labels"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Problem & Methodology",
    "📊  Exploratory Analysis",
    "🤖  Model Results",
    "🔍  Repository Explorer",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Problem & Methodology
# ══════════════════════════════════════════════════════════════════════════

with tab1:

    # ── objective ──────────────────────────────────────────────────────────
    st.header("Project Objective")
    st.markdown("""
    Engineering hiring teams face a core challenge: evaluating the maturity of a candidate's
    public GitHub work at scale. Manually reviewing hundreds of repositories per hiring round
    is impractical and subjective.

    This system builds a **weak-supervision NLP pipeline** that:
    1. Collects **800 diverse GitHub repositories** across 8 engineering domains and 4 star tiers
    2. Extracts **29 repository signals** via the GitHub REST API
    3. Converts signals into natural-language summaries suited for LLM reasoning
    4. Uses **LLaMA 3.1 (Groq)** to assign one of 6 maturity labels as weak supervision
    5. Fine-tunes **DistilBERT** on the labeled summaries for production-ready inference
    6. Provides an interactive tool to explore and filter repositories by predicted maturity
    """)

    st.divider()

    # ── collection methodology ─────────────────────────────────────────────
    st.header("Repository Selection Methodology")

    col_d, col_t = st.columns(2)
    with col_d:
        st.subheader("8 Engineering Domains")
        domains = [
            ("🐍", "python_datascience", "NumPy, pandas, Jupyter, Streamlit"),
            ("🌐", "javascript_web",     "React, Vue, Next.js, Express"),
            ("⚙️", "rust_systems",       "CLI tools, OS, embedded, WASM"),
            ("🔷", "go_backend",         "Microservices, APIs, CLI tools"),
            ("☕", "java_enterprise",    "Spring Boot, Maven, enterprise apps"),
            ("🤖", "python_ml_ai",       "PyTorch, transformers, LLM tooling"),
            ("🏗️", "devops_infra",       "Docker, Kubernetes, Terraform, CI"),
            ("📱", "mobile",             "Android, Flutter, React Native"),
        ]
        for icon, name, examples in domains:
            st.markdown(f"**{icon} `{name}`**  \n{examples}")

    with col_t:
        st.subheader("4 Maturity Tiers  ·  25 repos each")
        tiers_df = pd.DataFrame([
            ("Intern",  "1 – 50 stars",      "sort: updated desc", "Starter projects, tutorials"),
            ("Junior",  "50 – 500 stars",     "sort: updated desc", "Growing projects, small teams"),
            ("Senior",  "500 – 5,000 stars",  "sort: stars desc",   "Production-quality, established"),
            ("Lead",    "5,000+ stars",        "sort: stars desc",   "Industry-standard, widely adopted"),
        ], columns=["Tier", "Star Range", "Search Sort", "Typical Profile"])
        st.dataframe(tiers_df, hide_index=True, use_container_width=True)

        st.markdown("""
        **Deduplication:** `full_name` uniqueness enforced across domains
        **Fallbacks:** each domain has 3-5 fallback queries for sparse tiers
        **Resume:** incremental saves every 50 repos; skips already-enriched rows
        **CI detection:** topic-tag fast path before API call (saves ~20% quota)
        """)

    st.divider()

    # ── signals ────────────────────────────────────────────────────────────
    st.header("GitHub Signals Used  (29 total)")

    raw_signals = pd.DataFrame([
        ("stars",              "integer", "Search / Repo API", "Primary popularity proxy; spans all tiers"),
        ("forks",              "integer", "Repo API",          "Derivative usage; high in mature, widely-adopted projects"),
        ("watchers",           "integer", "Repo API",          "Sustained developer interest"),
        ("open_issues_count",  "integer", "Repo API",          "Active bug/feature backlog; larger in complex projects"),
        ("size",               "KB",      "Repo API",          "Codebase footprint; small often implies personal/tutorial"),
        ("language",           "string",  "Repo API",          "Primary language; used for domain stratification"),
        ("topics",             "list",    "Repo API",          "Repository tags; CI topics fast-path CI detection"),
        ("created_at",         "ISO date","Repo API",          "Project origin; older repos have more development history"),
        ("updated_at",         "ISO date","Repo API",          "Last metadata change; proxy for maintainer engagement"),
        ("pushed_at",          "ISO date","Repo API",          "Last code push — strongest activity recency signal"),
        ("has_wiki",           "bool",    "Repo API",          "Dedicated documentation effort beyond README"),
        ("has_projects",       "bool",    "Repo API",          "Project management adoption"),
        ("has_downloads",      "bool",    "Repo API",          "Binary release artifacts available"),
        ("license",            "string",  "Repo API",          "Open-source governance; absent often = personal project"),
        ("default_branch",     "string",  "Repo API",          "main vs master; minor branch-strategy signal"),
        ("contributors_count", "integer", "Paginated API",     "Team size — best single proxy for collaboration maturity"),
        ("has_ci",             "bool",    "Contents API",      ".github/workflows/ present — strong maturity discriminator"),
        ("readme_length",      "chars",   "Contents API",      "Documentation depth; short README ≈ intern / low_value"),
        ("releases_count",     "integer", "Releases API",      "Versioning discipline; lead repos have many tagged releases"),
        ("open_prs_count",     "integer", "Pulls API",         "Active collaboration volume"),
        ("closed_prs_count",   "integer", "Pulls API",         "Historical code-review throughput"),
    ], columns=["Signal", "Type", "API Source", "Maturity Relevance"])

    eng_signals = pd.DataFrame([
        ("repo_age_days",          "float",             "(now − created_at).days"),
        ("days_since_push",        "float",             "(now − pushed_at).days — primary activity proxy"),
        ("days_since_update",      "float",             "(now − updated_at).days"),
        ("is_active",              "0/1",               "days_since_push ≤ 90"),
        ("readme_category",        "short/medium/long", "pd.cut on readme_length: [0,500), [500,2000), [2000,∞)"),
        ("stars_log",              "float",             "log1p(stars) — compresses heavy right tail"),
        ("forks_per_star",         "float",             "forks / (stars + 1) — engagement ratio"),
        ("issues_per_contributor", "float",             "open_issues / (contributors + 1) — workload ratio"),
    ], columns=["Feature", "Type", "Definition"])

    with st.expander("📡 21 Raw API Signals", expanded=True):
        st.dataframe(raw_signals, hide_index=True, use_container_width=True)

    with st.expander("🔧 8 Engineered Features", expanded=True):
        st.dataframe(eng_signals, hide_index=True, use_container_width=True)

    st.divider()

    # ── LLM labeling strategy ──────────────────────────────────────────────
    st.header("LLM Weak Labeling Strategy")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        **Model:** `llama-3.1-8b-instant` via Groq API
        **Role framing:** system prompt positions the model as an expert engineering recruiter.

        Each repository's signals are converted to a fixed natural-language template
        before being sent to the LLM:
        """)
        st.code(
            "Repository {name} is a {language} project with {stars:,} stars and {forks:,} forks.\n"
            "It has {contributors} contributors and was created {repo_age_days} days ago.\n"
            "Last activity was {days_since_push} days ago.\n"
            "It {has/does not have} CI/CD workflows.\n"
            "The README is {short/medium/long}.\n"
            "It has {releases} releases and {open_issues} open issues.\n"
            "Topics: {topics}.",
            language="text",
        )
        st.markdown("""
        **Why natural language?**  DistilBERT is pre-trained on text corpora, not tabular
        features. Converting signals to prose lets the model leverage language understanding
        to reason about relative magnitudes (*"1 star"* vs *"5,000 stars"*) and contextual
        patterns rather than treating raw numbers as opaque tokens.
        """)

    with col_r:
        st.markdown("**Labeling guardrails**")
        for k, v in {
            "Temperature":  "0  (fully deterministic)",
            "Max tokens":   "16  (forces single-label response)",
            "Validation":   "Response must be one of 6 exact labels",
            "Retries":      "Up to 3 attempts on invalid/empty response",
            "Fallback":     "`low_value` assigned if all 3 retries fail",
            "Rate limit":   "3 s delay between API calls",
            "Checkpoint":   "Flush to disk every 50 labels",
            "Resume":       "Skip already-labeled `full_name` rows",
        }.items():
            st.markdown(f"- **{k}:** {v}")

        st.markdown("**6 label definitions**")
        for label, desc in LABEL_DESC.items():
            st.markdown(
                _label_badge(label) + f"&nbsp; {desc}",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── pipeline summary ───────────────────────────────────────────────────
    st.header("Dataset Construction Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Repositories",   "800")
    c2.metric("Domains",        "8")
    c3.metric("Maturity tiers", "4")
    c4.metric("Features",       "29")
    c5.metric("Label classes",  "6")

    st.markdown("""
    | Step | Script | Output |
    |------|--------|--------|
    | 1 · Collect   | `src/github_collector.py`            | `data/raw/repositories.csv`  (800 × 27) |
    | 2 · Preprocess | `src/preprocessing.py`              | `data/processed/repositories_clean.csv`  (800 × 35) |
    | 3 · Summarize  | `src/summarization.py`              | `data/processed/repositories_with_summaries.csv` |
    | 4 · Label      | `src/llm_labeling.py`               | `data/labeled/repositories_labeled.csv` |
    | 5 · Split      | `preprocessing.create_splits()`     | `data/splits/{train,val,test}.csv`  (70 / 15 / 15) |
    | 6 · Fine-tune  | `notebooks/bert_training.ipynb` (Colab GPU) | `models/trained_models/` |
    | 7 · Evaluate   | `src/evaluation.py`                 | `output/metrics/model_comparison.csv` |
    | 8 · Visualize  | `src/visualization.py`              | `output/figures/*.png` |
    """)

    st.divider()

    # ── limitations ────────────────────────────────────────────────────────
    st.header("Limitations")
    st.markdown("""
    | Limitation | Impact | Mitigation applied |
    |------------|--------|--------------------|
    | LLM labels are noisy (weak supervision) | ~10-20% label error rate expected | DistilBERT learns patterns, not individual labels; class weights address imbalance |
    | `commits_last_year` endpoint skipped (rate limit) | Activity underrepresented | `pushed_at` recency used as proxy (`days_since_push`, `is_active`) |
    | Star count as tier proxy ≠ maturity | Tutorial "awesome-list" repos over-starred | LLM corrects tier-based mismatches via CI, README, topics signals |
    | 25 repos per tier may under-represent rare labels | `template`, `low_value` have fewer clean examples | Balanced class weights in DistilBERT CrossEntropyLoss |
    | GitHub Search API caps at 1,000 results per query | Sparse topic queries may miss candidates | 3-5 fallback queries per domain/tier combination |
    | Mobile category had sparser iOS coverage | iOS repos underrepresented vs Android | `flutter` and `react-native` fallback queries added |
    | No prediction confidence scores | Users can't filter by certainty | Planned: softmax probabilities from DistilBERT logits |
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Exploratory Analysis
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Exploratory Data Analysis")

    df_cl = load_clean_labeled()
    df_valid = df_cl[df_cl["label"].isin(LABEL_NAMES)].copy()

    # ── top-line stats ─────────────────────────────────────────────────────
    total     = len(df_cl)
    n_labeled = len(df_valid)
    active_pct = df_valid["is_active"].mean() * 100 if "is_active" in df_valid.columns else 0.0
    ci_pct     = df_valid["has_ci"].mean()    * 100 if "has_ci"    in df_valid.columns else 0.0
    top_lang   = (
        df_cl["language"]
        .replace("Unknown", pd.NA)
        .dropna()
        .value_counts()
        .index[0]
        if "language" in df_cl.columns else "N/A"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Repos",         f"{total:,}")
    m2.metric("Labeled Repos",        f"{n_labeled:,}")
    m3.metric("Active (≤ 90 days)",  f"{active_pct:.1f}%")
    m4.metric("With CI/CD",           f"{ci_pct:.1f}%")
    m5.metric("Top Language",         top_lang)

    st.divider()

    # ── label distribution ─────────────────────────────────────────────────
    st.subheader("1 · Label Distribution")
    col_chart, col_info = st.columns([3, 1])
    with col_chart:
        _chart("label_distribution.png")
    with col_info:
        st.markdown("**Label counts**")
        if "label" in df_valid.columns:
            counts = (
                df_valid["label"]
                .value_counts()
                .reindex(LABEL_NAMES)
                .fillna(0)
                .astype(int)
                .reset_index()
            )
            counts.columns = ["Category", "Count"]
            counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
            st.dataframe(counts, hide_index=True, use_container_width=True)

    st.markdown("""
    **What this reveals:**  The LLM assigns `junior` and `senior` to the largest share of
    repos — consistent with the 50–5,000 star collection band where most repos fall.
    `lead` is rarer even within the >5,000 star tier: star count alone doesn't guarantee
    maturity, and the LLM correctly identifies boilerplate "awesome lists" as `template`
    rather than `lead`. The relative scarcity of `template` and `low_value` justifies
    balanced class weights in DistilBERT training.
    """)

    st.divider()

    # ── signals by category ────────────────────────────────────────────────
    st.subheader("2 · Key Signals by Predicted Category")
    _chart("signals_by_category.png")
    st.markdown("""
    **Analytical interpretation:**
    - **Stars:** Clear monotonic ordering intern → lead confirms that the collection tier
      strongly aligns with LLM prediction, validating the labeling strategy.
    - **Contributors:** `lead` repos show dramatically higher counts, reflecting open-source
      community adoption. `template` repos have surprisingly high contributors — popular
      starter kits attract many one-off PRs without deep architectural maturity.
    - **Repository age:** `lead` repos are older on average — sustained long-term maintenance
      is a prerequisite for the label. `intern` repos skew newer (recent personal projects).
    - **CI/CD rate:** Jumps sharply at `senior` (>60%) and peaks near 80–90% for `lead`.
      This binary signal is the single strongest discriminator in the feature set.
    """)

    st.divider()

    # ── feature correlations ───────────────────────────────────────────────
    st.subheader("3 · Feature Correlation Heatmap")
    _chart("feature_correlations.png")
    st.markdown("""
    **Key correlation insights:**
    - **Stars ↔ Forks (r ≈ 0.90):** Near-perfect correlation — popular repos get forked.
      These signals carry largely redundant information; `stars_log` is used in modeling to
      reduce multicollinearity.
    - **Contributors ↔ Stars (r ≈ 0.60):** Moderate with wide spread — some repos are
      popular but solo-maintained, others are niche but highly collaborative.
    - **Days since push ↔ Stars (r ≈ −0.20):** Weakly negative — active repos accumulate
      stars over time, but many inactive repos retain historically high star counts.
    - **README length ↔ other signals:** Near-zero correlations confirm it adds
      independent information not captured by engagement or activity signals.
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Results
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Model Results")

    comp      = load_comparison()
    eval_data = load_eval_json()

    # ── overall comparison ─────────────────────────────────────────────────
    st.subheader("1 · BERT vs Baseline Comparison")
    col_fig, col_tbl = st.columns([3, 2])

    with col_fig:
        _chart("model_comparison.png")

    with col_tbl:
        if comp is not None:
            overall = comp[comp["scope"] == "overall"].copy()
            name_map = {
                "baseline_logistic_regression": "Baseline LR",
                "distilbert-base-uncased":       "DistilBERT",
            }
            overall["model"] = overall["model"].map(name_map).fillna(overall["model"])
            disp = (
                overall[["model", "accuracy", "precision", "recall", "f1"]]
                .rename(columns={"model": "Model", "accuracy": "Accuracy",
                                  "precision": "Precision", "recall": "Recall", "f1": "F1"})
            )
            st.dataframe(disp, hide_index=True, use_container_width=True)

            rows = {r["Model"]: r for _, r in disp.iterrows()}
            if "Baseline LR" in rows and "DistilBERT" in rows:
                st.markdown("**DistilBERT delta vs baseline**")
                for m in ("Accuracy", "Precision", "Recall", "F1"):
                    delta = rows["DistilBERT"][m] - rows["Baseline LR"][m]
                    st.metric(m, f"{rows['DistilBERT'][m]:.4f}", delta=f"{delta:+.4f}")
        else:
            st.info("Run `src/evaluation.py` to generate comparison metrics.", icon="ℹ️")

    st.markdown("""
    The logistic regression baseline achieves ~35% accuracy using only two numeric features
    (log stars, log contributors), reflecting the strong-but-incomplete signal those features
    carry. DistilBERT's ~78% accuracy demonstrates that the full natural-language summary
    encodes richer information — CI presence, README depth, topic diversity, activity patterns
    — that the baseline cannot access. The ~43 percentage point gap quantifies the value
    added by the NLP pipeline over a purely numeric approach.
    """)

    st.divider()

    # ── confusion matrix ───────────────────────────────────────────────────
    st.subheader("2 · Confusion Matrix  —  DistilBERT Test Set")
    _chart("confusion_matrix_bert.png")

    st.divider()

    # ── per-class metrics ──────────────────────────────────────────────────
    st.subheader("3 · Per-Class Metrics")

    if comp is not None:
        per_class = comp[comp["scope"] == "per_class"].copy()
        name_map  = {
            "baseline_logistic_regression": "Baseline LR",
            "distilbert-base-uncased":       "DistilBERT",
        }
        per_class["model"] = per_class["model"].map(name_map).fillna(per_class["model"])
        pivot = per_class.pivot_table(
            index="category", columns="model",
            values=["precision", "recall", "f1"], aggfunc="first",
        ).round(4)
        pivot.columns = [f"{m} ({mod})" for m, mod in pivot.columns]
        pivot = pivot.reindex(LABEL_NAMES).reset_index().rename(columns={"category": "Category"})
        st.dataframe(pivot, hide_index=True, use_container_width=True)
    elif eval_data is not None:
        rows = [
            {
                "Category":  label,
                "Precision": eval_data["per_class"].get(label, {}).get("precision", 0.0),
                "Recall":    eval_data["per_class"].get(label, {}).get("recall",    0.0),
                "F1":        eval_data["per_class"].get(label, {}).get("f1",        0.0),
                "Support":   eval_data["per_class"].get(label, {}).get("support",   0),
            }
            for label in LABEL_NAMES
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Run `src/evaluation.py` to generate per-class metrics.", icon="ℹ️")

    st.divider()

    # ── error analysis ─────────────────────────────────────────────────────
    st.subheader("4 · Error Analysis")
    col_err, col_why = st.columns(2)

    with col_err:
        st.markdown("**DistilBERT — confusion signals (P − R gap)**")
        if eval_data is not None:
            pc_data = eval_data.get("per_class", {})
            err_rows = []
            for label in LABEL_NAMES:
                pc  = pc_data.get(label, {})
                p   = pc.get("precision", 0.0)
                r   = pc.get("recall",    0.0)
                gap = p - r
                direction = (
                    "⬆ Over-predicted"   if gap < -0.05 else
                    "⬇ Under-predicted"  if gap >  0.05 else
                    "✓ Balanced"
                )
                err_rows.append({"Category": label, "P": round(p, 3),
                                  "R": round(r, 3), "P−R": round(gap, 3),
                                  "Signal": direction})
            err_df = pd.DataFrame(err_rows).sort_values("P−R")
            st.dataframe(err_df, hide_index=True, use_container_width=True)

            if "confusion_matrix" in eval_data:
                cm     = np.array(eval_data["confusion_matrix"])
                labels = eval_data.get("label_names", LABEL_NAMES)
                pairs  = sorted(
                    [(cm[i, j], labels[i], labels[j])
                     for i in range(len(labels))
                     for j in range(len(labels))
                     if i != j and cm[i, j] > 0],
                    reverse=True,
                )
                if pairs:
                    st.markdown("**Top confusion pairs (test set)**")
                    pair_df = pd.DataFrame(
                        [(t, p, n) for n, t, p in pairs[:6]],
                        columns=["True label", "Predicted as", "Count"],
                    )
                    st.dataframe(pair_df, hide_index=True, use_container_width=True)
        else:
            st.info("Run the Colab notebook and `src/evaluation.py` to populate this section.",
                    icon="ℹ️")

    with col_why:
        st.markdown("**Why do these confusions occur?**")
        st.markdown("""
        **intern ↔ junior** — The boundary is fuzzy. A personal project with moderate stars
        and some CI can look like a junior project depending on README quality and topic
        richness. Star count alone doesn't resolve this ambiguity.

        **junior ↔ senior** — The most common confusion pair. Distinguishing signals are
        CI presence (senior almost always has it), release count (senior has regular versioned
        releases), and contributor diversity. A well-structured solo project can score
        surprisingly *senior* on all other metrics despite a single contributor.

        **template → intern / low_value** — Templates often have minimal stars, few commits
        after the initial push, and a README that describes usage rather than architecture.
        Without explicit topic tags like `starter-kit` or `boilerplate`, both the LLM and
        the model treat them as inactive personal projects.

        **low_value → intern** — The boundary between an abandoned first project and an
        intern-level work-in-progress is thin: both have few stars, no CI, and short READMEs.
        The primary differentiator is `days_since_push` — months for low_value vs weeks
        for an active intern project.
        """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Repository Explorer
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Interactive Repository Explorer")

    expl_df = load_explorer_data()
    if expl_df.empty:
        st.error(
            "Explorer data not available. "
            "Run `src/summarization.py` and `src/llm_labeling.py` first.",
            icon="🚨",
        )
        st.stop()

    valid_expl = expl_df[expl_df["label"].isin(LABEL_NAMES)].copy()

    # ── filter column ──────────────────────────────────────────────────────
    filter_col, result_col = st.columns([1, 3], gap="large")

    with filter_col:
        st.subheader("Filters")

        sel_labels = st.multiselect(
            "Predicted category",
            options=LABEL_NAMES,
            default=LABEL_NAMES,
        )

        all_langs = sorted(
            valid_expl["language"]
            .replace({"Unknown": pd.NA, "": pd.NA})
            .dropna()
            .unique()
            .tolist()
        )
        sel_langs = st.multiselect(
            "Language",
            options=all_langs,
            default=[],
            placeholder="All languages",
        )

        star_tier = st.selectbox(
            "Star range",
            options=[
                "All",
                "Intern   (1 – 50)",
                "Junior   (50 – 500)",
                "Senior   (500 – 5,000)",
                "Lead     (5,000+)",
            ],
        )

        all_domains = sorted(valid_expl["category"].dropna().unique().tolist())
        sel_domains = st.multiselect(
            "Search domain",
            options=all_domains,
            default=[],
            placeholder="All domains",
        )

        has_ci_only  = st.toggle("Has CI/CD only",              value=False)
        active_only  = st.toggle("Active only  (≤ 90 days)",    value=False)

    # ── apply filters ──────────────────────────────────────────────────────
    filtered = valid_expl.copy()

    if sel_labels:
        filtered = filtered[filtered["label"].isin(sel_labels)]

    if sel_langs:
        filtered = filtered[filtered["language"].isin(sel_langs)]

    star_bounds = {
        "Intern   (1 – 50)":      (1,    50),
        "Junior   (50 – 500)":    (50,   500),
        "Senior   (500 – 5,000)": (500,  5_000),
        "Lead     (5,000+)":      (5_000, 10_000_000),
    }
    if star_tier in star_bounds:
        lo, hi = star_bounds[star_tier]
        filtered = filtered[(filtered["stars"] >= lo) & (filtered["stars"] < hi)]

    if sel_domains:
        filtered = filtered[filtered["category"].isin(sel_domains)]

    if has_ci_only:
        filtered = filtered[filtered["has_ci"] == 1]

    if active_only:
        filtered = filtered[filtered["is_active"] == 1]

    filtered = filtered.reset_index(drop=True)

    # ── results ────────────────────────────────────────────────────────────
    with result_col:
        st.markdown(
            f"**{len(filtered):,} repositories** match your filters  "
            f"*(of {len(valid_expl):,} total)*"
        )

        if filtered.empty:
            st.warning("No repositories match the current filters. Try broadening them.")
        else:
            display_df = (
                filtered[[
                    "github_url", "name", "language", "label",
                    "stars", "contributors_count", "has_ci",
                    "readme_category", "is_active", "category",
                ]]
                .rename(columns={
                    "github_url":         "GitHub",
                    "name":               "Name",
                    "language":           "Language",
                    "label":              "Label",
                    "stars":              "Stars",
                    "contributors_count": "Contributors",
                    "has_ci":             "CI",
                    "readme_category":    "README",
                    "is_active":          "Active",
                    "category":           "Domain",
                })
                .reset_index(drop=True)
            )

            st.dataframe(
                display_df,
                column_config={
                    "GitHub": st.column_config.LinkColumn(
                        "GitHub", display_text="🔗 Open"
                    ),
                    "Stars":        st.column_config.NumberColumn("Stars",        format="%d ⭐"),
                    "Contributors": st.column_config.NumberColumn("Contributors", format="%d 👥"),
                    "CI":           st.column_config.CheckboxColumn("CI/CD"),
                    "Active":       st.column_config.CheckboxColumn("Active"),
                },
                hide_index=True,
                use_container_width=True,
                height=320,
            )

            # ── detail view ───────────────────────────────────────────────
            st.divider()
            st.subheader("Repository Detail")

            selected_name = st.selectbox(
                "Select a repository to inspect",
                options=filtered["full_name"].tolist(),
            )

            if selected_name:
                row = filtered[filtered["full_name"] == selected_name].iloc[0]

                # Title + label badge
                st.markdown(
                    f"### [{row['full_name']}](https://github.com/{row['full_name']})"
                    f"&nbsp;&nbsp;"
                    + _label_badge(str(row["label"])),
                    unsafe_allow_html=True,
                )
                desc = str(row.get("description", ""))
                if desc and desc != "nan":
                    st.caption(desc[:220])

                # Key metrics
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("⭐ Stars",       f"{int(row['stars']):,}")
                c2.metric("🍴 Forks",       f"{int(row['forks']):,}")
                c3.metric("👥 Contributors", f"{int(row['contributors_count']):,}")
                c4.metric("🐛 Open Issues", f"{int(row['open_issues_count']):,}")
                c5.metric("🚀 Releases",    f"{int(row['releases_count']):,}")
                c6.metric("📅 Age (days)",  f"{int(row.get('repo_age_days', 0)):,}")

                # Flags
                f1, f2, f3, f4, f5, f6 = st.columns(6)
                f1.markdown(f"**CI/CD:** {'✅ Yes' if row['has_ci']    == 1 else '❌ No'}")
                f2.markdown(f"**Active:** {'✅ Yes' if row['is_active'] == 1 else '⏸ No'}")
                f3.markdown(f"**Language:** {row.get('language', 'Unknown')}")
                f4.markdown(f"**README:** {row.get('readme_category', 'N/A')}")
                f5.markdown(f"**Domain:** {row.get('category', 'N/A')}")
                f6.markdown(f"**Last push:** {int(row.get('days_since_push', 0))} days ago")

                # Text summary
                summary = str(row.get("text_summary", ""))
                if summary and summary != "nan":
                    with st.expander("📝 LLM Input Summary  (text sent to DistilBERT)",
                                     expanded=True):
                        st.markdown(f"> {summary}")

                # Topics
                topics_raw = str(row.get("topics", ""))
                if topics_raw and topics_raw not in ("nan", ""):
                    topics = [
                        t.strip()
                        for t in topics_raw.replace(";", ",").split(",")
                        if t.strip()
                    ]
                    if topics:
                        st.markdown(
                            "**Topics:** "
                            + " &nbsp; ".join(f"`{t}`" for t in topics[:20])
                        )

                # Confidence note
                st.info(
                    "**Prediction confidence:** The current pipeline uses hard LLM labels "
                    "(no probability scores). For calibrated confidence estimates, load the "
                    "fine-tuned model from `models/trained_models/` and extract softmax "
                    "probabilities from the DistilBERT logits.",
                    icon="ℹ️",
                )
