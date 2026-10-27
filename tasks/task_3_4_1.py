"""Task 3.4.1: lexicon-based sentiment trends (VADER) across three business sectors.

Run from the project root:

    python -m tasks.task_3_4_1

Uses the same reproducible 10% review sample (shared seed) as tasks 3.2.2 and
3.3.1, scores the raw review text with VADER, and writes figures, tables and a
text summary to outputs/task_3_4_1/.

Extra dependencies (add to requirements.txt if missing):
    vaderSentiment, scipy, seaborn, matplotlib, pandas, numpy
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tasks.task_3_3_1 import DEFAULT_FRACTION, SAMPLE_SEED, sample_rows
from utils.load_cache import load_businesses, load_reviews

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task_3_4_1"

# Sector label -> Yelp category text used to match businesses.
SECTORS = {
    "Fast Food": "Fast Food",
    "Italian": "Italian",
    "Ice Cream": "Ice Cream & Frozen Yogurt",
}
SECTOR_ORDER = list(SECTORS)
SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
              5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
              10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
MIN_N = 100  # minimum reviews per sector-period group


def assign_sector(categories):
    """Return the sector label, or None if no match / more than one match."""
    if not isinstance(categories, str):
        return None
    hits = [label for label, key in SECTORS.items() if key in categories]
    return hits[0] if len(hits) == 1 else None


def load_sector_reviews() -> tuple[pd.DataFrame, int]:
    """Load the shared 10% sample and keep reviews of businesses in one sector."""
    reviews = load_reviews()
    rows = sample_rows(reviews.num_rows, DEFAULT_FRACTION)
    sampled = (reviews.select(rows)
               .select_columns(["review_id", "business_id", "stars", "date", "text"])
               .to_pandas())
    businesses = (load_businesses()
                  .select_columns(["business_id", "categories"]).to_pandas())
    print(f"Sampled reviews: {len(sampled):,} "
          f"(fraction={DEFAULT_FRACTION}, seed={SAMPLE_SEED})")

    businesses["Sector"] = businesses["categories"].apply(assign_sector)
    overlap = int(businesses["categories"].apply(
        lambda c: isinstance(c, str) and sum(k in c for k in SECTORS.values()) > 1
    ).sum())
    biz = businesses.dropna(subset=["Sector"])[["business_id", "Sector"]]
    df = sampled.merge(biz, on="business_id", how="inner")

    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df["date"].dt.year
    df["Season"] = df["date"].dt.month.map(SEASON_MAP)
    df["text_length"] = df["text"].str.len()
    print(df["Sector"].value_counts().to_string())
    print("Businesses excluded for overlapping sectors:", overlap)
    return df, overlap


def score_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Add a VADER compound score computed on the raw review text."""
    analyzer = SentimentIntensityAnalyzer()

    def compound(text):
        if not isinstance(text, str):
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    print("Scoring reviews with VADER...")
    df["sentiment_score"] = df["text"].apply(compound)
    print(df.groupby("Sector")["sentiment_score"].describe().round(3))
    return df


def summarise(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = (df.groupby(["Sector", by])["sentiment_score"]
         .agg(["mean", "std", "count"]).reset_index())
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g[g["count"] >= MIN_N]


def plot_yearly(yearly: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    for s in SECTOR_ORDER:
        g = yearly[yearly["Sector"] == s]
        plt.plot(g["Year"], g["mean"], marker="o", linewidth=2, label=s)
        plt.fill_between(g["Year"], g["mean"] - g["ci95"], g["mean"] + g["ci95"],
                         alpha=0.2)
    plt.title("Average VADER sentiment by year (95% CI)")
    plt.xlabel("Year")
    plt.ylabel("Mean VADER compound")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(OUTPUT_DIR / "sentiment_trends.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_seasonal(seasonal: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=seasonal, x="Season", y="mean", hue="Sector",
                order=SEASON_ORDER, hue_order=SECTOR_ORDER, errorbar=None)
    bar_w = 0.8 / len(SECTOR_ORDER)
    for i, s in enumerate(SECTOR_ORDER):
        d = seasonal[seasonal["Sector"] == s].set_index("Season").reindex(SEASON_ORDER)
        xs = np.arange(len(SEASON_ORDER)) - 0.4 + bar_w / 2 + i * bar_w
        plt.errorbar(xs, d["mean"], yerr=d["ci95"], fmt="none", color="black",
                     capsize=3)
    plt.title("Average VADER sentiment by season (95% CI)")
    plt.ylabel("Mean VADER compound")
    plt.legend(title="Sector", loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=len(SECTOR_ORDER))
    plt.savefig(OUTPUT_DIR / "sentiment_seasons.png", dpi=200, bbox_inches="tight")
    plt.close()


def stars_and_length(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["Sector", "Year"])
         .agg(mean_stars=("stars", "mean"), mean_length=("text_length", "mean"),
              count=("stars", "count"))
         .reset_index())
    g = g[g["count"] >= MIN_N]
    g.to_csv(OUTPUT_DIR / "yearly_stars_length.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SECTOR_ORDER:
        d = g[g["Sector"] == s]
        axes[0].plot(d["Year"], d["mean_stars"], marker="o", linewidth=2, label=s)
        axes[1].plot(d["Year"], d["mean_length"], marker="o", linewidth=2, label=s)
    axes[0].set_title("Mean Star Rating by Year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Mean Stars (out of 5)")
    axes[1].set_title("Mean Review Length (chars) by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mean Character Count")
    for ax in axes:
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stars_and_length_trends.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    return g


def mismatch_examples(df: pd.DataFrame) -> list[str]:
    """Save reviews where VADER and the star rating strongly disagree."""
    cols = ["Sector", "stars", "sentiment_score", "text"]
    fp = df[(df["stars"] == 1) & (df["sentiment_score"] > 0.8)]
    fn = df[(df["stars"] == 5) & (df["sentiment_score"] < -0.5)]
    fp[cols].sort_values("sentiment_score", ascending=False).head(10).to_csv(
        OUTPUT_DIR / "sarcasm_examples.csv", index=False)
    fn[cols].sort_values("sentiment_score").head(10).to_csv(
        OUTPUT_DIR / "false_negative_examples.csv", index=False)
    lines = [f"1-star but VADER > 0.8: {len(fp)} | 5-star but VADER < -0.5: {len(fn)}"]
    lines.append("1-star/VADER>0.8 by sector: "
                 + ", ".join(f"{k} {v}" for k, v in fp["Sector"].value_counts().items()))
    print("\n".join(lines))
    return lines


def agreement_and_tests(df: pd.DataFrame, yearly: pd.DataFrame) -> list[str]:
    lines = []
    rho = df["sentiment_score"].corr(df["stars"], method="spearman")
    vader_lab = pd.cut(df["sentiment_score"], [-1.01, -0.05, 0.05, 1.01],
                       labels=["neg", "neu", "pos"])
    star_lab = pd.cut(df["stars"], [0, 2, 3, 5], labels=["neg", "neu", "pos"])
    agree = (vader_lab == star_lab).mean()
    lines.append(f"Spearman(VADER, stars) = {rho:.3f}")
    lines.append(f"Agreement with star label = {agree:.3f}")

    # Pairwise sector comparison: Welch t-test, Bonferroni-adjusted.
    pairs = list(combinations(SECTOR_ORDER, 2))
    for s1, s2 in pairs:
        x = df.loc[df.Sector == s1, "sentiment_score"]
        y = df.loc[df.Sector == s2, "sentiment_score"]
        _, p = stats.ttest_ind(x, y, equal_var=False)
        d = (x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)
        lines.append(f"{s1} - {s2}: diff = {x.mean() - y.mean():.3f}, "
                     f"Welch p(adj) = {min(p * len(pairs), 1.0):.3g}, "
                     f"Cohen's d = {d:.3f}")

    # Linear trend per sector on yearly means.
    for s in SECTOR_ORDER:
        g = yearly[yearly["Sector"] == s]
        fit = stats.linregress(g["Year"], g["mean"])
        lines.append(f"{s} slope/yr = {fit.slope:.4f}, p = {fit.pvalue:.3g} "
                     f"(years: {g['Year'].min()}-{g['Year'].max()})")

    # Seasonality: Kruskal-Wallis per sector.
    for s in SECTOR_ORDER:
        sub = df[df.Sector == s]
        groups = [g["sentiment_score"].values for _, g in sub.groupby("Season")]
        h, p = stats.kruskal(*groups)
        means = sub.groupby("Season")["sentiment_score"].mean()
        lines.append(f"{s}: Kruskal-Wallis H = {h:.2f}, p = {p:.3g}, "
                     f"max-min seasonal mean gap = {means.max() - means.min():.3f}")
    print("\n".join(lines))
    return lines


def write_summary(df, yearly, seasonal, stars, mismatch, tests) -> None:
    lines = [f"Task 3.4.1 - Lexicon-based sentiment (VADER, shared "
             f"{DEFAULT_FRACTION:.0%} sample, seed {SAMPLE_SEED})", ""]

    lines.append(f"Reviews per sector (full sample, before the {MIN_N}-review threshold):")
    for s in SECTOR_ORDER:
        lines.append(f"  {s}: {int((df.Sector == s).sum())}")
    lines.append(f"  Total: {len(df)}")
    lines.append(f"  (Yearly/seasonal groups with fewer than {MIN_N} reviews are "
                 "excluded from the trend statistics below.)")
    lines.append("  (Seasons assume the northern hemisphere: Dec-Feb winter, "
                 "Mar-May spring, Jun-Aug summer, Sep-Nov autumn.)")

    lines += ["", f"Yearly trend (years with >={MIN_N} reviews):"]
    for s in SECTOR_ORDER:
        g = yearly[yearly["Sector"] == s]
        fit = stats.linregress(g["Year"], g["mean"])
        lines.append(f"  {s}: {g['mean'].iloc[0]:.3f} ({int(g['Year'].iloc[0])}) -> "
                     f"{g['mean'].iloc[-1]:.3f} ({int(g['Year'].iloc[-1])}), "
                     f"slope {fit.slope:+.4f}/yr, p={fit.pvalue:.3g}, n_years={len(g)}")

    lines += ["", f"Mean stars (first -> last year, >={MIN_N} reviews):"]
    for s in SECTOR_ORDER:
        g = stars[stars["Sector"] == s]
        lines.append(f"  {s}: {g['mean_stars'].iloc[0]:.2f} -> "
                     f"{g['mean_stars'].iloc[-1]:.2f}")

    lines += ["", "Seasonal means and gap (max-min):"]
    for s in SECTOR_ORDER:
        m = seasonal[seasonal["Sector"] == s].set_index("Season")["mean"]
        lines.append(f"  {s}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items())
                     + f" | gap {m.max() - m.min():.3f}")

    lines += ["", "VADER vs star rating and statistical tests:"]
    lines += [f"  {t}" for t in tests]
    lines += ["", "Disagreement counts:"]
    lines += [f"  {m}" for m in mismatch]

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUTPUT_DIR / 'summary.txt'}")


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = load_sector_reviews()
    print(pd.crosstab(df["Year"], df["Sector"]))
    df = score_vader(df)

    yearly = summarise(df, "Year")
    yearly.to_csv(OUTPUT_DIR / "yearly_sentiment.csv", index=False)
    plot_yearly(yearly)

    seasonal = summarise(df, "Season")
    seasonal.to_csv(OUTPUT_DIR / "seasonal_sentiment.csv", index=False)
    plot_seasonal(seasonal)

    stars = stars_and_length(df)
    mismatch = mismatch_examples(df)
    tests = agreement_and_tests(df, yearly)
    write_summary(df, yearly, seasonal, stars, mismatch, tests)
    print(f"Done in {time.time() - start:.0f} s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
