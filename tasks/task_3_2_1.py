"""Validate Zipf's Law and Heaps' Law on the full Yelp review dataset."""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

from utils.load_cache import load_reviews


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task_3_2_1"
NLTK_DATA_DIR = ROOT / "data" / "nltk_data"
SAMPLE_FRACTIONS = (0.10, 0.30, 0.50, 1.00)
SHUFFLE_SEED = 42
FIT_MIN_RANK = 10
FIT_MAX_RANK = 10_000


def prepare_nltk_data() -> None:
    """Download the sentence tokenizer once into the project data directory."""

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        print("Downloading NLTK punkt_tab data...", flush=True)
        nltk.download(
            "punkt_tab",
            download_dir=str(NLTK_DATA_DIR),
            quiet=True,
            raise_on_error=True,
        )


def tokenize(text: str) -> list[str]:
    """Tokenize and discard punctuation and stray one-letter fragments."""

    tokens = []
    for token in word_tokenize(text):
        token = token.casefold()
        if not any(character.isalnum() for character in token):
            continue
        if len(token) == 1 and token.isalpha() and token not in {"a", "i"}:
            continue
        tokens.append(token)
    return tokens


def main() -> None:
    prepare_nltk_data()
    reviews = load_reviews().select_columns(["text"])
    total_reviews = reviews.num_rows
    cutoffs = {int(total_reviews * fraction): fraction for fraction in SAMPLE_FRACTIONS}

    # One exact permutation makes the growing samples random, nested, and repeatable.
    reviews = reviews.shuffle(seed=SHUFFLE_SEED, keep_in_memory=True)

    term_frequencies = Counter()
    total_tokens = 0
    processed_reviews = 0
    heaps_points = []
    started = time.perf_counter()

    for batch in reviews.iter(batch_size=2_000):
        for text in batch["text"]:
            tokens = tokenize(text)
            term_frequencies.update(tokens)
            total_tokens += len(tokens)
            processed_reviews += 1

            if processed_reviews in cutoffs:
                heaps_points.append(
                    {
                        "percentage": int(cutoffs[processed_reviews] * 100),
                        "reviews": processed_reviews,
                        "tokens": total_tokens,
                        "vocabulary": len(term_frequencies),
                    }
                )

        if processed_reviews % 250_000 < len(batch["text"]):
            elapsed = time.perf_counter() - started
            print(
                f"Processed {processed_reviews:,}/{total_reviews:,} reviews "
                f"in {elapsed / 60:.1f} minutes",
                flush=True,
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Zipf: frequency f(r) should be proportional to rank r raised to -s.
    frequencies = np.array(sorted(term_frequencies.values(), reverse=True))
    ranks = np.arange(1, len(frequencies) + 1)
    zipf_fit = (ranks >= FIT_MIN_RANK) & (ranks <= FIT_MAX_RANK)
    zipf_x = np.log10(ranks[zipf_fit])
    zipf_y = np.log10(frequencies[zipf_fit])
    zipf_slope, zipf_intercept = np.polyfit(zipf_x, zipf_y, 1)
    zipf_predictions = zipf_intercept + zipf_slope * zipf_x
    zipf_r_squared = 1 - np.sum((zipf_y - zipf_predictions) ** 2) / np.sum(
        (zipf_y - zipf_y.mean()) ** 2
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(ranks, frequencies, linewidth=0.8, label="Observed frequencies")
    fit_ranks = np.array([ranks[zipf_fit][0], ranks[zipf_fit][-1]])
    fit_frequencies = 10 ** (
        zipf_intercept + zipf_slope * np.log10(fit_ranks)
    )
    axis.loglog(
        fit_ranks,
        fit_frequencies,
        "--",
        label=f"Fitted slope = {zipf_slope:.3f}",
    )
    axis.set(title="Zipf's Law in Yelp Reviews", xlabel="Term rank", ylabel="Frequency")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "zipf_law.png", dpi=200)
    plt.close(figure)

    # Heaps: fit log(V) = log(k) + beta*log(N).
    heaps_tokens = np.array([point["tokens"] for point in heaps_points], dtype=float)
    heaps_vocabulary = np.array(
        [point["vocabulary"] for point in heaps_points], dtype=float
    )
    heaps_beta, heaps_log_k = np.polyfit(
        np.log(heaps_tokens), np.log(heaps_vocabulary), 1
    )
    heaps_predictions = heaps_log_k + heaps_beta * np.log(heaps_tokens)
    heaps_r_squared = 1 - np.sum(
        (np.log(heaps_vocabulary) - heaps_predictions) ** 2
    ) / np.sum(
        (np.log(heaps_vocabulary) - np.log(heaps_vocabulary).mean()) ** 2
    )
    heaps_k = float(np.exp(heaps_log_k))
    predicted_double_vocabulary = heaps_k * (2 * total_tokens) ** heaps_beta

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(heaps_tokens, heaps_vocabulary, "o", label="Observed vocabulary")
    fitted_tokens = np.logspace(
        np.log10(heaps_tokens.min()), np.log10(heaps_tokens.max()), 100
    )
    axis.loglog(
        fitted_tokens,
        heaps_k * fitted_tokens**heaps_beta,
        "--",
        label=f"V = {heaps_k:.2f}N^{heaps_beta:.3f}",
    )
    for point in heaps_points:
        axis.annotate(
            f"{point['percentage']}%",
            (point["tokens"], point["vocabulary"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set(
        title="Heaps' Law in Yelp Reviews",
        xlabel="Tokens (N)",
        ylabel="Vocabulary size (V)",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "heaps_law.png", dpi=200)
    plt.close(figure)

    with (OUTPUT_DIR / "top_terms.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "term", "frequency"])
        for rank, (term, frequency) in enumerate(
            term_frequencies.most_common(100), start=1
        ):
            writer.writerow([rank, term, frequency])

    with (OUTPUT_DIR / "heaps_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=heaps_points[0].keys())
        writer.writeheader()
        writer.writerows(heaps_points)

    metrics = {
        "tokenizer": "nltk.tokenize.word_tokenize",
        "normalization": (
            "case-folded; punctuation-only tokens removed; single-letter "
            "alphabetic tokens removed except a and i; numbers retained"
        ),
        "shuffle_seed": SHUFFLE_SEED,
        "reviews": total_reviews,
        "tokens": total_tokens,
        "vocabulary": len(term_frequencies),
        "zipf": {
            "slope": float(zipf_slope),
            "exponent": float(-zipf_slope),
            "r_squared": float(zipf_r_squared),
            "fit_minimum_rank": FIT_MIN_RANK,
            "fit_maximum_rank": FIT_MAX_RANK,
            "fitted_terms": int(zipf_fit.sum()),
        },
        "heaps": {
            "k": heaps_k,
            "beta": float(heaps_beta),
            "r_squared": float(heaps_r_squared),
            "points": heaps_points,
            "predicted_vocabulary_at_double_tokens": round(
                predicted_double_vocabulary
            ),
        },
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Tokens: {total_tokens:,}")
    print(f"Vocabulary: {len(term_frequencies):,}")
    print(f"Zipf slope: {zipf_slope:.4f} (R^2 = {zipf_r_squared:.4f})")
    print(
        f"Heaps parameters: k = {heaps_k:.4f}, beta = {heaps_beta:.4f} "
        f"(R^2 = {heaps_r_squared:.4f})"
    )
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
