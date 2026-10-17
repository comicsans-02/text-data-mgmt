"""Summarise the Yelp business and review datasets for task 3.1."""

import json
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc

from utils.load_cache import load_businesses, load_reviews


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task_3_1"


def top_counts(values, limit: int = 10) -> list[dict]:
    """Return the most frequent values in an Arrow column."""

    counts = pc.value_counts(values).to_pylist()
    counts.sort(key=lambda item: item["counts"], reverse=True)
    return [
        {"value": item["values"], "count": item["counts"]}
        for item in counts[:limit]
    ]


def rating_counts(values) -> dict[str, int]:
    """Count records at each star rating, ordered from lowest to highest."""

    counts = pc.value_counts(values).to_pylist()
    counts.sort(key=lambda item: item["values"])
    return {str(item["values"]): item["counts"] for item in counts}


def main() -> None:
    businesses = load_businesses()
    reviews = load_reviews()
    business_table = businesses.data.table
    review_table = reviews.data.table

    category_counts = Counter()
    for categories in businesses["categories"]:
        if categories:
            category_counts.update(category.strip() for category in categories.split(","))

    text_lengths = pc.utf8_length(review_table["text"])
    quartiles = pc.quantile(text_lengths, q=[0.25, 0.50, 0.75, 0.95]).to_pylist()
    date_range = pc.min_max(review_table["date"]).as_py()

    business_required = ["business_id", "name", "categories"]
    review_required = ["review_id", "user_id", "business_id", "stars", "text", "date"]

    summary = {
        "businesses": {
            "rows": businesses.num_rows,
            "columns": {
                name: str(feature) for name, feature in businesses.features.items()
            },
            "missing_required_values": {
                name: businesses.num_rows - pc.count(business_table[name]).as_py()
                for name in business_required
            },
            "rating_counts": rating_counts(business_table["stars"]),
            "top_states": top_counts(business_table["state"], limit=10),
            "top_categories": [
                {"value": category, "count": count}
                for category, count in category_counts.most_common(15)
            ],
        },
        "reviews": {
            "rows": reviews.num_rows,
            "columns": {
                name: str(feature) for name, feature in reviews.features.items()
            },
            "missing_required_values": {
                name: reviews.num_rows - pc.count(review_table[name]).as_py()
                for name in review_required
            },
            "unique_users": pc.count_distinct(review_table["user_id"]).as_py(),
            "unique_businesses": pc.count_distinct(
                review_table["business_id"]
            ).as_py(),
            "date_range": {
                "first": date_range["min"].isoformat(sep=" "),
                "last": date_range["max"].isoformat(sep=" "),
            },
            "rating_counts": rating_counts(review_table["stars"]),
            "text_length_characters": {
                "mean": round(pc.mean(text_lengths).as_py(), 2),
                "minimum": pc.min(text_lengths).as_py(),
                "q25": quartiles[0],
                "median": quartiles[1],
                "q75": quartiles[2],
                "q95": quartiles[3],
                "maximum": pc.max(text_lengths).as_py(),
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dataset_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Businesses: {summary['businesses']['rows']:,}")
    print(f"Reviews: {summary['reviews']['rows']:,}")
    print(
        "Review dates: "
        f"{summary['reviews']['date_range']['first']} to "
        f"{summary['reviews']['date_range']['last']}"
    )
    print(f"Unique users: {summary['reviews']['unique_users']:,}")
    print(f"Unique reviewed businesses: {summary['reviews']['unique_businesses']:,}")
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()
