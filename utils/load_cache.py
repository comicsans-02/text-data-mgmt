"""Load the saved Yelp Arrow datasets."""

from pathlib import Path

from datasets import Dataset, load_from_disk


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"


def load_businesses() -> Dataset:
    """Load the complete business dataset."""

    path = CACHE_DIR / "businesses"
    if not path.is_dir():
        raise FileNotFoundError(f"Business cache not found: {path}")
    return load_from_disk(str(path))


def load_reviews() -> Dataset:
    """Load the complete review dataset."""

    path = CACHE_DIR / "reviews"
    if not path.is_dir():
        raise FileNotFoundError(f"Review cache not found: {path}")
    return load_from_disk(str(path))


def main() -> None:
    businesses = load_businesses()
    reviews = load_reviews()

    print(f"Businesses: {businesses.num_rows:,} rows")
    print(f"Reviews: {reviews.num_rows:,} rows")


if __name__ == "__main__":
    main()

