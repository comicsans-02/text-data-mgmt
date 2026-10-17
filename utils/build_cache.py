"""Build reusable Arrow caches from the two Yelp JSON files."""

from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"


def build_cache(source: Path, destination: Path) -> None:
    """Convert one JSON Lines file to a saved Arrow dataset."""

    if not source.is_file():
        raise FileNotFoundError(f"Missing dataset file: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Cache already exists: {destination}. Delete it before rebuilding."
    )

    print(f"Reading {source.name}...")
    dataset = Dataset.from_json(str(source))
    dataset.save_to_disk(str(destination))
    print(f"Saved {dataset.num_rows:,} rows to {destination}")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    build_cache(
        RAW_DIR / "yelp_academic_dataset_business.json",
        CACHE_DIR / "businesses",
    )
    build_cache(
        RAW_DIR / "yelp_academic_dataset_review.json",
        CACHE_DIR / "reviews",
    )


if __name__ == "__main__":
    main()
