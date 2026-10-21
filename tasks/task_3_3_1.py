"""Build a compressed-sparse inverted index over Yelp reviews for task 3.3.1."""

import argparse
import json
import random
import time
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tasks.task_3_2_2 import custom_tokenize
from utils.load_cache import load_reviews


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
OUTPUT_DIR = ROOT / "outputs" / "task_3_3_1"
DEFAULT_FRACTION = 0.10
SAMPLE_SEED = 42  # Same seed as task 3.2.2, so both tasks see the same reviews.
BATCH_SIZE = 2_000
PROGRESS_INTERVAL = 50_000


def term_counts(texts: list[str]) -> list[Counter[str]]:
    """Tokenize a batch of reviews into per-document term counts."""

    return [Counter(custom_tokenize(text)) for text in texts]


def sample_rows(total: int, fraction: float) -> list[int]:
    """Return sorted review rows for a reproducible random sample."""

    if fraction >= 1.0:
        return list(range(total))
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(range(total), int(total * fraction)))


def build_index(fraction: float, workers: int) -> dict:
    """Tokenize the sampled reviews and write the inverted index to disk."""

    all_reviews = load_reviews().select_columns(["text"])
    rows = sample_rows(all_reviews.num_rows, fraction)
    reviews = all_reviews.select(rows)
    num_docs = len(rows)

    # Postings are first collected in document order as parallel flat arrays,
    # then regrouped by term. This avoids millions of small Python lists.
    vocabulary: dict[str, int] = {}
    posting_terms = array("i")
    posting_docs = array("i")
    posting_tfs = array("i")
    doc_lengths = np.zeros(num_docs, dtype=np.int32)

    started = time.perf_counter()
    batches = (batch["text"] for batch in reviews.iter(batch_size=BATCH_SIZE))
    doc_id = 0
    with Pool(workers) as pool:
        for counts_batch in pool.imap(term_counts, batches):
            for counts in counts_batch:
                doc_lengths[doc_id] = sum(counts.values())
                for term, tf in counts.items():
                    term_id = vocabulary.setdefault(term, len(vocabulary))
                    posting_terms.append(term_id)
                    posting_docs.append(doc_id)
                    posting_tfs.append(tf)
                doc_id += 1

            if doc_id % PROGRESS_INTERVAL < len(counts_batch):
                elapsed = (time.perf_counter() - started) / 60
                print(
                    f"Indexed {doc_id:,}/{num_docs:,} reviews "
                    f"in {elapsed:.1f} minutes",
                    flush=True,
                )

    terms = np.frombuffer(posting_terms, dtype=np.int32)
    # A stable sort keeps each posting list ordered by document ID.
    order = np.argsort(terms, kind="stable")
    postings_doc = np.frombuffer(posting_docs, dtype=np.int32)[order]
    postings_tf = np.frombuffer(posting_tfs, dtype=np.int32)[order]
    document_frequency = np.bincount(terms, minlength=len(vocabulary))
    offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
    np.cumsum(document_frequency, out=offsets[1:])
    del terms, order, posting_terms, posting_docs, posting_tfs

    # The average document length is a global statistic fixed at indexing time,
    # as BM25 requires.
    average_length = float(doc_lengths.mean())

    # Cosine normalisation needs each document's TF-IDF vector length. Storing
    # it now makes query-time cosine scoring touch only the query's postings.
    idf = np.log(num_docs / document_frequency)
    posting_idf = np.repeat(idf, document_frequency)
    weights = (1.0 + np.log(postings_tf)) * posting_idf
    doc_norms = np.sqrt(
        np.bincount(postings_doc, weights=weights**2, minlength=num_docs)
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_DIR / "postings.npz",
        offsets=offsets,
        doc_ids=postings_doc,
        tfs=postings_tf.astype(np.uint16, copy=False)
        if postings_tf.max() <= np.iinfo(np.uint16).max
        else postings_tf,
        doc_lengths=doc_lengths,
        doc_norms=doc_norms.astype(np.float32),
        review_rows=np.asarray(rows, dtype=np.int32),
    )
    terms_by_id = sorted(vocabulary, key=vocabulary.get)
    with (INDEX_DIR / "vocabulary.json").open("w", encoding="utf-8") as file:
        json.dump(terms_by_id, file, ensure_ascii=False)

    stats = {
        "sample_fraction": fraction,
        "sample_seed": SAMPLE_SEED,
        "documents": num_docs,
        "vocabulary_size": len(vocabulary),
        "postings": int(offsets[-1]),
        "total_tokens": int(doc_lengths.sum()),
        "average_document_length": average_length,
        "empty_documents": int((doc_lengths == 0).sum()),
        "build_minutes": round((time.perf_counter() - started) / 60, 2),
        "tokenizer": "custom_tokenize from task 3.2.2",
    }
    with (INDEX_DIR / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "index_stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="share of reviews to index (default: 0.10; 1.0 indexes all)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stats = build_index(args.fraction, args.workers)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
