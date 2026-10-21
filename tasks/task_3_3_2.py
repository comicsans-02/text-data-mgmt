"""Rank reviews with from-scratch TF-IDF cosine and BM25 for task 3.3.2."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tasks.task_3_2_2 import custom_tokenize
from tasks.task_3_3_1 import INDEX_DIR
from utils.load_cache import load_businesses, load_reviews


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task_3_3_2"
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
TOP_K = 10
SNIPPET_CHARACTERS = 200

# Diverse by design: a single rare term, a single sentiment term, short and
# long multi-word needs, a phrase the custom tokenizer keeps as one hyphenated
# token, and a currency query that relies on the tokenizer's "$5" handling.
ANALYSIS_QUERIES = [
    "pho",
    "terrible",
    "dog friendly patio",
    "vegan gluten-free options",
    "rude staff long wait",
    "romantic dinner for a date night with great wine",
    "$5 happy hour",
    "best sushi in philadelphia",
]


class SearchEngine:
    """Query-time view of the inverted index built by task 3.3.1."""

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        if not (index_dir / "postings.npz").is_file():
            raise FileNotFoundError(
                f"Index not found in {index_dir}. Run: python -m tasks.task_3_3_1"
            )
        arrays = np.load(index_dir / "postings.npz")
        self.offsets = arrays["offsets"]
        self.doc_ids = arrays["doc_ids"]
        self.tfs = arrays["tfs"].astype(np.float64)
        self.doc_lengths = arrays["doc_lengths"].astype(np.float64)
        self.doc_norms = arrays["doc_norms"].astype(np.float64)
        self.review_rows = arrays["review_rows"]
        with (index_dir / "vocabulary.json").open(encoding="utf-8") as file:
            terms = json.load(file)
        self.term_ids = {term: term_id for term_id, term in enumerate(terms)}
        with (index_dir / "stats.json").open(encoding="utf-8") as file:
            self.stats = json.load(file)

        self.num_docs = len(self.doc_lengths)
        self.average_length = self.stats["average_document_length"]

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a term's document IDs and term frequencies."""

        term_id = self.term_ids.get(term)
        if term_id is None:
            return np.empty(0, dtype=np.int32), np.empty(0)
        start, end = self.offsets[term_id], self.offsets[term_id + 1]
        return self.doc_ids[start:end], self.tfs[start:end]

    def document_frequency(self, term: str) -> int:
        term_id = self.term_ids.get(term)
        if term_id is None:
            return 0
        return int(self.offsets[term_id + 1] - self.offsets[term_id])

    def boolean_and(self, query: str) -> np.ndarray:
        """Return documents containing every query term."""

        result = None
        for term in set(custom_tokenize(query)):
            docs, _ = self.postings(term)
            result = docs if result is None else np.intersect1d(result, docs)
        return np.empty(0, dtype=np.int32) if result is None else result

    def tfidf_idf(self, term: str) -> float:
        return math.log(self.num_docs / self.document_frequency(term))

    def bm25_idf(self, term: str) -> float:
        # The +1 inside the log keeps IDF positive for terms in over half of
        # the documents, avoiding negative contributions from very common terms.
        df = self.document_frequency(term)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_tfidf(self, query: str) -> np.ndarray:
        """Cosine similarity between log-TF x IDF query and document vectors."""

        scores = np.zeros(self.num_docs)
        query_norm_squared = 0.0
        for term, query_tf in Counter(custom_tokenize(query)).items():
            docs, tfs = self.postings(term)
            if len(docs) == 0:
                continue
            idf = self.tfidf_idf(term)
            query_weight = (1.0 + math.log(query_tf)) * idf
            query_norm_squared += query_weight**2
            scores[docs] += query_weight * (1.0 + np.log(tfs)) * idf

        if query_norm_squared == 0.0:
            return scores
        matched = scores > 0
        scores[matched] /= self.doc_norms[matched] * math.sqrt(query_norm_squared)
        return scores

    def score_bm25(
        self, query: str, k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> np.ndarray:
        """Okapi BM25 using the average document length stored at indexing."""

        scores = np.zeros(self.num_docs)
        for term, query_tf in Counter(custom_tokenize(query)).items():
            docs, tfs = self.postings(term)
            if len(docs) == 0:
                continue
            length_ratio = self.doc_lengths[docs] / self.average_length
            saturation = tfs * (k1 + 1.0) / (tfs + k1 * (1.0 - b + b * length_ratio))
            # Repeated query terms count once per occurrence, as in Lucene.
            scores[docs] += query_tf * self.bm25_idf(term) * saturation
        return scores

    def search(
        self,
        query: str,
        k: int = TOP_K,
        method: str = "bm25",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> list[tuple[int, float]]:
        """Return the top-k (document ID, score) pairs, best first."""

        if method == "bm25":
            scores = self.score_bm25(query, k1, b)
        elif method == "tfidf":
            scores = self.score_tfidf(query)
        else:
            raise ValueError(f"Unknown ranking method: {method}")

        matched = np.flatnonzero(scores > 0)
        if len(matched) > k:
            matched = matched[np.argpartition(-scores[matched], k - 1)[:k]]
        # Ties break by document ID, so repeated runs give identical rankings.
        ranked = sorted(matched, key=lambda doc: (-scores[doc], doc))
        return [(int(doc), float(scores[doc])) for doc in ranked]


def describe_results(engine, results, reviews, businesses) -> list[dict]:
    """Attach review text and business metadata to ranked document IDs."""

    rows = [int(engine.review_rows[doc]) for doc, _ in results]
    records = reviews.select(rows) if rows else []
    described = []
    for rank, ((doc, score), record) in enumerate(zip(results, records), start=1):
        business = businesses.get(record["business_id"], {})
        snippet = " ".join(record["text"].split())
        described.append(
            {
                "rank": rank,
                "doc_id": doc,
                "score": round(score, 4),
                "review_id": record["review_id"],
                "stars": record["stars"],
                "length": int(engine.doc_lengths[doc]),
                "business": business.get("name", ""),
                "city": business.get("city", ""),
                "categories": business.get("categories") or "",
                "snippet": snippet[:SNIPPET_CHARACTERS],
                "text": record["text"],
            }
        )
    return described


def verify_scores(engine, query: str, described: list[dict], method: str) -> float:
    """Recompute scores directly from raw text and return the largest error.

    This checks the vectorised index path against a straightforward reading of
    each formula applied to the re-tokenized review.
    """

    query_counts = Counter(custom_tokenize(query))
    largest_error = 0.0
    for result in described:
        doc_counts = Counter(custom_tokenize(result["text"]))
        if method == "bm25":
            length = sum(doc_counts.values())
            expected = sum(
                query_tf
                * engine.bm25_idf(term)
                * doc_counts[term]
                * (DEFAULT_K1 + 1)
                / (
                    doc_counts[term]
                    + DEFAULT_K1
                    * (1 - DEFAULT_B + DEFAULT_B * length / engine.average_length)
                )
                for term, query_tf in query_counts.items()
                if doc_counts[term] and engine.document_frequency(term)
            )
        else:
            def weight(tf, term):
                return (1 + math.log(tf)) * engine.tfidf_idf(term)

            known = [t for t in query_counts if engine.document_frequency(t)]
            dot = sum(
                weight(query_counts[t], t) * weight(doc_counts[t], t)
                for t in known
                if doc_counts[t]
            )
            query_norm = math.sqrt(sum(weight(query_counts[t], t) ** 2 for t in known))
            doc_norm = math.sqrt(
                sum(weight(tf, t) ** 2 for t, tf in doc_counts.items())
            )
            expected = dot / (query_norm * doc_norm)
        largest_error = max(largest_error, abs(expected - result["score"]))
    return largest_error


def overlap_metrics(first: list[dict], second: list[dict]) -> dict:
    """Compare two top-k lists by shared documents and rank agreement."""

    first_ids = [result["doc_id"] for result in first]
    second_ids = [result["doc_id"] for result in second]
    shared = set(first_ids) & set(second_ids)
    union = set(first_ids) | set(second_ids)
    rank_shift = [
        abs(first_ids.index(doc) - second_ids.index(doc)) for doc in shared
    ]
    return {
        "shared": len(shared),
        "jaccard": round(len(shared) / len(union), 3) if union else 1.0,
        "mean_rank_shift": round(float(np.mean(rank_shift)), 2) if shared else None,
    }


def load_business_lookup() -> dict[str, dict]:
    businesses = load_businesses().select_columns(
        ["business_id", "name", "city", "categories"]
    )
    return {row["business_id"]: row for row in businesses}


def run_analysis(engine: SearchEngine) -> None:
    """Run the analysis queries and save rankings, metrics, and checks."""

    reviews = load_reviews().select_columns(
        ["review_id", "business_id", "stars", "text"]
    )
    businesses = load_business_lookup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configurations = {
        "tfidf": {"method": "tfidf"},
        "bm25": {"method": "bm25"},
        # b = 0 switches off length normalisation, isolating its effect.
        "bm25_b0": {"method": "bm25", "b": 0.0},
        # A large k1 weakens term-frequency saturation.
        "bm25_k1_3": {"method": "bm25", "k1": 3.0},
    }

    result_rows = []
    summary = []
    for query in ANALYSIS_QUERIES:
        query_tokens = custom_tokenize(query)
        rankings = {
            name: describe_results(
                engine, engine.search(query, **options), reviews, businesses
            )
            for name, options in configurations.items()
        }
        for name, described in rankings.items():
            for result in described:
                result_rows.append(
                    {"query": query, "configuration": name}
                    | {key: value for key, value in result.items() if key != "text"}
                )

        summary.append(
            {
                "query": query,
                "tokens": query_tokens,
                "document_frequency": {
                    token: engine.document_frequency(token) for token in query_tokens
                },
                "matching_any_term": int(
                    np.count_nonzero(engine.score_bm25(query) > 0)
                ),
                "matching_all_terms": len(engine.boolean_and(query)),
                "mean_result_length": {
                    name: round(
                        float(np.mean([r["length"] for r in described])), 1
                    )
                    for name, described in rankings.items()
                },
                "mean_result_stars": {
                    name: round(float(np.mean([r["stars"] for r in described])), 2)
                    for name, described in rankings.items()
                },
                "tfidf_vs_bm25": overlap_metrics(rankings["tfidf"], rankings["bm25"]),
                "bm25_vs_bm25_b0": overlap_metrics(
                    rankings["bm25"], rankings["bm25_b0"]
                ),
                "bm25_vs_bm25_k1_3": overlap_metrics(
                    rankings["bm25"], rankings["bm25_k1_3"]
                ),
                "max_score_error": {
                    "tfidf": verify_scores(
                        engine, query, rankings["tfidf"], "tfidf"
                    ),
                    "bm25": verify_scores(engine, query, rankings["bm25"], "bm25"),
                },
            }
        )
        print(f"Ranked: {query}", flush=True)

    with (OUTPUT_DIR / "rankings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    with (OUTPUT_DIR / "query_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "index": engine.stats,
                "parameters": {"k1": DEFAULT_K1, "b": DEFAULT_B, "top_k": TOP_K},
                "queries": summary,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved results to {OUTPUT_DIR}")


def print_results(engine: SearchEngine, query: str, method: str, k: int) -> None:
    reviews = load_reviews().select_columns(
        ["review_id", "business_id", "stars", "text"]
    )
    described = describe_results(
        engine, engine.search(query, k, method), reviews, load_business_lookup()
    )
    if not described:
        print("No matching reviews.")
    for result in described:
        print(
            f"{result['rank']:>2}. {result['score']:.4f}  "
            f"{result['business']} ({result['city']}) "
            f"{result['stars']:.0f}★, {result['length']} tokens"
        )
        print(f"    {result['snippet']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    search = subcommands.add_parser("search", help="run one query")
    search.add_argument("query")
    search.add_argument("--method", choices=["bm25", "tfidf"], default="bm25")
    search.add_argument("-k", type=int, default=TOP_K)
    subcommands.add_parser("analyze", help="run the report query set")
    args = parser.parse_args()

    engine = SearchEngine()
    if args.command == "search":
        print_results(engine, args.query, args.method, args.k)
    else:
        run_analysis(engine)


if __name__ == "__main__":
    main()
