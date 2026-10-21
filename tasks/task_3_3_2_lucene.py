"""Compare the from-scratch rankers with Lucene's built-in similarities.

Lucene is reached through the Anserini jar bundled with Pyserini. Reviews are
handed to Lucene already tokenized by the custom tokenizer and indexed with a
whitespace analyzer, so both systems see identical terms and only the ranking
formulas differ.
"""

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tasks.task_3_2_2 import custom_tokenize
from tasks.task_3_3_2 import ANALYSIS_QUERIES, DEFAULT_B, DEFAULT_K1, OUTPUT_DIR
from tasks.task_3_3_2 import TOP_K, SearchEngine, overlap_metrics
from utils.load_cache import load_reviews


ROOT = Path(__file__).resolve().parents[1]
LUCENE_DIR = ROOT / "data" / "lucene_index"
BATCH_SIZE = 2_000
PROGRESS_INTERVAL = 100_000


def start_jvm():
    """Put the bundled Lucene jar on the classpath and return autoclass."""

    import pyserini

    jar_dir = Path(pyserini.__file__).parent / "resources" / "jars"
    jars = sorted(jar_dir.glob("anserini-*-fatjar.jar"))
    if not jars:
        raise FileNotFoundError(f"No Anserini jar found in {jar_dir}")

    import jnius_config

    if not jnius_config.vm_running:
        jnius_config.add_options("-Xmx4g")
        jnius_config.set_classpath(str(jars[-1]))
    from jnius import autoclass

    return autoclass


def tokenized_texts(texts: list[str]) -> list[str]:
    """Join each review's custom tokens with single spaces."""

    return [" ".join(custom_tokenize(text)) for text in texts]


def build_lucene_index(engine: SearchEngine, workers: int) -> None:
    """Index the same sampled reviews, in the same order, with Lucene."""

    if LUCENE_DIR.exists():
        raise FileExistsError(
            f"Lucene index already exists: {LUCENE_DIR}. Delete it to rebuild."
        )
    # Worker processes are forked before the JVM starts; forking a process
    # that already hosts a JVM is unsafe.
    pool = Pool(workers)
    autoclass = start_jvm()
    Paths = autoclass("java.nio.file.Paths")
    FSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
    WhitespaceAnalyzer = autoclass(
        "org.apache.lucene.analysis.core.WhitespaceAnalyzer"
    )
    IndexWriterConfig = autoclass("org.apache.lucene.index.IndexWriterConfig")
    IndexWriter = autoclass("org.apache.lucene.index.IndexWriter")
    Document = autoclass("org.apache.lucene.document.Document")
    TextField = autoclass("org.apache.lucene.document.TextField")
    StoredField = autoclass("org.apache.lucene.document.StoredField")
    Store = autoclass("org.apache.lucene.document.Field$Store")

    reviews = load_reviews().select_columns(["text"]).select(
        engine.review_rows.tolist()
    )
    directory = FSDirectory.open(Paths.get(str(LUCENE_DIR)))
    writer = IndexWriter(directory, IndexWriterConfig(WhitespaceAnalyzer()))

    started = time.perf_counter()
    doc_id = 0
    batches = (batch["text"] for batch in reviews.iter(batch_size=BATCH_SIZE))
    with pool:
        for texts in pool.imap(tokenized_texts, batches):
            for text in texts:
                document = Document()
                document.add(StoredField("doc_id", doc_id))
                document.add(TextField("contents", text, Store.NO))
                writer.addDocument(document)
                doc_id += 1
            if doc_id % PROGRESS_INTERVAL < len(texts):
                elapsed = (time.perf_counter() - started) / 60
                print(
                    f"Lucene indexed {doc_id:,}/{engine.num_docs:,} reviews "
                    f"in {elapsed:.1f} minutes",
                    flush=True,
                )
    writer.forceMerge(1)
    writer.close()
    print(f"Saved Lucene index to {LUCENE_DIR}")


class LuceneRanker:
    """Run bag-of-words queries against the Lucene index."""

    def __init__(self) -> None:
        if not LUCENE_DIR.is_dir():
            raise FileNotFoundError(
                f"Lucene index not found: {LUCENE_DIR}. "
                "Run: python -m tasks.task_3_3_2_lucene build"
            )
        autoclass = start_jvm()
        from jnius import cast

        Paths = autoclass("java.nio.file.Paths")
        FSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        DirectoryReader = autoclass("org.apache.lucene.index.DirectoryReader")
        IndexSearcher = autoclass("org.apache.lucene.search.IndexSearcher")
        BM25Similarity = autoclass(
            "org.apache.lucene.search.similarities.BM25Similarity"
        )
        ClassicSimilarity = autoclass(
            "org.apache.lucene.search.similarities.ClassicSimilarity"
        )
        self.BooleanQueryBuilder = autoclass(
            "org.apache.lucene.search.BooleanQuery$Builder"
        )
        self.Occur = autoclass("org.apache.lucene.search.BooleanClause$Occur")
        self.TermQuery = autoclass("org.apache.lucene.search.TermQuery")
        self.Term = autoclass("org.apache.lucene.index.Term")

        directory = FSDirectory.open(Paths.get(str(LUCENE_DIR)))
        # The cast selects IndexSearcher(IndexReader); pyjnius otherwise picks
        # the IndexReaderContext overload.
        reader = cast(
            "org.apache.lucene.index.IndexReader", DirectoryReader.open(directory)
        )
        self.searchers = {
            "bm25": IndexSearcher(reader),
            "tfidf": IndexSearcher(reader),
        }
        self.searchers["bm25"].setSimilarity(BM25Similarity(DEFAULT_K1, DEFAULT_B))
        self.searchers["tfidf"].setSimilarity(ClassicSimilarity())

    def search(self, query: str, k: int, method: str) -> list[tuple[int, float]]:
        searcher = self.searchers[method]

        # One SHOULD clause per query token mirrors our scorers, where a
        # repeated query term contributes once per occurrence.
        builder = self.BooleanQueryBuilder()
        for token in custom_tokenize(query):
            builder.add(self.TermQuery(self.Term("contents", token)), self.Occur.SHOULD)
        hits = searcher.search(builder.build(), k).scoreDocs
        stored = searcher.storedFields()
        return [
            (
                int(stored.document(hit.doc).getField("doc_id").numericValue().intValue()),
                float(hit.score),
            )
            for hit in hits
        ]


def compare(engine: SearchEngine) -> None:
    """Rank the analysis queries with both systems and save the comparison."""

    lucene = LuceneRanker()
    pairs = {"bm25": "BM25Similarity", "tfidf": "ClassicSimilarity"}
    rows = []
    summary = []
    for query in ANALYSIS_QUERIES:
        entry = {"query": query}
        for method, lucene_name in pairs.items():
            ours = engine.search(query, TOP_K, method)
            theirs = lucene.search(query, TOP_K, method)
            ours_ids = [{"doc_id": doc} for doc, _ in ours]
            theirs_ids = [{"doc_id": doc} for doc, _ in theirs]
            metrics = overlap_metrics(ours_ids, theirs_ids)
            metrics["same_order"] = [d for d, _ in ours] == [d for d, _ in theirs]
            if method == "bm25":
                # Lucene drops BM25's constant (k1 + 1) numerator factor, so the
                # ratio of scores for the same document should be k1 + 1.
                our_scores = dict(ours)
                ratios = [
                    our_scores[doc] / score
                    for doc, score in theirs
                    if doc in our_scores and score > 0
                ]
                metrics["mean_score_ratio"] = (
                    round(float(np.mean(ratios)), 4) if ratios else None
                )
            entry[f"{method}_vs_lucene_{lucene_name}"] = metrics

            for rank in range(max(len(ours), len(theirs))):
                rows.append(
                    {
                        "query": query,
                        "method": method,
                        "rank": rank + 1,
                        "ours_doc_id": ours[rank][0] if rank < len(ours) else "",
                        "ours_score": round(ours[rank][1], 4) if rank < len(ours) else "",
                        "lucene_doc_id": theirs[rank][0] if rank < len(theirs) else "",
                        "lucene_score": round(theirs[rank][1], 4)
                        if rank < len(theirs)
                        else "",
                    }
                )
        summary.append(entry)
        print(f"Compared: {query}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "lucene_rankings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "lucene_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build the Lucene index")
    build.add_argument("--workers", type=int, default=8)
    subcommands.add_parser("compare", help="compare rankings with Lucene")
    args = parser.parse_args()

    engine = SearchEngine()
    if args.command == "build":
        build_lucene_index(engine, args.workers)
    else:
        compare(engine)


if __name__ == "__main__":
    main()
