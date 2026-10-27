# AI6122 Yelp Analysis

Python implementation for assignment tasks 3.1, 3.2.1, 3.2.2, 3.3.1, and 3.3.2.

## Setup

Use Python 3.12 and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

Run all commands below from the project root.

Download the Yelp Open Dataset after accepting Yelp's dataset agreement. Put
the two required files in `data/raw/` using their official names:

```text
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

The raw data and generated caches are excluded from Git.

Allow about 16 GB of free disk space:

| Item | Size |
|---|---|
| Raw JSON files in `data/raw/` | 5.5 GB |
| Arrow cache in `data/cache/` | 4.7 GB |
| Temporary copy in `~/.cache/huggingface/datasets/json/` | 4.6 GB |
| Search indexes in `data/index/` and `data/lucene_index/` | 0.5 GB |

The temporary Hugging Face copy is not used after `data/cache/` has been
built, so it can be deleted.

## Build the reusable Arrow cache

```powershell
python -m utils.build_cache
```

This reads each JSON Lines file and writes memory-mapped Hugging Face Arrow
datasets under `data/cache/businesses/` and `data/cache/reviews/`. The cache
retains every source column for reuse by the other assignment tasks.

If a cache already exists, delete its directory before intentionally rebuilding
it.

To verify an existing cache:

```powershell
python -m utils.load_cache
```

## Task 3.1: dataset summary

```powershell
python -m tasks.task_3_1
```

The report-ready summary is saved to
`outputs/task_3_1/dataset_summary.json`.

## Task 3.2.1: Zipf's Law and Heaps' Law

```powershell
python -m tasks.task_3_2_1
```

The command processes the complete review cache and saves plots, fitted
parameters, Heaps sample points, and the 100 most frequent terms under
`outputs/task_3_2_1/`. It uses NLTK `word_tokenize`, removes punctuation-only
tokens and stray one-letter alphabetic fragments, and does not create an
`<UNK>` bucket.

The tokenizer and rare-token decision is documented in
`outputs/task_3_2_1/conclusions.md`.

## Task 3.2.2: custom tokenizer

```powershell
python -m tasks.task_3_2_2
```

This compares WordPunct, NLTK `word_tokenize`, and the custom regex tokenizer
on a reproducible random 10% sample. It writes vocabulary counts, frequent
tokens, real-review qualitative examples, and report-ready conclusions under
`outputs/task_3_2_2/`. The completed run takes about 15 minutes on the original
development machine.

## Task 3.3.1: inverted index

```powershell
python -m tasks.task_3_3_1
```

This tokenizes the same reproducible 10% review sample as task 3.2.2 with the
custom tokenizer and writes the inverted index to `data/index/`. It takes
about a minute with 8 worker processes. Use `--fraction 1.0` to index every
review, which needs much more memory, and `--workers` to change the process
count.

## Task 3.3.2: TF-IDF cosine and BM25 ranking

```powershell
python -m tasks.task_3_3_2 analyze
python -m tasks.task_3_3_2 search "dog friendly patio" --method bm25 -k 10
```

`analyze` runs the report query set through TF-IDF cosine, BM25
(k1 = 1.2, b = 0.75), and two BM25 parameter variants. It also verifies each
top-10 score against a direct recomputation from the raw text. Rankings,
per-query statistics, and conclusions are saved under `outputs/task_3_3_2/`.
`search` prints ranked reviews for any query. For example,
`search "dog friendly patio" -k 2` prints:

```text
 1. 17.7054  Risky Business (Tucson) 1★, 81 tokens
    When I called they said they have a dog friendly patio, so we took our ...
 2. 16.8855  Boathouse at Hendry's Beach (Santa Barbara) 2★, 19 tokens
    Their idea of dog friendly is tie your dog outside the patio! Just not ...
```

Each result shows:

- its rank and score (BM25 scores are unbounded; TF-IDF cosine scores lie
  between 0 and 1)
- the reviewed business and its city
- the review's star rating and its length in custom-tokenizer tokens
- the first 200 characters of the review

Both top results are complaints. Bag-of-words ranking matches the query terms
but cannot see negation or sentiment.

Other modules can import `SearchEngine` from `tasks.task_3_3_2` and call
`engine.search(query, k, method)`.

### Optional: comparison with Lucene

This step needs Java 21 (`JAVA_HOME` set, or `java` on the `PATH`) and the
Lucene jar bundled with Pyserini. Install Pyserini without its dependencies,
which include several gigabytes of deep-learning packages that this step does
not use:

```powershell
python -m pip install --no-deps pyserini==2.4.0
python -m pip install pyjnius
python -m tasks.task_3_3_2_lucene build
python -m tasks.task_3_3_2_lucene compare
```

`build` indexes the same sample with Lucene under `data/lucene_index/` in
about 1.5 minutes. `compare` ranks the analysis queries with Lucene's
`BM25Similarity` and `ClassicSimilarity` and saves the comparison under
`outputs/task_3_3_2/`.

### Task 3.4.1: Lexicon-Based Sentiment Mining

Run from the project root:

```bash
python -m tasks.task_3_4_1
```

This task uses VADER to score raw review text. It analyses Fast Food, Italian, and Ice Cream & Frozen Yogurt reviews. Businesses listed in more than one of these sectors are excluded.

The script saves yearly and seasonal sentiment trends, checks of star ratings and review length, example failure cases, and `summary.txt` under `outputs/task_3_4_1/`. The VADER score cache is local and should not be uploaded.

### Task 3.5: Comparison Sentence Detection

Run from the project root:

```bash
python -m tasks.task_3_5 detect
python -m tasks.task_3_5 text "Way cheaper than the Starbucks next door."
```

This task finds sentences in which a reviewer compares the reviewed business with other businesses, for example "Better than Olive Garden" or "Unlike other sushi places, ...". It splits each review into sentences and scores each sentence with hand-written rules: explicit comparison phrases, "<comparative> than", scoped superlatives such as "best in town", and names of other businesses found in the business cache. A sentence with a score of 2 or more counts as a comparison. Comparisons with the reviewer's expectations ("better than I expected") or with the same business at another time ("better than last time") are not counted. For each comparison, the rules also say whether the reviewed business comes out better, worse, or similar.

`detect` reads the shared Arrow cache and runs on the same reproducible 10% review sample as tasks 3.2.2, 3.3.1, and 3.4.1 (`sample_rows`, seed 42), using 8 worker processes (`--workers`). It saves every detected sentence with its evidence to `comparisons.jsonl` and the statistics to `summary.json` under `outputs/task_3_5/`. `text` classifies sentences typed on the command line and prints the evidence; add `--business` to also recognise competitor names.

#### Evaluation

```bash
python -m tasks.task_3_5 sample
python -m tasks.task_3_5 evaluate
```

Comparison sentences are rare, so a plain random sample of a few hundred sentences would contain only a handful of positives and give an unreliable recall. `sample` therefore draws a stratified sample:

- It skips review rows 0 to 49,999, which were used to tune the rules, so the labelled set is separate from the tuning examples.
- From the rest of the shared 10% sample, it reads 50,000 random reviews (seed 42) and classifies every sentence of 5 to 60 words.
- It keeps 100 random sentences from each of three strata: sentences the detector flags, unflagged sentences that contain a comparison-like word such as "than", "other" or "best" (where missed comparisons are most likely), and all other sentences.

The 300 sentences are shuffled into `to_label.tsv` without their stratum, so labellers cannot tell which sentences the detector flagged. Group members fill the `label` column with 1 (the sentence compares the business with another business) or 0. `sample_key.json` records the stratum of each sentence and the size of each stratum. `sample` will not overwrite a `to_label.tsv` that already exists.

`evaluate` compares the labels with the detector's predictions. It reports precision, recall, and F1 on the labelled sentences themselves, and estimates for all sentences in which each stratum is weighted by its size. The weighted estimates have 95% bootstrap confidence intervals and are the figures to report. Results and the list of errors are saved to `outputs/task_3_5/evaluation.json`.
