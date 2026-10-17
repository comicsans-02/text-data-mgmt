# AI6122 Yelp Analysis

Python implementation for assignment tasks 3.1, 3.2.1, and 3.2.2.

## Setup

Use Python 3.12 and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run all commands below from the project root.

Download the Yelp Open Dataset after accepting Yelp's dataset agreement. Put
the two required files in `data/raw/` using their official names:

```text
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

The raw data and generated caches are excluded from Git.

## Build the reusable Arrow cache

```powershell
python -m utils.build_cache
```

This reads each JSON Lines file and writes memory-mapped Hugging Face Arrow
datasets under `data/cache/businesses/` and `data/cache/reviews/`. The cache
retains every source column for reuse by the other assignment tasks.

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

If a cache already exists, delete its directory before intentionally rebuilding
it.
