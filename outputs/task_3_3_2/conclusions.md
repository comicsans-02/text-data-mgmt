# Task 3.3 conclusions

## Index (3.3.1)

The inverted index covers the same reproducible 10% review sample as task
3.2.2 (seed 42) and uses the task 3.2.2 custom tokenizer, so search terms and
the tokenizer analysis share one vocabulary.

- Documents: 699,028 reviews (3 contain no tokens)
- Vocabulary: 245,428 terms
- Postings: 49,209,004 (term, document, term frequency) entries
- Tokens: 73,564,962; average document length 105.24 tokens
- Build time: about 1.1 minutes with 8 tokenizer processes; 294 MB on disk

Postings are stored in compressed-sparse-row form: one `offsets` array
indexed by term ID points into flat `doc_ids` and `tfs` arrays. Each posting
list is sorted by document ID, so Boolean AND is a sorted-array intersection.
Document lengths, the global average length (for BM25), and each document's
TF-IDF vector norm (for cosine) are all computed once at indexing time.

## Ranking (3.3.2)

Both formulas are implemented from scratch with NumPy in
`tasks/task_3_3_2.py`; no ranking library is used.

- **TF-IDF cosine:** weight = (1 + ln tf) × ln(N / df) for both query and
  document; score = dot product / (|q| × |d|).
- **BM25:** idf = ln(1 + (N − df + 0.5) / (df + 0.5));
  score = Σ idf × tf(k1 + 1) / (tf + k1(1 − b + b·|d| / avgdl)), with
  k1 = 1.2 and b = 0.75. The +1 inside the logarithm keeps IDF positive for
  very common terms.

**Correctness check.** For every query, each top-10 score was recomputed
directly from the re-tokenized raw review with a plain, unvectorised version
of the formula. The largest difference was below 5 × 10⁻⁵, which equals the
4-decimal rounding of the stored scores.

## Query analysis

Eight queries cover a rare single term, a single sentiment term, short and long
multi-word needs, a hyphenated token, and a currency expression. The average
length of all indexed reviews is 105 tokens.

| Query | Docs with any / all terms | Mean top-10 length: TF-IDF / BM25 / BM25 b=0 | TF-IDF ∩ BM25 (of 10) |
|---|---|---|---|
| pho | 4,539 / 4,539 | 21 / 57 / 436 | 1 |
| terrible | 14,687 / 14,687 | 17 / 37 / 253 | 2 |
| dog friendly patio | 112,295 / 284 | 27 / 28 / 278 | 6 |
| vegan gluten-free options | 32,339 / 69 | 17 / 42 / 250 | 3 |
| rude staff long wait | 185,928 / 137 | 18 / 37 / 454 | 3 |
| romantic dinner for a date night with great wine | 643,681 / 7 | 23 / 43 / 499 | 1 |
| $5 happy hour | 154,861 / 1,100 | 32 / 60 / 311 | 4 |
| best sushi in philadelphia | 431,529 / 67 | 19 / 42 / 442 | 4 |

Full rankings are in `rankings.csv`, and per-query statistics are in
`query_summary.json`.

### Findings

1. **TF-IDF cosine strongly favours very short reviews.** Its top-10 results
   average 17–32 tokens, against a corpus average of 105. Cosine
   normalisation divides by the vector length of the whole document. A short
   review that is mostly query terms, such as "Good pho tastes good" (4
   tokens), therefore approaches a cosine of 1. These hits are on-topic but
   carry little information.
2. **BM25 returns longer, more informative reviews while staying on-topic.**
   With b = 0.75, top results average 28–60 tokens. For "pho", BM25 ranks
   reviews of dedicated pho restaurants that discuss broth and meat, while
   TF-IDF ranks one-line mentions. The two models agree on only 1–6 of the
   top-10 documents per query, so the choice of model changes the result
   list substantially.
3. **The b parameter controls length normalisation.** With b = 0 there is no
   length normalisation, and the top-10 average rises to 250–500 tokens: long
   essays that mention the term repeatedly. They usually share at most 1–2
   documents with standard BM25. This shows why the average document length
   must be computed globally at indexing time: it is the reference point that
   b scales against.
4. **k1 mainly matters for multi-term queries.** Raising k1 from 1.2 to 3.0
   left single-term rankings unchanged (10 of 10 shared) but changed up to 3 of
   10 results for longer queries. A larger k1 delays term-frequency
   saturation, so one heavily repeated term can outweigh coverage of several
   different query terms.
5. **Both models are bag-of-words, which has known limits.**
   - "dog friendly patio" ranks a 1-star complaint ("Their idea of dog
     friendly is tie your dog outside the patio!") near the top, because
     negation and sentiment are invisible to term matching.
   - "best sushi in philadelphia" matches reviews that merely contain the word
     "philadelphia". Location belongs to business metadata, not review text,
     so a field filter on `city` would serve this need better.
   - Stop words in the long query ("for", "a", "with") match 92% of documents
     but contribute almost nothing, because their IDF is near zero. This is
     the intended behaviour of IDF.
6. **Tokenizer choices carry through to retrieval.** "gluten-free" is a single
   token, so reviews that write "gluten free" without a hyphen do not match it.
   "$5" is indexed as `5` + `usd`. A review containing "$5" therefore matches
   both terms, but "5 dollars" matches only `5`.

## Comparison with Lucene

The same sampled reviews were indexed with Apache Lucene 10, through the
Anserini jar that ships with Pyserini (`tasks/task_3_3_2_lucene.py`). Each
review was passed to Lucene as custom-tokenizer output and indexed with a
whitespace analyzer, so both systems see identical terms. Queries are Boolean
OR queries with one clause per query token. Full results are in
`lucene_comparison.json` and `lucene_rankings.csv`.

| Query | Our BM25 vs Lucene `BM25Similarity`: shared / mean rank shift | Score ratio | Our TF-IDF vs Lucene `ClassicSimilarity`: shared |
|---|---|---|---|
| pho | 9 / 0.0 | 2.198 | 3 |
| terrible | 10 / 0.2 | 2.197 | 4 |
| dog friendly patio | 10 / 0.0 | 2.200 | 5 |
| vegan gluten-free options | 10 / 0.0 | 2.193 | 6 |
| rude staff long wait | 10 / 0.2 | 2.197 | 3 |
| romantic dinner for a date night with great wine | 10 / 0.6 | 2.193 | 4 |
| $5 happy hour | 10 / 0.2 | 2.195 | 3 |
| best sushi in philadelphia | 10 / 0.0 | 2.197 | 5 |

1. **BM25 agrees almost exactly.** The top-10 lists share 9–10 documents for
   every query, and three queries have an identical order. Every score ratio is
   about 2.2, which equals k1 + 1. Lucene omits the (k1 + 1) numerator factor
   because it is constant across documents and cannot change the ranking.
2. **The remaining BM25 differences come from Lucene's length encoding.**
   Lucene stores each document length in a single byte (`SmallFloat`), so
   lengths are approximate. Near-tied documents can therefore swap places, and
   for "pho" one tenth-place document is replaced. Our implementation uses
   exact lengths.
3. **TF-IDF agrees much less, by design.** Lucene's `ClassicSimilarity` is not
   cosine similarity. It uses √tf, idf = 1 + ln((N + 1) / (df + 1)), and a
   1/√length norm, and it never divides by the full document vector length.
   It therefore rewards short documents less extremely than our true cosine
   does. The two lists share only 3–6 of their top-10 documents.

## Limitations

- **Sample size:** the index covers a 10% sample. The same code indexes the
  full corpus with `--fraction 1.0`. That needs roughly 10× the memory (about
  490 million postings), which is near the limit of a 16 GB machine.
- **No relevance judgements:** the comparison is qualitative and relies on
  overlap statistics. No relevance labels exist for Yelp, so precision and
  NDCG cannot be computed.
