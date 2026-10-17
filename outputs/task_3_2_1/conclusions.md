# Task 3.2.1 conclusions

## Final result

The final analysis used NLTK `word_tokenize` on all 6,990,280 Yelp reviews.
Tokens were case-folded, punctuation-only tokens were removed, and isolated
single-letter alphabetic fragments were removed except `a` and `i`.

- Total tokens: 743,579,319
- Vocabulary size: 1,345,435
- Zipf exponent: 1.4358, with R² = 0.9892 over ranks 10–10,000
- Heaps parameters: k = 2.7870 and beta = 0.6404, with R² = 0.9999
- Predicted vocabulary after doubling the token count: 2,086,048

## Zipf's Law interpretation

Zipf's Law models term frequency as `f(r) = C × r^(-s)`, where `r` is a
term's frequency rank and `s` is the Zipf exponent. The fitted exponent of
1.4358 is the negative of the measured log-log slope, -1.4358. The resulting
R² of 0.9892 means that the fitted straight line explains 98.9% of the
variation in log frequency across ranks 10–10,000. The corpus therefore has a
very strong rank-frequency relationship over this range.

The exponent is higher than the textbook value of approximately 1.0. This
means frequency falls faster with rank in these processed Yelp reviews: a
tenfold increase in rank corresponds to approximately `10^1.4358 = 27.3`
times lower frequency under the fitted model. In practical terms, a relatively
small group of common words accounts for much of the corpus, followed by a
large tail of increasingly rare terms.

Ranks 1–9 were excluded because the very highest-frequency tokens form a
curved head rather than the approximately linear region used for the power-law
fit. Ranks beyond 10,000 were also excluded because the rare-token tail is more
strongly affected by discrete counts, names, spelling variants, and review
noise. The conclusion is therefore that Zipf's Law is a strong approximation
within the stated fitted range, not that every vocabulary rank follows one
perfect power law.

## Heaps' Law interpretation

Heaps' Law models vocabulary growth as `V(N) = k × N^beta`, where `N` is the
number of tokens and `V(N)` is the number of distinct tokens. The fitted
parameters `k = 2.7870` and `beta = 0.6404`, with R² = 0.9999 in log space,
closely describe the four nested random samples from 10% through 100% of the
review corpus.

Because beta is between 0 and 1, vocabulary growth is sublinear: additional
reviews continue to introduce unseen terms, but the number of new terms per
additional token gradually decreases. This is visible in the sample results.
Increasing the processed text from about 74.3 million tokens at 10% to 743.6
million at 100%—roughly ten times as much text—increased the vocabulary from
307,553 to 1,345,435 terms, or about 4.37 times.

The model predicts 2,086,048 distinct terms if the token count is doubled.
That is 740,613 more terms, or approximately 55% growth over the observed full
vocabulary, rather than a 100% increase. The prediction is an extrapolation
beyond the measured corpus and assumes future reviews have similar language
and preprocessing characteristics.

## Combined conclusion

The two results describe complementary properties of the same corpus. Zipf's
Law shows strong concentration among high-frequency terms and a long tail of
rare terms, while Heaps' Law shows that this tail keeps expanding as more text
is processed, although at a diminishing rate. Together, the high log-space R²
values provide strong empirical support for both laws in the Yelp reviews.

These numerical values depend on the chosen tokenization and normalization.
Case-folding merges capitalization variants, while retaining numbers and not
using stemming, lemmatization, or an `<UNK>` bucket preserves many distinct
surface forms. Those choices make the experiment reproducible and retain the
natural rare-token tail, but the fitted parameters should not be treated as
universal constants for every preprocessing method or text collection.

## Tokenizer and rare-token decision

A preliminary experiment compared WordPunctTokenizer and `word_tokenize` on a
reproducible random 10% sample. WordPunctTokenizer split contractions and noisy
punctuation too aggressively, whereas `word_tokenize` produced more useful word
boundaries after punctuation and stray-letter filtering.

A frequency threshold of 5 was also tested for mapping rare types to `<UNK>`:

| Tokenizer | Raw vocabulary | Rare types mapped | `<UNK>` frequency | `<UNK>` token share | Vocabulary after mapping | `<UNK>` rank |
|---|---:|---:|---:|---:|---:|---:|
| WordPunctTokenizer | 185,860 | 127,895 | 192,645 | 0.258% | 57,966 | 57 |
| `word_tokenize` | 307,592 | 241,392 | 334,840 | 0.450% | 66,201 | 31 |

Although the threshold removed many rare vocabulary types, those types made up
less than 0.5% of tokens. Combining them created an artificial high-frequency
`<UNK>` token at rank 57 or 31, distorting the frequency distribution that
Zipf's Law is intended to measure. The final analysis therefore uses
`word_tokenize` without an `<UNK>` bucket.
