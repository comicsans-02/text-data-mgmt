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

The high R² values show that the Yelp corpus closely follows both Zipf's Law
over the fitted rank range and Heaps' Law over the nested dataset samples.

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
