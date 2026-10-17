# Task 3.2.2 conclusions

## Method

The comparison used a reproducible random 10% subset of the review dataset:
699,028 of 6,990,280 reviews, selected without replacement using seed 42. All
three tokenizers case-folded their output, but otherwise retained their own
token boundaries and rules. The qualitative comparison used 18 real Yelp
excerpts selected from a separate random pool within that subset; no examples
were written by hand. Email local parts were replaced with `redacted` only in
the displayed qualitative excerpts; quantitative counts used the original text.

## Vocabulary comparison

| Tokenizer | Total tokens | Vocabulary size |
|---|---:|---:|
| WordPunctTokenizer | 85,523,718 | 189,457 |
| NLTK word_tokenize | 83,808,479 | 307,903 |
| Custom regex | 73,564,962 | 245,428 |

The custom tokenizer produced 13.98% fewer tokens than WordPunctTokenizer and
12.22% fewer than `word_tokenize`. This is mainly because it removes ordinary
standalone punctuation and mentions, while preserving contractions and
alphabetic hyphen chains instead of breaking them into several tokens.

Its vocabulary was 29.54% larger than WordPunctTokenizer's but 20.29% smaller
than `word_tokenize`'s. WordPunctTokenizer aggressively breaks complex forms
into a small set of reusable pieces, which produces the smallest vocabulary.
The custom tokenizer keeps meaningful forms such as `don't`, `go-to`, dates,
and Unicode words intact, so it retains more types than WordPunctTokenizer. At
the same time, URL/email categorisation, repeated-letter normalisation, unit
separation, and removal of punctuation noise prevent the vocabulary from
growing as large as that of `word_tokenize`.

## Qualitative findings

The real-review excerpts show that the custom tokenizer:

- preserves contractions and possessives as complete tokens;
- preserves alphabetic hyphen chains but splits hyphens containing digits;
- emits `<URL>` plus the domain, and maps emails to `<EMAIL>` or
  `<EMAIL_OTHER_DOMAIN>`;
- preserves dates with two slashes while splitting other slash expressions;
- separates currencies, percentages, ordinals, and recognised attached units;
- reduces letter runs of three or more characters to two;
- preserves accented and other Unicode words;
- removes single punctuation marks but maps repeated `!`, `?`, or `.` runs to
  `<EXCLAMATION>`; and
- handles repeated spaces, tabs, and newlines without producing empty tokens.

These outputs are cleaner and more semantically useful for the selected Yelp
noise patterns than the fragmented WordPunct output. They also avoid some of
the mixed behaviour of `word_tokenize`, such as splitting contractions while
retaining other irregular compounds.

## Limitations

Vocabulary size alone does not establish that one tokenizer is universally
better. The custom rules deliberately trade general linguistic analysis for
consistent Yelp-oriented normalisation. In particular, regex cannot determine
whether an alphabetic chain such as `state-of-the-art` or `the-best-food` is a
conventional compound or a run-on phrase, so both are preserved. The unit and
email-provider lists are finite, and `<EXCLAMATION>` merges different kinds of
expressive punctuation into one category.

No pictographic emoji occurred in the 100,000-review qualitative pool, so the
tokenizer's grapheme-preserving emoji rule could not be compared qualitatively
without introducing a constructed example. No such example was substituted.
