"""Compare a custom regex tokenizer with two standard NLTK tokenizers."""

import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import nltk
import regex
from nltk.tokenize import WordPunctTokenizer, word_tokenize

from utils.load_cache import load_reviews


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task_3_2_2"
NLTK_DATA_DIR = ROOT / "data" / "nltk_data"
SAMPLE_FRACTION = 0.10
SAMPLE_SEED = 42
QUALITATIVE_SEED = 43
PROGRESS_INTERVAL = 25_000
QUALITATIVE_CANDIDATES = 10
QUALITATIVE_POOL_SIZE = 100_000


# Personal email providers are grouped because their domains usually add little
# meaning. Business and organisation domains use a separate category.
COMMON_EMAIL_PROVIDERS = {
    "aol.com",
    "att.net",
    "comcast.net",
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "live.com",
    "me.com",
    "msn.com",
    "outlook.com",
    "proton.me",
    "protonmail.com",
    "sbcglobal.net",
    "verizon.net",
    "yahoo.com",
}

CURRENCY_NAMES = {"$": "usd", "€": "eur", "£": "gbp"}

# These suffixes can safely be interpreted as units rather than as parts of an
# arbitrary identifier such as "4Es".
UNIT_SUFFIXES = (
    "mins|hrs|secs|min|hr|sec|mph|kph|km|cm|mm|kg|mg|ml|cl|oz|"
    "lbs|lb|ft|gb|mb|am|pm|"
    "st|nd|rd|th|g|l|m|in"
)
NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
EMAIL_EXPRESSION = (
    r"[\p{L}\p{N}._%+\-]+@(?:[\p{L}\p{N}-]+\.)+[\p{L}]{2,}"
)
EMAIL_ADDRESS = regex.compile(EMAIL_EXPRESSION, regex.IGNORECASE)


TOKEN_PATTERN = regex.compile(
    rf"""
    (?P<URL>        (?:https?://|www\.)[^\s<>"']*[\p{{L}}\p{{N}}/#=_-])
  | (?P<EMAIL>      {EMAIL_EXPRESSION})
  | (?P<DATE>       (?<!\d)\d{{1,4}}/\d{{1,2}}/\d{{1,4}}(?!\d))
  | (?P<ACRONYM>    (?:\p{{L}}\.){{2,}})
  | (?P<CURRENCY>   [$€£]\s*{NUMBER})
  | (?P<PERCENT>    {NUMBER}%)
  | (?P<NUMBER_UNIT>{NUMBER}(?:{UNIT_SUFFIXES})(?![\p{{L}}\p{{N}}_]))
  | (?P<EMOJI>      (?=\p{{Extended_Pictographic}}|\p{{Regional_Indicator}}|[0-9#*]\uFE0F?\u20E3)\X)
  | (?P<HASHTAG>    \#[\p{{L}}\p{{N}}_]+)
  | (?P<MENTION>    @[\p{{L}}\p{{N}}_]+)
  | (?P<ALPHA_HYPHEN>
        \p{{L}}+(?:['’]\p{{L}}+)?
        (?:-\p{{L}}+(?:['’]\p{{L}}+)?)+
    )
  | (?P<CONTRACTION>\p{{L}}+(?:['’]\p{{L}}+)+)
  | (?P<ALPHANUMERIC>
        (?:\p{{L}}[\p{{L}}\p{{N}}_]*\d[\p{{L}}\p{{N}}_]*
        |\d[\p{{L}}\p{{N}}_]*\p{{L}}[\p{{L}}\p{{N}}_]*)
    )
  | (?P<NUMBER>     {NUMBER})
  | (?P<WORD>       \p{{L}}[\p{{L}}\p{{M}}]*)
  | (?P<EXCLAMATION>[!?.]{{2,}})
    """,
    regex.IGNORECASE | regex.VERBOSE,
)

REPEATED_LETTERS = regex.compile(r"(\p{L})\1{2,}")
EMOJI = regex.compile(
    r"\p{Extended_Pictographic}|\p{Regional_Indicator}|[0-9#*]\uFE0F?\u20E3"
)
DISPLAY_EMOJI = regex.compile(
    r"\p{Emoji_Presentation}|\p{Regional_Indicator}|"
    r"[0-9#*]\uFE0F?\u20E3|\p{Extended_Pictographic}\uFE0F"
)

RANDOM_QUALITATIVE_REVIEWS = 3
EXCERPT_CHARACTERS = 320

# Reservoir sampling selects random examples within useful feature strata.
QUALITATIVE_PATTERNS = {
    "Contractions and possessives": regex.compile(r"\b\p{L}+['’]\p{L}+\b"),
    "Alphabetic hyphens": regex.compile(r"\b\p{L}+(?:-\p{L}+)+\b"),
    "Hyphens containing digits": regex.compile(
        r"(?=[\p{L}\p{N}-]*\d)(?:[\p{L}\p{N}]+-)+[\p{L}\p{N}]+"
    ),
    "Abbreviations and acronyms": regex.compile(
        r"\b(?:dr|mr|mrs|ms|prof|sr|jr)\.|(?:\b\p{L}\.){2,}", regex.I
    ),
    "URLs": regex.compile(r"(?:https?://|www\.)[^\s<>\"']+", regex.I),
    "Emails": EMAIL_ADDRESS,
    "Hashtags or mentions": regex.compile(
        r"(?<![\p{L}\p{N}._%+\-])[@#][\p{L}\p{N}_]+", regex.I
    ),
    "Currency or percentages": regex.compile(rf"[$€£]\s*{NUMBER}|{NUMBER}%"),
    "Numbers with attached units": regex.compile(
        rf"{NUMBER}(?:{UNIT_SUFFIXES})\b",
        regex.I,
    ),
    "Dates": regex.compile(r"\b\d{1,4}/\d{1,2}/\d{1,4}\b"),
    "Non-date slashes": regex.compile(
        r"\b(?:\p{L}+/\p{L}+|\d+/\d+)\b"
    ),
    "Expressive punctuation": regex.compile(r"[!?.]{2,}"),
    "Repeated letters": REPEATED_LETTERS,
    "Unicode words": regex.compile(r"\p{L}*[^\x00-\x7F\W]\p{L}*"),
    "Pictographic emoji graphemes": DISPLAY_EMOJI,
    "Irregular whitespace": regex.compile(r" {2,}|\t|\n"),
}


def normalize_word(token: str) -> str:
    """Case-fold a word and shorten letter runs of length three or more."""

    token = token.casefold().replace("’", "'")
    return REPEATED_LETTERS.sub(r"\1\1", token)


def normalize_number(token: str) -> str:
    """Remove thousands separators while preserving decimal points."""

    return token.replace(",", "")


def url_domain(token: str) -> str:
    """Return a URL's lowercase domain without a leading www."""

    if not token.casefold().startswith(("http://", "https://")):
        token = "http://" + token
    domain = urlsplit(token).hostname or ""
    domain = domain.casefold()
    return domain.removeprefix("www.")


def custom_tokenize(text: str) -> list[str]:
    """Tokenize noisy review text according to the agreed custom rules."""

    tokens = []

    for match in TOKEN_PATTERN.finditer(text):
        kind = match.lastgroup
        value = match.group()

        if kind == "URL":
            tokens.extend(["<URL>", url_domain(value)])

        elif kind == "EMAIL":
            domain = value.rsplit("@", 1)[1].casefold()
            category = (
                "<EMAIL>"
                if domain in COMMON_EMAIL_PROVIDERS
                else "<EMAIL_OTHER_DOMAIN>"
            )
            tokens.append(category)

        elif kind == "ACRONYM":
            tokens.append(value.replace(".", "").casefold())

        elif kind == "CURRENCY":
            symbol = value[0]
            amount = value[1:].strip()
            tokens.extend([normalize_number(amount), CURRENCY_NAMES[symbol]])

        elif kind == "PERCENT":
            tokens.extend([normalize_number(value[:-1]), "percent"])

        elif kind == "NUMBER_UNIT":
            parts = regex.fullmatch(rf"({NUMBER})({UNIT_SUFFIXES})", value, regex.I)
            if parts is None:  # The master expression guarantees this match.
                raise ValueError(f"Could not separate number and unit: {value}")
            tokens.extend(
                [normalize_number(parts.group(1)), parts.group(2).casefold()]
            )

        elif kind == "HASHTAG":
            tokens.append(normalize_word(value[1:]))

        elif kind == "MENTION":
            continue

        elif kind in {"ALPHA_HYPHEN", "CONTRACTION", "ALPHANUMERIC", "WORD"}:
            token = normalize_word(value)
            if len(token) == 1 and token.isascii() and token.isalpha():
                if token not in {"a", "i"}:
                    continue
            tokens.append(token)

        elif kind == "NUMBER":
            tokens.append(normalize_number(value))

        elif kind == "EXCLAMATION":
            tokens.append("<EXCLAMATION>")

        elif kind == "DATE":
            tokens.append(value)

        elif kind == "EMOJI":
            tokens.append(value)

    return tokens


def prepare_nltk_data() -> None:
    """Make NLTK's sentence-tokenizer data available locally."""

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        print("Downloading NLTK punkt_tab data...", flush=True)
        nltk.download(
            "punkt_tab",
            download_dir=str(NLTK_DATA_DIR),
            quiet=True,
            raise_on_error=True,
        )


def standard_tokens(tokens: list[str]) -> list[str]:
    """Case-fold standard-tokenizer output without changing its boundaries."""

    return [token.casefold() for token in tokens]


def redact_email_local_parts(text: str) -> str:
    """Hide email local parts in qualitative output while retaining domains."""

    return EMAIL_ADDRESS.sub(
        lambda match: "redacted@" + match.group().rsplit("@", 1)[1], text
    )


def review_excerpt(text: str, match: regex.Match | None = None) -> str:
    """Return a short raw excerpt, centred on a feature when supplied."""

    if len(text) <= EXCERPT_CHARACTERS:
        return text.strip()

    if match is None:
        start = 0
    else:
        midpoint = (match.start() + match.end()) // 2
        start = max(0, midpoint - EXCERPT_CHARACTERS // 2)

    end = min(len(text), start + EXCERPT_CHARACTERS)
    start = max(0, end - EXCERPT_CHARACTERS)

    if start > 0:
        next_space = text.find(" ", start)
        if 0 <= next_space < end:
            start = next_space + 1
    if end < len(text):
        previous_space = text.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space

    return text[start:end].strip()


def add_to_reservoir(
    candidates: list[dict],
    candidate: dict,
    items_seen: int,
    capacity: int,
    rng: random.Random,
) -> int:
    """Maintain a uniform random sample from an incoming stream."""

    items_seen += 1
    if len(candidates) < capacity:
        candidates.append(candidate)
    else:
        replacement = rng.randrange(items_seen)
        if replacement < capacity:
            candidates[replacement] = candidate
    return items_seen


def collect_qualitative_candidates(
    text: str,
    source_row: int,
    state: dict,
    rng: random.Random,
) -> None:
    """Reservoir-sample real reviews overall and within each feature stratum."""

    random_candidate = {"source_row": source_row, "text": review_excerpt(text)}
    state["random_seen"] = add_to_reservoir(
        state["random_candidates"],
        random_candidate,
        state["random_seen"],
        RANDOM_QUALITATIVE_REVIEWS,
        rng,
    )

    for label, pattern in QUALITATIVE_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        feature_candidate = {
            "source_row": source_row,
            "text": review_excerpt(text, match),
        }
        state["feature_seen"][label] = add_to_reservoir(
            state["feature_candidates"][label],
            feature_candidate,
            state["feature_seen"][label],
            QUALITATIVE_CANDIDATES,
            rng,
        )


def finalize_qualitative_examples(state: dict) -> tuple[list[dict], dict[str, dict]]:
    """Choose distinct reviews from the sampled candidate pools."""

    used_rows = set()
    random_examples = []
    for number, candidate in enumerate(state["random_candidates"], start=1):
        example = {
            **candidate,
            "label": f"Unrestricted random review {number}",
        }
        random_examples.append(example)
        used_rows.add(candidate["source_row"])

    feature_examples = {}
    for label in QUALITATIVE_PATTERNS:
        candidate = next(
            (
                item
                for item in state["feature_candidates"][label]
                if item["source_row"] not in used_rows
            ),
            None,
        )
        if candidate is not None:
            feature_examples[label] = {**candidate, "label": label}
            used_rows.add(candidate["source_row"])

    return random_examples, feature_examples


def write_qualitative_comparison(
    wordpunct: WordPunctTokenizer,
    random_examples: list[dict],
    feature_examples: dict[str, dict],
) -> None:
    """Save side-by-side output for real, stratified Yelp excerpts."""

    lines = [
        "# Qualitative tokenizer comparison",
        "",
        (
            "All outputs are case-folded. Beyond that shared step, each "
            "tokenizer retains its native behaviour."
        ),
        (
            f"A random 10% subset was selected with seed {SAMPLE_SEED}. "
            f"A seeded pool of {QUALITATIVE_POOL_SIZE:,} reviews and reservoir "
            f"sampling ({QUALITATIVE_SEED}) then selected three unrestricted "
            "reviews and one distinct review per feature stratum."
        ),
        (
            "Email local parts are replaced with 'redacted' in this qualitative "
            "display before all three tokenizers run. Quantitative counts use "
            "the original review text."
        ),
        "",
    ]

    ordered_examples = random_examples + [
        feature_examples[label]
        for label in QUALITATIVE_PATTERNS
        if label in feature_examples
    ]
    for example in ordered_examples:
        label = example["label"]
        text = redact_email_local_parts(example["text"])
        outputs = {
            "WordPunctTokenizer": standard_tokens(wordpunct.tokenize(text)),
            "word_tokenize": standard_tokens(word_tokenize(text)),
            "Custom regex": custom_tokenize(text),
        }
        lines.extend(
            [
                f"## {label}",
                "",
                f"Source dataset row: {example['source_row']:,}",
                "",
                f"Input: {json.dumps(text, ensure_ascii=False)}",
                "",
            ]
        )
        for name, tokens in outputs.items():
            rendered = json.dumps(tokens, ensure_ascii=False)
            lines.append(f"- {name}: `{rendered}`")
        lines.append("")

    missing_strata = [
        label for label in QUALITATIVE_PATTERNS if label not in feature_examples
    ]
    if missing_strata:
        lines.extend(
            [
                "## Sample limitation",
                "",
                (
                    "No matching review was found in the random qualitative "
                    f"pool of {QUALITATIVE_POOL_SIZE:,} reviews for: "
                    + ", ".join(missing_strata)
                    + ". No constructed example was substituted."
                ),
                "",
            ]
        )

    (OUTPUT_DIR / "qualitative_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def vocabulary_style(term_frequencies: Counter[str]) -> dict[str, list[dict]]:
    """Collect frequent examples that show the custom vocabulary's form."""

    examples = {
        "category_tokens": [],
        "contractions": [],
        "hyphenated_words": [],
        "dates": [],
        "letter_number_tokens": [],
        "unicode_words": [],
        "emoji_or_pictographic_symbols": [],
        "decimal_numbers": [],
    }

    for token, frequency in term_frequencies.most_common():
        matching_categories = []
        if token.startswith("<") and token.endswith(">"):
            matching_categories.append("category_tokens")
        if "'" in token:
            matching_categories.append("contractions")
        if "-" in token:
            matching_categories.append("hyphenated_words")
        if regex.fullmatch(r"\d{1,4}/\d{1,2}/\d{1,4}", token):
            matching_categories.append("dates")
        if any(character.isalpha() for character in token) and any(
            character.isdigit() for character in token
        ):
            matching_categories.append("letter_number_tokens")
        if any(character.isalpha() and not character.isascii() for character in token):
            matching_categories.append("unicode_words")
        if EMOJI.search(token):
            matching_categories.append("emoji_or_pictographic_symbols")
        if regex.fullmatch(r"\d+\.\d+", token):
            matching_categories.append("decimal_numbers")

        for category in matching_categories:
            if len(examples[category]) < 15:
                examples[category].append(
                    {"token": token, "frequency": frequency}
                )

        if all(len(values) >= 15 for values in examples.values()):
            break

    return examples


def main() -> None:
    """Run the tokenizer comparison on a repeatable random 10% sample."""

    sys.stdout.reconfigure(encoding="utf-8")
    prepare_nltk_data()

    all_reviews = load_reviews().select_columns(["text"])
    source_reviews = all_reviews.num_rows
    sample_size = int(source_reviews * SAMPLE_FRACTION)
    sample_rng = random.Random(SAMPLE_SEED)
    sample_indices = sorted(sample_rng.sample(range(source_reviews), sample_size))
    reviews = all_reviews.select(sample_indices)

    wordpunct = WordPunctTokenizer()
    tokenizers = {
        "wordpunct": lambda text: standard_tokens(wordpunct.tokenize(text)),
        "word_tokenize": lambda text: standard_tokens(word_tokenize(text)),
        "custom_regex": custom_tokenize,
    }
    term_frequencies = {name: Counter() for name in tokenizers}
    token_totals = Counter()
    qualitative_rng = random.Random(QUALITATIVE_SEED)
    qualitative_positions = set(
        qualitative_rng.sample(
            range(sample_size), min(QUALITATIVE_POOL_SIZE, sample_size)
        )
    )
    qualitative_state = {
        "random_candidates": [],
        "random_seen": 0,
        "feature_candidates": {
            label: [] for label in QUALITATIVE_PATTERNS
        },
        "feature_seen": Counter(),
    }

    started = time.perf_counter()
    processed = 0
    for batch in reviews.iter(batch_size=2_000):
        for text in batch["text"]:
            if processed in qualitative_positions:
                collect_qualitative_candidates(
                    text,
                    source_row=sample_indices[processed] + 1,
                    state=qualitative_state,
                    rng=qualitative_rng,
                )
            for name, tokenize in tokenizers.items():
                tokens = tokenize(text)
                term_frequencies[name].update(tokens)
                token_totals[name] += len(tokens)
            processed += 1

        if processed % PROGRESS_INTERVAL < len(batch["text"]):
            elapsed = (time.perf_counter() - started) / 60
            print(
                f"Processed {processed:,}/{sample_size:,} reviews "
                f"in {elapsed:.1f} minutes",
                flush=True,
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = [
        {
            "tokenizer": name,
            "reviews": sample_size,
            "tokens": token_totals[name],
            "vocabulary": len(term_frequencies[name]),
        }
        for name in tokenizers
    ]

    with (OUTPUT_DIR / "vocabulary_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    with (OUTPUT_DIR / "top_tokens.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["tokenizer", "rank", "token", "frequency"])
        for name, frequencies in term_frequencies.items():
            for rank, (token, frequency) in enumerate(
                frequencies.most_common(100), start=1
            ):
                writer.writerow([name, rank, token, frequency])

    custom_examples = vocabulary_style(term_frequencies["custom_regex"])
    (OUTPUT_DIR / "custom_vocabulary_examples.json").write_text(
        json.dumps(custom_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    random_examples, feature_examples = finalize_qualitative_examples(
        qualitative_state
    )
    write_qualitative_comparison(wordpunct, random_examples, feature_examples)

    metrics = {
        "sample_fraction": SAMPLE_FRACTION,
        "sample_seed": SAMPLE_SEED,
        "qualitative_seed": QUALITATIVE_SEED,
        "qualitative_pool_reviews": len(qualitative_positions),
        "source_reviews": source_reviews,
        "sample_reviews": sample_size,
        "shared_normalization": "case-folding only",
        "qualitative_email_redaction": (
            "email local parts replaced with 'redacted'; quantitative counts unchanged"
        ),
        "qualitative_examples": len(random_examples) + len(feature_examples),
        "missing_qualitative_strata": [
            label for label in QUALITATIVE_PATTERNS if label not in feature_examples
        ],
        "comparison": comparison,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    for result in comparison:
        print(
            f"{result['tokenizer']}: {result['tokens']:,} tokens, "
            f"{result['vocabulary']:,} unique"
        )
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
