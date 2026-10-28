"""Task 3.4.2: yearly PMI associations for one target aspect term.

Place in tasks/ and run from the repository root:
    python -m tasks.task_3_4_2 --reviews "path/to/review.json" --target service

Python 3.10+; standard library only. JSON Lines is streamed, not loaded at once.
Context = an entire review; repeated words count once per review.
PMI_y(t,w) = log2(N_y * df_y(t,w) / (df_y(t) * df_y(w))).
All four counts use the SAME year's corpus after optional filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A deliberately small, explicit function-word list; retain no/not/never.
# These words are excluded only as candidate associates, not from corpus counts.
STOPWORDS = set("""a an the and or but if while as at by for from in into of on
onto to with without about above below between through during before after
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs this that these those who whom which what whose
am is are was were be been being do does did doing have has had having
can could may might must shall should will would also just very so too
than then there here when where why how all any both each few more most
other some such own same only up down out off over under again once
it's i'm i've i'd i'll you're you've you'd you'll he's she's we're we've
we'd we'll they're they've they'd they'll that's there's""".split())
# Unicode alphabetic words; normalize curly apostrophes and split hyphens.
# No stemming/lemmatization: price and prices remain different terms.
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return set(WORD.findall(text.casefold().replace("’", "'")))


def json_rows(path: Path, limit: int | None = None):
    """Fail clearly on broken input rather than silently changing the corpus."""
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit is not None and line_number > limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}, line {line_number}: expected a JSON object")
            yield line_number, row


def business_filter(args) -> set[str] | None:
    if not args.category and not args.business_id:
        return None
    ids = set(args.business_id) if args.business_id else None
    if args.category:
        if args.businesses is None:
            raise ValueError("--category requires --businesses pointing to business JSON")
        wanted = {c.strip().casefold() for c in args.category}
        matches = set()
        for _, row in json_rows(args.businesses):
            categories = row.get("categories") or ""
            tags = {c.strip().casefold() for c in categories.split(",")}
            # Repeated --category options require ALL specified exact tags.
            if wanted <= tags and isinstance(row.get("business_id"), str):
                matches.add(row["business_id"])
        ids = matches if ids is None else ids & matches
    if not ids:
        raise ValueError("No businesses match the optional filters")
    return ids


def count_reviews(args, allowed):
    totals, targets = Counter(), Counter()
    frequencies, joint = defaultdict(Counter), defaultdict(Counter)
    audit = Counter()
    started = time.perf_counter()
    for line_number, row in json_rows(args.reviews, args.max_reviews):
        audit["rows_read"] += 1
        if audit["rows_read"] % 100_000 == 0:
            print(f"Read {audit['rows_read']:,} reviews ({time.perf_counter()-started:.0f}s)", flush=True)
        if allowed is not None and row.get("business_id") not in allowed:
            audit["business_filtered"] += 1
            continue
        try:
            year = datetime.fromisoformat(row["date"]).year
        except (KeyError, TypeError, ValueError):
            audit["invalid_date"] += 1
            continue
        if ((args.start_year is not None and year < args.start_year)
                or (args.end_year is not None and year > args.end_year)):
            audit["year_filtered"] += 1
            continue
        if not isinstance(row.get("text"), str):
            audit["invalid_text"] += 1
            continue
        words = tokenize(row["text"])
        # Even an empty review contributes to the review-level sample space.
        totals[year] += 1
        candidates = {w for w in words if len(w) >= 2 and w not in STOPWORDS
                      and w != args.target}
        frequencies[year].update(candidates)
        if args.target in words:
            targets[year] += 1
            joint[year].update(candidates)
    audit["included_reviews"] = sum(totals.values())
    return totals, targets, frequencies, joint, audit


def rank_words(total, target_count, frequencies, joint, minimum, top_n):
    if not target_count:
        return []
    scores = []
    for word, co_count in joint.items():
        if co_count < minimum:
            continue
        pmi = math.log2((co_count * total) / (target_count * frequencies[word]))
        scores.append({"word": word, "pmi": pmi,
                       "cooccurrence_reviews": co_count,
                       "word_reviews": frequencies[word]})
    # Deterministic ties; retain signed PMI (do not replace it with PPMI).
    scores.sort(key=lambda r: (-r["pmi"], -r["cooccurrence_reviews"], r["word"]))
    return scores[:top_n]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(args, totals, targets, frequencies, joint, audit):
    output = args.output_dir or ROOT / "outputs" / "task_3_4_2" / args.target
    output.mkdir(parents=True, exist_ok=True)
    top_rows, year_rows, rankings = [], [], {}
    # Include missing calendar years explicitly so gaps are not called annual drift.
    for year in range(min(totals), max(totals) + 1):
        ranked = rank_words(totals[year], targets[year], frequencies[year],
                            joint[year], args.min_cooccurrence, args.top_n)
        rankings[year] = ranked
        status = ("no_reviews" if not totals[year] else
                  "no_target" if not targets[year] else
                  "no_eligible_words" if not ranked else
                  "fewer_than_top_n" if len(ranked) < args.top_n else "ok")
        year_rows.append(dict(year=year, reviews=totals[year],
                              target_reviews=targets[year], returned_words=len(ranked), status=status))
        for rank, row in enumerate(ranked, 1):
            top_rows.append(dict(year=year, rank=rank, target=args.target, **row,
                                 target_reviews=targets[year], reviews=totals[year]))
    drift_rows = []
    for year in sorted(rankings)[1:]:
        previous = {r["word"] for r in rankings[year - 1]}
        current = {r["word"] for r in rankings[year]}
        valid = bool(previous and current)
        overlap = len(previous & current) / len(previous | current) if valid else None
        drift_rows.append(dict(from_year=year - 1, to_year=year,
                               jaccard=overlap if valid else "",
                               drift=1 - overlap if valid else "",
                               entered="; ".join(sorted(current - previous)) if valid else "",
                               exited="; ".join(sorted(previous - current)) if valid else "",
                               status="ok" if valid else "insufficient_data"))
    write_csv(output / "top_words_by_year.csv",
              ["year", "rank", "target", "word", "pmi", "cooccurrence_reviews",
               "word_reviews", "target_reviews", "reviews"], top_rows)
    write_csv(output / "yearly_counts.csv",
              ["year", "reviews", "target_reviews", "returned_words", "status"], year_rows)
    write_csv(output / "yearly_drift.csv",
              ["from_year", "to_year", "jaccard", "drift", "entered", "exited", "status"], drift_rows)
    metadata = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), audit=dict(audit),
                    context="review presence/absence", log_base=2,
                    tokenizer="casefold; normalize apostrophes; Unicode alphabetic words; split hyphens; no stemming",
                    stopwords=sorted(STOPWORDS), sampling="full file" if args.max_reviews is None else "file prefix: smoke test only",
                    review_file_bytes=args.reviews.stat().st_size)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Task 3.4.2: associations with '{args.target}'", "",
             "Context: both words occur anywhere in the same review, once per review.",
             "PMI = log2(cooccurrence_reviews * reviews / (target_reviews * word_reviews)).",
             f"Minimum joint reviews per word per year: {args.min_cooccurrence}. Top N: {args.top_n}.",
             "Positive PMI means more co-occurrence than expected under independence; it is not sentiment.", ""]
    if args.max_reviews is not None:
        lines += ["SMOKE TEST ONLY: uses the file prefix, which is not a representative random sample.", ""]
    for row in year_rows:
        y = row["year"]
        terms = ", ".join(f"{r['word']} ({r['pmi']:.3f})" for r in rankings[y])
        lines += [f"- {y}: {totals[y]:,} reviews; {targets[y]:,} with target. "
                  + (terms or f"No ranking ({row['status']}).")]
    lines += ["", "## Changes in the top-word sets", "",
              "Jaccard = intersection / union; drift = 1 - Jaccard. This ignores rank order.",
              "Changes below describe membership in the top-N lists, not first/last appearances in the corpus.", ""]
    for row in drift_rows:
        if row["status"] == "ok":
            lines.append(f"- {row['from_year']} to {row['to_year']}: drift {row['drift']:.3f}; "
                         f"entered: {row['entered'] or '(none)'}; exited: {row['exited'] or '(none)' }.")
        else:
            lines.append(f"- {row['from_year']} to {row['to_year']}: insufficient data for comparison.")
    lines += ["", "## Interpretation for the report", "",
              "Use the yearly lists to describe changes in meaning or themes, supported by specific words and PMI values. "
              "Do not infer sentiment or causation from PMI alone. Inspect review contexts before interpreting ambiguous words.",
              "PMI favors rare words; the minimum co-occurrence threshold reduces but does not eliminate this. "
              "Small yearly counts, partial years, changing business composition and the top-N cutoff can mimic drift. "
              "Review-level co-occurrence can join words from unrelated sentences. Inflections remain separate. "
              "This simple tokenizer differs from the team's task 3.2.2 custom tokenizer.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Included {sum(totals.values()):,} reviews; {sum(targets.values()):,} contain '{args.target}'.")
    print(f"Results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=Path, default=ROOT / "data/raw/yelp_academic_dataset_review.json")
    parser.add_argument("--target", default="service", help="One aspect word; default: service")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-cooccurrence", type=int, default=20, help="Minimum joint reviews per year (default: 20)")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--businesses", type=Path, help="Business JSON; required only with --category")
    parser.add_argument("--category", action="append", default=[], help="Optional exact Yelp category; repeat to require all tags")
    parser.add_argument("--business-id", action="append", default=[], help="Optional business ID; repeat to include multiple businesses")
    parser.add_argument("--max-reviews", type=int, help="Read only first N lines for a smoke test; omit for final results")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/task_3_4_2/<target>")
    args = parser.parse_args()
    args.target = args.target.strip().casefold().replace("’", "'")
    if not WORD.fullmatch(args.target) or len(args.target) < 2:
        parser.error("--target must be one alphabetic word (apostrophes allowed), at least two letters")
    if args.top_n < 1 or args.min_cooccurrence < 1 or (args.max_reviews is not None and args.max_reviews < 1):
        parser.error("--top-n, --min-cooccurrence and --max-reviews must be positive")
    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    try:
        counts = count_reviews(args, business_filter(args))
        if not counts[0]:
            raise ValueError("No valid reviews remain after filtering; check paths, years and business filters")
        save_results(args, *counts)
        if not sum(counts[1].values()):
            print("Warning: target was not found. Counts and empty result tables were saved.", file=sys.stderr)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
