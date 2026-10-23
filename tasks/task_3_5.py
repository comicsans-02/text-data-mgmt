"""
Comparison Sentence Detector for Yelp Reviews   (AI6122 Assignment, Section 3.5)
=============================================================================

PURPOSE
    In some reviews the reviewer COMPARES the business being reviewed with other
    businesses they have visited ("Better than Olive Garden", "Unlike other sushi
    places...").  This tool finds those sentences, says why they were flagged,
    says whether the reviewed business comes out better / worse / similar, and
    lists which other businesses are mentioned.

METHOD  (rule-based NLP, standard library only, no deep learning)
    1. Split each review into sentences.
    2. For each sentence, add up evidence points from pattern rules:
         - explicit comparison phrases   ("compared to", "unlike", "not as good as") -> +2
         - comparative construction      ("cheaper than", "more expensive than")     -> +2
         - scoped superlatives           ("best in town", "best I've ever had")      -> +2
         - weaker superlative cues       ("hands down", "one of the best")          -> +1 each (max +2)
         - mention of ANOTHER business   (name found in business.json)              -> +1
    3. If total score >= threshold (default 2) the sentence is a "comparison".
    4. Infer the direction (better / worse / similar) from polarity words.

    Comparisons with the reviewer's expectations ("better than I expected") or with
    the same business at another time ("same as always", "better than last time")
    are NOT counted: only comparisons with OTHER businesses are.

SUB-COMMANDS
    text      classify sentences typed on the command line
    detect    run over review.json (+ business.json), write results and a summary
    sample    draw random REAL sentences from review.json for hand-labelling
    evaluate  precision / recall / F1 against that hand-labelled file

EXAMPLES  (run from the project root; the Yelp files are read from data/raw/)
    python -m tasks.task_3_5 text "Better than Olive Garden."
    python -m tasks.task_3_5 detect --limit 5000
    python -m tasks.task_3_5 sample --n 200
    python -m tasks.task_3_5 evaluate --gold outputs/task_3_5/to_label.tsv
"""
import argparse     # command-line interface
import collections  # Counter / defaultdict for statistics
import json         # the Yelp files are JSON-lines (one JSON object per line)
import re           # regular expressions: the core of the rule-based NLP
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "task_3_5"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"

# ==========================================================================
# 1. SENTENCE SPLITTING   (NLP step: sentence segmentation)
# ==========================================================================
# Split after . ! ? (optionally followed by closing quotes/brackets) when the
# next sentence starts with a capital letter or digit; also split at newlines.
# (?<=...) is a "look-behind" and (?=...) a "look-ahead": they test the text
# around the split point without consuming it.
_SENT_SPLIT = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])|\n+')

# Abbreviations ("Dr.", "vs.") end with a period but do NOT end a sentence.
_ABBREV = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|St|vs|Jr|Sr|No|etc|approx)\.$')


def split_sentences(text):
    """Split a review into a list of sentences using a regex plus an abbreviation guard."""
    parts, buf = [], ""
    for chunk in _SENT_SPLIT.split(text or ""):
        if not chunk or not chunk.strip():
            continue
        # Glue the chunk to any pending text left over from a false split.
        buf = (buf + " " + chunk).strip() if buf else chunk.strip()
        if _ABBREV.search(buf):
            continue            # e.g. "Dr." -> wrong split, keep accumulating
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return parts


# ==========================================================================
# 2. CUE LEXICONS / PATTERNS   (the "knowledge" of the rule-based system)
# ==========================================================================
def _rx(p):
    """Compile a case-insensitive regex."""
    return re.compile(p, re.I)


# Words right after "than" / "as" that mean the comparison is with an expectation
# or with the same business at another time, not with another business.
_SELF_AFTER = (r"i|we|you|expected|anticipated|thought|usual|always|before|ever|last|it|they|"
               r"average|advertised|necessary|needed|ideal|normal")

# (a) EXPLICIT comparison phrases.  One match is enough to add +2 to the score.
EXPLICIT = [
    _rx(r"\bcompar(?:ed|ing)\s+(?:to|with)\b"),                    # compared to / comparing with
    _rx(r"\bin\s+comparison\s+(?:to|with)\b"),                     # in comparison to
    _rx(r"\b(?:as\s+)?opposed\s+to\b"),                            # opposed to
    _rx(r"\bunlike\b"),                                            # unlike
    _rx(r"\b(?:versus|vs\.?)\s"),                                  # vs / versus
    _rx(r"\bon\s+par\s+with\b"),                                   # on par with
    _rx(r"\b(?:beats|beat|trumps|outshines|outclasses|blows\s+.{1,25}\s+out\s+of\s+the\s+water)\b"),
    # better/worse than ... but NOT "than I expected", "than ever", "than last time"
    _rx(r"\b(?:better|worse)\s+(?:than|then)\b(?!\s+(?:" + _SELF_AFTER + r")\b)"),
    _rx(r"\b(?:I(?:'ve|\s+have)\s+(?:been|had|eaten|tried|tasted)\s+(?:to\s+)?(?:much\s+|far\s+)?(?:better|worse))\b"),
    _rx(r"\bprefer\b.{0,40}\b(?:over|to|instead\s+of)\b"),         # I prefer X over Y
    _rx(r"\b(?:instead\s+of\s+going\s+to|rather\s+than\s+going\s+to|go\s+to\s+\w+\s+instead)\b"),
    _rx(r"\bnot\s+as\s+\w+\s+as\b(?!\s+(?:" + _SELF_AFTER + r")\b)"),     # not as good as
    _rx(r"\b(?:just\s+)?as\s+(?:good|great|bad|tasty|fresh|cheap|expensive|big|nice|clean)\s+as\b"
        r"(?!\s+(?:" + _SELF_AFTER + r")\b)"),
    _rx(r"\b(?:the\s+)?(?:same|similar)\s+(?:quality\s+)?(?:as|to)\b"
        r"(?!\s+(?:always|usual|before|ever|last|my\s+last|it|they)\b)"),
    _rx(r"\bbetter\s+(?:\w+\s+){0,2}(?:elsewhere|down\s+the\s+(?:street|road))\b"),
    # "other places ... better/cheaper/..." (other businesses + comparative word)
    _rx(r"\b(?:other|another)\s+(?:places?|restaurants?|spots?|shops?|stores?|salons?|hotels?|bars?|cafes?|"
        r"locations?|branches|joints?|dentists?|doctors?|mechanics?|gyms?)\b.{0,60}\b(?:better|worse|cheaper|"
        r"faster|slower|nicer|same|more|less)\b"),
    # "better than most other places"
    _rx(r"\b(?:better|worse|cheaper|nicer|cleaner|faster|slower|bigger|smaller|tastier|fresher)\b.{0,40}\b"
        r"(?:other|any\s+other|most|all\s+the)\s+(?:places?|restaurants?|spots?|\w+\s+(?:places?|restaurants?))\b"),
    _rx(r"\breminds?\s+me\s+of\b"),
    _rx(r"\b(?:like|similar\s+to)\s+(?:the\s+one|that\s+(?:place|restaurant)|a\s+(?:chain|chipotle|mcdonald'?s))\b"),
]

# (b) COMPARATIVE "<adjective> than".  Handled in code (_comparative_than) because
#     we must reject false positives such as "more than 2 hours".
#     group 1 = word(s) before "than" ("cheaper" or "more expensive"), group 2 = word after
_THAN = re.compile(r"\b((?:more|less)\s+\w+|\w+)\s+than\b\s*(\S+)?", re.I)
# Words before "than" that never signal a comparison between businesses.
_THAN_NEG_PREV = {"rather", "other", "later", "sooner", "no", "greater", "fewer", "earlier"}
# Nouns that merely end in -er and must not be taken for comparatives.
_ER_NOUNS = {"other", "rather", "water", "number", "dinner", "order", "customer", "server", "waiter",
             "never", "ever", "super", "after", "per", "under", "over", "burger", "manager", "owner",
             "bartender", "butter", "beer", "flower", "corner", "member", "together", "whether"}
# What counts as a comparative adjective: irregular forms, "more/less X", or any word ending in -er.
_COMP_ADJ = re.compile(r"^(?:better|worse|cheaper|nicer|cleaner|faster|slower|bigger|smaller|tastier|fresher|"
                       r"friendlier|quieter|louder|larger|tougher|softer|sweeter|saltier|spicier|greasier|"
                       r"hotter|colder|warmer|richer|thicker|thinner|healthier|greater|higher|lower|longer|"
                       r"shorter|easier|harder|stronger|weaker|prettier|safer|busier|more\s+\w+|less\s+\w+|"
                       r"\w{3,}er)$", re.I)

# (c) SCOPED SUPERLATIVES and personal-history cues.  The first two rules are worth +2,
#     the others are weaker evidence worth +1 each (max +2 from this group).
SUPERLATIVE = [
    # "best ... in town / in the area / in Las Vegas ..." -> ranks the business against all others nearby
    _rx(r"\b(?:best|worst|greatest|finest|cheapest|freshest)\b.{0,40}\b(?:in|of|around|within)\s+"
        r"(?:town|the\s+(?:city|area|state|neighbou?rhood|valley|strip|world)|\w+\s+county|"
        r"(?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))\b"),               # any capitalised place name
    _rx(r"\b(?:best|worst)\b.{0,40}\b(?:I(?:'ve|\s+have)\s+(?:ever\s+)?(?:had|been|tried|eaten|seen|visited))\b"),
    _rx(r"\b(?:hands\s+down|by\s+far|far\s+and\s+away)\b"),
    _rx(r"\bone\s+of\s+the\s+(?:best|worst)\b"),
    _rx(r"\b(?:other|every|any)\s+(?:place|restaurant|spot|shop|store|salon|hotel|bar|cafe)s?\s+(?:I|we)\b"),
    _rx(r"\b(?:my\s+)?(?:old|previous|usual|former|regular|other)\s+(?:place|restaurant|spot|dentist|doctor|"
        r"mechanic|salon|barber|gym|hotel)\b"),
    _rx(r"\bback\s+(?:home|in\s+\w+)\b.{0,40}\b(?:better|worse|same|cheaper|different)\b"),
]

# DIRECTION cues: used only AFTER a sentence has been accepted as a comparison.
BETTER_CUES = _rx(r"\b(?:better|best|beats?|superior|prefer|tastier|fresher|cheaper|nicer|cleaner|faster|"
                  r"friendlier|bigger\s+portions?|outshines|trumps|blows|hands\s+down|wins?|surpass)\b")
WORSE_CUES = _rx(r"\b(?:worse|worst|inferior|not\s+as\s+\w+\s+as|lacks?|overpriced|pricier|more\s+expensive|"
                 r"slower|dirtier|"
                 r"disappoint\w*|(?:don't|doesn't|didn't|can't|cannot|not)\s+(?:come\s+close|compare|measure|match|"
                 r"live\s+up)|go\s+elsewhere|better\s+(?:\w+\s+){0,2}elsewhere|"
                 r"I(?:'ve|\s+have)\s+(?:been|had)\s+(?:to\s+)?better)\b")
SIMILAR_CUES = _rx(r"\b(?:as\s+(?:good|great|bad|tasty|fresh|cheap|expensive)\s+as|on\s+par|(?:the\s+)?"
                   r"(?:same|similar)\s+(?:as|to)|reminds?\s+me|just\s+like|comparable)\b")

# ==========================================================================
# 3. OTHER-BUSINESS MENTION INDEX   (NLP step: named-entity lookup / gazetteer)
# ==========================================================================
# A "gazetteer" is a list of known names.  We build one from business.json so
# we can recognise when a review names a competitor.
_TOKEN = re.compile(r"[A-Za-z0-9&'\-]+")     # a word-like token (keeps &, ' and - inside names)
# One-word names that are also ordinary words -> too ambiguous, so not indexed.
_COMMON_SINGLE = set("""the a an and of cafe bar grill kitchen pizza sushi thai chinese mexican italian
burger burgers coffee bakery diner market deli house pub tavern restaurant salon spa gym hotel store shop
south north east west central downtown uptown ramen pho pumpkin tacos taco bagels bagel donuts donut
noodles noodle brunch breakfast lunch dinner dessert desserts cupcakes pancakes waffles wings bbq barbecue
steak seafood vegan organic fresh local express station garden village corner barber nails beauty
pharmacy dental clinic""".split())


class BusinessIndex:
    """Dictionary  lower-cased business name -> set of business_ids  plus metadata per business."""

    def __init__(self):
        self.names = collections.defaultdict(set)   # "olive garden" -> {"id1", "id2", ...}
        self.meta = {}                              # business_id -> (name, [categories], city)
        self.max_len = 1                            # longest indexed name in tokens (cap 6)
        self.cities = set()                         # city names, so "Philadelphia" is not a business

    def add(self, bid, name, categories, city):
        """Register one business from business.json."""
        self.meta[bid] = (name, [c.strip() for c in (categories or "").split(",") if c.strip()], city)
        if city:
            self.cities.add(city.strip().lower())
        toks = [t.lower() for t in _TOKEN.findall(name or "")]
        if not toks:
            return
        # Skip very short/ambiguous one-word names ("Joe", "Cafe", "Bar").
        if len(toks) == 1 and (len(toks[0]) < 5 or toks[0] in _COMMON_SINGLE):
            return
        self.names[" ".join(toks)].add(bid)
        self.max_len = max(self.max_len, min(len(toks), 6))

    def mentions(self, sentence, exclude_bid=None, exclude_name=None):
        """Return names of OTHER businesses mentioned in the sentence.

        Slides a window of up to max_len tokens over the sentence, longest first,
        and checks it against the name dictionary.  The first token must start with
        a capital letter (names are capitalised in text; this avoids matching
        ordinary words).  City names, one-word names that are capitalised only
        because they start the sentence, and the business being reviewed are skipped.
        """
        found = []
        raw = _TOKEN.findall(sentence)
        low = [t.lower() for t in raw]
        ex = (exclude_name or "").lower()
        i = 0
        while i < len(low):
            hit = None
            for n in range(min(self.max_len, len(low) - i), 0, -1):   # longest match first
                key = " ".join(low[i:i + n])
                if key not in self.names or not raw[i][0].isupper():
                    continue
                if key in self.cities:                  # a place, not a business
                    continue
                if n == 1 and i == 0:                   # capitalised only because it starts the sentence
                    continue
                # Skip the reviewed business itself.
                if key == ex or (exclude_bid and exclude_bid in self.names[key] and len(self.names[key]) == 1):
                    continue
                hit = (key, n)
                break
            if hit:
                found.append(hit[0])
                i += hit[1]          # jump past the matched name
            else:
                i += 1
        return found


# ==========================================================================
# 4. SENTENCE CLASSIFIER   (NLP step: rule-based text classification)
# ==========================================================================
def _comparative_than(sentence):
    """Return the matched '<adjective> than' text if it is a genuine comparison, else None.

    Rejects common false positives:
      'more than 2 hours', 'less than $10'   -> quantity, not comparison
      'other than', 'rather than'            -> not comparatives
      'longer than I expected'               -> compared with an expectation, not a business
      'busier than last time'                -> compared with the same business, not another one
    """
    for m in _THAN.finditer(sentence):
        prev, nxt = m.group(1).lower(), (m.group(2) or "")
        if prev in _THAN_NEG_PREV:
            continue
        # Compared with an expectation / the same business rather than another business.
        if re.match(r"^(?:" + _SELF_AFTER + r")\b", nxt, re.I):
            continue
        # "more/less/longer ... than <number>" is a quantity.
        if prev.split()[0] in {"more", "less", "fewer", "greater", "longer", "higher", "lower", "earlier",
                               "later"} and re.match(r"^\$?\d", nxt):
            continue
        # Must look like a comparative adjective, and not be a noun that merely ends in -er.
        if _COMP_ADJ.match(prev) and prev not in _ER_NOUNS:
            return m.group(0).strip()
    return None


def classify(sentence, index=None, bid=None, bname=None, threshold=2):
    """Classify ONE sentence.

    Returns a dict:
        is_comparison : bool   (score >= threshold)
        score         : int    total evidence points
        direction     : 'better'|'worse'|'similar'|'mixed'|'unspecified'|None
        evidence      : list of (cue type, matched text)  -> explains the decision
        others        : list of other-business names mentioned
    """
    score, evidence = 0, []

    # (a) explicit phrase: only the first matching phrase counts (+2)
    for rx in EXPLICIT:
        m = rx.search(sentence)
        if m:
            score += 2
            evidence.append(("explicit", m.group(0).strip()))
            break

    # (b) "<adjective> than": +2 on its own, +1 if an explicit phrase already matched
    t = _comparative_than(sentence)
    if t and not any(e[0] == "explicit" for e in evidence):
        score += 2
        evidence.append(("comparative", t))
    elif t:
        score += 1
        evidence.append(("comparative", t))

    # (c) scoped superlatives: the first two rules ("best ... in town", "best ... I've ever had")
    #     rank the business against all others, so they are worth +2; the rest +1.  Max +2 in total.
    sup = 0
    for i, rx in enumerate(SUPERLATIVE):
        m = rx.search(sentence)
        if m and sup < 2:
            pts = min(2 if i < 2 else 1, 2 - sup)
            sup += pts
            score += pts
            evidence.append(("scoped-superlative", m.group(0).strip()))

    # (d) another business named: +1 if other evidence exists, or a linking word is present
    others = index.mentions(sentence, bid, bname) if index else []
    if others and (score >= 1 or
                   re.search(r"\b(?:than|vs|versus|unlike|like|compared|instead|over)\b", sentence, re.I)):
        score += 1
        evidence.append(("other-business", ", ".join(others)))

    # Direction is only computed for accepted comparisons.
    direction = _direction(sentence) if score >= threshold else None
    return {"is_comparison": score >= threshold, "score": score, "direction": direction,
            "evidence": evidence, "others": others}


def _direction(s):
    """Decide whether the reviewed business comes out better, worse or similar."""
    # 'as good as', 'on par', 'same as' -> similar (unless a worse cue is present)
    if SIMILAR_CUES.search(s) and not WORSE_CUES.search(s):
        return "similar"
    w, b = bool(WORSE_CUES.search(s)), bool(BETTER_CUES.search(s))
    if w and not b:
        return "worse"
    if b and not w:
        # Negation flips polarity: "not better than", "isn't as good" -> worse
        if re.search(r"\b(?:not|n't|never)\s+(?:\w+\s+){0,2}(?:better|best|as\s+good)", s, re.I):
            return "worse"
        return "better"
    if w and b:
        # "better elsewhere" -> bad for the reviewed business
        return "worse" if re.search(r"elsewhere|than\s+(?:this|here)", s, re.I) else "mixed"
    return "unspecified"


# ==========================================================================
# 5. EVALUATION ON REAL YELP SENTENCES (labelled by the group, no invented data)
# ==========================================================================
def cmd_sample(args):
    """Draw random sentences from the REAL review.json so group members can label them.

    Output TSV columns:  review_id <TAB> label (empty: fill 1 or 0) <TAB> sentence
    A fixed random seed makes the sample reproducible.
    """
    import random
    rng = random.Random(args.seed)
    pool, seen = [], 0
    with open(args.review, encoding="utf-8") as f:
        for line in f:
            if args.limit and seen >= args.limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            for s in split_sentences(r.get("text", "")):
                if 5 <= len(s.split()) <= 60:          # skip fragments and run-on blocks
                    pool.append((r.get("review_id", ""), " ".join(s.split())))   # no tabs/newlines
    if not pool:
        sys.exit("No sentences found - check the --review file.")
    pick = rng.sample(pool, min(args.n, len(pool)))
    with open(args.out, "w", encoding="utf-8") as o:
        for rid, s in pick:
            o.write(f"{rid}\t\t{s}\n")
    print(f"Wrote {len(pick)} random sentences from {seen} reviews to {args.out}. "
          f"Fill column 2 with 1 (comparison) or 0 (not), then run 'evaluate'.")


def cmd_evaluate(args):
    """Compare the tool's predictions with the human labels and print precision/recall/F1.

    TP = comparison correctly found      FP = flagged but not a comparison
    FN = comparison the tool missed      TN = correctly ignored
    precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = harmonic mean of the two
    """
    tp = fp = fn = tn = 0
    errors = []
    # Use the same business index as 'detect', so the scores describe the same system.
    idx = load_businesses(args.business) if args.business else None
    with open(args.gold, encoding="utf-8-sig") as f:          # utf-8-sig: tolerates Excel's BOM
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or parts[1].strip() not in ("0", "1"):
                continue                                  # skip unlabelled rows
            y, s = int(parts[1].strip()), parts[2]
            p = int(classify(s, idx, threshold=args.threshold)["is_comparison"])
            if p and y:
                tp += 1
            elif p:
                fp += 1
                errors.append(("FP", s))
            elif y:
                fn += 1
                errors.append(("FN", s))
            else:
                tn += 1
    n = tp + fp + fn + tn
    if not n:
        sys.exit("No labelled rows found (column 2 must be 0 or 1).")
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"Labelled sentences: {n}  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  Accuracy={(tp + tn) / n:.3f}")
    for kind, s in errors[:args.show_errors]:
        print(f"  {kind}: {s}")


# ==========================================================================
# 6. COMMAND-LINE COMMANDS
# ==========================================================================
def cmd_text(args):
    """Classify sentences typed on the command line and show the evidence."""
    idx = load_businesses(args.business) if args.business else None
    for s in args.sentences:
        r = classify(s, idx, threshold=args.threshold)
        print(f"[{'COMPARISON' if r['is_comparison'] else 'no        '}] score={r['score']} "
              f"dir={r['direction']}  {s}")
        for kind, ev in r["evidence"]:
            print(f"      - {kind}: '{ev}'")


def load_businesses(path):
    """Read business.json (JSON-lines) into a BusinessIndex."""
    idx = BusinessIndex()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "business_id" in b:
                idx.add(b["business_id"], b.get("name", ""), b.get("categories"), b.get("city"))
    return idx


def cmd_detect(args):
    """Stream review.json, detect comparison sentences and print statistics.

    The review file is read line by line, so memory use stays small even for the full dataset.
    """
    idx = load_businesses(args.business)
    n_rev = n_sent = n_cmp = n_rev_cmp = 0            # counters
    dirs = collections.Counter()                      # direction distribution
    cat_tot, cat_cmp = collections.Counter(), collections.Counter()   # per-category review counts
    stars_all, stars_cmp = [], []                     # star ratings: all reviews / reviews with a comparison
    cited = collections.Counter()                     # most frequently compared-to businesses
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    shown = 0
    with open(args.review, encoding="utf-8") as f:
        for line in f:
            if args.limit and n_rev >= args.limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rev += 1
            bid = r.get("business_id")
            name, cats, city = idx.meta.get(bid, ("", [], ""))
            has = False                                # does this review contain >= 1 comparison?
            stars_all.append(r.get("stars", 0))
            for c in cats[:3]:                         # first three categories of the business
                cat_tot[c] += 1
            for s in split_sentences(r.get("text", "")):
                n_sent += 1
                res = classify(s, idx, bid, name, args.threshold)
                if not res["is_comparison"]:
                    continue
                has = True
                n_cmp += 1
                dirs[res["direction"]] += 1
                for o in res["others"]:
                    cited[o] += 1
                rec = {"review_id": r.get("review_id"), "business": name, "stars": r.get("stars"),
                       "sentence": s, "score": res["score"], "direction": res["direction"],
                       "evidence": res["evidence"], "other_businesses": res["others"]}
                if out:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if shown < args.show:                  # print a few examples to the screen
                    shown += 1
                    print(f"({r.get('stars')}★ {name}) [{res['direction']}, score {res['score']}] {s}")
            if has:
                n_rev_cmp += 1
                stars_cmp.append(r.get("stars", 0))
                for c in cats[:3]:
                    cat_cmp[c] += 1
    if out:
        out.close()

    # ---- summary statistics ----
    def avg(x):
        return sum(x) / len(x) if x else 0.0

    print("\n========== SUMMARY ==========")
    print(f"Reviews processed            : {n_rev}")
    print(f"Sentences processed          : {n_sent}")
    print(f"Comparison sentences         : {n_cmp} ({100 * n_cmp / max(n_sent, 1):.2f}% of sentences)")
    print(f"Reviews with >=1 comparison  : {n_rev_cmp} ({100 * n_rev_cmp / max(n_rev, 1):.2f}% of reviews)")
    print(f"Avg stars  all / with-comparison : {avg(stars_all):.2f} / {avg(stars_cmp):.2f}")
    print("Direction distribution       :", dict(dirs))
    print(f"\nTop categories by comparison rate (min {args.min_cat} reviews):")
    rates = [(c, cat_cmp[c] / t, t) for c, t in cat_tot.items() if t >= args.min_cat]
    for c, rt, t in sorted(rates, key=lambda x: -x[1])[:10]:
        print(f"  {c:<30} {100 * rt:5.1f}%  (n={t})")
    if cited:
        print("\nMost frequently compared-to businesses:")
        for k, v in cited.most_common(10):
            print(f"  {k:<30} {v}")
    if args.out:
        print(f"\nDetected sentences written to {args.out}")


def main():
    """Define the command-line interface and dispatch to the chosen sub-command."""
    # The Windows console cannot print "★" by default; force UTF-8 output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description="Comparison sentence detector for Yelp reviews")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("text", help="classify sentences given on the command line")
    t.add_argument("sentences", nargs="+")
    t.add_argument("--business", help="business.json, to also recognise named competitors")
    t.add_argument("--threshold", type=int, default=2)
    t.set_defaults(fn=cmd_text)

    d = sub.add_parser("detect", help="run on review.json")
    d.add_argument("--review", default=str(REVIEW_FILE))
    d.add_argument("--business", default=str(BUSINESS_FILE))
    d.add_argument("--limit", type=int, default=50000, help="max reviews (0 = all)")
    d.add_argument("--threshold", type=int, default=2)
    d.add_argument("--out", default=str(OUTPUT_DIR / "comparisons.jsonl"),
                   help="detected sentences as JSON lines (default: outputs/task_3_5/comparisons.jsonl)")
    d.add_argument("--show", type=int, default=15, help="number of examples printed")
    d.add_argument("--min-cat", type=int, default=30,
                   help="minimum reviews for a category to appear in the ranking")
    d.set_defaults(fn=cmd_detect)

    sm = sub.add_parser("sample", help="draw random real sentences from review.json to hand-label")
    sm.add_argument("--review", default=str(REVIEW_FILE))
    sm.add_argument("--n", type=int, default=200)
    sm.add_argument("--limit", type=int, default=50000, help="max reviews to read (0 = all)")
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--out", default=str(OUTPUT_DIR / "to_label.tsv"))
    sm.set_defaults(fn=cmd_sample)

    e = sub.add_parser("evaluate", help="P/R/F1 on a hand-labelled TSV made by 'sample'")
    e.add_argument("--gold", required=True)
    e.add_argument("--business", default=str(BUSINESS_FILE),
                   help="business.json, so other-business mentions count as in 'detect' ('' to skip)")
    e.add_argument("--threshold", type=int, default=2)
    e.add_argument("--show-errors", type=int, default=20)
    e.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    if getattr(args, "out", None):             # create outputs/task_3_5/ when writing there
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
