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
         - mention of ANOTHER business   (name found in the business cache)         -> +1
    3. If total score >= threshold (default 2) the sentence is a "comparison".
    4. Infer the direction (better / worse / similar) from polarity words.

    Comparisons with the reviewer's expectations ("better than I expected") or with
    the same business at another time ("same as always", "better than last time")
    are NOT counted: only comparisons with OTHER businesses are.

DATA
    Reviews and businesses are read from the shared Arrow cache (utils.load_cache),
    and 'detect' uses the same reproducible 10% review sample as tasks 3.2.2, 3.3.1
    and 3.4.1 (tasks.task_3_3_1.sample_rows, seed 42).

    The rules were tuned on the first 50,000 reviews of the dataset (rows 0-49,999,
    the old '--limit 50000' default).  'sample' never draws from those rows, so the
    hand-labelled evaluation set is kept separate from the tuning examples.

SUB-COMMANDS
    text      classify sentences typed on the command line
    detect    run over the shared review sample, write results and a summary
    sample    draw a stratified sample of REAL sentences for hand-labelling
    evaluate  precision / recall / F1 against that hand-labelled file

EXAMPLES  (run from the project root after 'python -m utils.build_cache')
    python -m tasks.task_3_5 text "Better than Olive Garden."
    python -m tasks.task_3_5 detect
    python -m tasks.task_3_5 sample
    python -m tasks.task_3_5 evaluate
"""
import argparse     # command-line interface
import collections  # Counter / defaultdict for statistics
import json         # results and sample metadata are written as JSON
import random       # reproducible stratified sampling and bootstrap
import re           # regular expressions: the core of the rule-based NLP
import sys
from multiprocessing import Pool
from pathlib import Path

from tasks.task_3_3_1 import DEFAULT_FRACTION, SAMPLE_SEED, sample_rows
from utils.load_cache import load_businesses, load_reviews

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "task_3_5"
TUNING_ROWS = 50_000      # review rows 0..49,999 were used to tune the rules
BATCH_SIZE = 2_000
PROGRESS_INTERVAL = 100_000

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
_SELF_AFTER = (r"i|we|you|he|she|expected|anticipated|thought|usual|always|before|ever|last|it|they|"
               r"average|advertised|necessary|needed|ideal|normal")

# Nouns that name a business (or a group of them).  If the thing compared against
# contains one of these, it is another business, not a part of the same one.
_BUSINESS_NOUN = (r"places?|restaurants?|spots?|joints?|shops?|stores?|bars?|pubs?|cafes?|bakery|bakeries|"
                  r"diners?|locations?|branch(?:es)?|chains?|hotels?|motels?|salons?|spas?|gyms?|dentists?|"
                  r"doctors?|barbers?|mechanics?|dealers(?:hips?)?|dealerships?|competition|competitors?|"
                  r"ones?|franchises?|buffets?|pizzerias?|steakhouses?|brewer(?:y|ies)|establishments?|"
                  r"business(?:es)?|clinics?|vets?|stalls?|trucks?")
# A target such as "the chicken", "the first floor", "our first visit" or "on my previous
# visit": starts with a determiner and contains no business noun, place cue or capitalised name.
_DETERMINER_TARGET = re.compile(r"^(?:(?:on|during|from|at|in)\s+)?(?:the|my|our|his|her|their|this|that)\s+",
                                re.I)
_OTHER_BUSINESS_CUE = re.compile(r"\b(?:" + _BUSINESS_NOUN + r"|elsewhere|other|another|any|most|every|all)\b"
                                 r"|\b(?:at|in|from|of)\s+(?-i:[A-Z])|(?-i:\b[A-Z][a-z]+)", re.I)


def _same_business_target(text_after):
    """True when the words after 'than' / 'compared to' name something of the SAME business.

    'the chicken', 'the first floor', 'our first visit', 'on my previous visit'  -> True
    'the Starbucks next door', 'the one on Main St', 'most places', 'my usual barber' -> False
    """
    words = text_after.strip()
    # Stop at the end of the clause: only the compared-against phrase matters.
    target = re.split(r"[.,;:!?()]|\b(?:and|but|because|so|which|who)\b", words, maxsplit=1)[0]
    target = " ".join(target.split()[:6])
    if not _DETERMINER_TARGET.match(target):
        return False
    rest = _DETERMINER_TARGET.sub("", target, count=1)
    return not _OTHER_BUSINESS_CUE.search(rest)

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
    _rx(r"\bcomparable\s+(?:to|with)\b"),                          # comparable to
    # doesn't compare to / can't come close to / doesn't hold a candle to
    _rx(r"\b(?:doesn'?t|does\s+not|don'?t|can'?t|cannot|won'?t|didn'?t)\s+(?:even\s+)?"
        r"(?:compare\s+(?:to|with)|come\s+close\s+to|hold\s+a\s+candle\s+to)\b"),
    _rx(r"\bstep\s+(?:up|down)\s+from\b"),                         # a huge step up from
    _rx(r"\briv(?:al|als|aled|alled)\b"),                          # rivals anything in New York
    _rx(r"\bputs?\s+.{1,40}?\s+to\s+shame\b"),                     # puts X to shame
    _rx(r"\b(?:miles|leaps\s+and\s+bounds|head\s+and\s+shoulders|light\s+years)\s+(?:ahead\s+of|above|beyond)\b"),
    _rx(r"\binstead\s+of\s+(?:coming|going)\b"),                   # I'll stick with X instead of coming here
    _rx(r"\b(?:it'?s|it\s+is|this\s+is)\s+no\s+(?-i:[A-Z])"),     # it is no Cheesecake Factory
    _rx(r"\bif\s+you\s+(?:like|love|enjoy)\s+(?-i:[A-Z])"),       # if you like Shake Shack, ...
    _rx(r"\b(?:like|prefer|love|enjoy)\b.{1,40}\b(?:more|better)\s+than\s+(?:this|here)\b"),
    # "we usually go to X but this was fresher"
    _rx(r"\b(?:usually|normally|always|used\s+to)\s+go\s+to\b.{0,60}\b(?:better|worse|more|less|\w{3,}er)\b"),
]

# Superlative forms: best, worst, most/least + adjective, favourite, and -est words
# (skipping ordinary words that merely end in -est, such as "honest" or "guest").
_SUP = (r"(?:best|worst|favou?rite|most\s+\w+|least\s+\w+|"
        r"(?!(?:honest|interest|rest|request|guest|west|suggest|forest|test|contest|protest|chest|nest|"
        r"pest|quest|vest|digest|modest|harvest|invest|arrest|manifest|attest|lest|zest|crest|jest|unrest|"
        r"priest)\b)\w{3,}est)")

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

# (c) SCOPED SUPERLATIVES and personal-history cues.  The first STRONG_SUPERLATIVES rules
#     rank the business against others, so they are worth +2; the others are weaker
#     evidence worth +1 each (max +2 from this group).
STRONG_SUPERLATIVES = 5
SUPERLATIVE = [
    # "best ... in town / in the area / in Las Vegas ..." -> ranks the business against all others nearby
    _rx(r"\b" + _SUP + r"\b.{0,40}\b(?:in|of|around|within|on)\s+"
        r"(?:town|the\s+(?:city|area|state|neighbou?rhood|valley|strip|world)|\w+\s+county|"
        r"(?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))\b"),               # any capitalised place name
    # "the best ... I've ever had", "the most comfortable bed I've ever slept in"
    _rx(r"\b" + _SUP + r"\b.{0,40}\b(?:I|we)(?:(?:'ve|\s+have|'d|\s+had)\s+(?:ever\s+)?|\s+ever\s+)"
        r"(?:had|been|tried|eaten|tasted|seen|visited|experienced|stayed|slept|used|encountered|taken|found)\b"),
    # "of all the hotels on the strip, this one has the best pool"
    _rx(r"\bof\s+all\s+(?:the\s+)?\w+.{0,60}?\b" + _SUP + r"\b"),
    # "other dentists I've been to were never this thorough"
    _rx(r"\b(?:other|every|any)\s+(?:\w+\s+)?(?:" + _BUSINESS_NOUN + r")\s+(?:I|we)(?:'ve|\s+have)?\s+"
        r"(?:been|tried|visited|gone|used|had)\b"),
    # "I've been to many Korean BBQ places and this ranks near the top"
    _rx(r"\b(?:I|we)(?:'ve|\s+have)\s+(?:been\s+to|tried|visited|eaten\s+at)\s+(?:many|most|lots\s+of|"
        r"a\s+lot\s+of|several|plenty\s+of|so\s+many|tons\s+of|all)\b"),
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
        # Compared with a dish, room or earlier visit of the SAME business ("than the chicken").
        if m.group(2) and _same_business_target(sentence[m.start(2):]):
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

    # (a) explicit phrase: only the first matching phrase counts (+2).  A phrase ending in
    #     "than / to / with / as" is skipped when it points at the same business
    #     ("better than the last two times", "compared to our first visit").
    for rx in EXPLICIT:
        m = rx.search(sentence)
        if not m:
            continue
        if re.search(r"\b(?:than|then|to|with|as)$", m.group(0).strip(), re.I) \
                and _same_business_target(sentence[m.end():]):
            continue
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

    # (c) scoped superlatives: the strong rules ("best ... in town", "best ... I've ever had")
    #     rank the business against all others, so they are worth +2; the rest +1.  Max +2 in total.
    sup = 0
    for i, rx in enumerate(SUPERLATIVE):
        m = rx.search(sentence)
        if m and sup < 2:
            pts = min(2 if i < STRONG_SUPERLATIVES else 1, 2 - sup)
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
# 5. DATA ACCESS   (shared Arrow cache and shared 10% review sample)
# ==========================================================================
def build_business_index():
    """Read the business cache into a BusinessIndex."""
    idx = BusinessIndex()
    biz = load_businesses().select_columns(["business_id", "name", "categories", "city"])
    for batch in biz.iter(batch_size=10_000):
        for bid, name, cats, city in zip(batch["business_id"], batch["name"], batch["categories"],
                                         batch["city"]):
            idx.add(bid, name, cats, city)
    return idx


def iter_review_batches(rows):
    """Yield lists of (review_id, business_id, stars, text) for the given rows of the review cache."""
    reviews = load_reviews().select_columns(["review_id", "business_id", "stars", "text"]).select(rows)
    for batch in reviews.iter(batch_size=BATCH_SIZE):
        yield list(zip(batch["review_id"], batch["business_id"], batch["stars"], batch["text"]))


# Classifying is the slow part (~40 regexes per sentence), so batches of reviews are
# shared out to worker processes.  Each worker builds its own business index once.
_WORKER = {}


def _init_worker(threshold):
    _WORKER["idx"] = build_business_index()
    _WORKER["threshold"] = threshold


def _classify_batch(batch):
    """Split and classify every review of a batch (runs in a worker process)."""
    idx, out = _WORKER["idx"], []
    for rid, bid, stars, text in batch:
        name = idx.meta.get(bid, ("",))[0]
        out.append((rid, bid, stars, [(s, classify(s, idx, bid, name, _WORKER["threshold"]))
                                      for s in split_sentences(text)]))
    return out


def classify_reviews(rows, threshold, workers):
    """Yield (review_id, business_id, stars, [(sentence, result), ...]) per review, in row order."""
    batches = iter_review_batches(rows)
    if workers <= 1:
        _init_worker(threshold)
        for res in map(_classify_batch, batches):
            yield from res
        return
    with Pool(workers, initializer=_init_worker, initargs=(threshold,)) as pool:
        for res in pool.imap(_classify_batch, batches):     # imap keeps the row order
            yield from res


# ==========================================================================
# 6. EVALUATION ON REAL YELP SENTENCES (labelled by the group, no invented data)
# ==========================================================================
# Comparison sentences are rare, so a plain random sample holds only a handful of
# positives.  'sample' therefore draws from three strata and 'evaluate' re-weights
# each stratum by its size, which gives estimates for ALL sentences:
#   flagged : the detector says "comparison"                -> precision
#   cue     : not flagged, but has a comparison-like word   -> where missed comparisons hide
#   other   : everything else                               -> checks that few are missed there
_CUE = _rx(r"\b(?:than|then|compar\w*|unlike|similar|same\s+as|versus|vs|prefer\w*|instead|elsewhere|"
           r"other|another|best|worst|favou?rite|remind\w*|just\s+like|rival\w*|beats?)\b")
STRATA = ("flagged", "cue", "other")


def _stratum(sentence, res):
    """Name of the sampling stratum of one classified sentence."""
    if res["is_comparison"]:
        return "flagged"
    if res["score"] >= 1 or _CUE.search(sentence):
        return "cue"
    return "other"


def cmd_sample(args):
    """Draw a stratified, blind sample of REAL sentences for the group to label.

    Pool: reviews of the shared 10% sample, minus the tuning rows (0..TUNING_ROWS-1).
    From --reviews random reviews of that pool, every sentence of 5-60 words is
    classified, and a fixed number per stratum is kept (reservoir sampling).
    The rows are shuffled so labellers cannot tell which stratum a sentence came from.

    to_label.tsv     id <TAB> label (empty: fill 1 or 0) <TAB> sentence
    sample_key.json  stratum sizes and the review / business / stratum of every id
    """
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists (it may hold labels). Use --force to overwrite it.")
    rng = random.Random(args.seed)
    pool = [r for r in sample_rows(load_reviews().num_rows, args.fraction) if r >= TUNING_ROWS]
    rows = sorted(rng.sample(pool, min(args.reviews, len(pool))))
    print(f"Reading {len(rows):,} random reviews from {len(pool):,} in the shared sample "
          f"(fraction={args.fraction}, seed={SAMPLE_SEED}, tuning rows < {TUNING_ROWS:,} excluded)")

    want = {"flagged": args.n_flagged, "cue": args.n_cue, "other": args.n_other}
    seen = collections.Counter()            # population size of each stratum
    picked = {st: [] for st in STRATA}      # reservoir per stratum
    for rid, bid, _stars, results in classify_reviews(rows, args.threshold, args.workers):
        for s, res in results:
            if not 5 <= len(s.split()) <= 60:  # skip fragments and run-on blocks
                continue
            st = _stratum(s, res)
            seen[st] += 1
            item = (rid, bid, " ".join(s.split()))   # no tabs/newlines
            if len(picked[st]) < want[st]:
                picked[st].append(item)
            else:                                    # keep each sentence with equal probability
                j = rng.randrange(seen[st])
                if j < want[st]:
                    picked[st][j] = item

    items = [(st, *item) for st in STRATA for item in picked[st]]
    rng.shuffle(items)
    with open(out, "w", encoding="utf-8") as o:
        o.write("id\tlabel\tsentence\n")
        for i, (_st, _rid, _bid, s) in enumerate(items, 1):
            o.write(f"{i}\t\t{s}\n")
    key = {"fraction": args.fraction, "sample_seed": SAMPLE_SEED, "seed": args.seed,
           "excluded_tuning_rows": TUNING_ROWS, "reviews_read": len(rows), "threshold": args.threshold,
           "population": {st: seen[st] for st in STRATA},
           "sampled": {st: len(picked[st]) for st in STRATA},
           "items": {str(i): {"review_id": rid, "business_id": bid, "stratum": st}
                     for i, (st, rid, bid, _s) in enumerate(items, 1)}}
    with open(args.key, "w", encoding="utf-8") as o:
        json.dump(key, o, indent=2)
    print("Sentences per stratum (population):", dict(key["population"]))
    print("Sampled per stratum               :", dict(key["sampled"]))
    print(f"Wrote {len(items)} sentences to {out} and the key to {args.key}. "
          f"Fill column 2 with 1 (comparison) or 0 (not), then run 'evaluate'.")


def _scores(c):
    """Precision, recall and F1 from a (possibly weighted) confusion Counter."""
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _confusion(rows, weight):
    """Confusion counts; each row counts 'weight[stratum]' times."""
    c = collections.Counter()
    for st, y, p, _s in rows:
        c[("tp" if y else "fp") if p else ("fn" if y else "tn")] += weight[st]
    return c


def cmd_evaluate(args):
    """Compare the tool's predictions with the group's labels and report precision/recall/F1.

    TP = comparison correctly found      FP = flagged but not a comparison
    FN = comparison the tool missed      TN = correctly ignored
    precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = harmonic mean of the two

    Each labelled sentence stands for  population(stratum) / labelled(stratum)  sentences,
    so the weighted scores estimate the detector's performance on all sentences.
    The 95% intervals come from a bootstrap that resamples within each stratum.
    """
    key = None
    if Path(args.key).is_file():
        with open(args.key, encoding="utf-8") as f:
            key = json.load(f)
    # Use the same business index as 'detect', so the scores describe the same system.
    idx = None if args.no_business else build_business_index()
    rows = []                                          # (stratum, label, prediction, sentence)
    with open(args.gold, encoding="utf-8-sig") as f:   # utf-8-sig: tolerates Excel's BOM
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or parts[1].strip() not in ("0", "1"):
                continue                               # skip the header and unlabelled rows
            item = key["items"].get(parts[0]) if key else None
            if key and item is None:
                sys.exit(f"id {parts[0]} is not in {args.key}; was the key file overwritten?")
            bid = item["business_id"] if item else None
            name = idx.meta.get(bid, ("",))[0] if idx and bid else None
            p = int(classify(parts[2], idx, bid, name, args.threshold)["is_comparison"])
            rows.append((item["stratum"] if item else "all", int(parts[1].strip()), p, parts[2]))
    if not rows:
        sys.exit("No labelled rows found (column 2 must be 0 or 1).")

    labelled = collections.Counter(r[0] for r in rows)
    positives = collections.Counter(r[0] for r in rows if r[1])
    population = key["population"] if key else {"all": len(rows)}
    weight = {st: population[st] / labelled[st] for st in labelled}
    raw = _confusion(rows, collections.defaultdict(lambda: 1))
    est = _confusion(rows, weight)

    rng = random.Random(args.seed)
    by_st = collections.defaultdict(list)
    for r in rows:
        by_st[r[0]].append(r)
    boot = []
    for _ in range(args.bootstrap):
        resample = [rng.choice(g) for g in by_st.values() for _ in g]
        boot.append(_scores(_confusion(resample, weight)))

    def ci(i):
        vals = sorted(b[i] for b in boot)
        return [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]] if vals else None

    n = len(rows)
    p, r, f1 = _scores(raw)
    wp, wr, wf1 = _scores(est)
    print(f"Labelled sentences: {n}")
    print(f"{'stratum':<8} {'population':>11} {'labelled':>9} {'positives':>10}")
    for st in labelled:
        print(f"{st:<8} {population[st]:>11,} {labelled[st]:>9} {positives[st]:>10}")
    print(f"\nOn the labelled sample (unweighted, oversampled positives inflate these):")
    print(f"  TP={raw['tp']} FP={raw['fp']} FN={raw['fn']} TN={raw['tn']}  "
          f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    print(f"Estimated for all sentences (weighted by stratum size, 95% bootstrap CI):")
    for label, val, i in (("Precision", wp, 0), ("Recall", wr, 1), ("F1", wf1, 2)):
        lo, hi = ci(i) or (val, val)
        print(f"  {label:<9} = {val:.3f}  [{lo:.3f}, {hi:.3f}]")
    errors = [("FP" if pr else "FN", st, s) for st, y, pr, s in rows if y != pr]
    for kind, st, s in errors[:args.show_errors]:
        print(f"  {kind} ({st}): {s}")

    result = {"labelled": n, "threshold": args.threshold, "bootstrap": args.bootstrap,
              "strata": {st: {"population": population[st], "labelled": labelled[st],
                              "positives": positives[st]} for st in labelled},
              "unweighted": {**{k: raw[k] for k in ("tp", "fp", "fn", "tn")},
                             "precision": p, "recall": r, "f1": f1},
              "weighted": {**{k: round(est[k], 1) for k in ("tp", "fp", "fn", "tn")},
                           "precision": wp, "recall": wr, "f1": wf1,
                           "precision_ci95": ci(0), "recall_ci95": ci(1), "f1_ci95": ci(2)},
              "errors": [{"type": k, "stratum": st, "sentence": s} for k, st, s in errors]}
    with open(args.out, "w", encoding="utf-8") as o:
        json.dump(result, o, indent=2, ensure_ascii=False)
    print(f"\nEvaluation written to {args.out}")


# ==========================================================================
# 7. COMMAND-LINE COMMANDS
# ==========================================================================
def cmd_text(args):
    """Classify sentences typed on the command line and show the evidence."""
    idx = build_business_index() if args.business else None
    for s in args.sentences:
        r = classify(s, idx, threshold=args.threshold)
        print(f"[{'COMPARISON' if r['is_comparison'] else 'no        '}] score={r['score']} "
              f"dir={r['direction']}  {s}")
        for kind, ev in r["evidence"]:
            print(f"      - {kind}: '{ev}'")


def cmd_detect(args):
    """Run the detector over the shared review sample and write the results and a summary.

    The review cache is memory-mapped and read in batches, so memory use stays small.
    """
    idx = build_business_index()
    rows = sample_rows(load_reviews().num_rows, args.fraction)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Reviews in sample: {len(rows):,} (fraction={args.fraction}, seed={SAMPLE_SEED})")
    n_rev = n_sent = n_cmp = n_rev_cmp = 0            # counters
    dirs = collections.Counter()                      # direction distribution
    cat_tot, cat_cmp = collections.Counter(), collections.Counter()   # per-category review counts
    stars_all, stars_cmp = [], []                     # star ratings: all reviews / reviews with a comparison
    cited = collections.Counter()                     # most frequently compared-to businesses
    examples = []                                     # a few examples printed to the screen
    with open(args.out, "w", encoding="utf-8") as out:
        for rid, bid, stars, results in classify_reviews(rows, args.threshold, args.workers):
            n_rev += 1
            if n_rev % PROGRESS_INTERVAL == 0:
                print(f"  {n_rev:,} reviews, {n_cmp:,} comparison sentences so far")
            name, cats, city = idx.meta.get(bid, ("", [], ""))
            has = False                                # does this review contain >= 1 comparison?
            stars_all.append(stars or 0)
            for c in cats[:3]:                         # first three categories of the business
                cat_tot[c] += 1
            for s, res in results:
                n_sent += 1
                if not res["is_comparison"]:
                    continue
                has = True
                n_cmp += 1
                dirs[res["direction"]] += 1
                for o in res["others"]:
                    cited[o] += 1
                rec = {"review_id": rid, "business": name, "stars": stars,
                       "sentence": s, "score": res["score"], "direction": res["direction"],
                       "evidence": res["evidence"], "other_businesses": res["others"]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if len(examples) < args.show:
                    examples.append(f"({stars}★ {name}) [{res['direction']}, score {res['score']}] {s}")
            if has:
                n_rev_cmp += 1
                stars_cmp.append(stars or 0)
                for c in cats[:3]:
                    cat_cmp[c] += 1

    # ---- summary statistics ----
    def avg(x):
        return sum(x) / len(x) if x else 0.0

    rates = [(c, cat_cmp[c] / t, t) for c, t in cat_tot.items() if t >= args.min_cat]
    summary = {"fraction": args.fraction, "sample_seed": SAMPLE_SEED, "threshold": args.threshold,
               "reviews": n_rev, "sentences": n_sent, "comparison_sentences": n_cmp,
               "comparison_sentence_pct": 100 * n_cmp / max(n_sent, 1),
               "reviews_with_comparison": n_rev_cmp,
               "reviews_with_comparison_pct": 100 * n_rev_cmp / max(n_rev, 1),
               "avg_stars_all": avg(stars_all), "avg_stars_with_comparison": avg(stars_cmp),
               "directions": dict(dirs),
               "top_categories": [{"category": c, "comparison_rate_pct": 100 * rt, "reviews": t}
                                  for c, rt, t in sorted(rates, key=lambda x: -x[1])[:10]],
               "most_compared_to": [{"business": k, "mentions": v} for k, v in cited.most_common(10)],
               "examples": examples}
    with open(args.summary, "w", encoding="utf-8") as o:
        json.dump(summary, o, indent=2, ensure_ascii=False)

    print("\n".join(examples))
    print("\n========== SUMMARY ==========")
    print(f"Reviews processed            : {n_rev}")
    print(f"Sentences processed          : {n_sent}")
    print(f"Comparison sentences         : {n_cmp} ({summary['comparison_sentence_pct']:.2f}% of sentences)")
    print(f"Reviews with >=1 comparison  : {n_rev_cmp} ({summary['reviews_with_comparison_pct']:.2f}% of reviews)")
    print(f"Avg stars  all / with-comparison : {avg(stars_all):.2f} / {avg(stars_cmp):.2f}")
    print("Direction distribution       :", dict(dirs))
    print(f"\nTop categories by comparison rate (min {args.min_cat} reviews):")
    for t in summary["top_categories"]:
        print(f"  {t['category']:<30} {t['comparison_rate_pct']:5.1f}%  (n={t['reviews']})")
    if cited:
        print("\nMost frequently compared-to businesses:")
        for k, v in cited.most_common(10):
            print(f"  {k:<30} {v}")
    print(f"\nDetected sentences written to {args.out}, summary to {args.summary}")


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
    t.add_argument("--business", action="store_true",
                   help="load the business cache, to also recognise named competitors")
    t.add_argument("--threshold", type=int, default=2)
    t.set_defaults(fn=cmd_text)

    d = sub.add_parser("detect", help="run on the shared review sample")
    d.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                   help="fraction of reviews in the shared sample (1.0 = all reviews)")
    d.add_argument("--limit", type=int, default=0, help="max reviews of the sample, for a quick run (0 = all)")
    d.add_argument("--threshold", type=int, default=2)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--out", default=str(OUTPUT_DIR / "comparisons.jsonl"),
                   help="detected sentences as JSON lines")
    d.add_argument("--summary", default=str(OUTPUT_DIR / "summary.json"))
    d.add_argument("--show", type=int, default=15, help="number of examples printed")
    d.add_argument("--min-cat", type=int, default=30,
                   help="minimum reviews for a category to appear in the ranking")
    d.set_defaults(fn=cmd_detect)

    sm = sub.add_parser("sample", help="draw a stratified sample of real sentences to hand-label")
    sm.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    sm.add_argument("--reviews", type=int, default=50_000, help="random reviews read from the pool")
    sm.add_argument("--n-flagged", type=int, default=100, help="sentences the detector flags")
    sm.add_argument("--n-cue", type=int, default=100, help="unflagged sentences with a comparison-like word")
    sm.add_argument("--n-other", type=int, default=100, help="all other sentences")
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--threshold", type=int, default=2)
    sm.add_argument("--workers", type=int, default=8)
    sm.add_argument("--out", default=str(OUTPUT_DIR / "to_label.tsv"))
    sm.add_argument("--key", default=str(OUTPUT_DIR / "sample_key.json"))
    sm.add_argument("--force", action="store_true", help="overwrite an existing to_label.tsv")
    sm.set_defaults(fn=cmd_sample)

    e = sub.add_parser("evaluate", help="P/R/F1 on the hand-labelled TSV made by 'sample'")
    e.add_argument("--gold", default=str(OUTPUT_DIR / "to_label.tsv"))
    e.add_argument("--key", default=str(OUTPUT_DIR / "sample_key.json"),
                   help="stratum key written by 'sample' (without it, rows are treated as a random sample)")
    e.add_argument("--no-business", action="store_true",
                   help="do not load the business cache (other-business mentions then never count)")
    e.add_argument("--threshold", type=int, default=2)
    e.add_argument("--bootstrap", type=int, default=1000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--show-errors", type=int, default=20)
    e.add_argument("--out", default=str(OUTPUT_DIR / "evaluation.json"))
    e.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
