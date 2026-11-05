AI6122 Text Data Management & Processing - Group Assignment
Review Data Analysis and Processing
Group 01 - Source code (Src_G01.zip)
=========================================================================

CONTENTS
  1. What is in this package
  2. Third-party libraries and download links
  3. Installation and setup
  4. How to use the system (commands, input format, parameters)
  5. Explanations of sample output


=========================================================================
1. WHAT IS IN THIS PACKAGE
=========================================================================

Src_G01.zip
  Readme.txt                  this file
  SourceCode/
    requirements.txt          Python packages to install
    utils/
      build_cache.py          converts the two Yelp JSON files to Arrow caches
      load_cache.py           loads (and checks) the Arrow caches
    tasks/
      task_3_1.py             3.1   dataset summary
      task_3_2_1.py           3.2.1 Zipf's Law and Heaps' Law
      task_3_2_2.py           3.2.2 custom regex tokenizer and comparison
      task_3_3_1.py           3.3.1 inverted index
      task_3_3_2.py           3.3.2 TF-IDF cosine and BM25 ranking (search)
      task_3_3_2_lucene.py    3.3.2 optional comparison with Lucene
      task_3_4_1.py           3.4.1 lexicon-based sentiment trends (VADER)
      task_3_4_2.py           3.4.2 term co-occurrence drift (PMI)
      task_3_5.py             3.5   application: comparison sentence detection
    outputs/task_3_5/
      to_label.tsv            our 300 hand-labelled sentences (input of 3.5
                              "evaluate")
      sample_key.json         stratum of each labelled sentence (input of
                              3.5 "evaluate")

As required, the Yelp dataset and the third-party libraries are NOT included.
All other output files are recreated when the commands in Section 4 are run.


=========================================================================
2. THIRD-PARTY LIBRARIES AND DOWNLOAD LINKS
=========================================================================

Runtime
  Python 3.12 ............ https://www.python.org/downloads/

Python packages (all installed by "pip install -r requirements.txt")
  datasets (Hugging Face)  https://pypi.org/project/datasets/
  pyarrow (Apache Arrow) . https://pypi.org/project/pyarrow/
  NLTK ................... https://pypi.org/project/nltk/
  regex .................. https://pypi.org/project/regex/
  NumPy .................. https://pypi.org/project/numpy/
  pandas ................. https://pypi.org/project/pandas/
                           (installed automatically with datasets/seaborn)
  SciPy .................. https://pypi.org/project/scipy/
  Matplotlib ............. https://pypi.org/project/matplotlib/
  seaborn ................ https://pypi.org/project/seaborn/
  vaderSentiment ......... https://pypi.org/project/vaderSentiment/

NLTK data
  punkt_tab .............. https://www.nltk.org/data.html
                           (downloaded automatically into data/nltk_data/
                           the first time task 3.2.1 or 3.2.2 runs)

Optional, only for the Lucene comparison in task 3.3.2
  Pyserini 2.4.0 ......... https://pypi.org/project/pyserini/
                           (bundles the Apache Lucene / Anserini jar;
                           Lucene: https://lucene.apache.org/)
  PyJNIus ................ https://pypi.org/project/pyjnius/
  Java 21 (JDK) .......... https://adoptium.net/temurin/releases/?version=21

Dataset
  Yelp Open Dataset ...... https://www.yelp.com/dataset
                           (only business.json and review.json are used)


=========================================================================
3. INSTALLATION AND SETUP
=========================================================================

3.1 Create an environment and install the packages
    Unzip Src_G01.zip and open a terminal in the SourceCode folder.
    Run ALL commands in this file from the SourceCode folder.

    Windows (PowerShell):
        python -m venv .venv
        .venv\Scripts\Activate.ps1
        python -m pip install -r requirements.txt

    macOS / Linux:
        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install -r requirements.txt

3.2 Add the Yelp data
    Download the Yelp Open Dataset (after accepting Yelp's dataset
    agreement) and put the two files, with their official names, in
    SourceCode/data/raw/ :

        data/raw/yelp_academic_dataset_business.json
        data/raw/yelp_academic_dataset_review.json

    Input format: JSON Lines (one JSON object per line), as distributed by
    Yelp. Allow about 16 GB of free disk space in total (raw files 5.5 GB,
    Arrow cache 4.7 GB, temporary Hugging Face copy 4.6 GB, indexes 0.5 GB).

3.3 Build the Arrow cache (once, before any task except 3.4.2)
        python -m utils.build_cache

    This writes memory-mapped Arrow datasets to data/cache/businesses/ and
    data/cache/reviews/, keeping every source column. To check the cache:
        python -m utils.load_cache

    If a cache already exists, delete its folder before rebuilding it. The
    temporary copy in ~/.cache/huggingface/datasets/json/ can be deleted
    after the cache is built.

3.4 Optional: Lucene comparison (task 3.3.2 only)
    Install Java 21 (set JAVA_HOME, or put java on the PATH), then:
        python -m pip install --no-deps pyserini==2.4.0
        python -m pip install pyjnius
    "--no-deps" avoids several gigabytes of deep-learning packages that
    this step does not use.

3.5 Order of running
    utils.build_cache -> 3.1, 3.2.1, 3.2.2 -> 3.3.1 -> 3.3.2 (-> Lucene)
    -> 3.4.1 -> 3.5. Task 3.4.2 reads the raw review JSON directly and can
    run at any time. Tasks 3.2.2, 3.3.1, 3.4.1 and 3.5 all use the same
    reproducible random 10% review sample (seed 42).


=========================================================================
4. HOW TO USE THE SYSTEM
=========================================================================

Every task writes its results to SourceCode/outputs/<task>/. Section 5
explains the main output files with sample values from our run.

-------------------------------------------------------------------------
Task 3.1 - Dataset summary
-------------------------------------------------------------------------
    python -m tasks.task_3_1
  Input:      Arrow cache (Section 3.3).
  Parameters: none.
  Output:     outputs/task_3_1/dataset_summary.json

-------------------------------------------------------------------------
Task 3.2.1 - Zipf's Law and Heaps' Law
-------------------------------------------------------------------------
    python -m tasks.task_3_2_1
  Input:      all 6,990,280 reviews from the Arrow cache.
  Parameters: none (fit ranks 10-10,000 and shuffle seed 42 are constants
              at the top of the script).
  Processing: NLTK word_tokenize, case-folding, punctuation-only tokens
              and stray one-letter fragments (except "a" and "i") removed;
              Heaps samples are nested 10/30/50/100% subsets.
  Output:     outputs/task_3_2_1/ - zipf_law.png, heaps_law.png,
              metrics.json, heaps_points.csv, top_terms.csv, conclusions.md

-------------------------------------------------------------------------
Task 3.2.2 - Custom tokenizer
-------------------------------------------------------------------------
    python -m tasks.task_3_2_2
  Input:      the shared 10% review sample (699,028 reviews).
  Parameters: none.
  Processing: compares NLTK WordPunctTokenizer, NLTK word_tokenize and our
              custom regex tokenizer (function custom_tokenize, reused by
              tasks 3.3.1 and 3.3.2). Takes about 15 minutes.
  Output:     outputs/task_3_2_2/ - vocabulary_comparison.csv,
              top_tokens.csv, qualitative_comparison.md,
              custom_vocabulary_examples.json, metrics.json, conclusions.md

-------------------------------------------------------------------------
Task 3.3.1 - Inverted index
-------------------------------------------------------------------------
    python -m tasks.task_3_3_1
    python -m tasks.task_3_3_1 --fraction 1.0 --workers 4
  Input:      the shared 10% review sample, tokenized with the custom
              tokenizer.
  Parameters: --fraction  share of reviews to index (default 0.10;
                          1.0 indexes all reviews and needs much more memory)
              --workers   number of worker processes (default 8)
  Output:     index files in data/index/ (used by task 3.3.2) and
              outputs/task_3_3_1/index_stats.json. Takes about 1 minute.

-------------------------------------------------------------------------
Task 3.3.2 - TF-IDF cosine and BM25 ranking
-------------------------------------------------------------------------
  Requires the index from task 3.3.1.

  Search with any query:
    python -m tasks.task_3_3_2 search "dog friendly patio" --method bm25 -k 10
  Parameters: "query"   the query text (in quotes)
              --method  bm25 (default) or tfidf
              -k        number of results to print (default 10)

  Run the report's query set:
    python -m tasks.task_3_3_2 analyze
  Runs 8 queries through TF-IDF cosine, BM25 (k1 = 1.2, b = 0.75) and two
  BM25 variants (b = 0 and k1 = 3), and checks every top-10 score against
  a direct recomputation from the raw text.
  Output: outputs/task_3_3_2/ - rankings.csv, query_summary.json,
          conclusions.md

  Optional comparison with Lucene (needs Section 3.4):
    python -m tasks.task_3_3_2_lucene build
    python -m tasks.task_3_3_2_lucene compare
  "build" indexes the same sample with Lucene in data/lucene_index/
  (--workers, default 8; about 1.5 minutes). "compare" ranks the same 8
  queries with Lucene's BM25Similarity and ClassicSimilarity.
  Output: outputs/task_3_3_2/lucene_rankings.csv, lucene_comparison.json

  Other code can also use the search engine directly:
    from tasks.task_3_3_2 import SearchEngine
    engine = SearchEngine(); engine.search(query, k, method)

-------------------------------------------------------------------------
Task 3.4.1 - Lexicon-based sentiment mining (VADER)
-------------------------------------------------------------------------
    python -m tasks.task_3_4_1
  Input:      reviews of Fast Food, Italian and Ice Cream & Frozen Yogurt
              businesses in the shared 10% sample (businesses listed in
              more than one of these sectors are excluded).
  Parameters: none.
  Output:     outputs/task_3_4_1/ - summary.txt, yearly_sentiment.csv,
              seasonal_sentiment.csv, yearly_stars_length.csv,
              sarcasm_examples.csv, false_negative_examples.csv,
              sentiment_trends.png, sentiment_seasons.png,
              stars_and_length_trends.png

-------------------------------------------------------------------------
Task 3.4.2 - Term co-occurrence drift (PMI)
-------------------------------------------------------------------------
    python -m tasks.task_3_4_2 --target service
    python -m tasks.task_3_4_2 --reviews "path/to/review.json" --target price
  Input:      the raw review JSON Lines file (NOT the Arrow cache); all
              reviews are streamed line by line. Uses the Python standard
              library only.
  Parameters:
    --reviews           review JSON file (default:
                        data/raw/yelp_academic_dataset_review.json)
    --target            target aspect word (default: service)
    --top-n             words kept per year (default: 10)
    --min-cooccurrence  minimum number of reviews per year that contain
                        both words (default: 20)
    --start-year, --end-year     optional year range
    --businesses        business JSON file; needed only with --category
    --category          optional exact Yelp category (repeat to require
                        all given categories)
    --business-id       optional business ID (repeatable)
    --max-reviews       read only the first N lines (quick test only)
    --output-dir        default: outputs/task_3_4_2/<target>/
  Method:     two words co-occur if both appear in the same review;
              PMI = log2(joint_reviews * reviews / (target_reviews *
              word_reviews)), computed separately for each year.
  Output:     outputs/task_3_4_2/<target>/ - top_words_by_year.csv,
              yearly_counts.csv, yearly_drift.csv, summary.md,
              run_metadata.json

-------------------------------------------------------------------------
Task 3.5 - Application: comparison sentence detection
-------------------------------------------------------------------------
  Classify sentences typed on the command line:
    python -m tasks.task_3_5 text "Way cheaper than the Starbucks next door."
  Parameters: one or more sentences in quotes
              --business   also recognise names of other businesses
                           (needs the Arrow cache)
              --threshold  minimum score for a comparison (default 2)

  Run on the shared 10% sample:
    python -m tasks.task_3_5 detect
  Parameters: --workers (default 8), --threshold (default 2),
              --limit N (only the first N sample reviews, for a quick run),
              --show N (examples printed, default 15),
              --out (default outputs/task_3_5/comparisons.jsonl),
              --summary (default outputs/task_3_5/summary.json)
  Output:     comparisons.jsonl (every detected sentence with its evidence)
              and summary.json in outputs/task_3_5/

  Evaluation:
    python -m tasks.task_3_5 evaluate
  Compares the detector with our labels in outputs/task_3_5/to_label.tsv
  (included in this package) and writes outputs/task_3_5/evaluation.json.
  It loads the business cache to recognise competitor names.
  Parameters: --bootstrap (resamples, default 1000), --seed (default 42),
              --show-errors (default 20), --no-business (skip the
              business cache), --gold, --key, --out.

  "python -m tasks.task_3_5 sample" draws a new stratified sample of 300
  sentences to label (100 flagged, 100 unflagged with a comparison-like
  word, 100 others). It will not overwrite the existing to_label.tsv
  unless --force is given.


=========================================================================
5. EXPLANATIONS OF SAMPLE OUTPUT
=========================================================================

Task 3.1 - outputs/task_3_1/dataset_summary.json
  Two parts, "businesses" and "reviews". Each lists the number of rows,
  the columns and their types, missing values in required fields, and the
  distribution of star ratings. Sample values:
    businesses: 150,346 rows; 103 have no categories
    reviews:    6,990,280 rows; 1,987,929 unique users; dates from
                2005-02-16 to 2022-01-19; median review length 406
                characters
  The business part also lists the top states and top categories.

Task 3.2.1 - outputs/task_3_2_1/
  metrics.json      the fitted parameters, e.g.
                      tokens 743,579,319; vocabulary 1,345,435
                      Zipf: exponent s = 1.4358, R^2 = 0.9892
                            (fitted on ranks 10-10,000, 9,991 terms)
                      Heaps: k = 2.7870, beta = 0.6404, R^2 = 0.9999
  heaps_points.csv  vocabulary size at each sample size, e.g.
                      percentage=10, reviews=699028, tokens=74329195,
                      vocabulary=307553
  top_terms.csv     the 100 most frequent terms (rank, term, frequency),
                    e.g. 1, the, 36756888
  zipf_law.png      log-log plot of frequency against rank with the
                    fitted line; heaps_law.png shows the four sample
                    points with the fitted Heaps curve.

Task 3.2.2 - outputs/task_3_2_2/
  vocabulary_comparison.csv   tokens and vocabulary size per tokenizer on
                              the same 699,028 reviews:
      tokenizer       tokens       vocabulary
      wordpunct       85,523,718   189,457
      word_tokenize   83,808,479   307,903
      custom_regex    73,564,962   245,428
  qualitative_comparison.md   three random reviews plus one real review
                              per difficult case (contractions, hyphens,
                              URLs, emails, currency, dates, repeated
                              letters, ...), tokenized by all three
                              tokenizers, e.g.
      Input "didn't" ->  WordPunct: didn ' t
                         word_tokenize: did n't
                         custom: didn't
  top_tokens.csv              the 100 most frequent tokens per tokenizer.

Task 3.3.1 - outputs/task_3_3_1/index_stats.json
  documents 699,028; vocabulary_size 245,428 terms; postings 49,209,004;
  average_document_length 105.24 tokens; build_minutes 1.12.

Task 3.3.2 - search output
  python -m tasks.task_3_3_2 search "dog friendly patio" -k 2 prints:

     1. 17.7054  Risky Business (Tucson) 1★, 81 tokens
        When I called they said they have a dog friendly patio, so we took our ...
     2. 16.8855  Boathouse at Hendry's Beach (Santa Barbara) 2★, 19 tokens
        Their idea of dog friendly is tie your dog outside the patio! Just not ...

  Each result shows its rank and score (BM25 scores are unbounded; TF-IDF
  cosine scores lie between 0 and 1), the business and its city, the
  review's star rating and its length in tokens, and the first 200
  characters of the review. Both top results are complaints: bag-of-words
  ranking matches the query terms but cannot see negation or sentiment.

  rankings.csv        top 10 results for each of the 8 queries under each
                      configuration (tfidf, bm25, bm25_b0, bm25_k1_3), with
                      score, star rating, length, business and a snippet.
  query_summary.json  per query: document frequencies, number of matching
                      reviews, mean length and stars of the results, and
                      the overlap between methods. For "pho", the mean
                      result length is 21.1 tokens with TF-IDF and 57.2
                      with BM25.
  lucene_comparison.json  per query: shared top-10 results, Jaccard
                      overlap, mean rank shift and score ratio between our
                      rankers and Lucene. For "pho", our BM25 and Lucene's
                      BM25Similarity share 9 of the top 10.

Task 3.4.1 - outputs/task_3_4_1/summary.txt (excerpt)
    Reviews per sector (full sample, before the 100-review threshold):
      Fast Food: 19993
      Italian: 42326
      Ice Cream: 13078
    Yearly trend (years with >=100 reviews):
      Fast Food: 0.630 (2009) -> 0.043 (2022), slope -0.0431/yr, p=7.73e-09, n_years=14
  The values are mean VADER compound scores (-1 = most negative,
  +1 = most positive). "slope" is the fitted change per year and p its
  significance. Years or seasons with fewer than 100 reviews are excluded.
  yearly_sentiment.csv / seasonal_sentiment.csv give mean, std, count and
  95% confidence interval per sector and year/season.
  sarcasm_examples.csv lists 1-star reviews that VADER scores above 0.8;
  false_negative_examples.csv lists 5-star reviews scored below -0.5.
  These show the limits of a lexicon-based method.

Task 3.4.2 - outputs/task_3_4_2/service/
  summary.md lists, for each year, the number of reviews, the number that
  contain the target, and the top words with their PMI, e.g.
    - 2007: 15,363 reviews; 2,696 with target. spotty (2.456),
      impeccable (2.236), prompt (2.065), slow (1.949), ...
  A positive PMI means the word appears in reviews with "service" more
  often than expected by chance; it does not mean positive sentiment.
  top_words_by_year.csv  year, rank, word, PMI and the counts behind it.
  yearly_counts.csv      reviews and target reviews per year.
  yearly_drift.csv       change between the top-word sets of consecutive
                         years: drift = 1 - Jaccard similarity (0 = same
                         set, 1 = no overlap), with the words that entered
                         and exited. E.g. 2006 -> 2007: drift 0.824.

Task 3.5 - command-line output of "text"
    $ python -m tasks.task_3_5 text "Way cheaper than the Starbucks next door." \
        "Better than I expected." "We waited more than 20 minutes." \
        "Best sandwiches in St.Louis!"
    [COMPARISON] score=2 dir=better  Way cheaper than the Starbucks next door.
          - comparative: 'cheaper than the'
    [no        ] score=0 dir=None  Better than I expected.
    [no        ] score=0 dir=None  We waited more than 20 minutes.
    [COMPARISON] score=2 dir=better  Best sandwiches in St.Louis!
          - scoped-superlative: 'Best sandwiches in St'

  Each line shows the decision, the rule score (2 or more = comparison),
  the direction (better / worse / similar / mixed / unspecified: how the
  reviewed business compares) and the sentence; the indented lines show
  the rules that matched. The second sentence compares with an
  expectation and the third states a quantity, so both are rejected.

  summary.json (detect): 699,028 reviews and 5,412,063 sentences;
    136,308 comparison sentences (2.52%); 119,256 reviews (17.06%)
    contain at least one. Directions: better 68,727, worse 12,713,
    similar 8,798, mixed 2,333, unspecified 43,737. It also lists the
    top categories, the most compared-to businesses and examples.

  evaluation.json (evaluate): precision, recall and F1 on the 300
    labelled sentences and, weighted by stratum size, for all sentences:
      precision 0.770 (95% CI 0.680-0.850)
      recall    0.340 (95% CI 0.224-0.573)
      F1        0.472 (95% CI 0.344-0.663)
    The "errors" list gives every false positive (FP) and missed
    comparison (FN) with its stratum.
