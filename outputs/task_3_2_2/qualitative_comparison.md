# Qualitative tokenizer comparison

All outputs are case-folded. Beyond that shared step, each tokenizer retains its native behaviour.
A random 10% subset was selected with seed 42. A seeded pool of 100,000 reviews and reservoir sampling (43) then selected three unrestricted reviews and one distinct review per feature stratum.
Email local parts are replaced with 'redacted' in this qualitative display before all three tokenizers run. Quantitative counts use the original review text.

## Unrestricted random review 1

Source dataset row: 4,208,889

Input: "Jerk chicken is awesome. They almost never have mac and cheese, but the other day I found out why when I had it for the first time: it's absolutely amazing."

- WordPunctTokenizer: `["jerk", "chicken", "is", "awesome", ".", "they", "almost", "never", "have", "mac", "and", "cheese", ",", "but", "the", "other", "day", "i", "found", "out", "why", "when", "i", "had", "it", "for", "the", "first", "time", ":", "it", "'", "s", "absolutely", "amazing", "."]`
- word_tokenize: `["jerk", "chicken", "is", "awesome", ".", "they", "almost", "never", "have", "mac", "and", "cheese", ",", "but", "the", "other", "day", "i", "found", "out", "why", "when", "i", "had", "it", "for", "the", "first", "time", ":", "it", "'s", "absolutely", "amazing", "."]`
- Custom regex: `["jerk", "chicken", "is", "awesome", "they", "almost", "never", "have", "mac", "and", "cheese", "but", "the", "other", "day", "i", "found", "out", "why", "when", "i", "had", "it", "for", "the", "first", "time", "it's", "absolutely", "amazing"]`

## Unrestricted random review 2

Source dataset row: 3,496,411

Input: "I just walked in to get a copy of a key that I had which could not be replicated by my local Home Depot. But this place was able to make a replica of the key with no problem at all. I called in advance to see if they did make copies and the associate Who picked up was very nice and also aided me and directions to the"

- WordPunctTokenizer: `["i", "just", "walked", "in", "to", "get", "a", "copy", "of", "a", "key", "that", "i", "had", "which", "could", "not", "be", "replicated", "by", "my", "local", "home", "depot", ".", "but", "this", "place", "was", "able", "to", "make", "a", "replica", "of", "the", "key", "with", "no", "problem", "at", "all", ".", "i", "called", "in", "advance", "to", "see", "if", "they", "did", "make", "copies", "and", "the", "associate", "who", "picked", "up", "was", "very", "nice", "and", "also", "aided", "me", "and", "directions", "to", "the"]`
- word_tokenize: `["i", "just", "walked", "in", "to", "get", "a", "copy", "of", "a", "key", "that", "i", "had", "which", "could", "not", "be", "replicated", "by", "my", "local", "home", "depot", ".", "but", "this", "place", "was", "able", "to", "make", "a", "replica", "of", "the", "key", "with", "no", "problem", "at", "all", ".", "i", "called", "in", "advance", "to", "see", "if", "they", "did", "make", "copies", "and", "the", "associate", "who", "picked", "up", "was", "very", "nice", "and", "also", "aided", "me", "and", "directions", "to", "the"]`
- Custom regex: `["i", "just", "walked", "in", "to", "get", "a", "copy", "of", "a", "key", "that", "i", "had", "which", "could", "not", "be", "replicated", "by", "my", "local", "home", "depot", "but", "this", "place", "was", "able", "to", "make", "a", "replica", "of", "the", "key", "with", "no", "problem", "at", "all", "i", "called", "in", "advance", "to", "see", "if", "they", "did", "make", "copies", "and", "the", "associate", "who", "picked", "up", "was", "very", "nice", "and", "also", "aided", "me", "and", "directions", "to", "the"]`

## Unrestricted random review 3

Source dataset row: 3,590,220

Input: "I had the best time today at Urban Oasis.  I received a great massage and manicure.  I will definitely be back.   Thanks Urban Oasis and Sharing Spree.  Great place!"

- WordPunctTokenizer: `["i", "had", "the", "best", "time", "today", "at", "urban", "oasis", ".", "i", "received", "a", "great", "massage", "and", "manicure", ".", "i", "will", "definitely", "be", "back", ".", "thanks", "urban", "oasis", "and", "sharing", "spree", ".", "great", "place", "!"]`
- word_tokenize: `["i", "had", "the", "best", "time", "today", "at", "urban", "oasis", ".", "i", "received", "a", "great", "massage", "and", "manicure", ".", "i", "will", "definitely", "be", "back", ".", "thanks", "urban", "oasis", "and", "sharing", "spree", ".", "great", "place", "!"]`
- Custom regex: `["i", "had", "the", "best", "time", "today", "at", "urban", "oasis", "i", "received", "a", "great", "massage", "and", "manicure", "i", "will", "definitely", "be", "back", "thanks", "urban", "oasis", "and", "sharing", "spree", "great", "place"]`

## Contractions and possessives

Source dataset row: 4,652,139

Input: "I really want to like this place.  I've been to this location quite a few times, but unfortunately nothing on their menu really sets them apart.  While the food is okay, the service from the staff is better.  Based on my own personal experiences, I would say that they are almost as friendly as the folks at Chic-Fil-A"

- WordPunctTokenizer: `["i", "really", "want", "to", "like", "this", "place", ".", "i", "'", "ve", "been", "to", "this", "location", "quite", "a", "few", "times", ",", "but", "unfortunately", "nothing", "on", "their", "menu", "really", "sets", "them", "apart", ".", "while", "the", "food", "is", "okay", ",", "the", "service", "from", "the", "staff", "is", "better", ".", "based", "on", "my", "own", "personal", "experiences", ",", "i", "would", "say", "that", "they", "are", "almost", "as", "friendly", "as", "the", "folks", "at", "chic", "-", "fil", "-", "a"]`
- word_tokenize: `["i", "really", "want", "to", "like", "this", "place", ".", "i", "'ve", "been", "to", "this", "location", "quite", "a", "few", "times", ",", "but", "unfortunately", "nothing", "on", "their", "menu", "really", "sets", "them", "apart", ".", "while", "the", "food", "is", "okay", ",", "the", "service", "from", "the", "staff", "is", "better", ".", "based", "on", "my", "own", "personal", "experiences", ",", "i", "would", "say", "that", "they", "are", "almost", "as", "friendly", "as", "the", "folks", "at", "chic-fil-a"]`
- Custom regex: `["i", "really", "want", "to", "like", "this", "place", "i've", "been", "to", "this", "location", "quite", "a", "few", "times", "but", "unfortunately", "nothing", "on", "their", "menu", "really", "sets", "them", "apart", "while", "the", "food", "is", "okay", "the", "service", "from", "the", "staff", "is", "better", "based", "on", "my", "own", "personal", "experiences", "i", "would", "say", "that", "they", "are", "almost", "as", "friendly", "as", "the", "folks", "at", "chic-fil-a"]`

## Alphabetic hyphens

Source dataset row: 709,746

Input: "justice. The truffle fries were a nice touch as well that complimented the burger perfectly.\n\nWhile the burger was the star of the show - there were many co-stars....all capable of being starts if you decide to cast them...\n\nCarpaccio - Man I love me some carpaccio and this one was a home run! \n \nOctopus - Perfectly"

- WordPunctTokenizer: `["justice", ".", "the", "truffle", "fries", "were", "a", "nice", "touch", "as", "well", "that", "complimented", "the", "burger", "perfectly", ".", "while", "the", "burger", "was", "the", "star", "of", "the", "show", "-", "there", "were", "many", "co", "-", "stars", "....", "all", "capable", "of", "being", "starts", "if", "you", "decide", "to", "cast", "them", "...", "carpaccio", "-", "man", "i", "love", "me", "some", "carpaccio", "and", "this", "one", "was", "a", "home", "run", "!", "octopus", "-", "perfectly"]`
- word_tokenize: `["justice", ".", "the", "truffle", "fries", "were", "a", "nice", "touch", "as", "well", "that", "complimented", "the", "burger", "perfectly", ".", "while", "the", "burger", "was", "the", "star", "of", "the", "show", "-", "there", "were", "many", "co-stars", "....", "all", "capable", "of", "being", "starts", "if", "you", "decide", "to", "cast", "them", "...", "carpaccio", "-", "man", "i", "love", "me", "some", "carpaccio", "and", "this", "one", "was", "a", "home", "run", "!", "octopus", "-", "perfectly"]`
- Custom regex: `["justice", "the", "truffle", "fries", "were", "a", "nice", "touch", "as", "well", "that", "complimented", "the", "burger", "perfectly", "while", "the", "burger", "was", "the", "star", "of", "the", "show", "there", "were", "many", "co-stars", "<EXCLAMATION>", "all", "capable", "of", "being", "starts", "if", "you", "decide", "to", "cast", "them", "<EXCLAMATION>", "carpaccio", "man", "i", "love", "me", "some", "carpaccio", "and", "this", "one", "was", "a", "home", "run", "octopus", "perfectly"]`

## Hyphens containing digits

Source dataset row: 608,462

Input: "decided to give it a try. This place is so cute and I loved watching the staff make the cookies. The line was very long, but moved swiftly. They allow about 7-10\nPeople to come in, at a time, which keeps the place from being overcrowded on the inside. The service is great and the staff were very helpful.  I loved"

- WordPunctTokenizer: `["decided", "to", "give", "it", "a", "try", ".", "this", "place", "is", "so", "cute", "and", "i", "loved", "watching", "the", "staff", "make", "the", "cookies", ".", "the", "line", "was", "very", "long", ",", "but", "moved", "swiftly", ".", "they", "allow", "about", "7", "-", "10", "people", "to", "come", "in", ",", "at", "a", "time", ",", "which", "keeps", "the", "place", "from", "being", "overcrowded", "on", "the", "inside", ".", "the", "service", "is", "great", "and", "the", "staff", "were", "very", "helpful", ".", "i", "loved"]`
- word_tokenize: `["decided", "to", "give", "it", "a", "try", ".", "this", "place", "is", "so", "cute", "and", "i", "loved", "watching", "the", "staff", "make", "the", "cookies", ".", "the", "line", "was", "very", "long", ",", "but", "moved", "swiftly", ".", "they", "allow", "about", "7-10", "people", "to", "come", "in", ",", "at", "a", "time", ",", "which", "keeps", "the", "place", "from", "being", "overcrowded", "on", "the", "inside", ".", "the", "service", "is", "great", "and", "the", "staff", "were", "very", "helpful", ".", "i", "loved"]`
- Custom regex: `["decided", "to", "give", "it", "a", "try", "this", "place", "is", "so", "cute", "and", "i", "loved", "watching", "the", "staff", "make", "the", "cookies", "the", "line", "was", "very", "long", "but", "moved", "swiftly", "they", "allow", "about", "7", "10", "people", "to", "come", "in", "at", "a", "time", "which", "keeps", "the", "place", "from", "being", "overcrowded", "on", "the", "inside", "the", "service", "is", "great", "and", "the", "staff", "were", "very", "helpful", "i", "loved"]`

## Abbreviations and acronyms

Source dataset row: 1,474,210

Input: "x-rays, and an MRI had done nothing to help. I ended up just googling chiropractors in Indianapolis, and ended up at Compass because of the good reviews. Dr. Bingham and the rest of his staff are miracle workers. My pain went away completely for the first time ever. It is a very science-based office. There's no"

- WordPunctTokenizer: `["x", "-", "rays", ",", "and", "an", "mri", "had", "done", "nothing", "to", "help", ".", "i", "ended", "up", "just", "googling", "chiropractors", "in", "indianapolis", ",", "and", "ended", "up", "at", "compass", "because", "of", "the", "good", "reviews", ".", "dr", ".", "bingham", "and", "the", "rest", "of", "his", "staff", "are", "miracle", "workers", ".", "my", "pain", "went", "away", "completely", "for", "the", "first", "time", "ever", ".", "it", "is", "a", "very", "science", "-", "based", "office", ".", "there", "'", "s", "no"]`
- word_tokenize: `["x-rays", ",", "and", "an", "mri", "had", "done", "nothing", "to", "help", ".", "i", "ended", "up", "just", "googling", "chiropractors", "in", "indianapolis", ",", "and", "ended", "up", "at", "compass", "because", "of", "the", "good", "reviews", ".", "dr.", "bingham", "and", "the", "rest", "of", "his", "staff", "are", "miracle", "workers", ".", "my", "pain", "went", "away", "completely", "for", "the", "first", "time", "ever", ".", "it", "is", "a", "very", "science-based", "office", ".", "there", "'s", "no"]`
- Custom regex: `["x-rays", "and", "an", "mri", "had", "done", "nothing", "to", "help", "i", "ended", "up", "just", "googling", "chiropractors", "in", "indianapolis", "and", "ended", "up", "at", "compass", "because", "of", "the", "good", "reviews", "dr", "bingham", "and", "the", "rest", "of", "his", "staff", "are", "miracle", "workers", "my", "pain", "went", "away", "completely", "for", "the", "first", "time", "ever", "it", "is", "a", "very", "science-based", "office", "there's", "no"]`

## URLs

Source dataset row: 5,653,849

Input: "and healthy food, even when you're in a rush. They post their menus online, so it's easy to see what they have as the daily specials (http://www.picniceats.com/daily-specials/), although their standard fare is quite food as well.\n\nThe only *potential* downside is that the good food does come at a price, but for"

- WordPunctTokenizer: `["and", "healthy", "food", ",", "even", "when", "you", "'", "re", "in", "a", "rush", ".", "they", "post", "their", "menus", "online", ",", "so", "it", "'", "s", "easy", "to", "see", "what", "they", "have", "as", "the", "daily", "specials", "(", "http", "://", "www", ".", "picniceats", ".", "com", "/", "daily", "-", "specials", "/),", "although", "their", "standard", "fare", "is", "quite", "food", "as", "well", ".", "the", "only", "*", "potential", "*", "downside", "is", "that", "the", "good", "food", "does", "come", "at", "a", "price", ",", "but", "for"]`
- word_tokenize: `["and", "healthy", "food", ",", "even", "when", "you", "'re", "in", "a", "rush", ".", "they", "post", "their", "menus", "online", ",", "so", "it", "'s", "easy", "to", "see", "what", "they", "have", "as", "the", "daily", "specials", "(", "http", ":", "//www.picniceats.com/daily-specials/", ")", ",", "although", "their", "standard", "fare", "is", "quite", "food", "as", "well", ".", "the", "only", "*", "potential", "*", "downside", "is", "that", "the", "good", "food", "does", "come", "at", "a", "price", ",", "but", "for"]`
- Custom regex: `["and", "healthy", "food", "even", "when", "you're", "in", "a", "rush", "they", "post", "their", "menus", "online", "so", "it's", "easy", "to", "see", "what", "they", "have", "as", "the", "daily", "specials", "<URL>", "picniceats.com", "although", "their", "standard", "fare", "is", "quite", "food", "as", "well", "the", "only", "potential", "downside", "is", "that", "the", "good", "food", "does", "come", "at", "a", "price", "but", "for"]`

## Emails

Source dataset row: 519,541

Input: "Was told to wear a mask but the employee didn't have one on when I walked in and as she was talking to me it was below her nose.  I have 8 kids and have spent thousands of dollars at this location and the previous location. My email is redacted@yahoo.com if you would like to talk about this issue."

- WordPunctTokenizer: `["was", "told", "to", "wear", "a", "mask", "but", "the", "employee", "didn", "'", "t", "have", "one", "on", "when", "i", "walked", "in", "and", "as", "she", "was", "talking", "to", "me", "it", "was", "below", "her", "nose", ".", "i", "have", "8", "kids", "and", "have", "spent", "thousands", "of", "dollars", "at", "this", "location", "and", "the", "previous", "location", ".", "my", "email", "is", "redacted", "@", "yahoo", ".", "com", "if", "you", "would", "like", "to", "talk", "about", "this", "issue", "."]`
- word_tokenize: `["was", "told", "to", "wear", "a", "mask", "but", "the", "employee", "did", "n't", "have", "one", "on", "when", "i", "walked", "in", "and", "as", "she", "was", "talking", "to", "me", "it", "was", "below", "her", "nose", ".", "i", "have", "8", "kids", "and", "have", "spent", "thousands", "of", "dollars", "at", "this", "location", "and", "the", "previous", "location", ".", "my", "email", "is", "redacted", "@", "yahoo.com", "if", "you", "would", "like", "to", "talk", "about", "this", "issue", "."]`
- Custom regex: `["was", "told", "to", "wear", "a", "mask", "but", "the", "employee", "didn't", "have", "one", "on", "when", "i", "walked", "in", "and", "as", "she", "was", "talking", "to", "me", "it", "was", "below", "her", "nose", "i", "have", "8", "kids", "and", "have", "spent", "thousands", "of", "dollars", "at", "this", "location", "and", "the", "previous", "location", "my", "email", "is", "<EMAIL>", "if", "you", "would", "like", "to", "talk", "about", "this", "issue"]`

## Hashtags or mentions

Source dataset row: 6,253,576

Input: "I can NOT wait for them to reopen in a couple months y'all! This is my favorite local spot in Nashville. Cash only, so be prepared! But such a fun joint, and the #1 place I've missed since covid !!! Friendly atmosphere, and just my favorite place to end a Nashville night."

- WordPunctTokenizer: `["i", "can", "not", "wait", "for", "them", "to", "reopen", "in", "a", "couple", "months", "y", "'", "all", "!", "this", "is", "my", "favorite", "local", "spot", "in", "nashville", ".", "cash", "only", ",", "so", "be", "prepared", "!", "but", "such", "a", "fun", "joint", ",", "and", "the", "#", "1", "place", "i", "'", "ve", "missed", "since", "covid", "!!!", "friendly", "atmosphere", ",", "and", "just", "my", "favorite", "place", "to", "end", "a", "nashville", "night", "."]`
- word_tokenize: `["i", "can", "not", "wait", "for", "them", "to", "reopen", "in", "a", "couple", "months", "y'all", "!", "this", "is", "my", "favorite", "local", "spot", "in", "nashville", ".", "cash", "only", ",", "so", "be", "prepared", "!", "but", "such", "a", "fun", "joint", ",", "and", "the", "#", "1", "place", "i", "'ve", "missed", "since", "covid", "!", "!", "!", "friendly", "atmosphere", ",", "and", "just", "my", "favorite", "place", "to", "end", "a", "nashville", "night", "."]`
- Custom regex: `["i", "can", "not", "wait", "for", "them", "to", "reopen", "in", "a", "couple", "months", "y'all", "this", "is", "my", "favorite", "local", "spot", "in", "nashville", "cash", "only", "so", "be", "prepared", "but", "such", "a", "fun", "joint", "and", "the", "1", "place", "i've", "missed", "since", "covid", "<EXCLAMATION>", "friendly", "atmosphere", "and", "just", "my", "favorite", "place", "to", "end", "a", "nashville", "night"]`

## Currency or percentages

Source dataset row: 2,079,434

Input: "offended, and shut me down with the inferred statement, \"That's our price... take it or leave it.\"  When I balked at the price Cecil initially offered me a $900 dollar discount.\"  Then, he suddenly became irate and said to me, \"I don't want to do business with you.\"  In  a matter of seconds, Cecil did a Jekyl and"

- WordPunctTokenizer: `["offended", ",", "and", "shut", "me", "down", "with", "the", "inferred", "statement", ",", "\"", "that", "'", "s", "our", "price", "...", "take", "it", "or", "leave", "it", ".\"", "when", "i", "balked", "at", "the", "price", "cecil", "initially", "offered", "me", "a", "$", "900", "dollar", "discount", ".\"", "then", ",", "he", "suddenly", "became", "irate", "and", "said", "to", "me", ",", "\"", "i", "don", "'", "t", "want", "to", "do", "business", "with", "you", ".\"", "in", "a", "matter", "of", "seconds", ",", "cecil", "did", "a", "jekyl", "and"]`
- word_tokenize: `["offended", ",", "and", "shut", "me", "down", "with", "the", "inferred", "statement", ",", "``", "that", "'s", "our", "price", "...", "take", "it", "or", "leave", "it", ".", "''", "when", "i", "balked", "at", "the", "price", "cecil", "initially", "offered", "me", "a", "$", "900", "dollar", "discount", ".", "''", "then", ",", "he", "suddenly", "became", "irate", "and", "said", "to", "me", ",", "``", "i", "do", "n't", "want", "to", "do", "business", "with", "you", ".", "''", "in", "a", "matter", "of", "seconds", ",", "cecil", "did", "a", "jekyl", "and"]`
- Custom regex: `["offended", "and", "shut", "me", "down", "with", "the", "inferred", "statement", "that's", "our", "price", "<EXCLAMATION>", "take", "it", "or", "leave", "it", "when", "i", "balked", "at", "the", "price", "cecil", "initially", "offered", "me", "a", "900", "usd", "dollar", "discount", "then", "he", "suddenly", "became", "irate", "and", "said", "to", "me", "i", "don't", "want", "to", "do", "business", "with", "you", "in", "a", "matter", "of", "seconds", "cecil", "did", "a", "jekyl", "and"]`

## Numbers with attached units

Source dataset row: 6,472,835

Input: "2nd time I've been there. Great service and very nice people! Chris, thank you for your help today! We will be back for sure!"

- WordPunctTokenizer: `["2nd", "time", "i", "'", "ve", "been", "there", ".", "great", "service", "and", "very", "nice", "people", "!", "chris", ",", "thank", "you", "for", "your", "help", "today", "!", "we", "will", "be", "back", "for", "sure", "!"]`
- word_tokenize: `["2nd", "time", "i", "'ve", "been", "there", ".", "great", "service", "and", "very", "nice", "people", "!", "chris", ",", "thank", "you", "for", "your", "help", "today", "!", "we", "will", "be", "back", "for", "sure", "!"]`
- Custom regex: `["2", "nd", "time", "i've", "been", "there", "great", "service", "and", "very", "nice", "people", "chris", "thank", "you", "for", "your", "help", "today", "we", "will", "be", "back", "for", "sure"]`

## Dates

Source dataset row: 5,546,931

Input: "Rented a car on 7/17/15 was in route out of state and noticed an expired registration, expired inspection, and no emission.  Was told we did not need emission and that they would fax evidence the registration was pending to the closest office to our destination on 7/17 LIES! It was not sent. Was told the local office"

- WordPunctTokenizer: `["rented", "a", "car", "on", "7", "/", "17", "/", "15", "was", "in", "route", "out", "of", "state", "and", "noticed", "an", "expired", "registration", ",", "expired", "inspection", ",", "and", "no", "emission", ".", "was", "told", "we", "did", "not", "need", "emission", "and", "that", "they", "would", "fax", "evidence", "the", "registration", "was", "pending", "to", "the", "closest", "office", "to", "our", "destination", "on", "7", "/", "17", "lies", "!", "it", "was", "not", "sent", ".", "was", "told", "the", "local", "office"]`
- word_tokenize: `["rented", "a", "car", "on", "7/17/15", "was", "in", "route", "out", "of", "state", "and", "noticed", "an", "expired", "registration", ",", "expired", "inspection", ",", "and", "no", "emission", ".", "was", "told", "we", "did", "not", "need", "emission", "and", "that", "they", "would", "fax", "evidence", "the", "registration", "was", "pending", "to", "the", "closest", "office", "to", "our", "destination", "on", "7/17", "lies", "!", "it", "was", "not", "sent", ".", "was", "told", "the", "local", "office"]`
- Custom regex: `["rented", "a", "car", "on", "7/17/15", "was", "in", "route", "out", "of", "state", "and", "noticed", "an", "expired", "registration", "expired", "inspection", "and", "no", "emission", "was", "told", "we", "did", "not", "need", "emission", "and", "that", "they", "would", "fax", "evidence", "the", "registration", "was", "pending", "to", "the", "closest", "office", "to", "our", "destination", "on", "7", "17", "lies", "it", "was", "not", "sent", "was", "told", "the", "local", "office"]`

## Non-date slashes

Source dataset row: 5,435,079

Input: "to absolutely nothing and good ol' BMW brake dust corroded into the wheel or some other form of German wizardry happened. So, I was left with this melted metal/rust/what have you on my wheels that would not come off. I used all sorts of cleaners and even steel wool, which at that point was just cutting down into bare"

- WordPunctTokenizer: `["to", "absolutely", "nothing", "and", "good", "ol", "'", "bmw", "brake", "dust", "corroded", "into", "the", "wheel", "or", "some", "other", "form", "of", "german", "wizardry", "happened", ".", "so", ",", "i", "was", "left", "with", "this", "melted", "metal", "/", "rust", "/", "what", "have", "you", "on", "my", "wheels", "that", "would", "not", "come", "off", ".", "i", "used", "all", "sorts", "of", "cleaners", "and", "even", "steel", "wool", ",", "which", "at", "that", "point", "was", "just", "cutting", "down", "into", "bare"]`
- word_tokenize: `["to", "absolutely", "nothing", "and", "good", "ol", "'", "bmw", "brake", "dust", "corroded", "into", "the", "wheel", "or", "some", "other", "form", "of", "german", "wizardry", "happened", ".", "so", ",", "i", "was", "left", "with", "this", "melted", "metal/rust/what", "have", "you", "on", "my", "wheels", "that", "would", "not", "come", "off", ".", "i", "used", "all", "sorts", "of", "cleaners", "and", "even", "steel", "wool", ",", "which", "at", "that", "point", "was", "just", "cutting", "down", "into", "bare"]`
- Custom regex: `["to", "absolutely", "nothing", "and", "good", "ol", "bmw", "brake", "dust", "corroded", "into", "the", "wheel", "or", "some", "other", "form", "of", "german", "wizardry", "happened", "so", "i", "was", "left", "with", "this", "melted", "metal", "rust", "what", "have", "you", "on", "my", "wheels", "that", "would", "not", "come", "off", "i", "used", "all", "sorts", "of", "cleaners", "and", "even", "steel", "wool", "which", "at", "that", "point", "was", "just", "cutting", "down", "into", "bare"]`

## Expressive punctuation

Source dataset row: 5,680,257

Input: "The quality of the food is fantastic and they accommodate to many different dietary restrictions. The inside doesn't have the greatest appeal, but once you look past that, you will be pleasantly surprised. Anytime my body is feeling depleted, this is my go-to spot! My two favorites... a #12 smoothie and a happy bowl!"

- WordPunctTokenizer: `["the", "quality", "of", "the", "food", "is", "fantastic", "and", "they", "accommodate", "to", "many", "different", "dietary", "restrictions", ".", "the", "inside", "doesn", "'", "t", "have", "the", "greatest", "appeal", ",", "but", "once", "you", "look", "past", "that", ",", "you", "will", "be", "pleasantly", "surprised", ".", "anytime", "my", "body", "is", "feeling", "depleted", ",", "this", "is", "my", "go", "-", "to", "spot", "!", "my", "two", "favorites", "...", "a", "#", "12", "smoothie", "and", "a", "happy", "bowl", "!"]`
- word_tokenize: `["the", "quality", "of", "the", "food", "is", "fantastic", "and", "they", "accommodate", "to", "many", "different", "dietary", "restrictions", ".", "the", "inside", "does", "n't", "have", "the", "greatest", "appeal", ",", "but", "once", "you", "look", "past", "that", ",", "you", "will", "be", "pleasantly", "surprised", ".", "anytime", "my", "body", "is", "feeling", "depleted", ",", "this", "is", "my", "go-to", "spot", "!", "my", "two", "favorites", "...", "a", "#", "12", "smoothie", "and", "a", "happy", "bowl", "!"]`
- Custom regex: `["the", "quality", "of", "the", "food", "is", "fantastic", "and", "they", "accommodate", "to", "many", "different", "dietary", "restrictions", "the", "inside", "doesn't", "have", "the", "greatest", "appeal", "but", "once", "you", "look", "past", "that", "you", "will", "be", "pleasantly", "surprised", "anytime", "my", "body", "is", "feeling", "depleted", "this", "is", "my", "go-to", "spot", "my", "two", "favorites", "<EXCLAMATION>", "a", "12", "smoothie", "and", "a", "happy", "bowl"]`

## Repeated letters

Source dataset row: 6,214,620

Input: "Everyone is so kind and welcoming. It is clean, comfortable, and their products are great quality. Everyone is great but I always see Lauren because she is so friendly and so patient- she always helps me bring my vision to life and is so so sooo talented. She takes her time with EVERYTHING and always delivers."

- WordPunctTokenizer: `["everyone", "is", "so", "kind", "and", "welcoming", ".", "it", "is", "clean", ",", "comfortable", ",", "and", "their", "products", "are", "great", "quality", ".", "everyone", "is", "great", "but", "i", "always", "see", "lauren", "because", "she", "is", "so", "friendly", "and", "so", "patient", "-", "she", "always", "helps", "me", "bring", "my", "vision", "to", "life", "and", "is", "so", "so", "sooo", "talented", ".", "she", "takes", "her", "time", "with", "everything", "and", "always", "delivers", "."]`
- word_tokenize: `["everyone", "is", "so", "kind", "and", "welcoming", ".", "it", "is", "clean", ",", "comfortable", ",", "and", "their", "products", "are", "great", "quality", ".", "everyone", "is", "great", "but", "i", "always", "see", "lauren", "because", "she", "is", "so", "friendly", "and", "so", "patient-", "she", "always", "helps", "me", "bring", "my", "vision", "to", "life", "and", "is", "so", "so", "sooo", "talented", ".", "she", "takes", "her", "time", "with", "everything", "and", "always", "delivers", "."]`
- Custom regex: `["everyone", "is", "so", "kind", "and", "welcoming", "it", "is", "clean", "comfortable", "and", "their", "products", "are", "great", "quality", "everyone", "is", "great", "but", "i", "always", "see", "lauren", "because", "she", "is", "so", "friendly", "and", "so", "patient", "she", "always", "helps", "me", "bring", "my", "vision", "to", "life", "and", "is", "so", "so", "soo", "talented", "she", "takes", "her", "time", "with", "everything", "and", "always", "delivers"]`

## Unicode words

Source dataset row: 860,242

Input: "wait to be seated. I feel like the waitresses should do a better job of watching for guests who need to be seated. \n\nPersonal note: my sister and her fiancé went here on their first date, and they also got engaged here (a couple years later, of course). So if you're looking to fall in love or get married,"

- WordPunctTokenizer: `["wait", "to", "be", "seated", ".", "i", "feel", "like", "the", "waitresses", "should", "do", "a", "better", "job", "of", "watching", "for", "guests", "who", "need", "to", "be", "seated", ".", "personal", "note", ":", "my", "sister", "and", "her", "fiancé", "went", "here", "on", "their", "first", "date", ",", "and", "they", "also", "got", "engaged", "here", "(", "a", "couple", "years", "later", ",", "of", "course", ").", "so", "if", "you", "'", "re", "looking", "to", "fall", "in", "love", "or", "get", "married", ","]`
- word_tokenize: `["wait", "to", "be", "seated", ".", "i", "feel", "like", "the", "waitresses", "should", "do", "a", "better", "job", "of", "watching", "for", "guests", "who", "need", "to", "be", "seated", ".", "personal", "note", ":", "my", "sister", "and", "her", "fiancé", "went", "here", "on", "their", "first", "date", ",", "and", "they", "also", "got", "engaged", "here", "(", "a", "couple", "years", "later", ",", "of", "course", ")", ".", "so", "if", "you", "'re", "looking", "to", "fall", "in", "love", "or", "get", "married", ","]`
- Custom regex: `["wait", "to", "be", "seated", "i", "feel", "like", "the", "waitresses", "should", "do", "a", "better", "job", "of", "watching", "for", "guests", "who", "need", "to", "be", "seated", "personal", "note", "my", "sister", "and", "her", "fiancé", "went", "here", "on", "their", "first", "date", "and", "they", "also", "got", "engaged", "here", "a", "couple", "years", "later", "of", "course", "so", "if", "you're", "looking", "to", "fall", "in", "love", "or", "get", "married"]`

## Irregular whitespace

Source dataset row: 1,765,076

Input: "it. I recommend the grilled cheese and fries - hot, greasy, and no nonsense. They have a good deal on 6 mini cheeseburgers too. Just don't think too much about the fact that they don't use gloves and they keep open meat piled high in the refrigerator. \n\nEat Rite is a St. Louis experience that I recommend to everyone"

- WordPunctTokenizer: `["it", ".", "i", "recommend", "the", "grilled", "cheese", "and", "fries", "-", "hot", ",", "greasy", ",", "and", "no", "nonsense", ".", "they", "have", "a", "good", "deal", "on", "6", "mini", "cheeseburgers", "too", ".", "just", "don", "'", "t", "think", "too", "much", "about", "the", "fact", "that", "they", "don", "'", "t", "use", "gloves", "and", "they", "keep", "open", "meat", "piled", "high", "in", "the", "refrigerator", ".", "eat", "rite", "is", "a", "st", ".", "louis", "experience", "that", "i", "recommend", "to", "everyone"]`
- word_tokenize: `["it", ".", "i", "recommend", "the", "grilled", "cheese", "and", "fries", "-", "hot", ",", "greasy", ",", "and", "no", "nonsense", ".", "they", "have", "a", "good", "deal", "on", "6", "mini", "cheeseburgers", "too", ".", "just", "do", "n't", "think", "too", "much", "about", "the", "fact", "that", "they", "do", "n't", "use", "gloves", "and", "they", "keep", "open", "meat", "piled", "high", "in", "the", "refrigerator", ".", "eat", "rite", "is", "a", "st.", "louis", "experience", "that", "i", "recommend", "to", "everyone"]`
- Custom regex: `["it", "i", "recommend", "the", "grilled", "cheese", "and", "fries", "hot", "greasy", "and", "no", "nonsense", "they", "have", "a", "good", "deal", "on", "6", "mini", "cheeseburgers", "too", "just", "don't", "think", "too", "much", "about", "the", "fact", "that", "they", "don't", "use", "gloves", "and", "they", "keep", "open", "meat", "piled", "high", "in", "the", "refrigerator", "eat", "rite", "is", "a", "st", "louis", "experience", "that", "i", "recommend", "to", "everyone"]`

## Sample limitation

No matching review was found in the random qualitative pool of 100,000 reviews for: Pictographic emoji graphemes. No constructed example was substituted.
