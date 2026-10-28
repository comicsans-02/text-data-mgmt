# Task 3.4.2: associations with 'service'

Context: both words occur anywhere in the same review, once per review.
PMI = log2(cooccurrence_reviews * reviews / (target_reviews * word_reviews)).
Minimum joint reviews per word per year: 20. Top N: 10.
Positive PMI means more co-occurrence than expected under independence; it is not sentiment.

- 2005: 854 reviews; 107 with target. excellent (1.784), menu (1.042), food (0.767), great (0.566), one (0.288), good (0.189), best (0.189), like (0.067), not (0.028), place (-0.115)
- 2006: 3,853 reviews; 605 with target. customer (2.256), slow (2.226), attentive (1.501), waiter (1.408), extremely (1.349), fast (1.314), quick (1.256), entree (1.234), waitress (1.190), asked (1.189)
- 2007: 15,363 reviews; 2,696 with target. spotty (2.456), impeccable (2.236), prompt (2.065), slow (1.949), refill (1.886), customer (1.814), promptly (1.695), attentive (1.571), efficient (1.544), notch (1.530)
- 2008: 48,226 reviews; 9,783 with target. spotty (2.022), impeccable (2.012), inattentive (1.799), prompt (1.768), slow (1.715), customer (1.694), speedy (1.670), refilled (1.662), intrusive (1.638), accommodated (1.591)
- 2009: 74,387 reviews; 17,878 with target. brad's (2.000), spotty (1.885), impeccable (1.730), attentiveness (1.642), prompt (1.629), unobtrusive (1.593), speedy (1.529), customer (1.528), inattentive (1.516), shoddy (1.516)
- 2010: 138,587 reviews; 34,757 with target. spotty (1.806), impeccable (1.737), loaner (1.673), bussers (1.593), prompt (1.576), maitre'd (1.548), customer (1.519), capellini (1.510), exemplary (1.504), slow (1.460)
- 2011: 230,813 reviews; 59,299 with target. spotty (1.719), impeccable (1.634), loaner (1.607), exemplary (1.598), unobtrusive (1.564), costumer (1.555), perfunctory (1.546), customer (1.516), prompt (1.504), slow (1.448)
- 2012: 286,570 reviews; 77,336 with target. impeccable (1.623), spotty (1.606), expedient (1.520), customer (1.466), loaner (1.423), slowness (1.395), slow (1.393), prompt (1.388), lackadaisical (1.387), busboy (1.378)
- 2013: 383,950 reviews; 105,070 with target. spork (1.670), spotty (1.649), impeccable (1.577), exemplary (1.489), customer (1.480), slowness (1.465), costumer (1.435), loaner (1.428), servicing (1.363), solicitous (1.359)
- 2014: 522,275 reviews; 146,935 with target. sloooow (1.660), adviser (1.552), impeccable (1.532), spotty (1.504), modem (1.493), costumer (1.457), elapsed (1.451), customer (1.447), loaner (1.430), unattentive (1.415)
- 2015: 688,415 reviews; 194,220 with target. impeccable (1.528), advisors (1.509), abominable (1.504), costumer (1.501), spotty (1.496), customer (1.475), abhorrent (1.442), loaner (1.439), dealership's (1.427), adviser (1.395)
- 2016: 758,882 reviews; 213,976 with target. centurylink (1.706), sloooow (1.527), impeccable (1.526), spotty (1.499), advisors (1.477), customer (1.473), costumer (1.459), loaner (1.435), fios (1.395), serviceman (1.341)
- 2017: 820,048 reviews; 229,788 with target. centurylink (1.643), impecable (1.620), adviser (1.536), costumer (1.536), fios (1.505), impeccable (1.500), customer (1.495), spotty (1.487), brighthouse (1.466), advisors (1.463)
- 2018: 906,362 reviews; 251,161 with target. returnability (1.851), fios (1.635), gaucha (1.598), adviser (1.584), airbag (1.571), sloooow (1.552), impeccable (1.540), costumer (1.519), customer (1.512), spotty (1.507)
- 2019: 907,284 reviews; 251,449 with target. centurylink (1.714), loaners (1.629), impeccable (1.539), sloooow (1.529), customer (1.521), costumer (1.508), advisors (1.502), fios (1.477), spotty (1.462), ehrlich (1.436)
- 2020: 554,557 reviews; 140,107 with target. costumer (1.670), advisor (1.657), customer (1.638), impeccable (1.627), advisors (1.570), centurylink (1.552), exemplary (1.532), airbag (1.530), loaner (1.499), tippers (1.499)
- 2021: 618,189 reviews; 160,534 with target. costumer (1.643), advisors (1.606), customer (1.600), loaner (1.587), impeccable (1.585), trugreen (1.576), advisor (1.561), spotty (1.544), recalls (1.440), flagging (1.412)
- 2022: 31,665 reviews; 7,909 with target. impeccable (1.665), customer (1.632), slower (1.603), costumer (1.586), slow (1.424), skills (1.362), representative (1.354), rep (1.319), prompt (1.214), stellar (1.209)

## Changes in the top-word sets

Jaccard = intersection / union; drift = 1 - Jaccard. This ignores rank order.
Changes below describe membership in the top-N lists, not first/last appearances in the corpus.

- 2005 to 2006: drift 1.000; entered: asked; attentive; customer; entree; extremely; fast; quick; slow; waiter; waitress; exited: best; excellent; food; good; great; like; menu; not; one; place.
- 2006 to 2007: drift 0.824; entered: efficient; impeccable; notch; prompt; promptly; refill; spotty; exited: asked; entree; extremely; fast; quick; waiter; waitress.
- 2007 to 2008: drift 0.667; entered: accommodated; inattentive; intrusive; refilled; speedy; exited: attentive; efficient; notch; promptly; refill.
- 2008 to 2009: drift 0.571; entered: attentiveness; brad's; shoddy; unobtrusive; exited: accommodated; intrusive; refilled; slow.
- 2009 to 2010: drift 0.750; entered: bussers; capellini; exemplary; loaner; maitre'd; slow; exited: attentiveness; brad's; inattentive; shoddy; speedy; unobtrusive.
- 2010 to 2011: drift 0.462; entered: costumer; perfunctory; unobtrusive; exited: bussers; capellini; maitre'd.
- 2011 to 2012: drift 0.571; entered: busboy; expedient; lackadaisical; slowness; exited: costumer; exemplary; perfunctory; unobtrusive.
- 2012 to 2013: drift 0.667; entered: costumer; exemplary; servicing; solicitous; spork; exited: busboy; expedient; lackadaisical; prompt; slow.
- 2013 to 2014: drift 0.667; entered: adviser; elapsed; modem; sloooow; unattentive; exited: exemplary; servicing; slowness; solicitous; spork.
- 2014 to 2015: drift 0.571; entered: abhorrent; abominable; advisors; dealership's; exited: elapsed; modem; sloooow; unattentive.
- 2015 to 2016: drift 0.571; entered: centurylink; fios; serviceman; sloooow; exited: abhorrent; abominable; adviser; dealership's.
- 2016 to 2017: drift 0.462; entered: adviser; brighthouse; impecable; exited: loaner; serviceman; sloooow.
- 2017 to 2018: drift 0.571; entered: airbag; gaucha; returnability; sloooow; exited: advisors; brighthouse; centurylink; impecable.
- 2018 to 2019: drift 0.571; entered: advisors; centurylink; ehrlich; loaners; exited: adviser; airbag; gaucha; returnability.
- 2019 to 2020: drift 0.667; entered: advisor; airbag; exemplary; loaner; tippers; exited: ehrlich; fios; loaners; sloooow; spotty.
- 2020 to 2021: drift 0.571; entered: flagging; recalls; spotty; trugreen; exited: airbag; centurylink; exemplary; tippers.
- 2021 to 2022: drift 0.824; entered: prompt; rep; representative; skills; slow; slower; stellar; exited: advisor; advisors; flagging; loaner; recalls; spotty; trugreen.

## Interpretation for the report

Use the yearly lists to describe changes in meaning or themes, supported by specific words and PMI values. Do not infer sentiment or causation from PMI alone. Inspect review contexts before interpreting ambiguous words.
PMI favors rare words; the minimum co-occurrence threshold reduces but does not eliminate this. Small yearly counts, partial years, changing business composition and the top-N cutoff can mimic drift. Review-level co-occurrence can join words from unrelated sentences. Inflections remain separate. This simple tokenizer differs from the team's task 3.2.2 custom tokenizer.
