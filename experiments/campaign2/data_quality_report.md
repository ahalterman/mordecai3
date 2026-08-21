# Data-quality audit of the Mordecai3 training and evaluation corpora

Written 2026-08-20, after the accuracy campaign closed at 0.9258 macro exact
match (`ACCURACY_CAMPAIGN.md`, `experiments/PLAN.md`). The question this report
answers is not "can the model be better" but "is the number real, and is it
measuring what we want it to measure."

Everything below is measured on the frozen split that every campaign number
uses, and — where predictions are needed — on the shipped checkpoint
`experiments/e29_swa_ep15/seed42.pt`, whose per-entity dump reproduces
`seed42.json` exactly (macro 0.9204; the 0.9258 headline is the 5-seed mean of
the last five epochs, so a single final checkpoint reads slightly lower).

**Headline verdict.** Leakage is *not* a problem: the split is effectively
document-level and cross-corpus duplication is two articles. The problems are
(a) the held-out sets are far easier and far less independent than their N
suggests — 38% of all held-out entities have a *country* as the gold answer and
score 99.85%, and 81% have their exact `(mention, gold id)` pair somewhere in
training, where the model scores 0.960 versus 0.771 on the 19% that are novel;
(b) ~40% of the residual error mass is not model error — 12% of sampled errors
have a wrong gold and 28% are granularity/duplicate-row convention
disagreements; and (c) the corpus identity is linearly recoverable at 93% from
an input the model actually consumes, and the model's A/P convention tracks each
corpus's convention to within a point. The macro-of-six headline is a blend of
one gazetteer-lookup benchmark, one sentence-level corpus, and 72% of its
entities coming from Wikipedia.

---

## 1. The data pipeline and the split

### How the split is made

`tools/train.py::load_es_data` (lines 160–317):

1. Each source's pickle is loaded. Entities are stored **in document order**:
   `format_source` flattens a list-of-lists produced per spaCy doc, so all
   entities of document *k* are contiguous and precede document *k+1*'s.
   Entities whose `correct_geonamesid` is `None` were already dropped at
   pickling time (line 775), so every entity in a pickle carries a gold id.
2. `split_list(es_data, 0.7)` takes a **positional prefix**: the first 70% of
   the *entity* list is train, the last 30% is held out. It is deterministic
   and depends on no seed.
3. Because entities are contiguous by document, an entity-level positional cut
   is *de facto* a document-level cut. Exactly one document straddles the
   boundary per source.
4. Source caps (`--source-limits`) are applied **after** the split, so a capped
   run and an uncapped run score on the same held-out set. The ship recipe caps
   nothing.
5. `Synth` is the exception and the one place the design breaks: the two
   synthetic pickles are **shuffled at entity level** (`random.seed(617)`) and
   then concatenated as `syn_cities[0:500] + syn_caps[0:500]` before the
   positional split. Two consequences below.

### Per source

| source | entities | documents | ents/doc | train ents | held-out ents | held-out docs | gold retrievable (all / held-out) | mean candidates |
|---|---|---|---|---|---|---|---|---|
| Prodigy | 1,668 | 1,220 | 1.37 | 1,168 | 500 | 321 | 0.998 / 1.000 | 73 |
| TR | 914 | 115 | 7.95 | 640 | 274 | 29 | 0.980 / 0.989 | 272 |
| LGL | 3,245 | 553 | 5.87 | 2,272 | 973 | 176 | 0.965 / 0.972 | 299 |
| GWN | 1,580 | 188 | 8.40 | 1,106 | 474 | 58 | 0.972 / 0.970 | 236 |
| Synth | 1,000 | 841 | 1.19 | 700 | 300 | 278 | 0.874 / 0.997 | 61 |
| WikiDocs | 21,521 | 3,373 | 6.38 | 15,065 | 6,456 | 1,019 | 0.984 / 0.985 | 168 |

"Gold retrievable" = at least one candidate flagged `correct`. Exact match is
computed only over retrievable entities (`error_utils.evaluate_results` skips
the rest), so the EM denominators are 500 / 271 / 946 / 460 / 299 / 6,362 =
**8,838 entities**, of which **6,362 (72%) are WikiDocs**. The macro-of-six
average therefore gives a WikiDocs entity 1/12th the weight of a TR entity.

### Two structural facts about the corpora behind the pickles

* **WikiDocs is 647 Wikipedia articles, not 3,556 documents.** The `chunk`
  field splits each article into up to 40 pieces. Titles are contiguous in
  `wiki/wiki_docs.jsonl`, so the split does not mix chunks of one article
  across train/held-out (only `List of earthquakes in 2021` straddles the
  boundary) — but the held-out set is **176 distinct Wikipedia articles**, all
  drawn from three categories (protest 1,837 chunks / disasters 1,049 /
  battles 670). Independent sample size is articles, not chunks.
* **Prodigy is not documents at all.** `orig_mordecai/loc_rank_db.jsonl` is one
  annotation *task* per (sentence, span): 2,664 records with 891 exact-duplicate
  texts and 1,450 near-duplicate pairs, because the same sentence reappears once
  per toponym in it. Median text length is 165 characters. 33.9% of Prodigy
  entities have an **all-zero `locs_tensor`** — no other place name in their
  "document" at all. Document-evidence features (`sib_*`, `geo`) are structurally
  blind here, which is the mechanism behind Prodigy's documented fight with the
  recipe (PLAN.md Wave 1b) and its 0.021 seed SD, the largest of any source.

---

## 2. Leakage hunt

### (a) Entity-level vs document-level splitting — clean except Synth

Document membership recovered by `doc_key` (sha1 of `doc_tensor`, the technique
from PLAN.md Wave 2b).

| source | documents shared by train and held-out | held-out entities in a train document | EM on those | EM on the rest |
|---|---|---|---|---|
| Prodigy | 0 | 0 | – | 0.8900 |
| TR | 1 | 5 | 1.000 | 0.8947 |
| LGL | 1 | 7 | 1.000 | 0.9020 |
| GWN | 1 | 2 | 1.000 | 0.9279 |
| **Synth** | **39** | **42 (14% of its held-out)** | **1.000** | 0.9728 |
| WikiDocs | 2 | 30 | 0.967 | 0.9278 |

Total **86 leaked entities out of 8,838 = 0.97%**, and the model gets 85 of 86
right. Removing them moves the macro from 0.9204 to **0.9192 (−0.0012)**.
Severity: **negligible** everywhere except Synth, where the entity-level shuffle
puts 14% of the held-out set's exact texts into training.

### (b) Cross-source duplication — two articles, and they do not cross the split

Normalised raw text of all 9,562 source records (TR-News.xml, lgl.xml, GWN.xml,
loc_rank_db.jsonl, both synth jsonls, wiki_docs.jsonl) hashed and shingled:

* **Exact duplicates across sources: 2** — `TR#15 ≡ LGL#82` ("Mason charged with
  sex assault. BANTAM — …") and `TR#16 ≡ LGL#83` ("Bingham to offer plan to
  combat drugs. TORRINGTON — …"). Both TR-News and LGL drew from the same US
  local-news pool. Neither pair crosses the train/held-out boundary.
* **Exact duplicates within a source:** Prodigy 891, SynCaps 10, WikiDocs 3.
* **Cross-source `doc_key` collisions: 0** (the trf tensors differ in the last
  bits between batches, so `doc_key` is a weak cross-corpus test — the text hash
  above is the real one).
* **Near-duplicate document pairs crossing train↔held-out: 13, all
  WikiDocs↔WikiDocs** (Jaccard 0.12–0.60 on 5-gram shingles) — Wikipedia
  boilerplate reused across articles: the ICRC history paragraph, "see also"
  link lists, the JNA-withdrawal paragraph. 13 of 1,019 held-out documents =
  **1.3%**, and they are boilerplate, not the toponym-bearing prose.

Severity: **none**. The often-feared "same syndicated article in TR and LGL" is
real but amounts to two articles that both sit on the same side of the split.

### (c) The leakage that *is* real: answer-key memorisation

Document leakage is the wrong thing to worry about. What actually inflates the
number is that the held-out sets re-ask questions the training set already
answered, in the same words.

| source | held-out entities | distinct `(mention, gold id)` pairs | held-out pairs also in **own** train | also in **any** source's train |
|---|---|---|---|---|
| Prodigy | 500 | 186 | 17.2% | 73.2% |
| TR | 274 | 144 | 39.4% | 73.0% |
| LGL | 973 | 360 | 44.4% | 55.1% |
| GWN | 474 | 194 | 50.4% | 76.6% |
| Synth | 300 | 247 | 17.7% | 57.0% |
| WikiDocs | 6,456 | 1,384 | 84.8% | 86.1% |

WikiDocs' 6,456 held-out entities are 1,384 distinct questions; LGL's 973 are
360. Accuracy conditioned on whether the answer was seen in training (any
source):

| source | seen, n | EM | novel, n | EM |
|---|---|---|---|---|
| Prodigy | 366 | 0.9399 | 134 | 0.7537 |
| TR | 199 | 0.9196 | 72 | 0.8333 |
| LGL | 525 | 0.9600 | 421 | 0.8314 |
| GWN | 359 | 0.9861 | 101 | 0.7228 |
| Synth | 171 | 0.9942 | 128 | 0.9531 |
| WikiDocs | 5,516 | 0.9603 | 846 | **0.7175** |
| **pooled** | **7,136** | **0.9602** | **1,702** | **0.7714** |

This is not leakage in the misconduct sense — real geoparsing also re-encounters
Denver — but it means **the headline is ~81% a memorisation benchmark and ~19%
a generalisation benchmark, and the two halves differ by 19 points.** Any claim
about performance on new text should quote the 0.77, not the 0.92. Note also
that 26–56% of each human corpus's held-out answers were supplied by a *different*
corpus's training half (Prodigy 56%, Synth 39%, TR 34%, GWN 26%, LGL 11%), so
the six sources are not six independent tests of the same model.

---

## 3. Label-quality adjudication

Stratified random sample, seed 20260820: 15 wrong + 10 right per source from TR,
LGL, GWN, WikiDocs (60 wrong, 40 right). Each item read with its full document
text, the document's other mentions, the gold row, the predicted row and the
top-5 candidates.

### Result on the 60 errors

| verdict | n | share |
|---|---|---|
| **gold-wrong** (gold indefensible; the model's answer is right or better) | 7 | 11.7% |
| **gold-defensible-but-convention** (A/P twin, duplicate gazetteer rows, historic-vs-current unit — either answer names the same place) | 17 | 28.3% |
| **model-wrong** (gold correct, genuine resolution failure) | 36 | 60.0% |

95% binomial CI on gold-wrong: 5–23%.

### Result on the 40 correct predictions

**0/40 false credits** — every gold was correct. The composition is the story:
10 of 40 are countries (`Republic of France`, `Federal Republic of Nigeria`,
`Somalia`), 5 are US/Indian states, the rest are unambiguous world cities
(`Beirut`, `Las Vegas`, `Houston`, `San Francisco`). All 40 had the gold at
rank 0. The half of the held-out set the model gets right is largely trivial.

### Gold-wrong examples, verbatim

* **TR #202 "Xinmo"** — *"A rescue mission was underway after approximately 40
  homes were buried in Maoxian county in Sichuan province… The landslide hit the
  village of Xinmo."* Gold `6919455` Xinmo, **Zigong Shi**. Model picked
  `10149157` Xinmo, **Aba Zangzu Qiangzu Zizhizhou** — the prefecture that
  contains Maoxian. The model is right; the gold is 300 km off.
* **LGL #586 "Lancaster"** — *"Morecambe and Lancaster shops give away festive
  bells… bells have been handed out in Morecambe and Lancaster… Marketgate…
  Arndale Centre."* Gold `5197079` **Lancaster, Pennsylvania**. Morecambe is in
  Lancashire. The model picked the UK Lancaster and was scored wrong.
* **GWN #392 "Overton"** — *"first date to the Pioneer cinema in Dewsbury… They
  moved to Thornhill, before switching to Middlestown, and then finally their
  bungalow in Overton."* Gold `7294409` Overton, **Cheshire West and Chester**.
  Model picked `12265641` Overton, **Wakefield** — the one 4 km from
  Middlestown. Model right, gold wrong.
* **LGL #779 "New York"** — *"outfit satellite offices for him in Jerusalem and
  in New York, where he lives."* Gold `5128638` = **New York State**. Model said
  New York City and was charged an error. Compare **TR #161**, the same phrase
  type — *"a New York-based FBI squad"* — where the gold *is* New York City and
  the model's state pick is charged an error. The two corpora use opposite
  conventions for the identical construction; the model cannot win both.
* **TR #246 "Varzaqan"** — *"The towns of Haris and Varzaqan in East Azerbaijan
  province were among those that suffered casualties."* Gold `112122` is a
  **PPLQ (abandoned populated place)** row; the model picked `112121`, the
  PPLA2 seat that is the actual town.
* **GWN #190 "South Side"** — *"A South Side man has been charged with murder…
  of the 6100 block of South Ingleside Avenue."* Gold `4903363` **Near South
  Side** (a different Chicago community area); model picked the `South Side` RGN
  row.
* **TR #164 "England"** — gold `2649994`, an `AREA` row for England; the model
  picked `6269131`, the canonical ADM1 England. A duplicate-row artefact.

### Convention examples (28% of errors)

`Moscow` PPLC vs `Moskva` ADM1 (0.6 km); `Shanghai` PPLA vs `Shanghai Shi` ADM1;
`Paris` PPLC vs `Paris` ADM2; `Beijing` PPLC vs `Beijing Shi` ADM1; `Menorca`
ISL vs `Menorca` ADMD; `Leipzig` PPLA3 vs `Kreisfreie Stadt Leipzig` ADM3;
`Snetterton` PPLA4 vs `Snetterton` ADM4; `Normandie` RGN vs `Normandie` ADM1;
`River Lune` — three separate STM/STMX rows for one river; `County of Cheshire`
RGN (historic) vs `Cheshire West and Chester` ADM2 (current).

Note that the campaign's conclusion "the A-side convention is a WikiDocs
phenomenon" (PLAN.md e12/e24) is **too narrow**: GWN annotates UK villages to
their ADM4 civil-parish rows (`Snetterton`, `Rockingham`) and UK counties to
current ADM2 rows, and WikiDocs itself is internally inconsistent (`Leipzig` →
ADM3 city-district, but `Bremen` in the *same document* → PPLA city).

### What this implies about the error budget

The error-distance decomposition over all 673 residual errors agrees with the
sample:

| gold↔prediction distance | n | share |
|---|---|---|
| < 1 km (duplicate gazetteer row) | 75 | 11.1% |
| 1–10 km (granularity: city inside its own unit) | 181 | 26.9% |
| 10–161 km | 177 | 26.3% |
| > 161 km (genuinely different place) | 240 | 35.7% |

38% of errors are same-country and under 10 km. Per source, "under 10 km" is
46% of WikiDocs errors, 36% GWN, 29% TR/Prodigy — but only **9% of LGL errors**,
78% of which are over 161 km. LGL's residual is real disambiguation; WikiDocs'
is half bookkeeping.

**Label-noise ceiling.** ~0.9% of held-out entities have a wrong gold (7/60 of
errors × 673/8,838), which caps strict EM near 0.99 in the abstract — but the
operative number is that **~40% of the remaining 673-error budget cannot be
recovered by better modelling under the current answer key.** The genuinely
addressable residual is ~400 entities, i.e. ~4.5 points, not 7.4.

---

## 4. Idiosyncrasy and overfit risk

### (a) Per-source spread

e29, `_last5`, 5 seeds:

| source | mean EM | seed SD | min | max |
|---|---|---|---|---|
| Synth | 0.9781 | 0.0077 | 0.9706 | 0.9886 |
| GWN | 0.9340 | 0.0062 | 0.9261 | 0.9404 |
| WikiDocs | 0.9284 | 0.0010 | 0.9273 | 0.9295 |
| TR | 0.9103 | 0.0084 | 0.9033 | 0.9247 |
| Prodigy | 0.9066 | 0.0211 | 0.8784 | 0.9260 |
| LGL | 0.8974 | 0.0060 | 0.8913 | 0.9055 |
| **macro** | **0.9258** | 0.0033 | | |

Spread across sources is **8.1 points**, 25× the macro's seed SD. The macro is
dominated by which sources are in it, not by the model.

### (b) Does the model receive a source-identifying feature?

Not by name, but yes in effect.

* `TrainData` feeds `doc_tensor` (the mean spaCy trf token tensor of the whole
  document) straight into `context_to_country` in the scoring path
  (`torch_model.py:501, 516`). A plain logistic regression on `doc_tensor`
  alone predicts **which of the six corpora a document came from with 93.3%
  5-fold accuracy** (majority baseline 53.6%). The corpus label is sitting in
  the model's input, linearly decodable.
* Structural fingerprints reinforce it: 1.37 mentions/doc and 33.9% all-zero
  `locs_tensor` in Prodigy, 1.19 in Synth, vs 6–8 in the news/wiki corpora; and
  median candidate-set size 19 (Prodigy) / 8 (Synth) / 337 (LGL).
* And the model *uses* it. On A/P twin entities (gold has a same-name,
  same-country, co-located partner across feature classes A and P), the model's
  A-side pick rate tracks each corpus's gold A-side rate:

| source | train gold A-share | held-out gold A-share | held-out **model** A-share |
|---|---|---|---|
| Prodigy | 0.068 | 0.122 | 0.111 |
| TR | 0.058 | 0.059 | 0.235 (n=51) |
| LGL | 0.090 | 0.084 | 0.110 |
| GWN | 0.072 | 0.096 | 0.074 |
| WikiDocs | 0.166 | 0.157 | 0.159 |

WikiDocs is annotated A-side twice as often as the news corpora, and the model
reproduces 0.159 against a gold rate of 0.157. This is the mechanism e24 found
("the model learns whichever convention it is trained on") shown to operate
*per corpus, simultaneously*. It is a genuine capability on this benchmark and a
liability off it: on new text of unknown provenance, the model has no convention
to condition on.

### (c) Does the unweighted mean overweight easy data?

Yes, three ways.

1. **Synth's held-out is not the Synth training distribution.** Because the two
   synthetic pickles are concatenated `syn_cities[0:500] + syn_caps[0:500]` and
   then cut at 700, training gets all 500 `syn_cities` + 200 `syn_caps` and the
   **held-out 300 are 100% `syn_caps`** — templates like `"Brasília, Brazil"`,
   `"Rome, Italy"`, `"the capital city of Togo"`, median 37 characters. 54% of
   its golds are countries. This is a gazetteer-lookup test worth 1/6 of the
   headline at 0.978, and it cannot move: no model change reachable from here
   will fail it or fix it.
2. **Country golds are 38% of the entire held-out set and score 99.85%.**

   | source | share of held-out whose gold is a country | share whose gold is a city (class P) |
   |---|---|---|
   | GWN | 53.7% | 30.4% |
   | Synth | 53.8% | 46.2% |
   | WikiDocs | 40.7% | 40.8% |
   | Prodigy | 34.4% | 32.0% |
   | TR | 22.1% | 42.8% |
   | LGL | 15.9% | 55.7% |

   EM by gold type, pooled: country 0.9985 (n=3,377), city 0.9050 (n=3,674),
   ADM1 0.9083, other-admin 0.7386, physical/other 0.6739 (n=276).
3. **The metric's alternatives.** Same checkpoint (seed42, macro 0.9204):

   | metric | value |
   |---|---|
   | macro of 6 sources (current headline) | **0.9204** |
   | …after removing the 86 doc-leaked entities | 0.9192 |
   | macro of TR+LGL+GWN (human news) | **0.9092** |
   | pooled TR+LGL+GWN | 0.9088 |
   | macro of 6, excluding country golds | 0.8776 |
   | macro of TR+LGL+GWN, excluding country golds | **0.8656** |
   | pooled TR+LGL+GWN, excluding country golds (n=1,220) | 0.8746 |
   | macro of 6, city (class P) golds only | 0.8931 |
   | macro of TR+LGL+GWN, city golds only | 0.8495 |
   | pooled, entities whose (mention, gold) is novel (n=1,702) | **0.7714** |

**Recommendation for the next campaign's target metric.** Report a triple, and
optimise the first:

1. **Primary — `TLG-hard`: macro over TR, LGL, GWN of exact match on entities
   whose gold is not a country** (n = 211 + 796 + 213 = 1,220; currently
   0.8656). It removes the two synthetic/pseudo-document corpora, removes the
   Wikipedia weighting distortion, and removes the 38% of the benchmark that is
   already solved. It is the metric with headroom, and it is the one that
   corresponds to the deployment target (news text).
2. **Secondary — twin-credit** on the same slice, so granularity convention
   swaps stop being charged (`tools/twin_credit_eval.py`, already built).
3. **Guardrail — novel-pair EM**: exact match restricted to held-out entities
   whose `(mention, gold id)` pair does not occur in any training source
   (currently 0.7714 pooled). This is the only number in the suite that cannot
   be moved by memorisation, and it should be watched for regressions when
   training data is added.

Keep the current macro-of-six reported for continuity with the campaign ledger,
but stop steering by it. It is 1/6 gazetteer lookup, 1/6 sentence fragments,
and 38% country identification.

---

## 5. Improvement opportunities, ranked

Cost is engineer-days; impact is on the proposed `TLG-hard` primary metric
unless stated.

### 1. Fix the Synth split and re-cut Prodigy at document level — impact: metric integrity, cost: 0.5 day

Two one-line-class changes in `load_es_data`: shuffle-then-split Synth by
document rather than entity (removes the only real leakage, 42 entities), and
either drop `syn_caps`-only held-out or interleave the two synthetic files so
the held-out matches the training distribution. Separately, group Prodigy's
duplicate-text tasks into real documents before splitting so its
`sib_*`/`geo` features have something to see. This does not raise accuracy — it
stops two of the six reported numbers from being uninterpretable. Do it first
because everything else is measured against it.

### 2. Re-adjudicate two specific slices, not the whole corpus — impact: +0.5–1.0 measured, cost: 2–3 days

The adjudication says 12% of errors have a wrong gold and 28% are convention.
Both are cheap to attack because they are concentrated:

* **Duplicate/superseded gazetteer rows** (11% of errors are sub-1 km):
  `England` AREA vs ADM1, `Normandie` RGN vs the 2016 ADM1, three `River Lune`
  rows, `Menorca` ISL vs ADMD, the D.C. cluster and Mauna Kea already named in
  ACCURACY_CAMPAIGN.md item 3. Collapse them in the ES index build and remap the
  gold ids. This is item 4 of PLAN.md's second-campaign list and it is still the
  best mechanical return in the repo.
* **The ~0.9% of golds that are simply wrong.** Screen candidates automatically —
  gold row is `PPLQ`/`*H`/historical, or gold is >100 km from every sibling
  anchor in its own document while a same-name candidate sits inside them — and
  hand-adjudicate the shortlist. On the TR/LGL/GWN slice that is a few hundred
  rows.

Do **not** attempt a global convention rewrite: e24 already proved that moving
golds within a twin class is a convention swap worth exactly 0.0000 under
twin-credit.

### 3. A held-out corpus the model has never seen the answers to — impact: the credibility of every future number, cost: 3–5 days

The single most valuable data artefact this project does not have. 1,702 of
8,838 held-out entities are novel `(mention, gold)` pairs and the model scores
0.771 on them; that is the real generalisation estimate and it rests on an
incidental slice. Build a small (300–600 entity) modern-news evaluation set,
annotated once, never trained on, drawn from outlets and years absent from
TR/LGL/GWN, deliberately dense in the failure classes we know about (US
same-name cities, UK/Commonwealth homographs like `Edmonton`/`Lancaster`, and
non-Anglophone toponyms). Annotate with an explicit written convention for A/P
granularity so it does not inherit the six-way convention muddle. Without this,
"0.926" cannot be defended in a paper against the memorisation numbers in §2(c).

### 4. Publication-metadata / corpus-prior features for the no-evidence residual — impact: +0.5–1.5 on LGL, cost: 3–4 days

LGL is the hardest source (0.897) and 78% of its errors are >161 km. The
adjudication shows why: *"Wal-Mart shooting threat dubious. COLUMBUS — …"* with
`Columbus` and `Columbus Police Department` as the only place names, gold
Columbus **Nebraska**. There is no document evidence; the answer is in the
outlet's location, which exists in `lgl.xml` and is currently discarded. Feed it
as a feature (not a rule), and check for the leakage it would create in the
other corpora before using it. This is PLAN.md item 3(b), and this audit
upgrades it: it is the identified cause of the largest single-source deficit.

Note the counter-example the same slice provides — *"Amy Booth, of Fowlerville,
**Mich.**"*, *"Thomas Smith and wife, Gwen of Detroit, **Texas**"*,
*"COLUMBUS … Nebraska Centennial Conference"* — where the evidence *is* present
and the model still missed. `sib_adm1` matches on full admin1 names only, so
`Mich.`, `Ind.`, `Calif.`, `Ky.` never fire. A US state-abbreviation alias table
in the sibling matcher is a half-day change with real error mass behind it.

### 5. Document-level Prodigy re-annotation — impact: +0.5–2.0 on Prodigy only, cost: 5+ days

Prodigy has the worst seed variance (SD 0.021) and structurally cannot use a
third of the feature set. Re-annotating at document level would fix it. But it
is 500 held-out entities behind 1/6 of a metric this report recommends
retiring; under `TLG-hard`, Prodigy does not appear at all. Do this only if
Prodigy is a deployment target in its own right — otherwise accept its ceiling
(PLAN.md item 8's own suggestion).

### 6. Deduplicate WikiDocs boilerplate — impact: ~0, cost: 0.5 day

13 near-duplicate documents crossing the split, all "see also" lists and reused
NGO history paragraphs. Measured effect on the headline is below noise. List it
in the methods section; do not spend time on it.

### Not recommended

* **Re-splitting the human corpora.** Document-level integrity is already
  intact; a re-split costs comparability with 60 experiment directories and buys
  0.001.
* **Cross-corpus dedup.** Two articles.
* **A global A/P label rewrite.** Settled and rejected in e24.

---

## Appendix: reproduction

Analysis scripts live in the session scratchpad (not committed):
`split_audit.py` (split replication and per-source stats), `rawtext.py` +
`dedup.py` (exact/near-duplicate hunt on raw texts), `align.py` (pickle
`doc_key` → raw record alignment; 115/115 TR, 553/553 LGL, 188/188 GWN,
3,371/3,373 WikiDocs), `leak2.py` (split-crossing duplication, cross-source
pair memorisation), `dump_preds.py` (per-entity predictions from
`experiments/e29_swa_ep15/seed42.pt`; reproduces `seed42.json` exactly),
`mention_overlap.py`, `difficulty.py`, `sample_adjudicate.py` + `show.py`
(the 100-item adjudication sample with document text and candidates).
Nothing in the repository was modified.
