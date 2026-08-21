# Scaling the place-span head's training labels (ladder step N2)

Campaign 2, the flagship NER track. Written 2026-08-20. Follows
`ner_retrain_scoping_report.md`, whose pilot put a 0.5 M-parameter span head on
the frozen `en_core_web_trf` token tensors, trained it on 5,034 TR/LGL/GWN gold
toponyms in 35 seconds, and found its **learning curve still climbing at 100% of
the available gold** — roughly +0.7 detection F1 and +4.5 nested recall per
doubling of labelled documents. This report is the doubling. It adds every other
label source on disk, one at a time, with a control for each.

Every number is on the **D2 denominator**: 2,084 non-demonym linked gold
toponyms in the 260 held-out documents of TR-News / LGL / GeoWebNews
(`phase0_report.md`). **No TEST split was touched.** Nothing in the shared tree
was modified except this file and `experiments/e55_ner_head/NOTES.md`. All code
and artifacts are in the session scratchpad and the worktree; paths at the end.

---

## Headline

**The pilot's learning curve does not continue, and the reason is not the
amount of data.** Adding all 20,302 WikiDocsFull documents and their 71,475
anchor-linked toponyms — **14x the gold documents and 14x the gold spans** —
moves held-out detection F1 by **+0.32 ± 0.43 (paired, 5 seeds, t = 1.66, not
significant)** and nested detection recall by **+3.60 ± 1.73 (t = 4.65,
significant)**. Once the two-stage *schedule* those arms need is controlled for
— the same warm restart run on gold alone — Wikipedia's own contribution falls
to **+0.18 F1 and +1.5 nested recall, neither significant.**

**Wikipedia's anchors are only 27.3% complete**, which is why. Of the 245,531
spaCy GPE/LOC entities in the corpus only 67,076 are exactly an anchor; the
other three quarters are toponyms nobody linked. And the anchors carry
essentially **no nested toponyms**: a head trained on WikiDocsFull alone reaches
det F1 77.51 on news — above today's shipping pipeline's 76.8 — with **nested
detection recall 4.9**.

**Loss design for partial annotation matters enormously, and then stops
mattering.** Across five treatments of the O-class, mixing Wikipedia into gold
training spans **3.32 detection F1 and 38 points of nested recall** (naive
83.96 / nested 34.6 → the best mask 85.30 / 57.5). But **every mixed arm loses
to the gold-only baseline**, and the recipe that does not — pre-train on the
scaled labels, then fine-tune on gold alone — makes the loss choice nearly
irrelevant (naive 87.26 vs the best mask 87.27). The correct answer to "how do
you weight the O-class on partially annotated text" turns out to be **don't let
partially annotated text set the decision boundary at all.**

**The cheapest source is the one that pays.** Explicit demonym negatives — the
666 D2 demonym gold spans plus 1,324 spaCy NORP entity spans already in the
training corpora, no new data, 16 s/run — cut the demonym false-positive count
from **73.2 to 59.6 at flat F1** (weight 10) or to **49.4 for −0.28 F1**
(weight 30, ΔdemFP t = −4.25, significant), and raise nested recall by
**+5.93 (t = 7.44, significant)** as a side effect. Today's serving path emits
59 demonym spans; the pilot's head emitted 73 and the scoping report called
demonym suppression "a real work item, not a freebie". It is a freebie.

**Recommendation: N2 does not clear the adoption bar and N1 should ship as
scoped.** The pilot's 87.16 F1 / 77.35 e2e is not clearly exceeded by any label
source here. Two changes are worth folding into the N1 artifact because they
cost nothing and are measured: **demonym negatives at weight 3–10**, and the
**warm-restart schedule**. Wikipedia is not worth the 10 GB cache and the 12x
training time for +0.18 F1.

---

## 0. Harness, baseline, and the noise floor

The pilot's harness was reused for the corpus reader, the D2 gold definition,
the detection scorer and the end-to-end path. The training loop was rewritten to
micro-batch chunks (the pilot ran one window per forward pass), which is **5x
faster** — 13 s per gold-only run against the pilot's 35–60 s — and is what
makes 100+ runs affordable.

It reproduces the pilot on both axes:

| | this harness | pilot |
|---|---|---|
| det F1, gold-only frozen head | **87.06 ± 0.36** (5 seeds) | 87.16 ± 0.12 (SE, 3 seeds) |
| nested det recall | 67.2 ± 2.0 | 69.1 ± 2.7 |
| e2e EM through the real ranker | **76.36 ± 1.35** (3 head seeds) | 77.35 (1 seed) |
| inference module ⇄ harness parity | exact (§4) | — |

**Two facts about noise that the pilot's error bars understate.**

1. **Runs are deterministic.** The same (config, seed) reproduces to every
   decimal of every metric, verified by rerunning the baseline. So a seed is a
   matched pair across arms and **every comparison below is paired** — which is
   what makes 3–5 seeds enough to say anything at all.
2. **The single-seed spread is larger than the pilot's SE suggests.** The
   baseline's five seeds are 87.62 / 86.87 / 86.71 / 87.19 / 86.91 (sd 0.36) and
   its demonym false-positive count ranges 50–81 (sd 13.4). With 5 paired seeds
   the detectable difference is about **±0.45 detection F1**, **±2.5 nested
   recall** and **±17 demonym FPs**. Nothing smaller than that is claimed here.

The pilot's reported spread (±0.12 SE) came from a batching configuration that
happened to be low-variance; the same recipe at a different micro-batch size
gives sd 0.36. Treat **0.4 detection F1 as this experiment's noise floor**.

---

## 1. What the label sources actually contain

### 1a. WikiDocsFull, cached and counted

The 9 already-spaCy'd DocBin shards (43 GB) were decoded once into a compact
per-token cache — **20,302 documents, 6,865,048 tokens, 71,475 anchors, 10 GB of
fp16 tensors, 110 s to build**. No spaCy re-run, no download, no Elasticsearch.

Two measurements decide how the corpus can be used:

| | |
|---|---|
| spaCy GPE/LOC entities in the corpus | 245,531 |
| ... of which exactly an anchor | **67,076 (27.3%)** |
| anchors | 71,475 |
| ... that spaCy tags GPE/LOC | 67,958 (95.1%) |

**The anchors are 27.3% complete.** They are not exotic — 95% are spans spaCy
would call a place anyway — they are simply sparse: an editor links "Iran" and
"Latin America" in a paragraph that also names Nigeria, Europe, Asia and the
Middle East. Three quarters of the visible toponyms in WikiDocsFull are
unlabelled, and under a naive objective every one of them is a false negative.

The second problem is what the anchors are *not*. A head trained on
WikiDocsFull alone, with no gold at all, scored on the news held-out set:

| | det P | det R | det F1 | **nested det R** |
|---|---|---|---|---|
| WikiDocsFull anchors only | 81.7 | 73.8 | **77.51** | **4.9** |
| today's serving path | 75.5 | 78.2 | 76.8 | 13.8 |
| gold-only head | 85.7 | 88.5 | 87.06 | 67.2 |

Wikipedia anchors alone transfer to news at roughly the level of the shipping
pipeline — a genuinely interesting result about domain transfer — but they
contain **almost no nested toponyms**, because a Wikipedia editor writing "the
University of Pittsburgh" links the university, not the city. Nested spans are
the miss class that owns half the pipeline's losses, and this corpus cannot
teach them.

### 1b. Silver nested labels at Wikipedia scale

`mordecai3.geoparse.nested_gazetteer_spans` was replayed over the cached arrays
(no spaCy re-run) and filtered as the scoping report §1d recommends:

```
467,875 raw proposals over 20,302 documents, 71,726 distinct strings
  4,376 strings are an exact geonames A/P name
  1,832 also occur standalone as GPE/LOC or as an anchor  -> 17,343 spans / 7,624 docs
  1,435 occur as an anchor-linked toponym somewhere        -> 12,180 spans / 6,065 docs
```

The tighter **anchor** filter is visibly cleaner than the scoping report's
**standalone** filter: the strings it additionally drops are `Council`,
`Taliban`, `EU`, `European Union`, `Human`, `Police`, `Liberation`, `Alliance`,
`Battle`, `Reuters`, `Al Jazeera`, `Duma` — and `Iraqi`, a demonym the
standalone filter kept because spaCy tagged it GPE somewhere in 20,000
documents. Both variants were carried into the arms.

### 1c. Demonym negatives

The hard-negative set is the D2 demonym gold rows (spaCy NORP with no
place-like token, ∪ GeoWebNews `Non_Literal_Modifier`) **plus every spaCy NORP
entity span** — 666 gold demonym spans and 1,324 NORP entities in the 646
training documents, and NORP spans in WikiDocsFull for the wiki arms. They are
already negatives under the pilot's objective; this arm multiplies their loss
weight.

---

## 2. Loss design for partial annotation

Five treatments of the unlabelled candidates in WikiDocsFull. `risky` is the set
a partial annotation cannot rule on: a span overlapping a spaCy GPE/LOC/FAC
entity, or a capitalised repeat of an anchor string in the same document.

| name | treatment of unlabelled candidates |
|---|---|
| `naive` | all are negatives at full weight — the trap the scoping report named |
| `mask_cap` | drop every span whose tokens are all capitalised alphabetic |
| `mask` | drop `risky` |
| `mask2` | drop `risky` ∪ capitalised sub-spans **inside an ORG/FAC/EVENT/WOA/LAW/PRODUCT host** |
| `pu` / `pu2` | positive-unlabelled: keep the same sets at weight 0.1 instead of dropping |
| `mask_prop` | `mask`, plus in-document anchor-string propagation (a capitalised repeat of an anchor becomes a positive) |

Negatives are subsampled at 12% with a compensating weight so the loss stays an
unbiased estimate of the full enumeration; masked candidates are dropped
outright, which is also a compute saving (`mask2` drops 7.38 M of 12.2 M).

### 2a. Mixed into gold training (wiki weight 1.0, 8 epochs, seed 42)

| loss | det P | det R | **det F1** | **nested det R** | demonym FP |
|---|---|---|---|---|---|
| `mask_prop` | 86.4 | 76.9 | 81.34 | 19.3 | 58 |
| `naive` | 86.6 | 81.5 | 83.96 | 34.6 | 51 |
| `mask_cap` | 84.9 | 83.4 | 84.17 | 57.3 | 68 |
| `mask` | 83.7 | 84.7 | 84.22 | 44.9 | 65 |
| `pu` (0.1) | 82.8 | 86.4 | 84.57 | 52.1 | 62 |
| `mask` at weight 0.3 | 83.3 | 87.8 | 85.51 | 57.8 | 67 |
| `mask` at weight 1.0, gold oversampled 4x | 85.7 | 87.2 | 86.47 | 55.1 | 67 |
| **`mask2`** | 83.1 | 87.6 | **85.30** | **57.5** | 71 |
| *gold-only baseline (5 seeds)* | *85.7* | *88.5* | ***87.06*** | ***67.2*** | *73* |

Four readings.

1. **The loss design is worth 3.32 F1 and 38 points of nested recall** between
   the worst and best treatment. It is not a detail.
2. **`mask2`'s extra clause is the single most valuable one**: masking the
   capitalised interiors of ORG/FAC hosts is +1.08 F1 and **+12.6 nested
   recall** over `mask`. This is the partial-annotation trap in its sharpest
   form — an unanchored "Pittsburgh" inside "University of Pittsburgh" is not a
   negative, it is the exact positive the whole campaign is chasing, and
   `naive`/`mask` label it negative 20,302 documents' worth of times.
3. **Anchor-string propagation is actively harmful** (81.34 / nested 19.3). It
   manufactures positives inside organisation names and the head learns the
   wrong convention wholesale. It was the most obvious "free yield" idea in the
   design space and it is the worst arm in the report.
4. **Every mixed arm loses to gold alone.** Down-weighting Wikipedia (0.3) and
   oversampling gold (4x) each help and neither closes the gap. Mixing 40,364
   Wikipedia windows with 1,244 gold windows means Wikipedia's span convention
   sets the decision boundary no matter how the loss is weighted.

### 2b. Two-stage: pre-train on the scaled labels, fine-tune on gold

4 epochs on gold + wiki, then a warm restart and 20 epochs on gold alone
(lr 3e-4), with dev-F1 checkpoint selection and threshold choice confined to the
gold stage. Seed 42:

| stage-1 loss | det P | det R | det F1 | nested det R | demonym FP |
|---|---|---|---|---|---|
| `pu2` | 86.6 | 87.4 | 86.99 | 61.7 | 74 |
| `naive` | 84.2 | 90.5 | **87.26** | 72.3 | 83 |
| `mask` | 85.3 | 89.3 | **87.29** | 70.1 | 80 |
| `mask2` | 84.5 | 90.3 | **87.27** | **73.1** | 81 |

**The 3.3-point spread collapses to 0.03.** Naive treatment of the O-class —
the thing the scoping report warned against — is indistinguishable from the best
mask once gold gets the last word. The one design that still hurts is `pu2`,
which keeps 7.4 M ambiguous candidates in the objective at low weight instead of
dropping them, and drifts the representation further for its trouble.

The honest conclusion is that **the loss design is a workaround for a mixing
strategy that should not be used.** `mask2` remains the right default — it costs
nothing, it is 20% cheaper per epoch, and it is the only design that is safe in
*both* recipes — but its value is insurance, not accuracy.

---

## 3. The arms

All arms below use the two-stage recipe with `mask2` unless stated. Deltas are
**paired by seed** against the gold-only baseline; `*` marks |t| > 2.776 at 5
seeds, `~` marks |t| > 4.303 at 3 seeds.

### 3a. The controls come first, because they explain most of the effect

| arm | seeds | det P | det R | det F1 | nested det R | demonym FP | s/run |
|---|---|---|---|---|---|---|---|
| **gold only, 20 ep (pilot N1)** | 5 | 85.7 | 88.5 | **87.06 ± 0.36** | **67.2 ± 2.0** | **73.2 ± 13.4** | 13 |
| gold only, 24 ep | 5 | 86.0 | 88.6 | 87.27 ± 0.48 | 66.5 ± 3.6 | 76.6 ± 8.8 | 25 |
| **gold only, 4 ep + warm restart + 20 ep** | 5 | 85.2 | 89.3 | 87.20 ± 0.49 | **69.3 ± 3.3** | 79.2 ± 3.0 | 28 |

More epochs alone buy nothing on the axis that matters (nested recall Δ −0.69,
t −0.45). **The warm restart alone
buys +2.12 nested recall** (paired, t = 1.66 — not significant on its own, but
consistently positive and free). This control is the reason the wiki arms cannot
be read against the pilot's number directly, and it was the single most
informative run in the report.

### 3b. Arm (a) — WikiDocsFull anchors

| arm | wiki docs | seeds | det F1 | Δ vs baseline | nested det R | Δ | demonym FP |
|---|---|---|---|---|---|---|---|
| A_wiki12 | ~2,550 | 3 | 87.42 ± 0.46 | +0.36 | 73.4 ± 4.9 | +6.91 | 72.7 |
| A_wiki25 | ~5,080 | 3 | 87.56 ± 0.20 | +0.49 | 70.9 ± 2.8 | +4.36 | 77.0 |
| A_wiki50 | ~10,150 | 3 | 87.60 ± 0.07 | +0.53 | 71.8 ± 2.3 | +5.27~ | 77.0 |
| **A_wiki100** | **20,302** | **5** | **87.38 ± 0.13** | **+0.32 (t 1.66)** | **70.8 ± 1.4** | **+3.60\*** | 79.4 |

**The wiki axis is flat from its first ~2,500 documents.** Eight-fold more
Wikipedia is worth nothing: 87.42 → 87.56 → 87.60 → 87.38, non-monotone and
inside the noise floor, with nested recall equally flat. Whatever out-of-domain
place-name pre-training is worth, 2,500 documents already deliver all of it.
**The answer to "is the curve still climbing" is: not on this axis.**

Against the **schedule control** rather than the baseline, arm (a) is:

| A_wiki100 − A_ctrl_2s (5 paired seeds) | det F1 | nested det R | demonym FP |
|---|---|---|---|
| | **+0.18 ± 0.39** (t 1.04) | **+1.48 ± 3.61** (t 0.92) | +0.20 |

**That is the number that matters, and it is not significant.** Most of the
apparent gain from 71,475 Wikipedia anchors is the warm restart they came
packaged with.

**Guardrail (per-source detection F1)**, baseline → A_wiki100:
TR 83.37 → 84.32, LGL 89.51 → 90.18, **GWN 83.63 → 82.77**. Only GeoWebNews
degrades, by 0.9 — the corpus with the most annotated non-literal place
mentions, i.e. the one whose convention Wikipedia is furthest from. Small, but
it is the direction the scoping report told us to watch.

**Residuals**, baseline seed 42 vs A_wiki100 seed 42 (219 → 203 misses,
308 → 346 false positives):

| category | baseline | +wiki |
|---|---|---|
| miss: nested in ORG, not found | 67 | **49** |
| miss: flat toponym, not found | 60 | 64 |
| miss: boundary error (overlapping span emitted) | 54 | 57 |
| FP: nothing annotated at that span | 82 | **109** |
| FP: overlaps a gold span, wrong boundary | 72 | 83 |
| FP: lands on a demonym gold row | 81 | 81 |
| FP: lands on an unlinked gold row | 73 | 73 |

Wikipedia does exactly the one thing it was expected to do — it teaches the head
to look inside organisation names, 18 fewer ORG-nested misses — and it pays for
that with 27 more spans on text nobody annotated. Net F1: nil.

### 3c. Arm (b) — filtered silver nested labels

| arm | silver spans | seeds | det F1 | Δ vs baseline | nested det R | Δ | demonym FP |
|---|---|---|---|---|---|---|---|
| A_wiki100 (no silver) | 0 | 5 | 87.38 ± 0.13 | +0.32 | 70.8 ± 1.4 | +3.60* | 79.4 |
| **+ silver, anchor filter** | 12,180 | 4 | **87.49 ± 0.20** | +0.40 (t 1.45) | 71.2 ± 1.3 | +4.81~ | **72.5** |
| + silver, standalone filter | 17,343 | 3 | **87.66 ± 0.17** | +0.59 (t 2.11) | 70.2 ± 2.0 | +3.70 | 80.0 |

Silver nested labels add **+0.11 to +0.28 F1 over the same arm without them** —
inside the noise floor at every seed count run, and the two filters are
indistinguishable from each other. Their one visible effect is on the demonym
false-positive count under the anchor filter (79.4 → 72.5), consistent with that
filter's habit of dropping demonym strings like `Iraqi`.

**Verdict: silver nested labels do not pay at this scale.** The gazetteer rule
they come from is measured at 74–80% precision, and 12,000–17,000 spans at 80%
precision cannot outweigh 5,034 gold ones — the more so because they are
Wikipedia spans, and §3b says Wikipedia spans are not the constraint.

### 3d. Arm (c) — explicit demonym negatives (gold only, no new data)

Loss-weight multiplier on the demonym negative set of §1c. 5 seeds each, paired
against the baseline.

| weight | det P | det R | det F1 | Δ F1 | nested det R | Δ nested | **demonym FP** | Δ demFP |
|---|---|---|---|---|---|---|---|---|
| 1 (baseline) | 85.7 | 88.5 | 87.06 ± 0.36 | — | 67.2 ± 2.0 | — | **73.2 ± 13.4** | — |
| **3** | 85.9 | 88.8 | **87.30 ± 0.17** | +0.24 | 71.0 ± 2.0 | +3.75* | **64.4 ± 5.7** | −8.8 |
| **10** | 85.4 | 89.1 | **87.17 ± 0.27** | +0.11 | **73.1 ± 1.4** | **+5.93\*** | **59.6 ± 0.5** | −13.6 |
| **30** | 85.8 | 87.8 | 86.78 ± 0.23 | −0.28 | 71.6 ± 1.9 | +4.35* | **49.4 ± 3.8** | **−23.8\*** |
| 100 | 85.2 | 85.3 | 85.21 ± 0.17 | −1.86~ | 68.6 ± 4.9 | +2.14 | **43.0 ± 2.0** | −27.7 |
| *serving path today* | *75.5* | *78.2* | *76.8* | | *13.8* | | *59* | |

This is the clean result of the report, and it is free.

1. **Demonym suppression is learnable after all.** The scoping report found the
   head emitting *more* demonym false positives than spaCy's label filter did
   (73 vs 59), concluded that "training demonyms as negatives did not teach the
   model to refuse them", and called a dedicated signal "a real work item, not a
   freebie". It is a freebie: **the same negatives, weighted 10x, take the count
   to 59.6 — level with the label filter the head replaced — at no cost in F1**;
   weighted 30x they reach 49.4, comfortably below it, for −0.28 F1 (not
   significant) and a significant −23.8 demonym FPs.
2. **There is a clean trade curve** with a knee between 10 and 30. Choosing on
   it is a product decision, not an accuracy one: under D2 these are the false
   positives the task most explicitly forbids, and the head is the only place
   left to forbid them once the NORP label filter is gone.
3. **Sharpening the demonym boundary also raises nested recall**, +5.93 at
   weight 10 (t = 7.44, significant) — as much as all of WikiDocsFull, out of
   data already in the training set. Both are teaching the same lesson: a
   capitalised place-shaped string is not automatically a place mention.
4. It costs **16 s/run** and no new data, against 110 s of cache building, 10 GB
   of disk and 153 s/run for Wikipedia.

### 3e. Arm (d) — self-training on the unlabelled Wikipedia text

The scoping report's own remedy for partial annotation: relabel rather than
mask. The gold-only head (seed 42) scored every ambiguous candidate in
WikiDocsFull; p ≥ 0.9 became a positive, p ≤ 0.05 a negative, the middle band
was dropped. One round, then the same two-stage recipe.

```
pseudo-positives 130,476   pseudo-negatives 7,600,127   dropped 149,598
```

Self-training nearly **triples** the wiki positive set (71,475 anchors →
201,950 spans), which is what the 27.3% coverage figure predicts it should.

| arm | seeds | det F1 | Δ vs baseline | nested det R | Δ | demonym FP |
|---|---|---|---|---|---|---|
| A_wiki100 (`mask2`) | 5 | 87.38 ± 0.13 | +0.32 | 70.8 ± 1.4 | +3.60* | 79.4 |
| **D_self (pseudo-labelled)** | 3 | **87.46 ± 0.07** | +0.39 (t 1.38) | 69.9 ± 1.5 | +3.37 | 77.7 |

Self-training recovers the sliver of F1 that masking gave up and gives back the
sliver of nested recall that masking bought. It lands on the same flat plateau
as everything else in §3b. **The plateau is not caused by the missing labels;
filling 130,476 of them in does not leave it.**

### 3f. Everything at once

Wikipedia + silver (anchor filter) + demonym negatives at weight 10, two-stage:

| arm | seeds | det P | det R | det F1 | Δ F1 | nested det R | Δ | demonym FP | Δ |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 85.7 | 88.5 | 87.06 ± 0.36 | — | 67.2 ± 2.0 | — | 73.2 ± 13.4 | — |
| **C_all** | 3 | 85.2 | 88.4 | 86.77 ± 0.37 | −0.30 (ns) | **73.7 ± 2.6** | **+7.24~** | **54.7 ± 1.2** | −16.0 |

The sources do stack on the axes they each move: **the best nested detection
recall in the report (73.7, Δ +7.24, t 8.43, significant) and a demonym
false-positive count below the serving path's (54.7 vs 59)** — for −0.30
detection F1, inside the noise floor. Against the schedule control the demonym
result is unambiguous: **ΔdemFP −24.00 ± 3.61, t = −11.53**, the tightest
effect in the report. It is the arm to take if nested recall and demonym
suppression are what you are buying; it is not an F1 win.

### 3g. End to end, through the real ranker

`experiments/e29_swa_ep15/seed42.pt` at `max_choices=100`, spans fed through
`add_es_data_batch` exactly as serving does, scored by
`tools/end_to_end_eval.score_examples`. **One ranker checkpoint and one ranker
seed**; the three figures per arm are three *head* seeds (42 / 101 / 202).

| span source | det R | **e2e EM** | nested e2e EM | output precision |
|---|---|---|---|---|
| serving path today (`phase0_report.md`) | 78.21 | 66.99 | 11.85 | 77.08 |
| serving + `nested_gazetteer_pass` | 88.77 | 75.96 | 51.60 | 68.44 |
| **gold-only head** | 89.5 / 86.5 / 89.3 | **76.36 ± 1.35** | 54.90 | 79.7 |
| **+ WikiDocsFull** | 90.3 / 88.6 / 89.2 | **77.37 ± 0.51** | 59.01 | 79.3 |
| **C_all (a+b+c)** | 88.9 / 88.2 / 88.3 | 76.33 ± 0.45 | **61.48** | 79.4 |
| *pilot, frozen head, 1 seed* | *89.44* | *77.35* | *59.75* | *79.02* |
| *pilot, fine-tuned encoder, 1 seed* | *90.79* | *78.31* | *60.74* | *80.63* |
| oracle spans | 99.47 | **83.45** | 78.02 | 89.50 |

Paired by head seed, WikiDocsFull is **+1.00 ± 0.93 e2e EM (t = 1.87, not
significant)** and **+4.11 ± 1.00 nested e2e EM (t = 7.1, significant)**.
C_all is **−0.03 ± 0.90 e2e EM** and **+6.58 nested e2e EM** — the highest
nested end-to-end exact match in either report, above the pilot's fine-tuned
encoder (60.74), from a frozen head that trains in 87 s.

The row that deserves the PI's attention is the baseline's spread: **74.81 to
77.26 e2e EM across three head seeds of the same recipe.** The pilot's
single-seed 77.35 is the top of that range. Any future end-to-end claim on this
denominator needs three head seeds; one is not a measurement, and the ladder's
"N1 = 77.4" figure should be restated as **76.4 ± 1.4**.

---

## 4. Ship staging (N1 prep)

Staged in the worktree, not applied — another agent is editing `geoparse.py`.

```
worktree: /home/andy/projects/mordecai3/.claude/worktrees/agent-a4c25ff5ea1f68094
          (branch accuracy-campaign, nothing committed)
artifact: experiments/e55_ner_head/ship/          <- in the worktree
  span_head.py            self-contained inference module (numpy + torch + a spaCy Doc)
  span_head_gold_42.pt    N1 as scoped: gold only, threshold 0.5, 1.6 MB
  span_head_C_all_42.pt   N2's combined arm (§3f), threshold 0.3, 1.6 MB
  test_span_head.py       parity test against the training harness
  INTEGRATION.md          the spec below, in full
full seed sets (5 gold, 3 C_all) in <scratchpad>/ner_scale/ship/
```

**Parity is verified, not asserted.** `test_span_head.py` runs the module over
the 260 held-out documents straight from the cached DocBins and reproduces each
arm's harness row exactly — the module and the harness enumerate the same
candidates over the same `._.tensor` values:

| checkpoint | det P | det R | det F1 | nested det R | demonym FP | predictions |
|---|---|---|---|---|---|---|
| `span_head_gold_42.pt` | 85.83 | 89.49 | **87.62** | 67.9 | 81 | 2,173 |
| `span_head_C_all_42.pt` | 83.95 | 88.87 | 86.34 | **76.5** | **56** | 2,206 |

Which to take is a product decision: `gold` is the best detection F1, `C_all`
trades 1.3 F1 for +8.6 nested recall and 25 fewer demonym false positives — the
two things D2 and the nested-miss class actually care about.

**What it replaces** (one call for three code paths):

| today | after |
|---|---|
| `doc_to_ex_expanded(..., geo_labels=GEO_LABELS)` — the label filter | `tagger.doc_to_ex(doc)` — every span the head scores above threshold |
| `trim_span_tokens` — the span trimmer, and `KEEP_LEADING_THE` with it | gone; the head is trained on gold character offsets and emits the trimmed span |
| `nested_gazetteer_spans` — the opt-in gazetteer pass and its ES round trip | gone; the head emits overlapping spans natively |

`guess_in_rel`, `doc_tensor`, `locs_tensor` and `sent` are computed exactly as
`doc_to_ex_expanded` computes them, so `add_es_data_batch`, `ProductionData`,
the ranker and `_resolve_results` are untouched. spaCy NER still runs, for
`locs_tensor`'s context set and the parse `guess_in_rel` reads. The old branch
should stay behind a flag for one release: it is the only way to reproduce the
frozen `ship` / `serving` numbers in the ledger.

**Cost, re-measured.** On 120 held-out documents (mean 349 tokens), idle GPU:

| | mean | median |
|---|---|---|
| `SpanTagger.doc_to_ex` (the head) | **6.86 ms** | 4.82 ms |
| `doc_to_ex_expanded` (today) | 1.68 ms | 1.16 ms |
| `nested_gazetteer_spans` (what it replaces) | **8.68 ms** | 6.28 ms |

So **+5.2 ms/doc** over today's extraction, not the pilot's +2.70 — the pilot
measured a leaner path — but still **less than the gazetteer pass it replaces**,
and without that pass's Elasticsearch round trip. Under GPU contention (five
concurrent training jobs) the head reads 8.70 ms and the gazetteer pass 9.00 ms,
so the ordering is stable. The head is one batched matmul over vectors the
pipeline already has; the cost is candidate enumeration, `O(n_tokens x 8)`.

**Two behaviour changes for callers**, both documented in `INTEGRATION.md`:
overlapping spans are now emitted (a consumer assuming a flat entity list must
say so; containment suppression was measured in the pilot at ~+0.1 F1 and is not
worth adding for accuracy), and NORP suppression is *learned* rather than
structural, so the residual demonym false-positive rate is non-zero — which is
why §3d exists and why it is a first-class metric.

**The gate is unchanged and still open:** every number here is on held-out
documents of the corpora the head trained on. Reproduce on D1's untouched
modern-news TEST corpus before the flag defaults on.

---

## 5. The unfreeze arm (e) was not run, and why

The brief gates it on the frozen head saturating *and* the scaled labels beating
the frozen arm by more than seed noise. **The second condition failed**: no label
source moved detection F1 outside the ±0.4 noise floor, so there is no "scaled
label set" on which N3's question differs materially from the pilot's. The
pilot's answer stands unmodified — fine-tuning the encoder on the same 5,034
gold spans is +1.27 ± 0.21 det F1 and +0.96 e2e EM for 25x the training time and
+11.1 ms/doc.

What this experiment *does* change is the prior. The pilot read its climbing
learning curve as "the head is data-limited". §3 says the head is limited by
**in-domain, complete, nested** annotation, of which there are 5,034 spans in
the world and no more on this disk. That is an argument for N3 being worth more
than the pilot thought, not less: when you cannot add data, capacity is the only
remaining lever. It is also an argument for the **annotation convention document
D1 asked for being the highest-value item left in the campaign** — 5,000 more
gold nested spans annotated to a written convention would be worth more than
everything measured here put together.

---

## 6. Which sources paid

| source | new data? | cost | det F1 | nested det R | demonym FP | verdict |
|---|---|---|---|---|---|---|
| (a) WikiDocsFull anchors, 71,475 spans | 10 GB cache, 110 s build | 153 s/run | +0.32 (ns); **+0.18 vs the schedule control (ns)** | +3.60*; +1.5 vs control (ns) | +6 | **did not pay** |
| (b) filtered silver nested, 12–17 k spans | ES pass, 6 min | +0 | +0.11 to +0.28 over (a), ns | +0.4 over (a) | −7 (anchor filter) | **did not pay** |
| (c) explicit demonym negatives | none | 16 s/run | +0.11 at w10, −0.28 at w30 | **+5.93\*** | **−13.6 at w10, −23.8\* at w30** | **paid** |
| (d) self-training, 130 k pseudo-positives | none | 173 s/run | +0.39 over baseline, ns | +3.37, ns | −2 | **did not pay** |
| warm-restart schedule (control) | none | +15 s/run | +0.14 (ns) | +2.12 (ns) | +6 | **free, mildly positive** |
| (a)+(b)+(c) together | as above | 87 s/run | −0.30 (ns) | **+7.24~ → 73.7** | **−16.0 → 54.7** | **pays on the two axes it targets, not on F1** |

Nothing here clears the adoption bar of "the pilot's 87.16 F1 / 77.35 e2e
clearly exceeded". **N1 should ship as scoped**, with demonym negatives at
weight 3–10 folded in and the warm restart taken because it is free.

---

## 7. Caveats

1. **One corpus family, again.** Held-out documents of the corpora the head
   trained on. This applies with *more* force here than in the pilot, because
   two of the four arms are explicitly about importing a different corpus's span
   convention, and the D2 held-out set is the only place that conflict is
   scored.
2. **The demonym definition is a judgement call** (scoping report §8.3) and the
   demonym-FP metric inherits it. All arms share the definition, so the
   comparisons hold; the absolute counts would move by ~1 point under the
   NORP-only rule.
3. **The demonym-FP metric is the noisiest thing in the report** — the
   baseline's five seeds span 50–81. The weight-30 result clears significance;
   the weight-3 and weight-10 results are consistent and monotone but do not,
   individually, at 5 seeds.
4. **e2e uses one ranker checkpoint and one ranker seed**, `e29_swa_ep15/
   seed42.pt` at `max_choices=100`, and the ranker was never trained on
   tagger-produced spans.
5. **Wikipedia tensors are cached in fp16**, the gold corpora in fp32. The
   difference is ~1e-3 relative and the wiki tensors are only ever used in
   stage 1, but it is not nothing and it was not ablated.
6. **The wiki curve was measured at fixed epochs**, so larger fractions get more
   gradient steps per epoch. Its flatness is a statement about "more Wikipedia
   under a fixed recipe", not about the asymptote of an optimally-tuned one.
7. **Screening rows are single-seed** and are marked as such; every claim in §6
   rests on 3–5 paired seeds.
8. **A cache-keying bug** made the first run of the standalone-filter silver arm
   a duplicate of the anchor-filter arm (both hashed to `bool(silver)` rather
   than the path). It was found, fixed, and the arm rerun; the §3c row labelled
   "standalone filter" is the rerun.
9. **The two-stage recipe's hyper-parameters were not swept.** 4 pre-train
   epochs and a 3e-4 fine-tune lr were the first values tried and the only ones
   run. A sweep could plausibly move arm (a) by more than the effect being
   measured, which is another way of saying the effect is small.

---

## Appendix: artifacts and reproduction

Session scratchpad, `.../scratchpad/ner_scale/`:

| file | what |
|---|---|
| `build_wiki.py` | decodes the 9 WikiDocsFull DocBin shards into the 10 GB token/tensor/anchor cache |
| `scaledata.py` | chunk builders, the D2 gold objective, and the partial-annotation loss designs |
| `scaletrain.py` | micro-batched span-head trainer, the two-stage schedule, the self-training relabeller |
| `sweep.py`, `cfg*.json` | the arm definitions actually run |
| `silver_wiki.py` | replays `nested_gazetteer_spans` over the cache and applies the filters |
| `show.py` | pooled tables, the per-source guardrail, and the paired-seed comparisons |
| `residual.py` | §3b's miss / false-positive breakdown |
| `latency.py` | §4's serving cost |
| `e2e.py`, `e2e_batch.sh` | end-to-end EM through the real ranker, wrapping the pilot's `e2e_tagger.py` |
| `results*.json`, `preds_*.json` | every arm's scores and predicted character spans |
| `ship/` | the deployable artifact: module, checkpoints, parity test, integration spec |

The pilot's harness (`.../scratchpad/ner/`) is imported read-only for
`evalcore.py`, `e2e_tagger.py` and the cached TR/LGL/GWN tensors.

```
uv run python <scratch>/ner_scale/build_wiki.py                    # 110 s, 10 GB
uv run python <scratch>/ner_scale/silver_wiki.py --filter anchor1 --out silver_anchor1.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_screen.json --out results.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_curve.json  --out results_curve.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_dem.json    --out results_dem.json
uv run python <scratch>/ner_scale/sweep.py --config cfg9.json       --out results_ctrl.json
uv run python <scratch>/ner_scale/show.py --paired gold_ship
uv run python <scratch>/ner_scale/ship/test_span_head.py ship/span_head_gold_42.pt
bash          <scratch>/ner_scale/e2e_batch.sh preds_gold_ship_42.json preds_A_wiki100_42.json
```
