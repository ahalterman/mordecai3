# Scoping a place-specialised NER: retrain the model, or put a head on the tensors?

Campaign 2, decision D5's ambitious alternative. Written 2026-08-20. Contains a
bounded pilot (5 architectures, 3 seeds each, real training runs on identical
data), a restatement of every end-to-end number on the D2 denominator, and a
measured latency budget.

Nothing in the shared tree was modified. All pilot code is in the session
scratchpad (`.../scratchpad/ner/`); paths at the end.

---

## Headline

**OntoNotes 5.0 is not on this machine and is not obtainable without an LDC
licence.** A "deeper retraining of the NER model on OntoNotes" in the literal
sense is off the table; what is available is better than a substitute for it,
because `en_core_web_trf`'s own transformer *is* an OntoNotes-fine-tuned
roberta-base, its weights are on disk, and this report extracts them
bit-exactly into a HuggingFace model that can be fine-tuned like any other.

**The D5 tagger head is not a consolation prize; it is most of the prize.** A
small span-classification head over the frozen trf tensors the pipeline already
computes, trained on 5,034 TR/LGL/GWN gold spans in **35 seconds**, moves the
held-out numbers (D2 denominator, pooled over TR/LGL/GWN, 2,084 gold toponyms):

| | det P | det R | det F1 | nested det R | **e2e EM** | output precision |
|---|---|---|---|---|---|---|
| serving path today (trim, no NORP) | 75.5 | 78.2 | 76.8 | 13.8 | **66.99** | 77.1 |
| serving + nested gazetteer pass | 66.5 | 88.8 | 76.0 | 61.5 | — | — |
| **span head over frozen trf tensors** | **85.3** | **89.1** | **87.2** | **69.1** | **77.35** | **79.0** |
| same head, encoder fine-tuned end to end | 86.6 | 90.3 | 88.4 | 73.7 | 78.31 | 80.6 |
| ... starting from **stock** roberta-base instead | 88.0 | 89.0 | 88.5 | 71.5 | — | — |
| oracle spans (ceiling) | 100 | 99.5 | 99.7 | 100 | **83.45** | 89.5 |

That is **+10.4 points of end-to-end exact match with precision going up, not
down**, and it closes 63% of the gap to the oracle-span ceiling. It costs
**2.7 ms/doc** at inference (about +5% of pipeline wall clock) because the
transformer forward pass is already paid for.

**The full retrain then fights for the remaining 6.1 points, and wins about one
of them.** Unfreezing the same OntoNotes-initialised roberta and training it
end-to-end on the same data reaches **det F1 88.43 +- 0.17 (vs 87.16 +- 0.12)
and e2e EM 78.31** — reproducible across 3 seeds, but +1.27 F1 and +0.96 EM over
a head that trains in 35 seconds, bought with 25x the training time and
**+11.1 ms/doc** at serving. It is not the lever the framing assumed.

**The lever the pilot actually found is data.** A learning curve on the frozen
head is still climbing at 100% of the available gold — roughly +0.7 detection
F1 and **+4.5 points of nested recall per doubling** of labelled documents —
and the next three doublings are already on disk: 20,302 Wikipedia documents
with 71,475 anchor-linked toponyms, already spaCy'd, already carrying tensors.

**And one result argues for the deep retrain on a completely different axis than
the one it was proposed for — the ranker's.** Fine-tuning for place spans makes
the mention vectors *better* at exactly what the encoder-scoping report found
the ranker depends on. Fine-tuning the incumbent: feature class +.028, feature
code +.045, admin1 +.047. Fine-tuning **stock roberta-base**, which that report
scored far below the incumbent on every probe: it comes back **level on feature
class (.885 vs .884)** and **ahead by +.096 on country and +.160 on admin1**.
The encoder track's dilemma — modern encoders know *where*, the incumbent knows
*what kind*, and the ranker needs *what kind* — dissolves under place
supervision. That makes the eventual encoder replacement (option b3) look like
an upgrade to the ranker's input rather than the regression the encoder track
feared. It is a probe, not a ranker run; no ranker was retrained.

Recommendation (§7): **N1 ship the frozen head; N2 scale the labels; N3 unfreeze
only if N2 saturates; N4 replace the encoder only if N3 wins.**

---

## 1. Data inventory

### 1a. OntoNotes 5.0: absent, and what stands in for it

An exhaustive filesystem sweep (`/` at depth 5-7, `$HOME` at depth 10-12
including `.cache`, `Dropbox`, `Music`, `projects`, `nltk_data`, `Downloads`,
plus `/data`, `/opt`, `/mnt`, `/srv`, `/usr/share`) found **zero** hits for
`ontonotes`, `LDC2013T19`, `conll-2012`, `conll-formatted-ontonotes-5.0`, or
any `.gold_conll` / `.onf` / `.name` / `.parse` file. The only occurrences of
the word on disk are prose in this repository's own planning documents. The
HuggingFace **datasets** cache is 16 KB (one stray loader script); there are no
`datasets--*` repos in the hub cache at all. No CoNLL-2003, WNUT, WikiANN,
Few-NERD or MultiNERD either. `~/nltk_data/corpora` holds only stopwords and
WordNet.

OntoNotes is LDC2013T19 and requires an LDC licence (non-member fee, or free to
member institutions); redistribution is prohibited, which is why nothing on
HuggingFace hosts the annotations.

**But the OntoNotes signal is already on disk, in weights.** `en_core_web_trf`
3.8.0 is a `spacy-curated-transformers` `RobertaTransformer` — roberta-base
(124.06 M parameters, `config.cfg`: `name = "roberta-base"`, `frozen = false`)
fine-tuned inside the spaCy pipeline on OntoNotes tagging, parsing and NER.
`scratchpad/ner/extract_roberta.py` ports those weights into a HF
`RobertaModel` (curated-transformers fuses Q/K/V into one `mha.input` matrix;
everything else is a rename) and verifies the port two ways:

* against the curated module run in-process on identical piece ids:
  **mean cosine 1.0000, max |diff| 3.6e-6** — the port is exact;
* against spaCy's own `._.tensor` on cached documents, using character-offset
  piece alignment: **mean cosine 1.00000, max |diff| 0.0000** on every document
  shorter than spaCy's 144-word-piece window (the handful that read 0.98 carry
  whitespace tokens, which spaCy's `with_non_ws_tokens` drops).

So the "deeper retrain" can be **initialised from the OntoNotes NER encoder
itself** rather than from a corpus we do not have. This is strictly better than
training on OntoNotes from scratch and it costs nothing.

### 1b. What *is* legally usable

| source | licence / status | scale | on disk? | what it gives |
|---|---|---|---|---|
| **`en_core_web_trf` weights** | MIT (model), OntoNotes-derived | 124 M params | **yes**, 478 MB | OntoNotes NER/typing knowledge as an initialisation, and as a *teacher* for distillation |
| **TR-News / LGL / GeoWebNews gold spans** | research corpora, already in-repo | **5,034 train + 2,084 held-out** D2 toponyms, 906 docs, 358 k tokens | **yes** | the only gold *nested* toponym annotation in existence for this task |
| **WikiDocs anchors** | CC-BY-SA (Wikipedia) | 3,556 docs / 8.5 M chars / **23,668** linked toponyms | **yes**, spaCy'd with tensors (10.5 GB DocBin) | anchor-gold place spans at scale |
| **WikiDocsFull anchors** | CC-BY-SA | 20,302 docs / 36.9 M chars (~6.7 M tokens) / **71,475** linked toponyms | **yes**, spaCy'd, 43 GB in 9 shards | the scale-up corpus, already encoded |
| **raw wiki scrape** (`raw_data/wiki/*.jsonl`) | CC-BY-SA | 128 k rows, ~1.2 GB | yes | 95 k anchor rows, per-name priors (poisoned for eval, see encoder report §5a) |
| Prodigy 2017 export | in-repo | 2,664 sentence-level rows | yes | small, sentence-level, Syria/Iraq-heavy |
| `syn_cities` / `syn_caps` templates | generated | 1,944 docs | yes | template text; useless for detection (spans are trivially findable) |
| silver-labelled news via `en_core_web_trf` | distillation of an MIT model over your own text | unbounded | — | transfers OntoNotes knowledge without the corpus |
| CoNLL-2003 | free for research, requires Reuters RCV1 agreement | 300 k tokens | no | LOC/ORG only, no nesting, 1996 newswire |
| WNUT-17 | CC-BY | 100 k tokens | no | noisy user text, `location` class |
| WikiANN / PAN-X (en) | ODC-BY | 20 k sentences | no | silver, from Wikipedia links — same signal as WikiDocs but worse |
| MultiNERD (en) | CC-BY-NC-SA | 164 k sentences | no | **non-commercial**; has fine location types |
| Few-NERD | CC-BY-SA | 188 k sentences | no | 66 fine types incl. `location-GPE`, `location-island`, `building-*` — the closest thing to *feature-class* supervision in a public corpus |

Download cost is small (all of the "no" rows together are under 1 GB), but
**none of them annotates nested toponyms**, which is the miss class that owns
52% of the current NER losses. They can only ever supply flat-span pretraining.

### 1c. The gold that matters, counted

Every gold row of TR/LGL/GWN, split as `tools/train.py` splits it (positional
70/30 on the flat entity list, recovered to document boundaries by
`tools/end_to_end_eval.heldout_doc_indices`; the straddling article is dropped):

| | docs | tokens | gold rows | linked | demonyms (D2, out) | **D2 gold** | of which nested | not token-alignable |
|---|---|---|---|---|---|---|---|---|
| TR train | 90 | 35,329 | 956 | 924 | 88 | **799** | 107 | 37 |
| TR held-out | 28 | 11,086 | 363 | 351 | 20 | **321** | 47 | 10 |
| LGL train | 413 | 158,212 | 3,585 | 3,114 | 190 | **2,919** | 660 | 5 |
| LGL held-out | 175 | 58,672 | 1,503 | 1,348 | 118 | **1,229** | 276 | 1 |
| GWN train | 143 | 69,165 | 4,486 | 1,708 | 388 | **1,316** | 282 | 4 |
| GWN held-out | 57 | 25,552 | 2,126 | 693 | 170 | **523** | 82 | 0 |
| **train total** | **646** | **262,706** | 9,027 | 5,746 | 666 | **5,034** | **1,049 (20.8%)** | 46 |
| **held-out total** | **260** | **95,310** | 3,992 | 2,392 | 308 | **2,073 (+11)** | **405 (19.5%)** | 11 |

"D2 gold" is the demonym-excluded denominator resolved by D2: a gold toponym
with a geonames id whose span is not a demonym. The demonym flag is frozen as
data, computed once as the union of two independent signals — spaCy tags the
whole gold span NORP with no GPE/LOC token, or GeoWebNews' own annotation type
is `Non_Literal_Modifier`. **The held-out D2 denominator is 2,084**
(2,073 alignable + 11 whose character offsets do not line up with any token, so
no token-span model can ever produce them; they stay in the denominator).

"nested" means the gold span sits strictly inside a larger spaCy entity. The
held-out breakdown by host label: ORG 255, GPE 67, FAC 35, LOC 26,
WORK_OF_ART 10, EVENT 6, LAW 5.

Note what this table also says about scale: **the entire gold training signal
for a place-name tagger is 5,034 spans in 646 documents.** That is enough for a
head (§5) and nowhere near enough to justify moving 124 M encoder parameters.

### 1d. Silver nested labels: the rule, measured

Gold nested spans exist only in TR/LGL/GWN. Scaling the tagger to Wikipedia,
CoNLL or silver news needs a nested-span annotator, and the obvious candidate
is the rule `mordecai3.geoparse.nested_gazetteer_spans` already implements:
sub-spans of 1-3 capitalised alphabetic tokens inside an
ORG/FAC/EVENT/WORK_OF_ART/LAW/PRODUCT entity, leftmost-longest, kept when a
geonames row's own `name`/`asciiname` equals the string exactly and its feature
class is A or P.

Because TR/LGL/GWN annotate nested toponyms in gold, that rule's quality as a
*silver annotator* can be measured directly. On the 646 training documents
(`scratchpad/ner/silver_nested.py`):

```
1,396 proposals, 543 of them D2 gold  ->  38.9% precision
                                          58.1% recall of the 837 nested gold
                    +13 land on unlinked gold rows, +20 on demonym gold rows
```

(837, not §1c's 1,049: the rule only looks inside `NESTED_LABELS` hosts, so the
210 gold toponyms nested inside a GPE/LOC entity are out of its reach by
construction — those are the ones the span trimmer already handles.)

**38.9% is far too dirty to train on**, and it is the same number that shows up
at serving as detection precision 75.5 -> 66.5. The false positives are two
kinds: real places that are not place *mentions* ("Ottawa" in *Ottawa Senators*,
"Madrid" in *Real Madrid*, "Toronto" in *Toronto Star*) and gazetteer junk
matching common nouns ("Police", "Superior", "Human", "Gender", "Code",
"Globe", "Alliance", "Chevrolet", "Ballon").

Cheap filters, all computable without gold, measured on the same proposals
(`scratchpad/ner/silver_filters.py`):

| filter | proposals kept | precision | keeps this much of the rule's gold |
|---|---|---|---|
| none | 1,396 | 38.9% | 100% |
| population >= 5,000 | 899 | 57.5% | 95.2% |
| the host entity has an institutional head word (*police, department, university, county, sheriff, ...*) | 424 | 51.2% | 40.0% |
| the same string occurs as a **standalone GPE/LOC somewhere in the corpus** | 644 | **74.2%** | **88.0%** |
| ... occurring standalone **twice or more** | 519 | **80.3%** | 76.8% |
| corpus-standalone >= 1 AND (institutional head OR pop >= 50 k) | 559 | 78.5% | 80.8% |
| corpus-standalone >= 1 AND span is multi-token | 109 | 87.2% | 17.5% |

**The corpus-standalone test is the filter to use**: a nested candidate is kept
only if the same surface string is also emitted as a bare GPE/LOC entity
elsewhere in the corpus. It raises silver precision from 38.9% to 74-80% while
keeping 77-88% of the rule's yield, needs no gazetteer thresholds, and is a
one-pass Counter over the corpus's own spaCy entities. On WikiDocsFull, where
the anchor already tells you the geonames id, it can be tightened further by
requiring the anchor and the nested string to agree.

Two more precision levers worth spec'ing but not measured here: (i) train the
tagger on the *filtered* silver and use its own high-confidence predictions as
round-2 labels (self-training, standard for this exact problem), and (ii) score
the candidate with the ranker and drop it when the ranker abstains — the
abstention path already runs at 86% precision (calibration report §4).

### 1e. Disk

The box is at **94% (113 GB free on a single 1.8 TB volume)**, and
`~/.cache/huggingface` is already 141 GB. Budget:

* nothing new is needed for the tagger head: it reads
  `raw_data/spacyed/source_{tr,lgl,gwn}.spacy` (2.3 GB, already there);
* the pilot's cached tensors + docs json for TR/LGL/GWN: **1.1 GB**;
* the extracted roberta: **474 MB** (and the same again per saved fine-tuned
  encoder);
* WikiDocsFull is already spaCy'd (43 GB, already there) — a scale-up needs
  **no new download and no new spaCy run**;
* the external corpora in §1b together are < 1 GB if ever wanted.

So the whole programme fits in ~2 GB of new disk. The thing to watch is not
this work but the 43 GB of `source_wiki_docs_full.*.spacy` already resident.

---

## 2. The D2 restatement (new, and needed by everything else)

D2 removed demonyms from the task, so every published end-to-end number is
stated against the wrong denominator. Restated here on the 2,084 non-demonym
linked gold toponyms of the same 260 held-out documents. Detection is measured
from the cached DocBins with no Elasticsearch (`baseline_detect.py`);
end-to-end runs the real retrieval + ranker path with
`experiments/e29_swa_ep15/seed42.pt` at `max_choices=100` (`e2e_tagger.py`,
which imports `tools/end_to_end_eval.score_examples` read-only).

| configuration | det P | det R | det F1 | nested det R | predictions | e2e EM |
|---|---|---|---|---|---|---|
| `ship` (pre-fix: untrimmed) | 73.0 | 75.7 | 74.3 | 0.0 | 2,160 | — |
| **`serving` (today's default)** | **75.5** | **78.2** | **76.8** | 13.8 | 2,159 | **66.99** |
| `include_fac=False` | 86.8 | 76.6 | **81.4** | 12.3 | 1,839 | — |
| `accept_norp=True` (D2-forbidden, for reference) | 64.6 | 78.2 | 70.7 | 13.8 | 2,524 | — |
| `nested_gazetteer_pass=True` | 66.5 | **88.8** | 76.0 | **61.5** | 2,782 | — |
| **oracle spans** | — | 99.5 | — | 100 | 2,084 | **83.45** |

Per corpus, today's serving path on the D2 denominator: TR P 78.9 / R 75.8,
LGL 73.3 / 77.3, GWN 78.8 / 81.8; e2e EM TR 65.0, LGL 65.0, GWN 72.8.
(`ship`'s nested recall of 0.0 is not a typo: every nested gold the untrimmed
path can reach is one where spaCy's span carries a leading article, so the
trimmer is the entire 13.8.)

Four things change once demonyms leave the denominator:

1. **Detection recall was understated by ~8 points** (68.1 -> 75.7 on `ship`,
   70.6 -> 78.2 on `serving`) and **e2e EM by ~6.4** (60.58 -> 66.99). The gap
   to the ceiling shrinks from 21.5 points to **16.5**.
2. **`accept_norp` becomes strictly bad**, as D2 intends: it buys no recall
   (the demonyms it finds are no longer scored) and costs 10.9 points of
   detection precision. Removing the code path is the right call.
3. **`include_fac=False` is the best-F1 spaCy configuration** on the D2
   denominator (81.4 vs 76.8): +11.3 precision for −1.6 recall, and 43% faster.
   It was also the best-F1 arm on the old denominator (75.7 vs 71.6) but cost
   −0.7 e2e EM there, so this is an F1-versus-EM trade rather than a free win,
   and its e2e was **not** re-measured on the D2 denominator here. Worth
   settling in Phase 1 — and moot if the tagger ships, since the tagger
   subsumes the label question entirely.
4. **The nested gazetteer pass is measured squarely**: 13.8 -> 61.5 nested
   recall for 9 points of overall precision. The tagger head does better on
   both axes (§5).

Note also `fp_on_demonym`: today's serving path emits 59 spans that land on
demonym gold rows even with `accept_norp=False`, because spaCy tags those spans
GPE. A trained detector was expected to learn not to; it does not (§5a).

---

## 3. Architecture options

Five candidates, judged against the constraints the campaign has already
established: nested spans are mandatory, the ranker consumes the same
transformer's tensors, demonyms must not be emitted, and ES owns 57-65% of
latency so the NLP stage has some headroom but not multiples.

| | (a) spaCy-native retrain (spancat / new NER) | (b1) span head over frozen trf tensors (**D5**) | (b2) fine-tune the OntoNotes roberta, serve as a 2nd model | (b3) fine-tune it and **replace** the encoder, re-embed + retrain the ranker | (c) HF span model on a different encoder |
|---|---|---|---|---|---|
| **training cost** | spaCy config + `spacy train`, ~1 h/run, awkward to sweep | **35 s/run**, 5,034 spans, 20 epochs on a 4090 | ~13 min/run/seed (measured) | same, plus 80 s re-embed + 50 s ranker run per arm | same as b2 |
| **serving latency** | replaces spaCy's NER; spancat adds a suggester + classifier over the existing tok2vec, ~few ms | **+2.70 ms/doc measured** (~+5% pipeline) | **+11.1 ms/doc** for the second roberta pass (~+23% pipeline, +55% NLP stage) | potentially *cheaper* than today (one transformer instead of spaCy's full pipe) if a light tokenizer+sentencizer replaces the parser | +11 ms/doc, and a different width breaks `bert_size` |
| **nested spans** | spancat: yes (span suggester, overlapping); transition NER: **no** | **yes** — arbitrary overlapping spans, 69.1% nested recall measured | yes | yes | yes |
| **demonyms** | learnable | negatives by construction, but **not actually learned** — the head still emits 66-73 spans on demonym golds vs the serving path's 59 (§5a) | same | same | same |
| **ranker interaction** | none if the transformer is untouched; **breaks the tensors** if the transformer is retrained | **none at all** — same tensors, same ranker, same checkpoint | none (spaCy still runs, its tensors unchanged) | **first-class risk**: the mention slot's feature-class signal is what a swap loses (−0.035 EM measured in the encoder report). Must re-probe and re-train the ranker | worst case: the encoder-scoping pilot already measured −0.035 for a naive swap |
| **risk** | spaCy training loop is a second, unfamiliar harness; the `curated_transformer` component is not easily fine-tuned outside spaCy | very low; reversible; no new serving dependency | medium: two 124 M models resident, 2x transformer compute | high: touches the ranker's input distribution, the one thing the campaign proved is fragile | high, and dominated by b3 |
| **verdict** | not worth the harness | **do this now** | it does beat b1, by +1.27 det F1 / +0.96 e2e EM (§5) — but that is a thin return for 11 ms/doc, so gate it behind a data scale-up | the *right* long-run shape, and §6 says it may improve the ranker rather than hurt it — but only after b2 has proven itself | no |

Two structural points behind the table.

**The transition-based NER spaCy ships cannot represent the target at all.**
`en_core_web_trf`'s `ner` is a `spacy.TransitionBasedParser.v2` over a BILUO
action space; "Paris" inside "Paris Police Department" is not expressible. Any
spaCy-native option therefore means `spancat`, which is a span suggester plus a
span classifier — architecturally the same thing as (b1), but expressed in
Thinc, trained by `spacy train`, and harder to sweep. **(b1) is spancat with
the suggester replaced by exhaustive in-sentence enumeration and the tok2vec
frozen.** There is no accuracy argument for the spaCy version, only an
integration one, and integration is already solved because the tensors are on
the `Doc`.

**Option (b3) is the only one that changes the ranker**, and it is the one the
encoder-scoping report's finding bites hardest. That report measured a naive
swap of the mention tensor at −0.035 exact match and diagnosed the cause as
loss of GeoNames feature-class signal. A place-specialised fine-tune could
plausibly *improve* that signal rather than destroy it — §6 measures whether it
does.

---

## 4. The pilot: design

**Question.** Given identical data, an identical objective, an identical head
and an identical decode, does moving the encoder's 124 M parameters buy
anything over a head on the frozen tensors?

**Task.** Binary span classification. Every token span of 1-8 tokens that does
not cross a sentence boundary is a candidate (1.53 M candidates in train,
646 k in held-out); a candidate is positive iff it is exactly a D2 gold
toponym. Demonyms are therefore **negatives by construction** — the detector is
asked to learn D2's policy rather than have it filtered downstream — and nested
toponyms are positives, so nesting is learned rather than ruled.

**Head** (identical in every arm): `Linear(768, 256) + GELU + LayerNorm +
Dropout(0.2)`, span representation `concat(h_start, h_end, mean_pool,
width_embedding(32))`, then `Linear(800, 256) + GELU + Dropout + Linear(256, 1)`,
BCE over all candidates. ~0.5 M parameters.

**Arms** (only the source of the per-token 768-d vector differs):

| arm | token vectors |
|---|---|
| `frozen` | the `en_core_web_trf` tensors already cached in the DocBins — **D5's plan** |
| `frozen_ctx` | the same, plus a 2-layer transformer encoder over the window inside the head |
| `hf_onto` | the extracted roberta, **frozen**, run under this script's windowing (bridge control: isolates windowing/tokenisation from fine-tuning) |
| `ft_onto` | the extracted roberta, **unfrozen** — the deep retrain, initialised from OntoNotes |
| `ft_raw` | stock `roberta-base`, unfrozen — does the OntoNotes initialisation matter? |

**Data and splits.** TR/LGL/GWN only. Windows are sentence-aligned and capped
at 384 word pieces (so a token sees up to ~2.6x spaCy's 144-piece window).
Train/held-out is the campaign's own positional document split; **15% of the
training documents are held out again as a dev set**, fixed across arms and
seeds, used for early stopping and for choosing the decision threshold. The
held-out 260 documents are touched once per run, by the selected checkpoint at
the dev-selected threshold.

**Decode.** Every span above threshold is emitted, overlaps included — that is
what makes the output nested-capable. No non-maximum suppression.

**Optimisation.** AdamW, one-cycle schedule, head lr 1e-3, encoder lr 2e-5,
gradient accumulation 4, grad-norm clip 1.0. Frozen arms 20 epochs, fine-tuned
arms 12. Seeds {42, 101, 202}; tables are mean +/- SE over the three.

**Scoring.** `scratchpad/ner/evalcore.py`, D2 denominator, exact character-span
match for the headline and one-to-one greedy overlap as a secondary. Predicted
spans landing on demonym or unlinked gold rows count as false positives (D2
says the detector should not emit them) and are also reported separately.
End-to-end EM feeds the span set through the real ES + ranker path.

**Ceiling.** 2,072 of the 2,084 held-out D2 golds are expressible as an
in-sentence token span of <= 8 tokens, so the enumeration ceiling on recall is
**99.4%**.

---

## 5. The pilot: results

Pooled over the 260 held-out documents, 2,084 D2 gold toponyms. spaCy rows are
single deterministic configurations; pilot rows are mean +/- SE over 3 seeds.

| arm | seeds | det P | det R | **det F1** | overlap R | **nested det R** | flat det R | predictions | FP on demonym | FP on unlinked | train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ship` (pre-fix spaCy) | — | 73.0 | 75.7 | 74.3 | 82.9 | 0.0 | 93.9 | 2,160 | 59 | 191 | — |
| `serving` (today) | — | 75.5 | 78.2 | 76.8 | 82.9 | 13.8 | 93.7 | 2,159 | 59 | 191 | — |
| `nested_gazetteer_pass` | — | 66.5 | 88.8 | 76.0 | 94.0 | 61.5 | 95.4 | 2,782 | 75 | 229 | — |
| **`frozen`** (D5) | 3 | 85.3 +- 0.7 | 89.1 +- 0.5 | **87.16 +- 0.12** | 91.7 +- 0.6 | 69.1 +- 2.7 | 93.9 | 2,177 | 73 | 65 | **35-60 s** |
| `frozen_ctx` | 3 | 85.2 +- 0.9 | 85.1 +- 0.9 | 85.14 +- 0.40 | 87.9 +- 0.8 | 52.6 +- 2.1 | 93.0 | 2,083 | 79 | 48 | 75 s |
| `hf_onto` (port control) | 1 | 84.3 | 90.0 | 87.05 | 92.8 | 69.4 | 95.0 | 2,226 | 81 | 76 | 320 s |
| **`ft_onto`** (deep retrain) | 3 | 86.6 +- 0.1 | 90.3 +- 0.3 | **88.43 +- 0.17** | 93.2 +- 0.3 | **73.7 +- 0.9** | 94.3 | 2,173 | 72 | 63 | 450-1040 s |
| `ft_raw` (stock roberta init) | 3 | 88.0 +- 0.6 | 89.0 +- 0.6 | **88.47 +- 0.04** | 91.7 +- 0.7 | 71.5 +- 1.7 | 93.2 | 2,109 | 58 | 53 | 410-650 s |

Per source, detection F1: `serving` TR 77.3 / LGL 75.2 / GWN 80.3;
`frozen` TR 83.6 / LGL 89.7 / GWN 83.5; `ft_onto` TR 83.6 / LGL 91.3 / GWN 84.6;
`ft_raw` TR 84.6 / LGL 91.1 / GWN 84.7. Every arm gains most on LGL, which is
the corpus with the most nested toponyms and the worst spaCy recall.

End to end, through the real ES + ranker path, same 2,084 gold toponyms
(`e2e_tagger.py`, one seed each):

| span source | det R | **e2e EM** | nested e2e EM | locations emitted | correct | output precision |
|---|---|---|---|---|---|---|
| `serving` (today) | 78.21 | **66.99** | 11.85 | 1,811 | 1,396 | 77.08 |
| `serving` + `nested_gazetteer_pass` | 88.77 | **75.96** | 51.60 | 2,313 | 1,583 | 68.44 |
| `frozen` seed 42 | 89.44 | **77.35** | 59.75 | 2,040 | 1,612 | 79.02 |
| `ft_onto` seed 42 | 90.79 | **78.31** | 60.74 | 2,024 | 1,632 | 80.63 |
| oracle spans | 99.47 | **83.45** | 78.02 | 1,943 | 1,739 | 89.50 |

Six readings.

1. **The frozen head is the big move.** +10.4 detection F1, +11 detection
   recall, +9.8 detection *precision*, +55 points of nested recall, and
   **+10.36 end-to-end exact match** over the shipping configuration — every
   axis at once, which no configuration arm in `end_to_end_report.md` managed.
   It also **strictly dominates the strongest no-training configuration**, the
   nested-gazetteer pass: detection F1 87.2 vs 76.0, nested recall 69.1 vs
   61.5, detection precision 85.3 vs 66.5, e2e EM 77.35 vs 75.96, and output
   precision 79.0 vs 68.4 — the gazetteer flag emits 2,313 locations to get
   1,583 right, the head emits 2,040 to get 1,612. The head **replaces** that
   flag; it does not complement it.
2. **It closes 63% of the gap to the oracle-span ceiling** (66.99 -> 77.35 of a
   possible 83.45) and **it raises output precision at the same time**
   (77.1 -> 79.0): the user gets 1,612 correct locations instead of 1,396, for
   fewer wrong ones per emitted location.
3. **The fine-tuned encoder wins, modestly and reproducibly**: +1.27 det F1
   (88.43 +- 0.17 vs 87.16 +- 0.12, non-overlapping at 3 seeds each), +4.6
   nested recall, +0.96 e2e EM. That is a real effect, not seed noise — and it
   is a thin return for 25x the training time and a second 124 M forward pass
   at serving.
4. **The gain is fine-tuning, not windowing or the port.** `hf_onto` — the same
   extracted roberta, frozen, under the pilot's own 384-piece windowing and
   tokenisation — lands at F1 87.05 / nested 69.4, i.e. on top of `frozen`
   (87.16 / 69.1). The 384-vs-144-word-piece context window is worth nothing
   here, and the weight port introduces nothing.
5. **Extra capacity on frozen features hurts.** `frozen_ctx` adds two
   transformer layers over the window inside the head and loses 2.02 F1 and 16.5
   points of nested recall. With 551 training documents the parameters are the
   problem, not the solution — which is the same message as §5b.
6. **The OntoNotes initialisation buys nothing for detection.** `ft_raw` starts
   from stock `roberta-base` — the encoder the encoder-scoping report measured
   as *far* worse than the incumbent on every probe — and after the same twelve
   epochs it reaches **88.47 +- 0.04**, statistically identical to `ft_onto`'s
   88.43 +- 0.17. It trades a little nested recall (71.5 vs 73.7) for a little
   precision (88.0 vs 86.6) and emits the fewest demonym false positives of any
   arm (58). Whatever the fine-tune is learning, it is learning it from 5,034
   place spans, not from OntoNotes. That has consequences for the ranker (§6).

### 5a. What the head fixes, and what it does not

Residual analysis of `frozen` seed 42 (220 misses of 2,084; 337 false
positives of 2,177 predictions):

| misses | n | | false positives | n |
|---|---|---|---|---|
| flat toponym, not found at all | 77 | | nothing annotated at that span | 122 |
| nested in ORG, not found | 60 | | overlaps a gold span, wrong boundary | 82 |
| boundary error (overlapping span emitted) | 46 | | lands on an unlinked gold row | 67 |
| nested in FAC/LOC/GPE/WOA/EVENT | 26 | | lands on a demonym gold row | 66 |
| gold offsets align to no token | 11 | | | |

The visible pattern in the false positives is the enclosing span emitted
alongside the nested one — `Pittsburgh` *and* `University of Pittsburgh`,
`Tehran` *and* `Tehran University` — because the decode emits every span above
threshold with no suppression. **That fix was tried and it is worth almost
nothing**: dropping a kept span that strictly contains another kept span when
the extra tokens are institutional head words removes 17 predictions and moves
F1 from 87.00 to 87.11; dropping *every* such container costs recall and is
net-neutral (86.99). The containment pairs are simply not numerous. **67 false
positives land on unlinked gold rows** ("church", "the building"), which the
corpora annotate as toponyms without ids; whether those are errors is an
annotation-scope decision, not a model one.

**Demonym suppression is barely learned.** The frozen head emits 73 spans that
land on demonym gold rows against the serving path's 59, `ft_onto` 72, and only
`ft_raw` improves on spaCy at 58. Training demonyms as negatives did *not*
teach the model to refuse them, presumably because "American" is a negative in
*American troops* and a positive nowhere, while the frozen tensor for it is
nearly indistinguishable from a GPE mention's. A dedicated demonym negative
signal (an explicit lexicon feature on the span, or a second output head
trained on the 666 demonym gold spans in train) is a real work item, not a
freebie — and it matters, because under D2 these are the false positives the
task most explicitly forbids.

### 5b. The head is data-limited, and the data exists

Same head, same protocol, only the number of training documents changes
(`curve.py`, 3 seeds each). "100%" is 551 documents / 4,343 spans — the 646
training documents of §1c minus the 95 held back as dev, which every arm shares:

| training documents | training spans | det P | det R | det F1 | nested det R |
|---|---|---|---|---|---|
| 68 (12.5%) | 551 | 85.1 | 85.1 | 85.11 +- 0.47 | 55.6 +- 0.9 |
| 137 (25%) | 1,065 | 86.1 | 85.6 | 85.84 +- 0.29 | 56.5 +- 2.7 |
| 275 (50%) | 2,146 | 85.4 | 87.9 | 86.59 +- 0.24 | 65.6 +- 2.4 |
| **551 (100%)** | **4,343** | 85.3 | 89.1 | **87.16 +- 0.12** | **69.1 +- 2.7** |

**The curve has not flattened**: roughly **+0.7 detection F1 and +4.5 points of
nested recall per doubling** of labelled documents, with no sign of saturation
at 551. Precision is flat and recall does all the work, which is what you would
expect from a detector that has simply not seen enough place names.

That is the single most important number in this report for planning purposes,
because the corpus that supplies the next three doublings is **already on disk,
already spaCy'd, and already carries tensors**: WikiDocsFull is 20,302
documents with 71,475 anchor-linked toponyms (§1b). Adding it does not need a
download, an ES pass, or a spaCy run.

---

## 6. The ranker interaction

The encoder-scoping report's central finding was that the ranker's mention slot
depends on GeoNames **feature class**, that `en_core_web_trf`'s OntoNotes
fine-tuning supplies it, and that a naive swap to a generic modern encoder
costs −0.035 exact match because it does not. Any retrain that *replaces* the
encoder (option b3) inherits that risk.

Linear probe on the **mention vector alone** — the slot in question — mean-pooled
over the gold span, trained on the same positional train split and scored on
the same held-out documents, 7,081 mentions with gazetteer labels, 2,069 held
out, mean of two probe seeds (`probe.py`):

| mention encoder | feature class | feature code | country | country\|admin1 |
|---|---|---|---|---|
| majority baseline | .488 | .231 | .586 | .039 |
| **spaCy `._.tensor` (incumbent)** | **.884** | **.664** | **.781** | **.429** |
| the extracted roberta, frozen (port control) | .882 | .664 | .776 | .425 |
| `ft_onto` — that roberta after place-span fine-tuning | **.910** | .709 | .791 | .476 |
| `ft_raw` — **stock roberta-base** after the same fine-tuning | .885 | **.738** | **.877** | **.589** |

Three readings.

1. **The port control lands on the incumbent to within 0.005 on all four
   labels**, which independently confirms §1a's parity check and confirms that
   this probe is measuring the same thing the encoder report measured.
2. **Fine-tuning the incumbent on place-span detection improves every axis**,
   including the one the encoder report identified as the fragile one:
   feature class +.028, feature code +.045, admin1 +.047. Nothing is traded
   away.
3. **The `ft_raw` row is the interesting one.** Stock roberta-base is the
   encoder the encoder report scored *far below* the incumbent on every probe
   (country .271 / .461 / .197 against spaCy's .559 / .722 / .525). After
   twelve epochs of place-span supervision on 551 documents it **matches the
   incumbent on feature class (.885 vs .884)** — the signal the encoder report
   attributed specifically to OntoNotes fine-tuning — while beating it by
   **+.096 on country, +.160 on admin1 and +.074 on feature code**.

That last row is the strongest result in this report for the ranker's future.
The encoder track's dilemma was that modern encoders know *where* a place is
and the incumbent knows *what kind* of place it is, and the ranker needs the
second. **Place-span supervision buys the second without giving up the first**,
on an encoder that had neither to start with. The way to a better mention
representation is not a newer encoder and not the OntoNotes weights — it is
**more place supervision**, on whichever encoder you like.

It is a proxy, not a verdict. The probe fits 768 free dimensions against the
label, while the ranker sees four cosine scalars through a fixed country table
and an 8-d learned code table, and the encoder report is explicit that a
representation can be linearly decodable and still unusable through that
bottleneck. **No ranker was retrained here**, deliberately — that is N4's job,
and N4's gate.

---

## 7. Recommendation: the experiment ladder

The pilot changes D5's justification without changing D5's decision. The head
over frozen tensors is 63% of the available prize for 35 seconds of training,
2.7 ms/doc and zero risk to the ranker; the deeper retrain is worth roughly one
more point of end-to-end EM today, and its real value shows up only once the
training data is scaled past what TR/LGL/GWN can supply — at which point it
also becomes the route to a better ranker input. Stage it accordingly.

### N0 — make detection a first-class metric (prerequisite, hours)

Nothing in the ledger scores detection. Add: D2-denominator detection P/R/F1
(exact and overlap), **nested recall broken out**, output precision, and e2e EM,
all restated as in §2 — the published 58.6 / 72.9 / 80.1 numbers are on a
denominator D2 retired. Decide explicitly whether a prediction on an *unlinked*
gold row is a false positive (67 of the head's 337). Delete `accept_norp`
(§2 confirms it is strictly dominated once demonyms leave the denominator).
Take `include_fac=False` while you are there: on the D2 denominator it is the
best-F1 spaCy configuration (81.4 vs 76.8) and 43% faster.

### N1 — ship the frozen span head (the flagship; ~1 day)

Exactly D5. Exhaustive in-sentence span enumeration up to 8 tokens over the
`._.tensor` values the pipeline already computes, 0.5 M-parameter span head,
trained on the 5,034 TR/LGL/GWN D2 gold spans, threshold chosen on a held-out
15% of the training documents. (Containment suppression is measured in §5a and
is not worth adding.) It **replaces** `doc_to_ex_expanded`'s label filter, the
span trimmer and `nested_gazetteer_pass`; spaCy NER stays only for the context
tensor and `guess_in_rel`.

Expected, on the D2 dev denominator (measured here): **det F1 76.8 -> 87.2,
e2e EM 67.0 -> 77.4, nested detection recall 13.8 -> 69.1, output precision
77.1 -> 79.0, +2.7 ms/doc.** Every one of those is an improvement on both axes
at once, which no configuration arm has managed.

**Gate before publication:** the head is trained on TR/LGL/GWN's own span
conventions and evaluated on held-out documents *of the same corpora*. The
data-quality track already showed the ranker can identify the corpus from
`doc_tensor` alone (93.3%) and adapts to per-corpus convention; a span detector
has even more opportunity to. **This number must be reproduced on D1's untouched
modern-news TEST corpus before it is claimed.** I would expect a real but
smaller gain there.

### N2 — scale the labels (cheap, high expected value; ~2 days)

§5b says the head is data-limited and §1b says the data is on disk. In order:

1. **WikiDocs / WikiDocsFull anchors** as positives (23,668 / 71,475 spans,
   already spaCy'd with tensors). The trap is that Wikipedia anchors are
   *incomplete*: an unlinked "Paris" in the same article becomes a false
   negative. Use partial-annotation training — restrict the negative set to
   candidates inside sentences that contain at least one anchor, or mask
   candidates whose string matches an anchor elsewhere in the article — rather
   than treating every unlabelled span as negative.
2. **Silver nested labels** from the gazetteer rule plus the corpus-standalone
   filter (§1d: 38.9% -> 74-80% precision, keeping 77-88% of the yield). Weight
   them below gold.
3. **Self-training**: relabel with the N1+N2 model, keep high-confidence spans,
   retrain. This is the standard remedy for partial annotation and it is nearly
   free at 35 s/run.
4. **A demonym negative signal** (§5a): the head does not learn demonym refusal
   from negatives alone. Cheapest form is an explicit lexicon feature on the
   span (geonames `alternativenames` carry the adjectival forms, which is why
   `accept_norp` worked at all), or a second binary head trained on the 666
   demonym gold spans in train.

5. **Distillation over unlabelled news**, only if 1-4 run out. Silver-label
   your own text with `en_core_web_trf` (an MIT-licensed model, so this is the
   legal route to OntoNotes-style supervision) and train the head on it. This
   is the weakest of the five — the teacher's label set is exactly the one
   whose failures started this campaign — but it is unbounded in volume and it
   is how you would adapt the detector to a new domain without annotating it.

Guardrail: per-source detection deltas on TR/LGL/GWN. If they degrade while the
pooled number rises, Wikipedia's span convention is displacing the corpora's.

### N3 — the deep retrain, gated (only if N2 saturates; ~1 week)

Unfreeze, in increasing order of cost and risk:

1. **Top-N layers only.** Unfreeze the last 4 of the 12 roberta layers. This is
   the cheapest test of "is encoder capacity the constraint", and it keeps the
   lower layers — and therefore approximately the ranker's tensors — intact.
2. **Full fine-tune, served as a second model.** Measured here at +1.27 det F1
   (3 seeds, 88.43 +- 0.17 vs 87.16 +- 0.12) and +0.96 e2e EM (1 seed) over the
   frozen head, for 25x training time
   (1,037 s vs 35-60 s) and **+11.1 ms/doc** at serving (≈ +23% of pipeline wall
   clock, ≈ +55% of the NLP stage — consistent with the encoder report's +60%).

**Gate:** must beat N2's frozen head by more than seed noise (the frozen arm's
SE is 0.12 F1; treat 3 seeds and t(2) as the minimum) on the D2 detection
metric **and** on D1's untouched TEST corpus. If it clears N2's head by less
than a point, do not take the second model; the latency is not worth it.

**Which base?** For *detection* it does not matter: fine-tuned stock
roberta-base reaches the same F1 as the fine-tuned OntoNotes encoder (§5), so
the OntoNotes initialisation buys nothing here. For the *ranker* the two differ
in profile (§6): the OntoNotes base keeps more feature-class signal, the stock
base gains far more country/admin1. Carry both into N4 and let the ranker
decide — and note that the stock base is the one that frees the pipeline from
`en_core_web_trf` altogether.

### N4 — the consolidation, and the only step that touches the ranker

If and only if N3 wins: **replace the encoder wholesale**. Serve *one* roberta
forward pass that produces both the spans and the ranker's three tensors, drop
`en_core_web_trf` entirely, and keep a light spaCy (`en_core_web_sm`) for
tokenisation, sentences and the dependency parse that `guess_in_rel` reads.
That is strictly *cheaper* than today, because today's pipeline runs a 124 M
transformer plus a tagger, parser and NER whose outputs the tagger has made
redundant.

Economics are already established: re-embedding all six sources is ~80 s of GPU
and 1.6 GB (no ES traffic, no spaCy re-run — encoder report §2), and a ranker
run is 50 s, so the ranker can be retrained once per arm. The evidence that
this is worth trying is §6: place-span fine-tuning gives the mention slot a
strictly better probe profile than the incumbent's — the OntoNotes base gains
+.028 feature class and +.045 feature code, and the *stock roberta* base comes
back level on feature class while gaining +.096 country and +.160 admin1, where
a naive generic swap lost 0.03-0.12 on exactly that axis. Run E40 (the
representation-harness fix) first: the encoder report shows the current recipe
loses 0.010-0.016 to provably-free reparameterisations, and this arm has to be
judged above that floor.

**Gate:** the ranker's TLG-hard must not regress, measured on 5 seeds against
e29 with the harness fix (E40) already in place — the encoder report's ~0.015
"noise floor for any representation arm" applies here too.

### What each stage is worth

| stage | detection F1 (D2) | e2e EM (D2) | evidence |
|---|---|---|---|
| today | 76.8 | 66.99 | measured, §2 |
| today + `nested_gazetteer_pass` | 76.0 | 75.96 (output precision 68.4) | measured, §5 |
| N0 `include_fac=False` | 81.4 | not measured | measured, §2 |
| **N1 frozen head** | **87.2** | **77.35** | measured, §5 |
| N2 scaled labels | +0.7 F1 per doubling, extrapolated | — | curve, §5b |
| N3 full retrain | +1.27 +- 0.21 (3 seeds) | +0.96 (1 seed) | measured, §5 |
| N4 encoder replacement | — | ranker-side, unmeasured | probe only, §6 |
| oracle spans | 99.7 | 83.45 | measured, §2 |

If the deeper retrain is **deferred indefinitely**, N1+N2 still take end-to-end
exact match from 67.0 to ~77-79 on the D2 denominator, close two thirds to
three quarters of the gap to the oracle-span ceiling, and cost 2.7 ms/doc and
nothing in risk. That is roughly three ranker campaigns' worth of accuracy for
a day of work, and it is available now.

If the deeper retrain is **taken**, the additional prize is about a point of
detection today, an unknown amount at N2's data scale, and — uniquely — a
mention representation measurably better than the incumbent's at the one thing
the ranker's text pathway depends on. **That last item is the real reason to do
it**, and it means the NER campaign and the representation campaign (E41-E43)
are the same campaign: the "typed encoder" E43 asked for is exactly the encoder
N3 trains.

### Explicitly not recommended

* Acquiring OntoNotes. It has no nested toponyms; its knowledge is already in
  the incumbent's weights; and `ft_raw` shows that after place-span supervision
  a stock roberta-base matches the OntoNotes-initialised one anyway. There is
  no version of this project that needs the corpus.
* CoNLL-2003 / WikiANN / MultiNERD pretraining. Flat spans only, wrong domain,
  and MultiNERD is non-commercial. Few-NERD is the only one with a plausible
  angle (fine location types as feature-class supervision) and it should wait
  behind N2.
* A spaCy-native `spancat` rebuild. It is the same architecture as N1 expressed
  in a second training harness.
* Adding transformer layers on top of the frozen tensors: measured, and it
  *loses* (`frozen_ctx`, −2.1 F1 and −16.5 nested recall against `frozen`) —
  551 documents do not support the extra parameters.
* Retraining the ranker before N3 clears its gate.

---

## 8. Caveats

1. **One corpus family.** Everything is TR/LGL/GWN, held-out documents of the
   same corpora the head trained on. This is the same weakness D1 created the
   TEST corpus to fix, and it applies with more force to a span detector than
   to a ranker.
2. **Seeds.** Three per arm, except `hf_onto` (one; it is a control whose job
   is only to land on `frozen`). Three seeds is enough to separate `frozen`
   from `ft_onto` (87.16 +- 0.12 vs 88.43 +- 0.17, non-overlapping) and nowhere
   near enough to separate `ft_onto` from `ft_raw`.
3. **The demonym definition is a judgement call with a measurable cost.** The
   D2 flag is the union of "spaCy tags the whole span NORP" (215 held-out
   golds) and "GeoWebNews types it `Non_Literal_Modifier`" (153). The union is
   2,084; the NORP-only rule would be **2,177**. The 93-gold difference is
   mostly GWN rows whose surface *is* a place name used attributively ("U.S.
   troops", "EU sanctions"), 59 of which spaCy tags GPE. Every number in this
   report moves by up to ~1 point under the other rule; the *comparisons* do
   not, since all arms share the definition.
4. **Threshold selection.** Each run picks its threshold on its own dev split
   by micro-F1 over candidate spans, which is a slightly different objective
   from span-level F1 on the held-out set. A per-arm threshold sweep on
   held-out would raise every pilot row, and is not done for that reason.
5. **The head's input is one hyper-parameter away from spaCy's.** The pilot
   windows at 384 word pieces, spaCy at 144; the `hf_onto` control shows this
   is worth essentially nothing here (F1 87.1 vs 87.2), so the frozen arm's
   result transfers to the real pipeline unchanged.
6. **Precision is measured strictly.** Predictions on demonym and unlinked gold
   rows count as false positives. Under a lenient convention that excludes both,
   every arm's precision rises by 5-7 points and the ordering is unchanged.
7. **The e2e numbers use one ranker checkpoint** (`e29_swa_ep15/seed42.pt`), one
   seed, at `max_choices=100`, and the tagger's spans are fed through
   `add_es_data_batch` exactly as the serving path does. The ranker was never
   trained on tagger-produced spans, so N1 is being scored slightly out of
   distribution — which, if anything, understates it.

---

## Appendix: artefacts

All under
`/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/`
(session scratchpad; nothing was written into `raw_data/`, `mordecai3/`,
`tools/` or `experiments/PLAN.md`).

| file | what |
|---|---|
| `build_data.py` | reads the corpora + DocBins, writes per-source token/tensor/gold caches (1.1 GB) |
| `inventory.py` | the §1c counts |
| `evalcore.py` | the D2 gold definition and the detection scorer |
| `baseline_detect.py` | §2's spaCy detection variants, from cached DocBins |
| `extract_roberta.py` | ports `en_core_web_trf`'s curated roberta into HF, with the parity check |
| `spanpilot.py` | the pilot: chunking, span head, five arms, train/dev/test loop |
| `curve.py` | §5b's learning curve |
| `summarize.py` | pooled and per-source tables, mean +/- SE |
| `silver_nested.py`, `silver_filters.py` | §1d's silver-annotator measurement and filter sweep |
| `probe.py` | §6's mention-vector linear probe |
| `e2e_tagger.py` | end-to-end EM for any span set, through the real ES + ranker path |
| `pilot_results.json`, `pilot_ft.json`, `pilot_all.json`, `curve.json`, `baseline_detect.json`, `probe_results.json`, `silver_*.json`, `e2e_*.json` | raw results |
| `roberta_onto/`, `enc_ft_*/` | the extracted encoder and the fine-tuned ones |
| `preds_*.json` | predicted character spans per held-out document, per arm |

Reproduce the pilot with:

```
uv run python <scratchpad>/ner/build_data.py
uv run python <scratchpad>/ner/extract_roberta.py
uv run python <scratchpad>/ner/baseline_detect.py
uv run python <scratchpad>/ner/spanpilot.py \
    --arms frozen,frozen_ctx,hf_onto,ft_onto,ft_raw --seeds 42,101,202 \
    --epochs 12 --frozen-epochs 20 --save-preds --save-encoder
uv run python <scratchpad>/ner/summarize.py
uv run python <scratchpad>/ner/e2e_tagger.py preds_frozen_42.json
uv run python <scratchpad>/ner/probe.py --encoders spacy,roberta_onto,enc_ft_onto_42,enc_ft_raw_42
```
