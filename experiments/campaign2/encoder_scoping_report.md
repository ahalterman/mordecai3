# Campaign 2, finding 1: scoping the frozen-encoder replacement

Written 2026-08-20. Scopes `experiments/PLAN.md` "A second campaign" items 1
(replace the frozen encoder) and 3 (the no-evidence residual). Contains a
bounded pilot: 5 seeds x 15 arms of real training runs on a reduced mix, plus
an offline linear-probe screen in the Wave-2b style.

**Headline: do not do a straight encoder swap.** A modern long-context encoder
is measurably *better* at saying **where** a mention is (linear probe for gold
country +0.10 to +0.23, for gold admin1 +0.13 to +0.21) and measurably *worse*
at saying **what kind of place** it is (feature class -0.03 to -0.12), and the
ranker's text pathway is dominated by the second. Swapping all three text
tensors costs **-0.035 to -0.043 exact match**; the entire loss is in the
mention slot, while the two *context* slots swap for roughly free
(-0.007 overall, **+0.009 on GWN**). The bet is still live, but the experiment
that tests it is "carry both mention representations" or "fine-tune the new
encoder with an entity-typing objective", not "replace".

Nothing in the shared tree was modified. All pilot code is in the session
scratchpad (`.../scratchpad/pilot/`); paths are listed at the end.

---

## 1. The text pathway, exactly

### What the model consumes

Three `float32` vectors per **entity** (not per candidate), each 768-d today:

| key | pooled over | built in |
|---|---|---|
| `tensor` | the mention's own tokens | `tools/train.py:377` (prodigy), `:451` (wiki), `:524` (wiki_docs), `:697` (TR/LGL/GWN); `mordecai3/geoparse.py:193` at inference |
| `locs_tensor` | the *other* GPE/LOC tokens in the document | same lines |
| `doc_tensor` | every token in the document | same lines |

`mordecai3/torch_model.py:117-125` stacks them into `ProductionData`;
`__getitem__` hands them to the model as `placename_tensor`,
`other_locs_tensor`, `doc_tensor`.

### Where they enter the model

`geoparse_model.forward`, `mordecai3/torch_model.py:510-561`:

```
x            = text_to_country(placename_tensor)     Linear(bert_size -> 24)
x_code       = text_to_code(placename_tensor)        Linear(bert_size ->  8)
x_other_locs = context_to_country(other_locs_tensor) Linear(bert_size -> 24)  # shared
x_doc        = context_to_country(doc_tensor)        Linear(bert_size -> 24)  # shared
```

then four cosine similarities per candidate, against embeddings of the
candidate's country and feature code:

```
cos_sim_country   = cos(x,            country_embed_transform(country_emb[cand]))
cos_sim_code      = cos(x_code,       code_emb[cand])
cos_sim_other_locs= cos(x_other_locs, country_embed_transform(country_emb[cand]))
cos_sim_doc       = cos(x_doc,        country_embed_transform(country_emb[cand]))
```

Those four scalars are concatenated with the 9 base + 17 enriched gazetteer
features and go into the mix MLP (`:561`).

**The whole text pathway is a four-scalar bottleneck per candidate**, and two
of the four scalars only ever compare context to a *country* embedding. There
is no channel by which document context can say "this is the county, not the
city" — only `cos_sim_code`, off the mention vector, can.

### The interface a replacement must satisfy

1. **Three vectors of equal width per entity**, mean-pooled the same way; the
   loader casts to float32 and `bert_size` is read off
   `es_train_data[0]['tensor'].shape[0]` (`tools/train.py:1092`, `:1326`).
2. **`bert_size` is overloaded.** `country_embed_transform =
   nn.Linear(bert_size, country_size)` (`torch_model.py:449`) is applied to the
   **frozen 768-d country table** `mordecai3/assets/country_bert_768.npy`, not
   to the text. So an encoder of width != 768 does not just change one number:
   the country-table projection must be split off into its own
   `Linear(768, country_size)`. Today's ModernBERT-class encoders are 768-d, so
   the pilot needed no code change at all — but any 384-d or 1024-d candidate,
   and *any* widened/concatenated mention slot, does.
3. **A rebuild path that exists at inference.** `geoparse.py:doc_to_ex_expanded`
   is the live version, so a new encoder has to run inside or beside the spaCy
   pipeline. spaCy NER is still needed regardless: it supplies the mention spans
   and the GPE/LOC set that define the pooling.
4. **Checkpoint compatibility.** `bert_size` is already written to the
   `<ckpt>.json` sidecar (`train.py:1326`); an encoder name/revision should join
   it, or a model will silently load against the wrong tensors.

### Two defects found while mapping it (not fixed here)

- **Train/serve mismatch in `locs_tensor`.** `geoparse.py:190` builds
  `loc_ents` from `['GPE','LOC','EVENT_LOC','NORP']`; `train.py:684` (and the
  other three formatters) use `['GPE','LOC']` only. The served
  `other_locs_tensor` is therefore pooled over a *different* token set than the
  trained one — NORP tokens ("Turkish", "Israeli") are in at serve time and out
  at train time.
- **The `locs_tensor` slot earns nothing.** Pilot arm `nolocs` replaces it with
  a copy of `doc_tensor` (i.e. deletes the other-locs signal): **-0.0019 +/-
  0.0030, not significant**. Given the mismatch above, that is not surprising.
  Either fix the mismatch and re-measure, or reclaim the slot for something
  else — it is 768 free input dimensions.

### What the incumbent actually is

`en_core_web_trf` 3.8.0 is **roberta-base**, *fine-tuned inside the spaCy
pipeline* (`config.cfg:105 frozen = false`, `:247 name = "roberta-base"`), read
at its **last hidden layer**, under
`WithStridedSpans.v1 stride=104 window=144` (`config.cfg:129-132`).

Two consequences the campaign should internalise:

- **No token has ever seen more than ~144 word-pieces of context.** The
  "document tensor" is a mean of 110-word-window vectors. Genuine
  document-level context is not in these features at all.
- **The weights were tuned on OntoNotes tagging/parsing/NER.** These are
  entity-typing features by construction. That is exactly what the pilot finds
  they are unusually good at, and exactly what a general-purpose encoder lacks.

---

## 2. Rebuild economics: much cheaper than PLAN.md assumes

The key discovery: **a re-embed needs neither a spaCy re-run nor an
Elasticsearch re-query.** The DocBins in `raw_data/spacyed/` already carry the
text, tokenisation, NER and character offsets; the entity order inside each
pickle is exactly reproducible by replaying the formatter loop. So the rebuild
is: read DocBin -> encode text with the new model -> re-pool -> splice the three
vectors into a copy of the `_enriched_compact` pickle.

`scratchpad/pilot/reembed.py --parity` proves the replay is aligned by
re-pooling the **original** spaCy tensors and diffing against the pickle:

```
tr   914 entities   max|diff| tensor / locs_tensor / doc_tensor = 0.0 / 0.0 / 0.0
lgl 3245 entities   0.0 / 0.0 / 0.0
gwn 1580 entities   0.0 / 0.0 / 0.0
```

Bit-for-bit, on all 5,739 entities, plus a check that the `search_name` and
`correct_geonamesid` sequences match position for position.

### Measured and projected cost, all six sources

Encode throughput measured at **~30k spaCy tokens/s** (ModernBERT-base, fp32,
one document at a time, RTX 4090 — several times faster with bf16 + batching).

| source | docs | tokens | `.spacy` on disk | pickle | encode |
|---|---|---|---|---|---|
| TR | 118 | 46k | 287 MB | 63 MB | 3.9 s (measured) |
| LGL | 588 | 217k | 1.4 GB | 245 MB | 7.3 s (measured) |
| GWN | 200 | 95k | 600 MB | 96 MB | 2.9 s (measured) |
| Prodigy | 2,664 | ~90k | 564 MB | 45 MB | ~5 s |
| Synth (2 files) | 1,944 | ~20k | 170 MB | 116 MB | ~3 s |
| WikiDocs | 3,556 | ~1.5M | 9.8 GB | 971 MB | ~55 s |
| **total (ship recipe)** | **9,070** | **~2.0M** | **12.8 GB read** | **1.54 GB** | **~80 s GPU** |
| (WikiDocsFull, if ever) | 20,302 | ~6.7M | 43 GB | — | ~4 min |

So: **~1.6 GB of extra disk for a complete parallel set of pickles, and a
wall-clock rebuild dominated by DocBin I/O rather than the GPU — call it 10-20
minutes end to end.** No ES traffic, no spaCy re-run, no touching the 56 GB of
`spacyed/`.

Work still required for a full rebuild: the replay tool implements the
TR/LGL/GWN formatter only. Prodigy, wiki, wiki_docs and synth use three other
formatter functions (`train.py:334`, `:409`, `:486`) that need the same
treatment — mechanical, and each gets the same bit-parity assertion for free.

### The economics of option (a) vs option (b)

**Confirmed: the "50s per run" regime survives a frozen swap.** The encoder
stays frozen, so it is one re-embed pass, once, and every downstream training
run reads precomputed vectors exactly as today. Pilot runs on the reduced mix
took **13 s** each; nothing about the loop changed.

**Fine-tuning end-to-end (option b) is what changes the economics**, and
"precompute per epoch" does not rescue it — the encoder weights move, so the
vectors must be recomputed every step, not every epoch. At the measured
throughput, one epoch of the ship mix is ~2.0M tokens ≈ 70 s of forward pass
alone in fp32, plus backward (~2-3x), so **15-25 min per 15-epoch run in the
best case versus 50 s today** — a 20-30x slowdown. That is still tractable on
one 4090, but it ends the regime in which 60 experiment arms fit into a day. If
option (b) is ever taken, budget it as *one* arm, not a wave, and plan option
(c) (distil back to frozen features) before anything ships.

**Serving latency (measured, LGL's 588 news documents, GPU):** spaCy
`en_core_web_trf` full pipe 20.3 ms/doc; a second ModernBERT-base pass
12.4 ms/doc fp32 batch-1. A dual-encoder serving path therefore costs about
**+60% on the NLP stage** before any optimisation — and the NLP stage is not
where end-to-end latency lives (ES retrieval is). Acceptable.

---

## 3. Encoder candidates

Everything below was already in the local HF cache except ModernBERT-base
(600 MB, downloaded). The box has internet.

| candidate | params | dim | ctx | why |
|---|---|---|---|---|
| **`answerdotai/ModernBERT-base`** | 149M | **768** | 8192 | 2024-12 MLM encoder; 768-d means **zero code change**; 8192 context ends the 144-token window entirely |
| **`nomic-ai/modernbert-embed-base`** | 149M | **768** | 8192 | ModernBERT-base + contrastive tuning; best probe scores of anything tried |
| `BAAI/bge-small-en-v1.5` | 33M | 384 | 512 | the latency option — but 384-d forces the `bert_size` split, and 512 context needs windowing |

Rejected after the probe: **vanilla `roberta-base`** (the un-fine-tuned version
of the incumbent) and **`BAAI/bge-base-en-v1.5`** both score *far below* the
spaCy tensors on every probe (see below). That is the cleanest evidence that
what makes the incumbent good is its **spaCy fine-tuning**, not roberta.

### Offline probe screen (Wave-2b methodology)

A multinomial linear probe on `concat(tensor, locs_tensor, doc_tensor)`
(2304-d), predicting the gold candidate's country / admin1 / feature class.
Same positional 70/30 split per source that training uses. 2000 full-batch
AdamW steps, wd 1e-3, mean of seeds 42 and 101. Held-out accuracy:

| encoder | country TR / LGL / GWN | admin1 TR / LGL / GWN | feature class TR / LGL / GWN |
|---|---|---|---|
| majority baseline | .517 / .702 / .263 | .092 / .081 / .096 | .554 / .557 / .661 |
| **spaCy (incumbent)** | .559 / .722 / .525 | .234 / .372 / .363 | **.858 / .921 / .827** |
| roberta-base (raw) | .271 / .461 / .197 | .055 / .105 / .051 | — |
| bge-base-en-v1.5 | .096 / .468 / .210 | .020 / .075 / .053 | — |
| ModernBERT-base | .720 / .829 / .683 | .363 / .491 / .516 | .784 / .871 / .712 |
| **modernbert-embed-base** | **.797 / .823 / .753** | **.435 / .555 / .575** | .803 / .895 / .776 |

Read that table twice. On *where the place is*, the ModernBERT family beats the
incumbent by 10-23 points of country accuracy and 13-21 points of admin1. On
*what kind of place it is*, the incumbent wins by 3-12 points. The first is what
PLAN.md item 1 predicted. The second is what the training pilot then measured
as the thing that actually matters.

**What the probe does not predict:** it fits 2304 free dimensions against the
gold label directly, whereas the model gets four cosine scalars against a fixed
country table and a learned 8-d code table. A representation can be linearly
decodable and still be unusable through that bottleneck. The pilot below is the
demonstration.

---

## 4. The pilot

**Design.** Reduced mix TR + LGL + GWN (5,739 entities, ~4,000 train /
1,700 held out), ship recipe verbatim (`--mix-dim 512 --logits --mask-padding
--oov-bucket-fix --modern-mlp --label-smoothing 0.05 --enriched
--feature-blocks "prom,name,cue,sib,geo,shape" --epochs 15 --avg-params
--avg-mode swa`), **5 seeds {42, 101, 202, 617, 1848}**, score = mean of the
last five epochs, paired per-seed deltas, `*` = |mean| > 2 SE. Only the three
text tensors differ between arms; candidates, labels and all 26 enriched
features are byte-identical. Each run takes 13 s.

Baseline on this mix: **0.9116 exact match / 0.9425 acc@161** (the full six-source
ship model reads 0.9258 — the reduced mix is a different, easier/smaller
problem; only the *deltas* transfer).

### 4a. The straight swap loses

| arm | text tensors | exact match | delta (paired, 5 seeds) |
|---|---|---|---|
| base | spaCy (incumbent) | 0.9116 | — |
| **rand** | **gaussian noise, matched norm** | **0.8351** | **-0.0764 +/- 0.0062 \*** |
| mbert | ModernBERT-base, all three | 0.8768 | -0.0348 +/- 0.0046 \* |
| nomicmb | modernbert-embed-base, all three | 0.8682 | -0.0434 +/- 0.0038 \* |
| mbertz | ModernBERT-base, z-scored | 0.8791 | -0.0325 +/- 0.0057 \* |

The `rand` arm is the important control: **the text pathway is worth +7.6
points** on this mix. The new encoders recover only 4.2-3.3 of those 7.6. They
are not neutral — they carry real signal — they are just worse than the
incumbent through this interface.

Longer training does not rescue them (60 epochs: base -0.0226, mbert -0.0440,
nomicmb -0.0504 — everything overfits the small mix, the gap widens).

### 4b. The loss is entirely in the mention slot

| arm | mention slot | context slots (locs, doc) | exact match | delta |
|---|---|---|---|---|
| base | spaCy | spaCy | 0.9116 | — |
| **hybctx** | **spaCy** | **modernbert-embed** | **0.9048** | **-0.0068 +/- 0.0038 \*** |
| hybmen | modernbert-embed | spaCy | 0.8740 | -0.0376 +/- 0.0012 \* |
| nomicmb | modernbert-embed | modernbert-embed | 0.8682 | -0.0434 +/- 0.0038 \* |

Per source, `hybctx`: TR -0.0204\*, LGL -0.0090 (n.s.), **GWN +0.0090\*** exact
match and **GWN +0.0085\* acc@161**. So upgrading the two *context* vectors to a
long-context modern encoder is close to free on the news corpora and a real
gain on GWN — and `hybmen` reproduces essentially the whole regression on its
own. Combined with the feature-class probe row, the mechanism is not in doubt:

> **the mention slot's job is entity typing (city vs county vs river vs
> province) through `cos_sim_code`, spaCy's OntoNotes fine-tuning supplies it,
> and a general-purpose encoder does not.**

That is the same city-vs-same-named-admin-unit axis the campaign's twin-credit
metric was built around.

### 4c. How much of this is the recipe being tuned to the incumbent?

Controls, all applied to the mention slot only, all on the same 5 seeds:

| control | what it does | delta |
|---|---|---|
| `rot` | random 768x768 **orthogonal rotation** (exactly absorbable by the existing Linear+bias, zero information lost) | **+0.0001 +/- 0.0005** |
| `ctr` | subtract the training mean (also exactly absorbable) | -0.0100 +/- 0.0041 \* |
| `basez` | per-dimension z-score (also exactly absorbable) | -0.0162 +/- 0.0033 \* |
| `pca384` | keep the top 384 principal components (98.5% of variance) | -0.0245 +/- 0.0045 \* |
| `+LayerNorm` on all three text inputs (patched model) | | -0.0006 +/- 0.0015 (n.s.) |

Reading:

- **The pathway is rotation-invariant in practice** (+0.0001). So the encoder
  results above are *not* a coordinate-system artefact. This was the first
  thing I suspected and it is ruled out.
- But it is **not invariant to centering or per-dimension rescaling**, both of
  which the Linear+bias could absorb exactly. -0.010 and -0.016 for pure
  reparameterisations means the 15-epoch / flat-1e-3 recipe leaves the text
  projections underfitted, and it is tuned around the incumbent's geometry.
  **Treat ~0.015 as the noise floor for any "different representation" arm.**
- **Dimensionality is precious**: discarding the low-variance 1.5% of the
  spaCy tensor costs 0.025. Any compression, distillation or 384-d encoder
  should expect to pay this.
- **An input LayerNorm does not help** (it was the obvious harness fix; it is
  not the fix).

The -0.035 to -0.043 encoder deltas are well outside that 0.015 floor, so the
finding stands — but a future arm claiming +0.01 from a representation change
must clear the floor, which means the harness needs work first (see 6a).

### 4d. What was tried and is inconclusive

`mixmen` (mention = PCA-384(spaCy) concat PCA-384(ModernBERT), each L2-normed)
scored -0.0353\*, but its own control `pcactl` (the same construction with the
spaCy half duplicated, i.e. no ModernBERT at all) scored -0.0280\* — the
truncation and per-example normalisation dominate. **The concatenation
hypothesis is untested**, and it cannot be tested without widening the
interface (see 2, constraint 2). `bothmen` (spaCy mention kept, ModernBERT
mention routed through the unused-anyway `locs` slot) scored -0.0128\*, worse
than `hybctx` — but that slot only reaches the *country* table, never the code
table, so it does not test the hypothesis either.

### 4e. Honest caveats

- **Reduced mix.** 5,739 entities from three corpora, versus ~29,000 from six.
  No Prodigy, Synth or WikiDocs. Absolute EM is not comparable to the campaign
  ledger; only paired deltas are, and even those could differ at full scale
  (more data is exactly what a richer representation needs).
- **The recipe is the incumbent's.** Epochs, LR and mix width were tuned in
  Waves 1-4 against spaCy tensors. The `ctr`/`basez` controls show that matters
  by ~0.015. A fair fight gives each encoder its own short LR sweep on the text
  projections.
- **One pooling scheme.** Mean over sub-words then mean over tokens, copied
  exactly from the incumbent so the comparison is clean. Better options exist
  for a modern encoder (first-subword, span-boundary concat, attention pooling)
  and none were tried.
- **Last hidden layer only** (layer -5 of ModernBERT-base was probed and was
  worse than -1 on `where`, so -1 was used).
- **Nomic's models expect a task prefix** (`search_document: `) which was not
  used, since the token states are wanted, not the pooled embedding.
- **The probe and the pilot disagree, and the pilot wins.** Anyone re-reading
  the probe table as encouragement should re-read 4b.

---

## 5. Other text-derived signals (analysis only)

### 5a. Corpus-level per-name priors: derivable, but the honest version is a build

`raw_data/orig_mordecai/loc_rank_db.jsonl` is **not** a rank database — it is
the 2017 Prodigy annotation export (2,664 rows; keys `text, spans, options,
accept, answer, meta`) that doubles as the "prodigy" training source
(`train.py:832`). No frequency field. Its implicit prior covers 438 mention
strings, only 29 ambiguous, all Syria/Iraq news. Dead end.

`raw_data/wiki/*.jsonl` **is** Wikipedia link data and does support
P(geonameid | anchor text): 95,249 rows across five raw scrape files carry
`ent_text` (anchor) + `correct_geonamesid`. Computed: 11,446 distinct anchors,
only 537 ambiguous. `tools/prepare_wiki.py` never aggregated by anchor, so the
counts are intact but unused.

**It is unusable as-is, and this is measured, not hypothetical.** Scoring the
anchor prior's argmax against the existing per-entity error dump
(`~/.claude/jobs/51d882cd/tmp/erroranalysis2/records_seed42.pkl`, 8,977 held-out
entities, 914 errors):

| source | anchor coverage | prior top-1 = gold, on that source's errors |
|---|---|---|
| LGL | 0.683 | 11.8% of 170 errors |
| TR | 0.807 | 11% of 35 |
| GWN | 0.842 | 33% of 63 |
| Prodigy | 0.818 | 13% of 53 |
| **WikiDocs** | **1.000** | **89.8% of 579** |

WikiDocs is 72% of the eval set and these files *are* its label-generating
process. A prior built from them posts a large fake gain by pure leakage. An
honest version requires rebuilding the anchor table from the enwiki dump
(`/home/andy/projects/wiki/enwiki-latest-pages-articles.xml.bz2`, 25 GB, on
disk) **excluding every title in `wiki_docs.jsonl`**, and judging per source with
WikiDocs excluded. Honest headroom: ~12% of LGL errors, ~33% of GWN's.

This is **not** the rejected "Wikipedia fame prior" (that screen —
`~/.claude/jobs/51d882cd/tmp/featscreen/` — tested per-*place* prominence:
redirect counts, intro length, primary-topic flags, best residual accuracy
0.192 against `log_min_km_anchor`'s 0.519). Per-*name* conditional priors have
never been tried. But it is a medium-cost build with a severe evaluation trap
and modest upside — rank it below 5b.

### 5b. LGL outlet metadata: present, and it bites exactly where nothing else does

Every one of LGL's 588 articles carries `<domain>` and `<feedid>`
(`raw_data/Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml`,
a symlink into `~/projects/mordecai/`): 85 distinct outlets, 6.9 articles each,
100% coverage. TR-News has `<domain>` too (36 outlets, but national/international
— cbc, bbc, cnn, reuters). GWN has no domain field. Prodigy, Synth and WikiDocs
have none.

Concentration, over all LGL linked toponyms: **80.8%** fall in their outlet's
modal admin1 (leave-one-article-out: 79.6%); on ambiguous phrases, **90.2%**.
TR-News, same computation: 35.9%. Real distributions:
`pal-item.com -> Indiana 62, Ohio 33`; `ajc.com -> Georgia 88`;
`palestineherald.com -> Texas 83`.

Quantified against the actual residual errors (176 held-out LGL documents
joined to articles by `(phrase, geonameid)` multiset; all 170 LGL errors
covered):

| LGL held-out errors (n=170) | count | share |
|---|---|---|
| **gold in outlet's home admin1, prediction elsewhere (fixable)** | **79** | **46.5%** |
| neither in home admin1 | 55 | 32.4% |
| both in home admin1 (feature inert) | 25 | 14.7% |
| prediction in home admin1, gold elsewhere (feature would hurt) | 10 | 5.9% |

And cross-tabbed against PLAN.md item 3's no-document-evidence class
(`not has_sib and not has_anchor`, 38 of the 170 LGL errors):
**30 of those 38 (79%) would be resolved by the outlet's home state.**
Examples: `parispi.net` (Tennessee) "Paris" -> gold Paris TN, predicted Paris,
France, three times; `pal-item.com` (Indiana) "Richmond", "Hagerstown",
"Centerville" all predicted to the bigger same-name city elsewhere.

Two design constraints:

1. **Home location must not come from training labels.** LGL's file order
   groups articles by feed and the positional split hands whole outlets to one
   side: only 9 of the 32 held-out outlets appear in training at all. Use the
   domain string (34 of 85 domains contain a place name; among held-out
   outlets, 13 domains cover 48 of the 79 fixable errors) or an external
   newspaper gazetteer. Note the pleasing inversion: a local paper called
   *Paris* is evidence *against* the population prior.
2. **It is a source indicator.** The feature is structurally absent for 81% of
   eval entities (Prodigy, Synth, WikiDocs). PLAN.md already records
   corpus-identity recalibration as a screen killer. Give it a neutral null
   (mask channel + within-set mean, not a sentinel) and check per-source deltas
   — if WikiDocs moves at all, the model is reading the mask.

Plumbing is cheap: `read_file` already parses the whole XML with `xmltodict`,
so `domain` reaches `data_formatter` untouched; or join onto existing pickles
by the `doc_key` that `tools/enrich_pickles.py:397` already computes.

---

## 6. Recommended experiment specs

Ordered by expected value per unit of effort. Numbered from e40 because
`experiments/e30_absw100` and `e31_abstain_w3` were created concurrently by
another line of work.

### 6a. E40 — fix the harness before judging any encoder (prerequisite, cheap)

PLAN.md item 1 says "build the frozen-feature comparison harness first". This
is what that means concretely: the recipe currently loses 0.010-0.016 to
transformations that are provably free, so any representation arm is measured
through a ~0.015 artefact. Arms: separate (higher) LR and/or a short warmup for
`text_to_country` / `text_to_code` / `context_to_country`; re-run the `ctr` and
`basez` controls as the acceptance test. **The harness is fixed when `ctr` and
`basez` read 0.000.** Cost: ~20 runs, minutes. `scratchpad/pilot/train_ln.py`
shows the monkeypatch pattern for testing model changes without touching
`mordecai3/`.

### 6b. E41 — widen the mention slot to carry both representations (the real test of item 1)

Split `bert_size` into `text_dim` and `country_emb_dim=768` in
`geoparse_model.__init__` (`torch_model.py:449` is the only line that conflates
them), then set the mention tensor to `concat(spaCy 768, ModernBERT 768)` at
full width — no PCA, no per-example normalisation, both of which 4d showed are
disqualifying. Keep context slots on the modern encoder (4b says that is
already free-to-positive). Judge per source; the prediction is LGL/TR gains and
flat WikiDocs. Cost: one small code change, one re-embed (~15 min), 5 seeds.

### 6c. E42 — ship the free half now: context-slot swap

`hybctx` is -0.0068 overall but **+0.0090\* on GWN** with the *incumbent-tuned*
recipe and no harness fix. Re-run it on the full six-source mix after 6a. If it
comes out non-negative there, it is a free 8192-token upgrade to the two
context vectors and it removes the 144-token window from the document features.
Cost: one re-embed of all six sources, 5 seeds.

### 6d. E43 — give the new encoder the entity typing it lacks

The diagnosis in 4b says the missing ingredient is OntoNotes-style typing, not
capacity. Cheapest form: fine-tune ModernBERT-base as a token classifier on
GeoNames feature class using the existing training candidates as distant
supervision (or on OntoNotes NER), then freeze it and re-embed. That keeps the
50 s/run economics. Only if that fails should option (b), full end-to-end
fine-tuning, be attempted — see 2 for its 20-30x cost.

### 6e. E44 — outlet home-location feature (from 5b)

Feature: `is_outlet_home_adm1`, `is_outlet_home_country`, and
`log_km_to_outlet_home`, with an explicit `has_outlet` mask channel. Home
location from the domain string plus an external gazetteer, never from
training labels. Screen offline first (Wave-2b style) against the existing
`records_seed42.pkl` dump; the ceiling is already measured at 46.5% of LGL
errors and 79% of LGL's no-evidence errors, against 5.9% that would move the
wrong way. Report per-source deltas; treat any WikiDocs movement as a mask leak.

### 6f. E45 — fix or reclaim the `locs_tensor` slot (from 1)

Align the train and serve definitions of `loc_ents` (NORP in or out — pick
one), re-measure `nolocs`. If the slot is still worth -0.002, reclaim its 768
input dimensions for something that is not free.

### Not recommended

- A straight three-tensor swap on the full mix. It has now been measured five
  ways and loses.
- The per-name Wikipedia prior (5a) before its leakage-free table exists.
- Any 384-d encoder: `pca384` says the pathway pays 0.025 for that much
  compression before the encoder even matters.

---

## Appendix: artefacts

All under
`/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/pilot/`
(session scratchpad; nothing was written into `raw_data/` or `mordecai3/`):

| file | what |
|---|---|
| `reembed.py` | replays the TR/LGL/GWN formatter with a different encoder; `--parity` proves bit-exact alignment |
| `dump_vecs.py` | writes the three vectors + labels as npz, for the probe |
| `probe.py` | the linear probe (country / admin1 / feature class) |
| `variants.py` | `rand` / `zscore` / `l2` tensor variants |
| `train_ln.py` | monkeypatched trainer (LayerNorm on text inputs) — the pattern for model changes without editing `mordecai3/` |
| `summarize.py` | paired per-seed deltas, campaign house style |
| `data/pickled_es/` | 14 pickle variants (symlinks for the baselines) |
| `runs/*.json` | 90 training runs (18 arms x 5 seeds), each with `_last5` and `_history` |
| `vecs/*.npz` | probe vectors for 6 encoders x 3 sources |

**Concurrency note.** `tools/train.py` was edited by another researcher at
08:19 while these arms were running (08:01-08:44) — the `--window` /
`--abstain-weight` additions, whose defaults are documented as behaviour-
preserving. Verified rather than trusted: re-running `base` seed 42 against the
edited file reproduces the pre-edit run's `_last5` **bit-identically**
(exact_match_avg 0.914003485958198 both times, every metric to 1e-12). All arms
are comparable.

Reproduce any arm with:

```
uv run python tools/train.py train --data-dir <scratchpad>/data --mix-dim 512 \
  --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
  --enriched --feature-blocks "prom,name,cue,sib,geo,shape" --epochs 15 \
  --avg-params --avg-mode swa --seed 42 --dataset-names "TR, LGL, GWN" \
  --pickle-suffix _nomicmb --metrics-out <out>.json
```
