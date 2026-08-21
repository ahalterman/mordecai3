# e55_ner_head — scaling the place-span head's labels (ladder step N2)

Campaign 2, the flagship NER track. This is step **N2** of the ladder in
`experiments/campaign2/ner_retrain_scoping_report.md` §7: the pilot (N1) put a
0.5 M-parameter span head on the frozen `en_core_web_trf` tensors, trained it on
5,034 TR/LGL/GWN gold toponyms in 35 s, and measured its learning curve **still
climbing** at +0.7 detection F1 and +4.5 nested recall per doubling. N2 adds
every other label source on disk and measures each one's marginal contribution.

**No checkpoint is promoted, no shared code was changed, no TEST split was
touched.** Full report with tables, ablations and residuals:
`experiments/campaign2/ner_head_scaling_report.md`.

## Denominator

D2, 2,084 non-demonym linked gold toponyms in the 260 held-out documents of
TR-News / LGL / GeoWebNews (`experiments/campaign2/phase0_report.md`). Held-out
documents **of the corpora the head trained on** — the N1 gate (reproduce on
D1's modern-news TEST corpus) is unchanged and still open.

## Method

Frozen encoder throughout: the head reads the `._.tensor` values the pipeline
already computes for the ranker. Candidates are every span of ≤ 8 tokens that
does not cross a sentence boundary; a candidate is positive iff it is exactly a
D2 gold toponym, so demonyms are negatives by construction and nested toponyms
are positives.

The training loop was rewritten to micro-batch chunks — **13 s/run against the
pilot's 35–60 s** — and reproduces the pilot: det F1 **87.06 ± 0.36** (5 seeds)
vs 87.16 ± 0.12, e2e EM **76.36 ± 1.35** (3 head seeds) vs 77.35 (1 seed).

Runs are **deterministic** given (config, seed), verified by rerun, so every
comparison is **paired by seed**. `*` = |t| > 2.776 at 5 seeds, `~` = |t| > 4.303
at 3 seeds. The noise floor is ~0.4 detection F1, ~2.5 nested recall and ~17
demonym false positives.

## Label sources added

| source | scale | on disk before this arm? |
|---|---|---|
| WikiDocsFull anchors | 20,302 docs / 6.87 M tokens / **71,475 anchors** | yes, spaCy'd; cached here to 10 GB fp16 in 110 s |
| filtered silver nested | **12,180** spans (anchor filter) / 17,343 (standalone filter) | generated here from the cached arrays + one ES pass |
| explicit demonym negatives | 666 D2 demonym golds + 1,324 spaCy NORP spans | already in the training corpora |
| self-training pseudo-labels | **130,476** pseudo-positives over WikiDocsFull | generated here, one round |

## Results (D2 held-out, paired vs the gold-only baseline)

| arm | seeds | det F1 | Δ F1 | nested det R | Δ | demonym FP | Δ |
|---|---|---|---|---|---|---|---|
| **gold only, 20 ep (pilot N1)** | 5 | **87.06 ± 0.36** | — | **67.2 ± 2.0** | — | **73.2 ± 13.4** | — |
| gold only, 24 ep *(control)* | 5 | 87.27 ± 0.48 | +0.05 | 66.5 ± 3.6 | −0.62 | 76.6 | +2.0 |
| gold only, 4 ep + warm restart + 20 ep *(control)* | 5 | 87.20 ± 0.49 | +0.14 | 69.3 ± 3.3 | +2.12 | 79.2 | +6.0 |
| **(a) + WikiDocsFull, two-stage** | 5 | 87.38 ± 0.13 | +0.32 | 70.8 ± 1.4 | **+3.60\*** | 79.4 | +6.2 |
| **(b) + silver nested (anchor filter)** | 4 | 87.49 ± 0.20 | +0.40 | 71.2 ± 1.3 | **+4.81~** | 72.5 | −0.8 |
| (b) + silver nested (standalone filter) | 3 | 87.66 ± 0.17 | +0.59 | 70.2 ± 2.0 | +3.70 | 80.0 | +9.3 |
| **(c) demonym negatives, weight 10** | 5 | 87.17 ± 0.27 | +0.11 | **73.1 ± 1.4** | **+5.93\*** | **59.6 ± 0.5** | −13.6 |
| (c) demonym negatives, weight 30 | 5 | 86.78 ± 0.23 | −0.28 | 71.6 ± 1.9 | +4.35* | **49.4 ± 3.8** | **−23.8\*** |
| **(d) + self-training (one round)** | 3 | 87.46 ± 0.07 | +0.39 | 69.9 ± 1.5 | +3.37 | 77.7 | +7.0 |
| **(a)+(b)+(c) together** | 3 | 86.77 ± 0.37 | −0.30 | **73.7 ± 2.6** | **+7.24~** | **54.7 ± 1.2** | −16.0 |
| *serving path today* | — | *76.8* | | *13.8* | | *59* | |

**Nothing moved detection F1 outside the noise floor.** Wikipedia's 71,475
anchors are worth **+0.32 F1 (t = 1.66, ns)** against the baseline and
**+0.18 F1 (t = 1.04, ns)** against the warm-restart control that its recipe
requires — i.e. most of the apparent gain is the schedule, not the data.

End to end, one ranker checkpoint (`e29_swa_ep15/seed42.pt`, `max_choices=100`),
three head seeds:

| span source | e2e EM | nested e2e EM |
|---|---|---|
| gold-only head | **76.36 ± 1.35** | 54.90 |
| + WikiDocsFull | **77.37 ± 0.51** (+1.00, t 1.87, ns) | 59.01 (+4.11, t 7.1, sig) |
| (a)+(b)+(c) together | 76.33 ± 0.45 (−0.03, ns) | **61.48** (+6.58) |
| oracle spans | 83.45 | 78.02 |

61.48 nested end-to-end exact match is the highest in either report — above the
pilot's fine-tuned encoder (60.74) — from a frozen head that trains in 87 s.

## Reading

1. **The pilot's learning curve was an in-domain curve and it does not
   continue.** The wiki axis is flat from its first ~2,500 documents: 12.5% /
   25% / 50% / 100% of WikiDocsFull score 87.42 / 87.56 / 87.60 / 87.38.
2. **Wikipedia's anchors are 27.3% complete** (67,076 of 245,531 spaCy GPE/LOC
   entities) and carry **almost no nested toponyms** — a head trained on them
   alone reaches det F1 77.51 with **nested recall 4.9**. That is the whole
   story: the head is limited by *in-domain, complete, nested* labels, of which
   there are 5,034 in existence, not by label volume.
3. **Loss design for partial annotation matters 3.32 F1 / 38 nested recall when
   the partial labels are mixed into training, and ~0.03 F1 when they are
   confined to a pre-training stage.** Naive O-class treatment — the documented
   trap — is indistinguishable from the best mask under the two-stage recipe.
   Anchor-string propagation is the worst idea in the design space (81.34 F1,
   nested 19.3).
4. **Explicit demonym negatives are the only source that paid, and they cost
   nothing.** The scoping report §5a called demonym suppression "a real work
   item, not a freebie"; weighting the negatives already present 10x takes the
   demonym FP count from 73.2 to **59.6** at flat F1 — level with the label
   filter the head replaces — and 30x takes it to **49.4** for −0.28 F1 (ns) and
   a significant −23.8 FPs. It also buys +5.93 nested recall, as much as all of
   WikiDocsFull.
5. **The combined arm is the one to look at if nested recall and demonym
   suppression are the goal**: nested detection recall **73.7** and demonym FPs
   **54.7**, both the best in the report, for −0.30 F1 (ns).
6. **The e2e number in the ladder needs restating.** Three head seeds of the
   pilot's own recipe give 74.81 / 77.02 / 77.26 — the published single-seed
   77.35 is the top of that range. N1 is **76.4 ± 1.4**, not 77.4.

## Ship staging (N1 prep, not applied)

`experiments/e55_ner_head/ship/` **in the worktree** (full seed sets in
`<scratchpad>/ner_scale/ship/`): a self-contained `span_head.py` inference
module, two head checkpoints — `span_head_gold_42.pt` (N1 as scoped, det F1
87.62 / nested 67.9 / demonym FP 81) and `span_head_C_all_42.pt` (the combined
arm, 86.34 / **76.5** / **56**), 1.6 MB each with the threshold inside — a
parity test that reproduces each harness row **exactly** from the cached
DocBins, and `INTEGRATION.md` specifying
the replacement of `doc_to_ex_expanded`'s label filter, `trim_span_tokens` and
`nested_gazetteer_spans`. Measured serving cost on 120 held-out documents (idle
GPU): **6.86 ms/doc mean, 4.82 ms median**, against 1.68 ms for today's
extraction and **8.68 ms for `nested_gazetteer_spans`, the pass it replaces**
(plus that pass's Elasticsearch round trip). That is **+5.2 ms/doc**, not the
pilot's +2.7 — worth restating in the ladder. Staged, not applied —
`geoparse.py` is being edited by another arm.

## Gate

The unfreeze arm (N3 on scaled labels) was **not run**: it is gated on the
scaled labels beating the frozen arm by more than seed noise, and none did. The
pilot's N3 estimate (+1.27 det F1, +11.1 ms/doc) stands unmodified — but its
*prior* improves, because "add more data" is now measured and closed as an
option. The highest-value remaining item is D1's written annotation convention
plus fresh in-domain nested gold, not another corpus off the disk.

## Reproducing

All code in the session scratchpad `.../scratchpad/ner_scale/` (see the report's
appendix for the file table). The pilot's harness `.../scratchpad/ner/` is
imported read-only.

```
uv run python <scratch>/ner_scale/build_wiki.py                     # 110 s, 10 GB
uv run python <scratch>/ner_scale/silver_wiki.py --filter anchor1 --out silver_anchor1.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_screen.json --out results.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_curve.json  --out results_curve.json
uv run python <scratch>/ner_scale/sweep.py --config cfg_dem.json    --out results_dem.json
uv run python <scratch>/ner_scale/sweep.py --config cfg9.json       --out results_ctrl.json
uv run python <scratch>/ner_scale/sweep.py --config cfg11.json      --out results_c.json
uv run python <scratch>/ner_scale/show.py --paired gold_ship
uv run python <scratch>/ner_scale/ship/test_span_head.py ship/span_head_gold_42.pt
```
