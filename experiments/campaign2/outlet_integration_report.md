# Outlet feature: mainline integration and ship candidate (e54_outlet_ship)

Campaign 2, Phase 2. Written 2026-08-20. This is the third and last outlet
report. `outlet_feature_report.md` built the block (e50) and discharged its two
pre-ship conditions (e53); both were done in a worktree, on an older base, with
the *curated* home table. This one lands the feature in the mainline on top of
Phase 0, re-enriches the shipped pickles, retrains the ship arm on the
**researched** table, builds the serving path §8 only specified, and names a
checkpoint.

**Verdict: the merge is clean and the arm is stronger than e53 measured it.
Identity gates pass at the strongest available standard — all five e29 seeds
reproduce both their checkpoint and their metrics file md5-identically on the
re-enriched pickles, before and after every code change. The ship arm is
+0.0368 ± 0.0070 TLG-hard (t = 11.80) over e29, against e53's +0.0281, because
the researched table leaves no LGL or TR document without a home and makes the
TR gain significant for the first time (+0.0333, t = 5.72). End to end from raw
text on the D2 denominator: 66.99 → 67.95 with no outlet supplied, → 70.40 with
outlets. Ship candidate: seed 42, staged and packaged; the flip is one line and
is not made here.**

One caveat, stated up front: the withheld-outlet number is
**−0.0047 ± 0.0018, t = −2.65** against a criterion of 2.776. It passes, and
the point estimate matches e53's −0.0051 to four decimals, but the seed spread
is half as wide so there is less margin than the e53 write-up implies. §6.

---

## 1. What was merged, and how

The source worktree (`agent-abf0c3af0f35ebb78`) was cut before Phase 0 and had
mainline tooling hand-copied into it, so its diff against its own base mixes
outlet work with a stale snapshot of other people's files. Nothing was
git-merged. Each outlet-specific change was re-applied by hand onto the current
mainline file, and the rest of the worktree's diff was discarded.

| file | what landed | how |
|---|---|---|
| `mordecai3/outlet_features.py` | new, unchanged from e50 | copied |
| `mordecai3/torch_model.py` | `FEATURE_BLOCKS["outlet"]` (appended last), one `_PAD_SENTINEL`, `TrainData.set_outlet_dropout` + `_prepare_outlet_dropout` | patch applied clean — mainline `torch_model.py` was still at the base commit |
| `tools/enrich_pickles.py` | `--outlet-only`, `--permute-homes`, `outlet_pass`, `permute_homes` | patch applied clean, plus a new `--outlet-table researched\|curated` |
| `tools/train.py` | `--outlet-dropout` (flag, validation, per-epoch call, sidecar record) | 4 hunks re-applied by hand; the worktree's 5th hunk was a *stale* copy of Phase 0's scoreboard printer and was dropped |
| `mordecai3/geoparse.py` | the serving path (§4) | written here — the worktree's `geoparse.py` was byte-identical to the mainline's, i.e. §8 really was a spec |
| `tools/end_to_end_eval.py` | `--feature-blocks`, `--outlet-sources`, `<domain>` read out of the corpus XML | written here (§7) |
| `tools/outlet_*.py`, `tools/e53_aggregate.py`, `tools/data/outlet_homes_researched.tsv` | copied; `outlet_conditions_eval.py` generalised off its hard-coded scratch paths | |
| `experiments/e50_outlet/`, `experiments/e53_outlet_dropout/` | the worktree's `seed*.json` / `metrics2` / `conditions_*.json` records | copied, so the ledger survives the worktree |

The **researched** table (`tools/data/outlet_homes_researched.tsv`, a source URL
per row) is now the default everywhere. The curated table stays in
`tools/outlet_home_table.py` as provenance and is reachable with
`--outlet-table curated`, which is what reproduces e50/e53.

## 2. Identity gate — the merge is a no-op with the block off

The gate was run three times: after the code merge (pickles untouched), after
re-enriching the pickles in place, and after the last code edit. Each time, the
e29 recipe at seed 42, written to a scratch path:

| | checkpoint md5 | metrics md5 |
|---|---|---|
| frozen `experiments/e29_swa_ep15/seed42.*` | `22785b60360b7c21edc1ca8b7261167e` | `1d84e9529c77db0ef5b9c639d2dff96f` |
| gate 1 — after code merge | **match** | **match** |
| gate 2 — after re-enrichment | **match** | **match** |
| gate 3 — after `end_to_end_eval` / `compact_candidates` edits | **match** | **match** |

And then across the whole seed set, because the paired baseline had to be rerun
anyway (§5):

| seed | `seed*.pt` | `seed*.json` |
|---|---|---|
| 42, 101, 202, 617, 1848 | **5/5 md5-identical to the frozen e29 checkpoints** | **5/5 md5-identical to the frozen e29 metrics** |

That is the strongest form of the check e50 ran: not just the metrics file but
the weights, on all five seeds, on pickles that now carry five extra columns.
No frozen artifact was written to at any point; every gate wrote to
`$SCRATCH/e54/gate*/`.

## 3. Re-enrichment

```
uv run python tools/enrich_pickles.py --outlet-only \
  --data-dir raw_data/pickled_es --out-dir raw_data/pickled_es \
  --outlet-table researched --outlet-home-cache experiments/e54_outlet_ship/outlet_homes.json
```

About 40 seconds for all seven sources — it reads the already-enriched pickles
and re-derives nothing from Elasticsearch except the 120 home rows. The seven
`*_enriched_compact.pkl` in `raw_data/pickled_es` were backed up to scratch
first (1.5 GB), rewritten in place with `feat_matrix` widened from 42 to 47
columns, and the backup deleted as soon as gate 2 passed. Net cost on disk:
the compact set went 1.50 GB → 1.65 GB.

Coverage, which is where this build differs from e53's:

| | curated (e50/e53) | **researched (e54)** |
|---|---|---|
| domains resolved | 109 of 120 | **120 of 120** |
| point-level homes | 94 | **105** |
| LGL entities: point / country / **none** | 2,790 / 338 / **117** | 3,107 / 138 / **0** |
| TR entities: point / country / **none** | 237 / 337 / **340** | 275 / 639 / **0** |

The geocoded table this produced is **byte-identical to the worktree's
`outlet_homes_v2.json`** — all 120 rows equal — so the two builds agree on the
homes and differ only in which table trained the model.

**One trap, noted and guarded.** The outlet block is attached by a pass over the
*already enriched* pickles, so it lives in the compacted caches the trainer
reads and not in the 4.3 GB uncompacted `*_enriched.pkl`. Rebuilding the caches
(`train.py compact-cache`, or the on-the-fly path in `load_es_data`) therefore
silently drops it. `compact_candidates` now fills the missing keys with the
block's null encoding and logs a warning naming the fix, rather than raising a
`KeyError` a thousand entities deep. `tests/test_feature_parity.py` reads the
uncompacted pickles and is unaffected.

## 4. Serving (report §8, built)

`Geoparser.__init__(..., outlet_homes=<path|dict|None>)` loads the 120-row
table — from `mordecai3/assets/outlet_homes.json` by default, and **only when
the checkpoint's sidecar says the block is there** (`self.uses_outlet`). A
checkpoint without it never loads the table, never writes the keys, and ignores
the argument with a warning.

`geoparse_doc(text, ..., outlet=None)` and
`geoparse_batch(texts, ..., outlets=None)` take one domain per document. They
flow into `add_es_data_batch(..., outlet_homes=[...])`, which after the existing
`add_document_features(doc_es)` call does exactly what §8 specified: one
`add_outlet_features(entity["es_choices"], home)` per entity, or
`clear_outlet_features(...)` when the document has no outlet. **A document with
no outlet is written to the null encoding, not left with the keys missing** —
`ProductionData` indexes them by name and would raise.

Lookup is forgiving in the two ways real provenance data is messy:
`normalize_outlet` accepts a bare domain, a host, or a whole URL; and
`lookup_outlet_home` tries both `www.`-prefixed and bare forms, because LGL
writes `ajc.com` and TR-News writes `www.cbc.ca`. An unknown domain returns
`None` and takes the no-outlet path.

Working, on the marquee case:

```python
>>> geo = Geoparser(model_path="experiments/e54_outlet_ship/seed42.pt")
>>> geo.geoparse_doc(TEXT)["geolocated_ents"]              # Paris, Île-de-France 2988507
>>> geo.geoparse_doc(TEXT, outlet="parispi.net")           # Paris, Tennessee   4647963
>>> geo.geoparse_doc(TEXT, outlet="post-gazette.com")      # Paris, Île-de-France 2988507
```

The third line is the serving-side echo of the §6.4 permutation control: a
different American local paper does *not* pull Paris to America, so what the
model reads is the article-to-newsroom correspondence and not a US prior.

## 5. Results — 5 paired seeds, t(4) > 2.776

Recipe: e29_swa_ep15 + `--feature-blocks "…,outlet"` + `--outlet-dropout 0.5`,
seeds {42, 101, 202, 617, 1848}, checkpoints and sidecars in
`experiments/e54_outlet_ship/`.

Both arms are read from `seed*.metrics2.json` — the Phase-0 standard suite — so
one metric implementation under one reserved-row convention scores both sides.
The e29 side is `e54_outlet_ship/e29_ref/`, the recipe rerun *in this tree*
(§2); the frozen e29 directory predates the metric suite and has no
`metrics2.json`, and the worktree's e50 baseline `metrics2.json` are not
comparable because the mainline changed the twin-credit denominator after they
were written. Aggregator: `tools/e54_aggregate.py`.

| metric | e29 | e54 | paired Δ | t | |
|---|---|---|---|---|---|
| **TLG-hard (primary)** | 0.8676 | **0.9043** | **+0.0368 ± 0.0070** | **11.80** | **\*** |
| LGL non-country EM | 0.8739 | 0.9452 | +0.0714 ± 0.0081 | 19.79 | \* |
| LGL EM | 0.8937 | 0.9533 | +0.0596 ± 0.0068 | 19.62 | \* |
| LGL novel-pair EM | 0.8181 | 0.9278 | +0.1097 ± 0.0155 | 15.84 | \* |
| LGL twin-credit | 0.8956 | 0.9545 | +0.0590 ± 0.0068 | 19.32 | \* |
| novel-pair EM, all sources (guardrail) | 0.7731 | 0.8040 | +0.0309 ± 0.0072 | 9.56 | \* |
| twin-credit macro (no Synth) | 0.9241 | 0.9405 | +0.0164 ± 0.0066 | 5.54 | \* |
| **macro EM, 5 sources (Synth excluded, D4)** | 0.9109 | 0.9271 | +0.0162 ± 0.0095 | 3.83 | \* |
| macro EM, 6 sources (continuity) | 0.9214 | 0.9359 | +0.0145 ± 0.0071 | 4.57 | \* |

The e29 column reproduces `phase0_report.md` §1 exactly (TLG-hard 0.8676,
novel-pair 0.7731, twin-credit 0.9241, macro-5 0.9109, macro-6 0.9214), which is
the check that the baseline here is the campaign's baseline.

**Novel-pair EM deserves its line again**: it is the number answer-key
memorisation cannot move, and it rises 3.1 points overall and 11.0 on LGL.

### 5.1 TLG-hard's three ingredients, and the new TR result

| | e29 | e54 | Δ | t | e53 d50, for comparison |
|---|---|---|---|---|---|
| LGL non-country | 0.8739 | 0.9452 | +0.0714 | 19.79 \* | +0.0719 \* |
| **TR non-country** | 0.8810 | 0.9143 | **+0.0333** | **5.72 \*** | +0.0114 / +0.0074, both n.s. |
| GWN non-country | 0.8479 | 0.8535 | +0.0056 | 1.18 | +0.0085 n.s. |

**The whole gain over e53 is TR.** Pairing e54 directly against e53's d50 seed
for seed — legitimate here because TLG-hard's definition did not change between
the two trees, and both score the same e29 baseline at 0.8676:

| | e53 d50 (curated) | e54 (researched) | paired Δ | t | |
|---|---|---|---|---|---|
| TLG-hard | 0.8956 | 0.9043 | **+0.0087 ± 0.0030** | **6.54** | **\*** |
| TR non-country | 0.8914 | 0.9143 | +0.0229 | 12.83 | \* |
| LGL non-country | 0.9457 | 0.9452 | −0.0005 | −0.25 | n.s. |
| GWN non-country | 0.8498 | 0.8535 | +0.0038 | 1.63 | n.s. |

LGL is flat; TR carries all of it, because the researched table resolves all 35
TR domains and leaves none of its 914 entities homeless where the curated table
left 340. That was e50 §9's first "also worth doing" and it arrived free with
condition 1's fix — the table rebuild was run to retire a provenance risk and
turned out to be worth a significant accuracy gain of its own.

### 5.2 Guardrails — the four sources that structurally cannot have an outlet

| source | e29 | e54 | Δ | t | |
|---|---|---|---|---|---|
| Prodigy | 0.9016 | 0.8944 | −0.0072 ± 0.0350 | −0.46 | n.s. |
| GWN | 0.9283 | 0.9304 | +0.0022 ± 0.0049 | 1.00 | n.s. |
| Synth | 0.9739 | 0.9799 | +0.0060 ± 0.0069 | 1.96 | n.s. |
| WikiDocs | 0.9270 | 0.9276 | +0.0006 ± 0.0020 | 0.69 | n.s. |

The encoder report's tripwire — "if WikiDocs moves at all, the model is reading
the mask" — reads +0.0006 at t = 0.69. Prodigy's ±0.0350 is its usual n = 250
noise.

### 5.3 Per seed

| seed | TLG e29 | TLG e54 | Δ | LGL-nc Δ | novel-all | twin | macro-5 | LGL withheld |
|---|---|---|---|---|---|---|---|---|
| **42** | 0.8597 | **0.9069** | +0.0471 | +0.0704 | **0.8042** | **0.9456** | **0.9326** | **0.8996** |
| 101 | 0.8716 | 0.9014 | +0.0298 | +0.0704 | 0.8014 | 0.9340 | 0.9217 | 0.8901 |
| 202 | 0.8641 | 0.8979 | +0.0338 | +0.0729 | 0.7956 | 0.9389 | 0.9254 | 0.8975 |
| 617 | 0.8653 | 0.9057 | +0.0404 | +0.0829 | 0.8168 | 0.9421 | 0.9269 | 0.8901 |
| 1848 | 0.8771 | **0.9098** | +0.0328 | +0.0603 | 0.8019 | 0.9419 | 0.9290 | 0.8901 |

## 6. Withheld outlet — the e53 blocker, re-checked on the mainline build

`tools/outlet_conditions_eval.py`, LGL held-out, 946 scored entities, one
scorer for every arm and condition. "Withheld" overwrites the five outlet
columns with the exact null encoding a no-outlet source carries.

| arm | present | withheld | withheld − e29 baseline | |
|---|---|---|---|---|
| e29 baseline (no block) | 0.8981 | 0.8981 | (reference) | |
| **e54 (d50, researched table)** | **0.9609** | **0.8934** | **−0.0047 ± 0.0018, t = −2.65** | **n.s.** |
| *e53 d50 (curated), from its NOTES* | 0.9605 | 0.8930 | −0.0051 ± 0.0036, t = −1.41 | n.s. |
| *e50 (no dropout), from its NOTES* | 0.9584 | 0.8698 | −0.0283 ± 0.0052, t = −5.45 | \* |

Per seed (e29 / e54 present / e54 withheld): 42 `.9027/.9662/.8996`,
101 `.8964/.9577/.8901`, 202 `.9027/.9630/.8975`, 617 `.8890/.9598/.8901`,
1848 `.8996/.9577/.8901`.

**Read honestly.** The point estimate reproduces e53 exactly (−0.0047 vs
−0.0051); the *t* is nearly twice as large only because this build's seed
spread is half as wide, and four of five seeds are negative. It clears the
adopt criterion, and it is 6× smaller than the regression that killed e50 — but
the "flat, not merely non-significant" version of this number is still not in
hand. e53 §11.4 already flagged the fix: **one p = 0.7 probe**, five runs,
~5 minutes. It is not a blocker (the e2e numbers in §7 show the no-outlet
condition is a net *improvement* over e29 end to end), but it is the one thing
a shipping decision might want first.

**Since answered — see §11.** The probe was run and p = 0.7 does *not* flatten
this number (−0.0057, if anything marginally worse, and statistically identical
to p = 0.5). The residual is a floor of about five entities that more dropout
cannot reach, not an under-trained fallback. **p = 0.5 stands.**

## 7. End to end from raw text

`tools/end_to_end_eval.py` gained `--feature-blocks` (which must match the
checkpoint) and `--outlet-sources` (which corpora hand their `<domain>` to the
geoparser). `read_corpus` now carries the domain; TR-News and LGL have one on
every article, GeoWebNews has none. The plumbing is the same
`_outlet_homes_for` the library uses, so the harness cannot drift from serving.

Seed 42, `serving` variant, 260 held-out documents, D2 denominator (2,084 gold
toponyms), `max_choices=100`. Raw JSON in `experiments/e54_outlet_ship/e2e/`.

| configuration | e2e EM | acc@161 | det R | det P | emitted-loc precision | oracle-span EM | campaign-parity |
|---|---|---|---|---|---|---|---|
| e29 seed42 — the Phase-0 reference | **66.99** | 69.19 | 78.22 | 75.50 | 77.08 | 83.45 | 90.96 |
| e54 seed42, **no outlet supplied** | **67.95** | 69.96 | 78.22 | 75.50 | 76.09 | 84.42 | 92.25 |
| e54 seed42, LGL outlets | 70.01 | 71.98 | 78.22 | 75.50 | 81.37 | 87.17 | 94.68 |
| e54 seed42, LGL + TR outlets | **70.40** | 72.36 | 78.22 | 75.50 | 82.32 | 87.41 | 94.87 |

The e29 row reproduces `phase0_report.md` §1 cell for cell (66.99 / 69.19 /
78.22 / 75.50 / 83.45 / 90.96), so the harness changes are a no-op on the
existing path.

Per corpus, e2e EM:

| | TR-News | LGL | GeoWebNews |
|---|---|---|---|
| e29 seed42 | 64.95 | 65.04 | 72.85 |
| e54, no outlet | 64.35 | 66.26 | 74.19 |
| e54, LGL outlets | 64.35 | **69.76** | 74.19 |
| e54, LGL + TR outlets | **66.77** | **69.76** | 74.19 |

Three things worth stating:

* **Detection is byte-identical in all four rows.** Same NER, same spans, same
  recall and precision — every point of movement is resolution.
* **The no-outlet row is +0.96 over e29**, not a regression. Whatever the
  ranker-conditioned withheld number in §6 says, the pipeline a caller gets
  with no outlet metadata at all is better than today's, mostly because GWN
  (which has no outlets and never will) rises 1.3 points.
* **Outlets are worth +3.4 e2e EM** over e29 and +2.5 over the same checkpoint
  without them, and they buy 5 points of emitted-location precision on top.
  The gap from serving to the oracle-span ceiling does **not** close: 16.5
  points at Phase 0, 16.5 without outlets, 17.0 with them. The ceiling rises
  with the floor, which is exactly what a *ranker* improvement should do — the
  16-point gap is NER's, and only D5 can spend it.

## 8. Ship candidate

**seed 42.** It leads the arm on novel-pair EM (0.8042), twin-credit (0.9456),
macro-5 (0.9326), macro-6 (0.9394) and the withheld condition (0.8996), and is
second on TLG-hard by 0.0029 — under a third of the seed sd (0.0047). It is
also the seed the e2e numbers above were measured on and the seed Phase 0's
reference used, so nothing has to be re-measured to compare them.

Staged, so promoting it is one line:

* `mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt` (md5
  `1b16db5c8d43cc0c7a0132aa133fd80a`, identical to
  `experiments/e54_outlet_ship/seed42.pt`) **+ its `.pt.json` sidecar**, which
  records `feature_blocks: [...,"outlet"]` and `outlet_dropout: 0.5` — so
  `Geoparser()` configures the block by itself.
* `mordecai3/assets/outlet_homes.json`, the 120-row geocoded table, loaded
  automatically and only when the block is present.
* Both listed in `pyproject.toml`'s package data.

**The flip** — the owner's decision, deliberately not made here:

```python
# mordecai3/geoparse.py, line ~135
DEFAULT_MODEL_ASSET = "assets/mordecai_2026-08-20_e54_seed42.pt"
```

Nothing else changes. `geoparse.py` today still points at
`assets/mordecai_2026-08-20_seed101.pt` and the comment above that line says so.

What the flip would buy and cost, for the record: +0.0368 TLG-hard, +1.0 e2e EM
with no outlet metadata and +3.4 with it; the five Phase-0 `xfail`s are
untouched by it (they are e29-seed101 behaviours and were not re-attributed
against seed 42 — a promotion should re-run them).

## 9. Tests

`tests/test_outlet_features.py`, 21 tests, all passing. The two the brief asked
for, plus the unit layer:

* **outlet passed → features change → a known case improves.**
  `test_a_local_papers_own_town_beats_the_population_prior` drives the real
  `geoparse_doc`: no outlet → Paris, France; `parispi.net` → Paris, Tennessee
  (4647963), the case `outlet_feature_report.md` §7 names.
  `test_outlet_features_actually_reach_the_candidates` reads the numbers off
  the candidates with `debug=True` and asserts the five outlet columns move
  *and nothing else does*.
* **no outlet → identical to pre-merge behavior.**
  `test_the_default_checkpoint_ignores_the_outlet_argument` asserts full dict
  equality of `geoparse_doc(text)` and `geoparse_doc(text, outlet="parispi.net")`
  on the packaged e29 checkpoint, and that no outlet key appears on any result.
  `test_batch_without_outlets_is_unchanged` does the same for `geoparse_batch`.
* Plus: the block is registered last and shifts no earlier column; the null
  encoding is the `NO_ANCHOR_KM` sentinel and not zero; `add_outlet_features(…,
  None)` and `clear_outlet_features` agree exactly; a country-level home fires
  only the country channel; URL/`www.` normalisation; an unknown domain is
  score-for-score the no-outlet path; batch outlets are positional and a length
  mismatch raises.

Full suite: **79 passed, 2 failed, 1 skipped, 5 xfailed** — the two failures are
the known allowances `test_miss_oxford` and `test_prague`, and the five xfails
are Phase 0's.

## 10. Surprises

1. **Condition 1's fix was also condition "also worth doing" #1.** The
   researched table was built to retire a provenance risk; it bought +0.0087
   TLG-hard (t = 6.54) over the curated one, all of it TR, purely by resolving
   11 more domains. Coverage, not curation quality, was the binding constraint
   — and e53 §10.2 measured the swap only at *evaluation* time (−0.0007), which
   is why nobody saw this coming: swapping the table under a trained model
   changes nothing, but training on the better table is worth a third as much
   again as the whole dropout arm.
2. **The withheld number got tighter, not looser.** Same point estimate as e53,
   half the spread, so |t| went 1.41 → 2.65. Non-significance that rests on
   noisy seeds is not the same as flatness, and this is the second report to
   defer the p = 0.7 probe that would tell them apart.
3. **The outlet block lives only in the compacted caches.** Nothing in the
   pipeline announced that, and `compact_candidates` would have raised
   `KeyError` on the uncompacted pickles. Guarded and logged, but it is a real
   footgun for anyone who rebuilds the caches.
4. **The worktree's `geoparse.py` was byte-identical to the mainline's.** §8 was
   labelled "spec, not built" and it meant it — worth knowing when reading a
   worktree's `git status`, where a modified `geoparse.py` looked like work.
5. **e2e improves even with no outlet supplied** (+0.96 EM), driven by GWN,
   which has no outlet metadata in any training example. That is the dropout
   regulariser paying out on a source it was not aimed at, and it is the
   opposite of what §8's blocker looked like.

---

## 11. Follow-up: the p = 0.7 probe (asked twice, now answered)

e53 §11.4 and §6 above both deferred the same cheap question. It was run:
five seeds, identical e54 setup (researched table, the same re-enriched
pickles, no re-enrichment), `--outlet-dropout 0.7`. Checkpoints, metrics and
both aggregations are in `experiments/e54_outlet_ship/p07/`.

```
for S in 42 101 202 617 1848; do bash tools/run_e54.sh $S d70; done
uv run python tools/e54_aggregate.py --arm experiments/e54_outlet_ship/p07
uv run python tools/e54_aggregate.py --base experiments/e54_outlet_ship \
                                     --arm experiments/e54_outlet_ship/p07
uv run python tools/outlet_conditions_eval.py --source lgl --data-dir raw_data/pickled_es \
  --arm "e29:…:experiments/e29_swa_ep15/seed{seed}.pt" \
  --arm "d50:…,outlet:experiments/e54_outlet_ship/seed{seed}.pt" \
  --arm "d70:…,outlet:experiments/e54_outlet_ship/p07/seed{seed}.pt" \
  --baseline e29 --out experiments/e54_outlet_ship/p07/conditions_lgl.json
```

### 11.1 The withheld condition does not flatten

| arm | present | withheld | withheld − e29 | t | negative seeds |
|---|---|---|---|---|---|
| e29 baseline | 0.8981 | 0.8981 | (reference) | | |
| **d50 (p = 0.5)** | 0.9609 | 0.8934 | **−0.0047 ± 0.0039** | −2.65 n.s. | 4 / 5 |
| **d70 (p = 0.7)** | 0.9600 | 0.8924 | **−0.0057 ± 0.0071** | −1.79 n.s. | 4 / 5 |

Per seed, withheld (e29 / d50 / d70): 42 `.9027/.8996/.8996`,
101 `.8964/.8901/.8827`, 202 `.9027/.8975/.9017`, 617 `.8890/.8901/.8911`,
1848 `.8996/.8901/.8869`.

**The point estimate moves the wrong way** — −0.0047 → −0.0057 — and the *t*
falls only because the spread nearly doubles (±0.0039 → ±0.0071). Paired d70
against d50 directly: **−0.0011 ± 0.0044, t = −0.53**, i.e. the two are
indistinguishable. Four of five seeds stay negative in both.

**So the e53 monotonicity reading was wrong.** d30 −0.0068 → d50 −0.0051 looked
like a trend that more dropout would drive to zero; it is not. The residual is a
**floor at about −0.005**, which on this denominator is 946 × 0.005 ≈ **5
entities**. Whatever those five mentions are, they are not fixed by showing the
model more no-outlet news documents — the model has already learned that policy
by p = 0.5, and the remainder is something else (a plausible read: LGL golds
whose *only* disambiguating evidence in the document was the newsroom, where no
fallback policy can help).

### 11.2 The present-side cost is real but not significant

| metric | e54 (p = 0.5) | p = 0.7 | paired Δ | t | |
|---|---|---|---|---|---|
| **TLG-hard** | **0.9043** | **0.9026** | **−0.0017 ± 0.0063** | **−0.61** | **n.s.** |
| LGL non-country EM | 0.9452 | 0.9457 | +0.0005 ± 0.0032 | 0.36 | n.s. |
| novel-pair EM, all | 0.8040 | 0.8005 | −0.0035 ± 0.0072 | −1.11 | n.s. |
| twin-credit macro | 0.9405 | 0.9390 | −0.0015 ± 0.0026 | −1.28 | n.s. |
| macro EM, 5 sources | 0.9271 | 0.9263 | −0.0008 ± 0.0021 | −0.90 | n.s. |
| macro EM, 6 sources | 0.9359 | 0.9341 | −0.0018 ± 0.0015 | −2.69 | n.s. |
| **Synth (guardrail)** | 0.9799 | 0.9732 | **−0.0067 ± 0.0047** | **−3.16** | **\*** |

Against e29, p = 0.7 is still a large win (TLG-hard +0.0351 ± 0.0114, t = 6.88),
so this is a choice between two good settings, not a rejection. But **every
aggregate cell is worse-or-equal at p = 0.7, none is better, and the only
significant movement anywhere in the comparison is a Synth regression** of
−0.0067 (t = −3.16). Macro-of-six at −0.0018 (t = −2.69) sits just under the
criterion pointing the same way. Seven small negatives and no positives is not
what noise usually looks like.

### 11.3 Verdict — **p = 0.5 stands**

The decision rule was: adopt p = 0.7 if the withheld condition goes to ≈ 0 *and*
the present-side cost is non-significant. The second clause holds (t = −0.61);
**the first fails outright** — withheld is −0.0057, no closer to zero than
p = 0.5's −0.0047 and statistically identical to it. p = 0.7 buys nothing where
it was supposed to buy something, and consistently gives back a little
everywhere else.

**The ship setting remains `--outlet-dropout 0.5`, and the ship candidate
remains `experiments/e54_outlet_ship/seed42.pt`** — unchanged, already staged
at `mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt`. Nothing in §8 moves.

The useful by-product is that the withheld residual is now **characterised
rather than merely tolerated**: it is a ~5-entity floor that dropout cannot
reach, not an under-trained fallback policy. If anyone wants it gone, the lead
is an error analysis of those five LGL mentions, not another value of *p* — and
§7 remains the reason it is not urgent, since end to end the no-outlet
condition is +0.96 EM *over* e29 rather than a regression at all.
