# Independent audit of the accuracy campaign (2026-08-20)

Auditor's brief: recompute the headline from primary data, quantify
selection-on-test risk, check the evaluation code, and confirm the ship
checkpoint reproduces. Raw data was treated as authoritative wherever it
disagreed with `NOTES.md`. Nothing was committed; no repository file was
modified. Working scripts live in the session scratchpad
(`repro.py`, `leak.py`, `denom.py`).

**Bottom line.** The headline arithmetic is exactly right and the ship
checkpoint reproduces bit-for-bit. The campaign's three big claims (enriched
features, the loss/recipe fix, weight averaging) survive every test I threw at
them. What does not survive is (a) the identity of the artifact at the repo
root, (b) several fine-grained tiebreaks inside the recipe, (c) the Wikipedia
reversal, and (d) the framing of the reported number as something a deployment
would see.

---

## 1. Recomputation of the headline — **PASS, with three reporting defects**

Recomputed from `experiments/*/seed*.json` `_last5` blocks, five seeds
{42, 101, 202, 617, 1848}.

| quantity | ledger | recomputed | verdict |
|---|---|---|---|
| e0 exact match | 0.8808 ± 0.0038 | 0.8808, sd 0.0038, se 0.0017 | matches (± is the **SD**) |
| e0 acc@161 | 0.9262 ± 0.0029 | 0.9262, sd 0.0029 | matches |
| e29 exact match | 0.9258 | 0.9258, sd 0.0033 | matches |
| e29 acc@161 | 0.9661 | 0.9661, sd 0.0004 | matches |
| paired ΔEM | +0.0450 ± 0.0051 * | +0.0450, sd 0.0057, **2 SE = 0.0051** | matches (± is **2 SE**) |
| paired Δacc@161 | +0.0399 ± 0.0024 * | +0.0399, 2 SE 0.0024 | matches |
| LGL | +0.098 * | +0.0982, 2 SE 0.0056, 5/5 seeds | matches |
| TR | +0.062 * | +0.0615, 2 SE 0.0112, 5/5 | matches |
| WikiDocs | +0.060 * | +0.0602, 2 SE 0.0025, 5/5 | matches |
| GWN / Prodigy / Synth "all positive" | — | +0.0137 / +0.0182 / +0.0179 | means positive; see below |

Every intermediate baseline also reproduces: e11b_combo_ls 0.8845, e14_no_cf
0.9191, and the whole Wave-2 and Wave-4 delta tables in the NOTES files
reproduce to the last digit I checked. No arithmetic errors found anywhere.

Three defects in how the numbers are *presented*:

**1a. The `±` symbol means two different things in adjacent cells.** On the
absolute rows it is the standard deviation across seeds; on the delta rows it
is twice the standard error of the paired difference. A reader comparing
"0.8808 ± 0.0038" with "+0.0450 ± 0.0051" will assume the same quantity. Pick
one and label it.

**1b. "Seed-to-seed spread was cut roughly in half" is false as written.**
Exact-match seed SD went 0.0038 (e0) → 0.0033 (e29): a 13% reduction, not a
halving. The halving is real against *e14_no_cf* (0.0058 → 0.0033), which is
the comparison the e29 NOTES actually makes, but the sentence in
`ACCURACY_CAMPAIGN.md` sits immediately after the e0 table. acc@161 spread did
collapse (0.0029 → 0.0004, 7×). And per source the claim inverts: Prodigy's
seed SD **tripled**, 0.0066 → 0.0211.

**1c. "Every source improved" is a statement about means only.** Prodigy is
+0.0182 with 2 SE = 0.0233 (t = 1.57, 4/5 seeds, one seed at −0.0208). It is
not distinguishable from zero. The other five sources are (t = 3.5 to 47).

---

## 2. Selection-on-test risk — **MODERATE and quantified; the winner is stable, its ingredients are not**

There is no dev set: all 57 comparable configs (58 counting `e24_rstar`, which
is scored on a different answer key) were selected on the same six held-out
sources the headline reports.

### How big is the max-selection bias?

Three independent estimates:

* **Empirical, leave-2-seeds-out.** For each of the 10 ways to split the five
  seeds 3/2, pick the argmax config on the 3 seeds and score it on the other 2.
  Mean optimism **+0.0020 EM**. `e29_swa_ep15` was picked in **8 of 10** splits
  (`e27_strip_ep15` twice) and is also the all-5-seed argmax. *The identity of
  the winner is stable; only its margin is inflated.*
* **Order statistics / simulation.** Median per-config seed SD is 0.0034, so
  SE(mean of 5) = 0.0015. If the top *K* configs were truly identical, the
  best-of-*K* score exceeds the truth by +0.0018 (K=5), +0.0024 (K=10),
  +0.0029 (K=20), +0.0035 (K=57). Cross-config correlation of seed residuals
  is only ρ ≈ 0.05–0.12, so these are near the independent case.
* **Per-decision.** Median paired-delta SD among top arms is 0.0049
  (SE 0.0022). Picking the best of 3 / 4 / 6 equal arms yields an expected
  reported delta of +0.0019 / +0.0023 / +0.0028.

**Estimate: ~0.002–0.004 of the +0.0450 headline is winner's curse (5–8%).**
The campaign's structure protects it — the ladder made ~10 sequential decisions
over 3–6 arms each, not one selection over 57 — and the two dominant effects
(features +0.0345, SWA +0.0067) are far outside any plausible selection bias.

**But the reported SEs understate the real uncertainty for a reason nobody
flagged.** `load_es_data` calls `random.seed(617)` unconditionally, so the
train/held-out split, the Synth subsample, and the training data order are
*identical* across all five "seeds". The seed varies only torch weight init and
the DataLoader shuffle. Every ± in this campaign is therefore optimization
noise **conditional on one frozen split**. It carries no information about
whether the recipe generalizes to a different sample.

### Do the winning recipe's ingredients replicate?

| ingredient | contrast | Δ EM | 2 SE | seeds + | t | verdict |
|---|---|---|---|---|---|---|
| enriched feature block | e11b → e14_no_cf | +0.0345 | 0.0065 | 5/5 | 10.7 | **solid** |
| SWA | e14_no_cf → e29 | +0.0067 | 0.0032 | 5/5 | 4.2 | **solid** |
| Wave-1b bundle | e11_combo → e11b | +0.0052 | 0.0026 | 5/5 | 4.0 | **solid** |
| label smoothing **= 0.05** | e4_mlp → e7_ls005 | +0.0016 | 0.0021 | 4/5 | 1.5 | **not replicated** |
| label smoothing 0.1 | e4_mlp → e7_ls01 | +0.0010 | 0.0048 | 2/5 | 0.4 | indistinguishable from 0.05 |
| label smoothing 0.2 | e4_mlp → e7_ls02 | +0.0004 | 0.0025 | 2/5 | 0.3 | indistinguishable from 0.05 |
| epochs 15 (no SWA) | e14(15ep) → e23_ep12 | **+0.0013** | 0.0029 | 3/5 | 0.9 | **12 epochs scored higher** |
| epochs 15 (no SWA) | e14(15ep) → e23_ep10 | **+0.0013** | 0.0036 | 3/5 | 0.7 | **10 epochs scored higher** |
| SWA 8–15 vs SWA 7–12 | e25_swa → e29 | +0.0012 | 0.0010 | 4/5 | 2.4 | **winner's-curse-sized** |
| SWA 8–15 vs EMA | e25_ema → e29 | +0.0018 | 0.0022 | 3/5 | 1.6 | not replicated |
| SWA 8–15 vs SWA+cosine | e29_swa_cosine → e29 | +0.0020 | 0.0026 | 4/5 | 1.5 | not replicated |
| drop `cf` block | e13_all → e14_no_cf | +0.0011 | 0.0052 | 3/5 | 0.4 | **not replicated** (tiebreak was acc@161 +0.0013 ± 0.0013, t = 2.08) |
| keep `prom` block | e14_no_prom → e14_no_cf | +0.0053 | 0.0063 | 4/5 | 1.7 | not replicated |
| keep `prom,name` | e14_core → e14_no_cf | +0.0067 | 0.0072 | 4/5 | 1.9 | not replicated |

**Read:** the recipe that is actually supported by the data is *"enriched
candidate features + the fixed loss + some label smoothing + a short schedule
+ some form of weight averaging."* Everything finer — 0.05 rather than 0.1,
15 epochs rather than 12, SWA rather than EMA, 26 features rather than 30 — is
inside the noise. Note the internal tension: Wave 3 explicitly recommended
`--epochs 12` ("+0.0013 EM n.s., 20% cheaper"), and the ship recipe silently
went back to 15 on the strength of a +0.0012 ± 0.0010 SWA-window comparison
that is smaller than the measured selection optimism.

### Runner-up daylight

e29_swa_ep15 0.9258 · e25_swa 0.9246 · e25_ema 0.9240 · e29_swa_cosine 0.9237 ·
e27_strip_ep15 0.9227 · e28_cosine 0.9219. The top four are within 0.0021 —
about 1.4 SE of a single config's mean-of-5. There is essentially no daylight.

### One trap in the ledger's directory listing

`e24_rstar` has the **highest `_last5` exact match in the whole campaign**
(0.9259, above e29's 0.9258). It is correctly discounted in PLAN.md — it is
scored on the rewritten `_r2` answer key, so the number is not comparable — but
anyone sorting the experiment directories by score will land on the rejected
arm first. Worth a `REJECTED` marker in that directory.

### A methodological point about the `*` rule

`* = |mean| > 2 SE` is a z-test applied to n = 5. The 95% two-sided critical
value for t(4) is **2.776**, not 2. Recomputing, these survive comfortably:
e0→e29 (t = 17.6), enriched features (10.7), SWA vs e14 (4.2), the Wave-1b
bundle (4.0), TR/LGL/WikiDocs/GWN/Synth in the headline table (3.5–47).
These **lose their star**: SWA 8–15 vs SWA 7–12 (2.4), SWA 7–12 vs ep12 (2.7),
`e12_ls005_full22` (2.7), e14_no_cf vs e13_all on acc@161 (2.08), and LGL (2.4)
and Synth (2.05) in the "e29 improves all six sources" table. The campaign's
conclusions are unaffected; roughly ten of its tiebreak stars are.

### Pairing does not do what the ledger says it does

Both `ACCURACY_CAMPAIGN.md` and `PLAN.md` justify paired deltas with *"training
is bit-for-bit reproducible at a fixed seed, so same-seed differences between
configs are pure signal."* Measured across the 30 post-Wave-2 arms, the
paired-difference SD is **0.93–0.97×** the unpaired SD. The seed effect shared
across configs has SD 0.0013 against 0.0037 within-config (ρ ≈ 0.12). Pairing
buys a 3–7% variance reduction, not a qualitative change. Determinism at a
fixed seed means a *run* is reproducible; it does not mean two different
configurations share their optimization noise. The reported SEs are honest
(they are computed from the actual differences) — the justification is not.

---

## 3. Evaluation-code correctness

### (a) Denominator — consistent across configs, but the number is not deployment accuracy

`error_utils.evaluate_results` skips an entity when `np.where(ent['correct'])`
is empty, i.e. when the gold geonames id is not among the retrieved candidates.
Those entities never enter `correct_geoid` or `dists`, so **exact match and
acc@161 share a denominator of "entities whose gold is retrievable."**

The denominator is determined entirely by the pickles, not by the model, so it
is **identical across every config and every seed** — all comparisons in the
campaign are fair. Verified independently: `n_val` is the same in every config
json, and the eval denominator pooled to **8,838**, which is exactly the figure
`ACCURACY_CAMPAIGN.md` quotes ("4 of 8,838 held-out entities").

Exclusion rate per source (held-out split):

| source | n_val | gold not retrievable | excluded | eval N |
|---|---|---|---|---|
| Prodigy | 500 | 0 | 0.00% | 500 |
| TR | 274 | 3 | 1.09% | 271 |
| LGL | 973 | 27 | **2.77%** | 946 |
| GWN | 474 | 14 | **2.95%** | 460 |
| Synth | 300 | 1 | 0.33% | 299 |
| WikiDocs | 6456 | 94 | 1.46% | 6362 |
| **pooled** | **8977** | **139** | **1.55%** | **8838** |

The gap between the reported number and what a deployment sees, measured on
`experiments/e29_swa_ep15/seed42.pt`:

| metric | macro EM |
|---|---|
| **as reported** (retrievable-gold denominator) | **0.9204** |
| same argmax, denominator = all held-out mentions | 0.9072 (**−1.32 pt**) |
| through `geoparse.py`'s abstention rules | 0.9019 (**−1.85 pt**) |

And none of this includes spaCy NER misses, which are invisible to this harness
entirely (PLAN item 6 is right to flag it).

`evaluate_results` does compute `total_missing` and `missing_correct`, but
`make_wandb_dict` never propagates them — so the exclusion rate appears in
**none** of the 290 experiment jsons. It should be logged.

### (b) NULL / reserved-row handling — **train, eval, and serving all disagree**

Three distinct rows are in play and the three code paths pick different ones.

1. **Training** (`TrainData.create_labels`): when no candidate is correct, the
   label is index `max_choices - 1` = **499**, a row that `create_mask` forces
   live and whose gaz features are overwritten to −1. This is the abstention
   class the model is trained to emit.
2. **Eval** (`evaluate_results`): `predicted_position` is
   `argmax([c['score'] for c in es_choices if 'score' in c])`, and scores are
   only written for `n < len(es_choices)`. For the ~94% of entities with fewer
   than 500 candidates, **row 499 is never in the argmax at all** — the class
   the model is trained on is invisible to the metric. Separately, eval's
   "credit for the no-match option" branch tests index `len(es_choices)-1`,
   which is the gazetteer's own `_null_choice` row — a *different* row from 499.
3. **Serving** (`geoparse.py`, `_get_best`): the first thing it does is
   `if pred[-1] == pred.max(): abstain`. The reserved row **is** selectable at
   inference. It then abstains a second time if the gazetteer NULL row wins.

Measured on e29 seed42 over the 8,977 held-out entities: the reserved row wins
the raw argmax on **121 (1.35%)** and the gazetteer NULL row on another 26.
Applying both serving rules is what takes macro EM from 0.9204 to 0.9019.

Also in this area:

* `error_utils.py` line 51 (`correct_position == len-1 and predicted == len-1`)
  is **dead code**. `ex['correct']` is `c['geonameid'] == gold`, and
  `_null_choice`'s geonameid is the literal string `"NULL"`, which never equals
  a gold id. The only live "missing" branch is `correct_position is None`.
* An entity whose prediction lands on the NULL row gets
  `dist = haversine(gold, (0, 0))`. That is correct for acc@161 (it counts as
  wrong) but makes `avg_dist` essentially uninterpretable — e.g. Prodigy reads
  `avg_dist 192 km` at `acc@161 = 0.98`.
* `create_labels` would raise `IndexError` if a gold ever landed past index 499
  without `--full-null-row`. Measured: **0** such entities in all six held-out
  sets, so it is latent only. (43% of LGL entities do have >500 candidates.)

**Net effect on the campaign: none of this biases any comparison** — it is
identical across configs. It does mean the campaign's own metric cannot see the
abstention signal that campaign-2 item 5 proposes to productionize.

### (c) `_last5` — matches the docs, but is **not schedule-neutral**

Implementation (`train.py` 1222–1223) is exactly `history[-5:]`, per-key mean,
over every key including `avg_dist` and `loss`. As documented.

The problem is what the rule does to configs with different schedules:

| config | epochs | `_last5` | peak epoch EM | peak epoch | `_last5` deficit |
|---|---|---|---|---|---|
| e0_baseline | 30 | 0.8808 | 0.8876 | 26.4 | **−0.0068** |
| e11b_combo_ls | 15 | 0.8845 | 0.8903 | 12.6 | −0.0057 |
| e14_no_cf | 15 | 0.9191 | 0.9260 | 10.4 | −0.0070 |
| e25_swa | 12 | 0.9246 | 0.9269 | 10.2 | −0.0023 |
| **e29_swa_ep15** | 15 | 0.9258 | 0.9277 | 14.0 | **−0.0019** |

The baseline runs 30 epochs and overfits downward after epoch 26, so its last
five epochs sit 0.0068 below its own peak. The winner runs 15 epochs with SWA,
whose curve is flat at the end, so it loses only 0.0019. **Peak-to-peak the
headline delta is +0.0401, not +0.0450** — roughly 0.005 of the headline is the
scoring rule being harsher on the baseline's tail than on the winner's.

Peak-picking is itself selection-on-test and would be worse, so `_last5` is the
right *kind* of rule. But the honest statement is "+0.040 to +0.045 depending on
the summarization rule," and the "peak EM / peak epoch" columns in the NOTES
tables are max-over-30-epochs on the test set and are optimistic by
construction.

### (d) Train / held-out split and leakage

**How the split is determined.** `load_es_data` → `split_list(data, 0.7)`: a
**positional** 70/30 cut of each source's pickle, which is in document reading
order. Caps (`--source-limits`) are applied *after* the split, so data-scaling
arms are scored on the same held-out set — verified. Deterministic given the
pickles, and `twin_credit_eval.EXPECTED_VAL` hard-asserts the six held-out
sizes, which is good practice.

**Weakness:** the split key is *position*, not a document or article id. A
pickle rebuild that adds or drops one entity shifts the boundary and silently
re-partitions everything. The `EXPECTED_VAL` guard catches the size change but
only in the two standalone tools, not in `train.py`.

**Leakage measured** by hashing `doc_tensor` (identical hash ⇒ identical
document) and intersecting train with held-out:

| source | shared docs | held-out entities in a shared doc | % of held-out |
|---|---|---|---|
| Prodigy | 0 | 0 | 0.0% |
| TR | 1 | 5 | 1.8% |
| LGL | 1 | 7 | 0.7% |
| GWN | 1 | 2 | 0.4% |
| WikiDocs | 2 | 30 | 0.5% |
| **Synth** | **39** | **42** | **14.0%** |

TR/LGL/GWN/WikiDocs each show exactly the one or two documents that straddle the
70/30 boundary — an artefact of splitting by entity rather than document, but
≤0.5–1.8% of any held-out set and not material.

**Synth is different and is a real finding.** 39 documents appear on both sides,
covering 14% of its held-out entities. These are duplicate texts inside the
synthetic corpus, not a boundary effect. Synth carries a full 1/6 of the
headline macro weight, PLAN.md already warns that "syn_cities is a geometry
cheat sheet," and e29 scores 0.9781 on it. It should be dropped from the
headline or explicitly down-weighted.

---

## 4. Checkpoint reproduction

### e29_swa_ep15/seed42.pt — **PASS, exactly**

Re-implemented the load/split/eval path independently (no import of
`train.py`), loaded `seed42.pt` through its `.pt.json` sidecar, and re-ran
`make_wandb_dict`. **All 42 metric keys reproduce with diff exactly 0.0**,
including `avg_dist` to 6 decimals. This also confirms that the sidecar is
complete enough to reconstruct the model, and that the split is deterministic
across processes.

**One caveat the ledger does not state.** The saved checkpoint is the
epoch-15 SWA snapshot, and its actual EM is **0.9204**. The headline 0.9258 is
the mean over 5 seeds of the mean of each run's last 5 epochs. Seed 42 is the
*worst* of the five checkpoints:

| checkpoint | EM |
|---|---|
| seed42.pt | 0.9204 |
| seed101.pt | 0.9300 |
| seed202.pt | 0.9225 |
| seed617.pt | 0.9270 |
| seed1848.pt | 0.9290 |
| **mean** | **0.9258** |

The mean of the five saved checkpoints does equal the headline (independently
confirmed: `ensemble_eval` K=1 = 92.58%), so the recipe is fine. But if a single
artifact is shipped, it will not be 0.9258 — and shipping seed42 would ship the
one that is half a point light.

### mordecai_2026-08-20.pt — **FAIL: this is not the ship model**

Its sidecar `mordecai_2026-08-20.json` says `"pickle_suffix": "_r2"` and has no
`weight_avg` key at all. Re-evaluated:

| scored against | macro EM | acc@161 |
|---|---|---|
| `_r2` labels (its own key) | 0.9207 | 0.9591 |
| frozen labels (the campaign's key) | **0.9150** | 0.9591 |

Those match `experiments/e24_rstar/seed1848.json` (0.92073 / 0.95905) **exactly**.
This file is the leftover checkpoint from the last run of the **rejected R\*\*
label-rewrite arm** — trained on rewritten labels, no SWA. It is 1.1 points
below the headline on the campaign's own answer key.
`experiments/e24_rstar/NOTES.md:107` already notes that those ten runs clobbered
the file, so this is known — but the file is still sitting at the repo root
where it reads as the deliverable, and re-running `train.py train` without
`--checkpoint-out` on 2026-08-20 would have overwritten it again.

Related: `mordecai_2026-08-19.pt` has **no sidecar at all** and cannot be loaded
correctly without guessing the behaviour flags — exactly the failure mode the
sidecar was introduced to prevent.

Related, and larger: `mordecai3/geoparse.py:323` still loads
`assets/mordecai_2025-08-27.pt` by default. **No campaign checkpoint has been
installed anywhere the library will find it.** Both root `.pt` files and all
`experiments/**/*.pt` are gitignored, so the campaign's model currently exists
only as a documented command line.

### Secondary tools — **PASS**

* `ensemble_eval.py` on the five e29 checkpoints reproduces the e26/e29 tables
  exactly: K=1 92.58%, K=3 93.13%, K=5 93.17%; twin K=1 93.79%, K=5 94.28%.
* `ACCURACY_CAMPAIGN.md`'s "3-seed ensemble ~93.1% strict / 94.3% twin-credit"
  is **correct** — twin at K=3 is 94.30%; that cell simply isn't in the NOTES
  table (which shows only twin K=1 and K=5).
* "the final model reads 93.8%" twin-credit = the K=1 (expected single model)
  figure of 93.79%. Correct.
* `twin_credit_eval.py` on seed42: strict macro 92.04% → twin 93.55%, and its
  internally recomputed strict EM agrees with `error_utils` to **0.000000** on
  every source. The denominator claim in its docstring holds.

Note on the twin-credit definition: `twin_classes` builds connected components
over pairwise A–P edges within a stripped-name group, so a class can *chain*
(A₁–P₁, P₁–A₂ merges A₁ with A₂ even at 0.29° apart). In practice the credit
granted is small and concentrated (LGL: 326 twin-involved entities, only 3
swaps credited), so this is a theoretical rather than practical concern. The
gate is a lat/lon box, not a great-circle radius, so its effective size varies
with latitude.

---

## 5. Other findings

### 5a. The Wikipedia reversal is a domain-match artifact — **HIGH severity, a memory note depends on it**

`e11c_ls_short` and `e11b_combo_ls` differ by exactly one flag (WikiDocs cap
2000 vs uncapped, +15,065 train entities). The isolated effect:

| | Δ EM | 2 SE | t |
|---|---|---|---|
| macro exact match | **+0.0027** | 0.0042 | **1.26 (n.s.)** |
| macro acc@161 | +0.0049 | 0.0031 | 3.18 |
| WikiDocs | **+0.0417** | 0.0053 | 15.7 |
| LGL | +0.0105 | 0.0020 | 10.4 |
| **TR** | **−0.0207** | 0.0067 | **6.2** |
| Prodigy | −0.0110 | 0.0128 | 1.7 |
| GWN | −0.0023 | 0.0063 | 0.7 |
| Synth | −0.0024 | 0.0084 | 0.6 |

More Wikipedia training data buys 4.2 points on the Wikipedia held-out set and
**costs 2.1 points on TR**. It is a wash on macro exact match and negative on
two of the three human-annotated news corpora. It survives only on acc@161,
which is the metric that most rewards WikiDocs' "same admin unit, ~0 km" cases.

`ACCURACY_CAMPAIGN.md` says "Uncapped WikiDocs is part of the winning recipe"
and "WIKI_TRAINING_DATA.md's conclusion is repealed"; the auto-memory note says
"REVERSED: uncapped WikiDocs helps under the fixed loss." **The evidence does
not support a general reversal.** It supports "more Wikipedia improves
Wikipedia." The original memo's conclusion may well still stand for the news
corpora. This should be corrected before campaign 2 plans any further data
scaling.

### 5b. Macro vs pooled weighting changes individual arms' stories — **evaluation debt, already flagged in PLAN item 8**

WikiDocs is **72%** of held-out entities but **1/6** of the headline weight;
Prodigy is 5.6% of entities and also 1/6.

| config | macro EM | pooled EM |
|---|---|---|
| e0_baseline | 0.8808 | 0.8672 |
| e11b_combo_ls | 0.8845 | 0.8989 |
| e14_no_cf | 0.9191 | 0.9191 |
| e29_swa_ep15 | 0.9258 | 0.9252 |
| **headline delta** | **+0.0450** | **+0.0581** |

The macro choice is *conservative* for the headline. But it badly distorts
individual arms: the Wave-1b winner's "+0.0037" macro is **+0.0317** pooled.
For a geoparsing audience the number to lead with is probably TR+LGL+GWN
pooled: **0.8404 → 0.9095 (+0.069)**.

### 5c. The two "deliberate default" bugs check out, but the phrasing is stronger than the data

* **Inert dropout: confirmed.** `model.train()` is called once before the epoch
  loop; `evaluate_results` calls `model.eval()` and nothing restores training
  mode, so dropout is live only during epoch 1. The fix arm `e28_trainmode`
  measures **−0.0008 ± 0.0044 (t = 0.4)** — i.e. *no measurable effect*, not
  "fixing it hurts." Consequence worth noting: `dropout: 0.3` is recorded in
  every config sidecar and in every `mordecai_*.json`, and it is a no-op. A
  future run that trains from that config with the bug fixed gets different
  behaviour than the number attached to it.
* **Cosine LR never applied: confirmed.** `scheduler.step()` is now gated behind
  `--lr-schedule`, default off. Every reported run trained at flat 1e-3.
  `e28_cosine` (+0.0028 ± 0.0027, t = 2.1) and `e29_swa_cosine` (−0.0020 vs
  e29, t = 1.5) are both inside noise, so "applying it makes SWA worse" is also
  overstated — it makes no measurable difference either way.

### 5d. Smaller items

* PLAN.md's Wave-3 note "country head at 0.2 = best acc@161 of campaign
  (0.9655)" is **stale**: e29_swa_ep15 reaches 0.9661.
* `evaluate_results` mutates the held-out candidate dicts (`c['score'] = ...`),
  so the eval is not reentrant and depends on the same dict objects being
  reused. Harmless today; a trap for anyone refactoring.
* The masked-smoothing trap was checked: `e7_ls005/ls01/ls02` show epoch-1
  losses of 1.52 / 1.88 / 2.37 against e4_mlp's 1.12 — sane, so those runs used
  `masked_smoothed_ce`, not the exploding path. No stale numbers there.
* `e0_baseline` uses `--mix-dim 512` and `WikiDocs=2000`, i.e. it is a
  reconstructed baseline rather than the shipped 2025-08-27 model. That is a
  reasonable choice and is disclosed, but the campaign has never measured its
  gain against the model actually in production.

---

## Recommendations, in priority order

1. **Deal with the root checkpoint before anything else.** Delete or rename
   `mordecai_2026-08-20.pt`/`.json` (it is the rejected R\*\* arm, 0.9150 on
   the frozen key) and `mordecai_2026-08-19.pt` (no sidecar). Promote a named
   e29 checkpoint — seed101 at 0.9300, or better the 3-seed probability
   ensemble at 0.9313 — into `mordecai3/assets/` with its sidecar, and point
   `geoparse.py:323` at it. Right now the campaign has shipped nothing.
2. **Carve out a real dev set before campaign 2 runs a single arm.** Cheapest:
   re-split each source 60/15/25 and select on dev, report on test once.
   Better: freeze the current six sources as *dev* and acquire a genuinely
   untouched test set (GeoVirus, WikToR, or a held-back slice of LGL
   documents). Without this, campaign 2 inherits campaign 1's selection debt
   and adds its own.
3. **Rotate the split.** Because `random.seed(617)` freezes the split and the
   data order, every ± in the ledger is optimization noise on one partition.
   Run e0 and the ship recipe over 3 different 70/30 splits; expect the honest
   generalization interval to be several times wider than ±0.0015. Also switch
   the split key from list position to a document id.
4. **Report three numbers, not one.** For e29 seed42: 0.9204 on the
   retrievable-gold denominator, 0.9072 over all held-out mentions, 0.9019
   through the serving path. Propagate `total_missing`/`missing_correct` into
   `make_wandb_dict` so the exclusion rate is in every json. And report the
   *checkpoint's own* measured accuracy alongside the 5-seed `_last5` mean.
5. **Decide the aggregation and freeze it** (macro / pooled / TR+LGL+GWN) before
   more arms are run, and **drop Synth from the headline** — 14% of its
   held-out entities sit in documents that are also in train, on top of PLAN's
   own "geometry cheat sheet" warning.
6. **Make eval and serving agree on the reserved row.** Either have
   `evaluate_results` consult `pred[-1]` the way `geoparse.py` does, or have
   serving ignore it. Until they agree, the abstention signal that campaign-2
   item 5 wants to productionize cannot be measured on the campaign's own
   metric — and the served EM is ~1.9 points below the reported one.
7. **Use t(4), not 2 SE — or run 10 seeds.** Ten of the ladder's stars sit at
   2.0–2.8 SE and do not survive a correct small-sample test. The big
   conclusions are untouched; the tiebreaks (`cf` dropped, 15 epochs not 12,
   SWA not EMA, ls 0.05 not 0.1) are not evidence-backed and should be
   presented as arbitrary-but-harmless choices rather than findings.
8. **Correct the Wikipedia claim** in `ACCURACY_CAMPAIGN.md`, `PLAN.md` and the
   auto-memory note: uncapped WikiDocs is +0.042 on WikiDocs, −0.021 on TR, and
   +0.0027 ± 0.0042 (n.s.) on the macro. It is a domain-match effect, not a
   repeal of WIKI_TRAINING_DATA.md.
9. **Correct two smaller sentences:** "seed spread cut roughly in half" (true
   vs e14_no_cf and for acc@161; not true for EM vs e0, and Prodigy's tripled),
   and "fixing [the dropout bug] hurts" / "applying [cosine] makes SWA worse"
   (both are t < 2.1 — they do nothing measurable).

---

## Appendix: what was actually run

* `repro.py` — independent reimplementation of `load_es_data`'s eval path
  (no `train.py` import), used to re-evaluate `e29_swa_ep15/seed42.pt` and
  `mordecai_2026-08-20.pt`.
* `leak.py` — `doc_tensor` MD5 hashing, train ∩ held-out per source.
* `denom.py` — eval denominator, reserved-row/NULL-row argmax rates, and the
  reported-vs-deployment-vs-serving EM ladder.
* Recomputation of all `_last5` means, paired deltas, 2 SE and t statistics
  over the 58 experiment directories.
* Leave-2-seeds-out selection-stability experiment (10 splits) and a 20,000-draw
  order-statistic simulation of max-selection bias.
* `tools/ensemble_eval.py --checkpoints "experiments/e29_swa_ep15/seed*.pt" --k 1 3 5`
  and `tools/twin_credit_eval.py --checkpoint experiments/e29_swa_ep15/seed42.pt`,
  both re-run from scratch.
