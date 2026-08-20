# e24_rstar — the guarded A/P label rewrite (`R**`)

Wave 4 / e12_labels, implementation arm. The e12 analysis (PLAN.md) recommended
rewriting A-side gold labels onto their co-located P twin, under guards, on the
theory that WikiDocs' annotation convention was taxing a model that already
resolved such pairs to the populated place. This arm builds the rewritten data,
trains on it, and separates the two things a naive arm comparison confounds.

**Verdict: do not adopt.** The rewrite is a *convention swap*, not a correction.
The model learns whichever convention it is handed: on the 229 held-out WikiDocs
labels it moves, the baseline is right 74.6% of the time and the rewritten arm
0.9%, and the two arms score **identically** under a granularity-blind metric
(twin-credit learning effect +0.0021 ± 0.0049 macro, -0.0001 ± 0.0015 on
WikiDocs). On the frozen answer key the macro effect is not significant and
WikiDocs regresses -0.0232*. Details in section 5.

## 1. What was built

| file | what it does |
|---|---|
| `tools/rewrite_labels.py` | reads the frozen `*_enriched.pkl`, applies `R**`, writes `*_enriched_r2.pkl` with only `correct` / `correct_geonamesid` changed. `--dry-run` counts, `--verify` re-reads and diffs. |
| `tools/twin_credit_eval.py` | standalone scorer: strict EM (via the untouched `error_utils.evaluate_results`) and twin-credit EM per source, on either label set. Does not modify `error_utils.py`. |
| `tools/train.py` | new `--pickle-suffix` (default `""`, a no-op). Selects a label variant of the enriched pickles; the compact-cache probe and `compact-cache --pickle-suffix` both key on it, so a variant never reads the base cache. Recorded in the metrics sidecar and the checkpoint config. |

### The rule

Relabel a gold answer to the most populous **P** member of its `nstrip_cc` twin
class when all of:

1. the gold is `feature_class == "A"`;
2. the mention string carries no admin cue word — the *Aleppo Governorate* /
   *Jefferson County* exemption;
3. the gold's feature code is not country- or first-order-admin level
   (`PCL*`, `ADM1*`, `TERR`) — without this, *Maryland* (the state) becomes
   *Maryland City* (pop 8k);
4. a P member exists in the class;
5. the A unit's population is not more than 3x the P twin's — without this,
   LGL's bare county mentions (*Kanawha*, *Otoe*, *Braxton*, *Muscogee*) move to
   namesake pop-0 hamlets.

A twin class is a connected component over the *model-visible* candidates: same
de-accented, admin-word-stripped name, within 0.15 degrees in both lat and lon,
same country, joined only across the A/P boundary, kept only if it spans both
classes.

## 2. Flip counts (match the e12 report exactly)

`uv run python tools/rewrite_labels.py --verify`

| source | N entities | rewrites (train) | rewrites (val) | total |
|---|---|---|---|---|
| Prodigy | 1668 | 6 | 7 | 13 |
| TR | 914 | 9 | 0 | 9 |
| LGL | 3245 | 3 | 1 | 4 |
| GWN | 1580 | 6 | 5 | 11 |
| Synth | 5701 | 0 | 0 | 0 |
| WikiDocs | 21521 | **458** | **229** | 687 |
| **all** | 34629 | **482** | **242** | 724 |

The e12 report predicted 458 WikiDocs train and 18 TR+LGL+GWN train
(9 + 3 + 6 = 18 here). Both exact.

Biggest WikiDocs groups: *geneva* x123 (`Genève` ADM3 → `Genève` PPLA),
*paris* x123 (ADM2 → PPLC), *madrid* x43, *barcelona* x42, *cali* x25,
*pamplona* x19, *antwerp* x18, *stuttgart* x16.

### Integrity checks

- `--verify` re-read all 34,629 entities of both pickle sets and compared every
  key: **0 problems**. Only `correct` and `correct_geonamesid` differ, every
  rewritten entity has exactly one `correct` row, and it agrees with
  `correct_geonamesid`.
- The `_enriched_r2_compact` caches were diffed against the base caches:
  `feat_matrix`, `es_choices` and `tensor` identical on every entity; only the
  labels differ. The two training arms therefore see byte-identical inputs.
- `tools/twin_credit_eval.py` reproduces `experiments/e14_no_cf/seed42.json`'s
  per-source `exact_match` and `acc_at_161` to 1e-12, so its twin-credit numbers
  sit on the campaign metric's exact denominator. (Incidentally this also
  confirms `_gaz_from_matrix` and the candidate-dict path produce identical
  features.)

## 3. Training arms

Recipe: e14_no_cf, unchanged.

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train \
    --epochs 15 --mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp \
    --label-smoothing 0.05 --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --seed $S --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --pickle-suffix _r2 \
    --metrics-out experiments/e24_rstar/seed$S.json > experiments/e24_rstar/seed$S.log 2>&1
done
```

`seed<S>.pt` / `seed<S>.pt.json` are each seed's final checkpoint and its model
config (train.py overwrites `mordecai_<date>.pt`, so they are copied out per
run). The e14_no_cf baseline was rerun the same way to obtain its checkpoints;
that rerun is bit-identical to the frozen `experiments/e14_no_cf/seed*.json`
(full `_history` and `_last5`), which is also the regression test that
`--pickle-suffix ""` changes nothing. Its checkpoints now sit beside the frozen
metrics as `experiments/e14_no_cf/seed<S>.pt` (+ `.pt.json`), so the next arm
that needs baseline checkpoints does not have to retrain for them.

> **Heads-up:** train.py's fixed `mordecai_<date>.pt` output name means any run
> clobbers the previous one. These ten runs overwrote the `mordecai_2026-08-20.pt`
> that a Wave-3 `aux_country` run had left in the repo root; it now holds
> e24_rstar seed1848. A `--checkpoint-out` flag would be worth adding.

## 4. Results

### 4.1 What a normal arm comparison shows (and why it is wrong here)

`_last5`, 5 seeds, paired per seed, `*` = |mean| > 2 SE -- the campaign convention.

| metric | e14_no_cf | e24_rstar | paired delta ± 2 SE |
|---|---|---|---|
| EM (macro) | 0.9191 | 0.9259 | +0.0069 ± 0.0048* |
| acc@161 (macro) | 0.9632 | 0.9632 | +0.0001 ± 0.0006 |
| Prodigy EM | 0.9034 | 0.9017 | -0.0018 ± 0.0156 |
| TR EM | 0.9051 | 0.9139 | +0.0089 ± 0.0076* |
| LGL EM | 0.8843 | 0.8905 | +0.0062 ± 0.0071 |
| GWN EM | 0.9243 | 0.9383 | +0.0139 ± 0.0045* |
| Synth EM | 0.9740 | 0.9785 | +0.0044 ± 0.0028* |
| WikiDocs EM | 0.9231 | 0.9328 | +0.0097 ± 0.0017* |

Read this way, e24_rstar looks like a win: +0.0069 EM* with WikiDocs +0.0097*,
GWN +0.0139*, TR +0.0089*. It is not. train.py scores each arm against the labels
it trained on, so the WikiDocs column compares two different answer keys.

### 4.2 Decomposition

Every cell below is the *same* forward pass over the *same* candidate features on
the *same* held-out split; only the answer key changes between the `orig` and `r2`
columns. Final-epoch checkpoints, 5 seeds, paired.

* **metric effect** = e14 checkpoints re-scored on the rewritten key minus the same
  checkpoints on the original key. No model changed: this is the answer key moving.
* **learning effect** = e24 minus e14 on one fixed key. The only number that says
  whether training on rewritten labels produced a better model.

#### 4.2.1 Exact match

| source | e14 / orig key | e14 / r2 key | e24 / orig key | e24 / r2 key | metric effect | learning effect (orig key) | learning effect (r2 key) | naive total |
|---|---|---|---|---|---|---|---|---|
| Prodigy | 0.9136 | 0.9088 | 0.9176 | 0.9100 | -0.0048 ± 0.0085 | +0.0040 ± 0.0277 | +0.0012 ± 0.0187 | -0.0036 ± 0.0230 |
| TR | 0.9018 | 0.9018 | 0.9070 | 0.9070 | +0.0000 ± 0.0000 | +0.0052 ± 0.0127 | +0.0052 ± 0.0127 | +0.0052 ± 0.0127 |
| LGL | 0.8765 | 0.8765 | 0.8829 | 0.8829 | +0.0000 ± 0.0000 | +0.0063 ± 0.0066 | +0.0063 ± 0.0066 | +0.0063 ± 0.0066 |
| GWN | 0.9183 | 0.9239 | 0.9339 | 0.9396 | +0.0057 ± 0.0011* | +0.0157 ± 0.0155* | +0.0157 ± 0.0167 | +0.0213 ± 0.0163* |
| Synth | 0.9672 | 0.9672 | 0.9799 | 0.9799 | +0.0000 ± 0.0000 | +0.0127 ± 0.0119* | +0.0127 ± 0.0119* | +0.0127 ± 0.0119* |
| WikiDocs | 0.9243 | 0.9034 | 0.9011 | 0.9343 | -0.0209 ± 0.0057* | -0.0232 ± 0.0026* | +0.0310 ± 0.0047* | +0.0100 ± 0.0034* |
| **macro** | 0.9170 | 0.9136 | 0.9204 | 0.9256 | -0.0033 ± 0.0017* | +0.0034 ± 0.0083 | +0.0120 ± 0.0072* | +0.0087 ± 0.0077* |
| **human macro** | 0.9026 | 0.9028 | 0.9103 | 0.9099 | +0.0002 ± 0.0019 | +0.0078 ± 0.0111 | +0.0071 ± 0.0087 | +0.0073 ± 0.0102 |
| **human pooled** | 0.8970 | 0.8971 | 0.9046 | 0.9041 | +0.0001 ± 0.0018 | +0.0076 ± 0.0103 | +0.0070 ± 0.0079 | +0.0071 ± 0.0094 |

On the frozen key the macro effect is **+0.0034 ± 0.0083 -- not significant**, and it
is a trade: WikiDocs **-0.0232\***, everything else slightly positive. The human
corpora pool to +0.0076 ± 0.0103, also not significant, which is what the e12
analysis predicted (ceiling +0.5 EM against ~1.0 seed noise).

#### 4.2.2 Twin-credit EM

| source | e14 / orig key | e14 / r2 key | e24 / orig key | e24 / r2 key | metric effect | learning effect (orig key) | learning effect (r2 key) | naive total |
|---|---|---|---|---|---|---|---|---|
| Prodigy | 0.9292 | 0.9292 | 0.9284 | 0.9284 | +0.0000 ± 0.0000 | -0.0008 ± 0.0223 | -0.0008 ± 0.0223 | -0.0008 ± 0.0223 |
| TR | 0.9196 | 0.9196 | 0.9144 | 0.9144 | +0.0000 ± 0.0000 | -0.0052 ± 0.0089 | -0.0052 ± 0.0089 | -0.0052 ± 0.0089 |
| LGL | 0.8808 | 0.8808 | 0.8852 | 0.8852 | +0.0000 ± 0.0000 | +0.0044 ± 0.0054 | +0.0044 ± 0.0054 | +0.0044 ± 0.0054 |
| GWN | 0.9374 | 0.9374 | 0.9435 | 0.9435 | +0.0000 ± 0.0000 | +0.0061 ± 0.0076 | +0.0061 ± 0.0076 | +0.0061 ± 0.0076 |
| Synth | 0.9759 | 0.9759 | 0.9839 | 0.9839 | +0.0000 ± 0.0000 | +0.0080 ± 0.0086 | +0.0080 ± 0.0086 | +0.0080 ± 0.0086 |
| WikiDocs | 0.9390 | 0.9390 | 0.9390 | 0.9390 | +0.0000 ± 0.0000 | -0.0001 ± 0.0015 | -0.0001 ± 0.0015 | -0.0001 ± 0.0015 |
| **macro** | 0.9303 | 0.9303 | 0.9324 | 0.9324 | +0.0000 ± 0.0000 | +0.0021 ± 0.0049 | +0.0021 ± 0.0049 | +0.0021 ± 0.0049 |
| **human macro** | 0.9167 | 0.9167 | 0.9179 | 0.9179 | +0.0000 ± 0.0000 | +0.0011 ± 0.0062 | +0.0011 ± 0.0062 | +0.0011 ± 0.0062 |
| **human pooled** | 0.9087 | 0.9087 | 0.9111 | 0.9111 | +0.0000 ± 0.0000 | +0.0024 ± 0.0064 | +0.0024 ± 0.0064 | +0.0024 ± 0.0064 |

Two things to read off this table. First, **every metric effect is exactly zero**:
twin credit is provably invariant to moving a gold label within its own twin class,
and the arithmetic confirms it to the last digit. Second, the **learning effect
under twin credit is +0.0021 ± 0.0049 -- nothing**, and WikiDocs is
-0.0001 ± 0.0015. Once granularity is forgiven, the two arms are the same model.

#### 4.2.3 acc@161km

| source | e14 / orig key | e14 / r2 key | e24 / orig key | e24 / r2 key | metric effect | learning effect (orig key) | learning effect (r2 key) | naive total |
|---|---|---|---|---|---|---|---|---|
| Prodigy | 0.9852 | 0.9852 | 0.9828 | 0.9828 | +0.0000 ± 0.0000 | -0.0024 ± 0.0048 | -0.0024 ± 0.0048 | -0.0024 ± 0.0048 |
| TR | 0.9402 | 0.9402 | 0.9292 | 0.9292 | +0.0000 ± 0.0000 | -0.0111 ± 0.0062* | -0.0111 ± 0.0062* | -0.0111 ± 0.0062* |
| LGL | 0.9034 | 0.9034 | 0.9017 | 0.9017 | +0.0000 ± 0.0000 | -0.0017 ± 0.0104 | -0.0017 ± 0.0104 | -0.0017 ± 0.0104 |
| GWN | 0.9696 | 0.9696 | 0.9726 | 0.9726 | +0.0000 ± 0.0000 | +0.0030 ± 0.0029* | +0.0030 ± 0.0029* | +0.0030 ± 0.0029* |
| Synth | 0.9920 | 0.9920 | 0.9967 | 0.9967 | +0.0000 ± 0.0000 | +0.0047 ± 0.0027* | +0.0047 ± 0.0027* | +0.0047 ± 0.0027* |
| WikiDocs | 0.9808 | 0.9808 | 0.9808 | 0.9808 | +0.0000 ± 0.0000 | +0.0000 ± 0.0025 | +0.0000 ± 0.0025 | +0.0000 ± 0.0025 |
| **macro** | 0.9619 | 0.9619 | 0.9606 | 0.9606 | +0.0000 ± 0.0000 | -0.0012 ± 0.0014 | -0.0012 ± 0.0014 | -0.0012 ± 0.0014 |
| **human macro** | 0.9496 | 0.9496 | 0.9466 | 0.9466 | +0.0000 ± 0.0000 | -0.0030 ± 0.0022* | -0.0030 ± 0.0022* | -0.0030 ± 0.0022* |
| **human pooled** | 0.9407 | 0.9407 | 0.9387 | 0.9387 | +0.0000 ± 0.0000 | -0.0020 ± 0.0042 | -0.0020 ± 0.0042 | -0.0020 ± 0.0042 |

acc@161km is granularity-blind for free (twin classes span a median 3 km), and it
agrees: macro -0.0012 ± 0.0014.

#### 4.2.4 Accuracy restricted to the labels the rewrite moved

These are the only held-out entities where the two answer keys can disagree, so everything in 4.2 is this table diluted by the untouched majority. `n` is per seed and identical across arms.

| source | N moved (held-out) | e14 / orig key | e14 / r2 key | e24 / orig key | e24 / r2 key |
|---|---|---|---|---|---|
| Prodigy | 7 | 0.600 | 0.257 | 0.714 | 0.171 |
| TR | 0 | -- | -- | -- | -- |
| LGL | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| GWN | 5 | 0.000 | 0.520 | 0.000 | 0.520 |
| Synth | 0 | -- | -- | -- | -- |
| WikiDocs | 229 | 0.746 | 0.164 | 0.009 | 0.933 |

**This is the whole result.** On the 229 WikiDocs held-out labels the rewrite
touched, e14 is right 74.6% of the time on the original key and 16.4% on the
rewritten one; e24 is right **0.9%** on the original key and **93.3%** on the
rewritten one. The model learns whichever convention it is handed, and learns the
rewritten one slightly better because the rewrite made WikiDocs internally more
consistent. Neither arm is more correct about where these places are -- the two
candidates are the same settlement, a median 3 km apart.

#### 4.2.5 Does any source regress?

| source | e24 - e14 on the ORIGINAL key | seeds worse | verdict |
|---|---|---|---|
| Prodigy | +0.0040 ± 0.0277 | 3/5 | flat (within noise) |
| TR | +0.0052 ± 0.0127 | 1/5 | flat (within noise) |
| LGL | +0.0063 ± 0.0066 | 1/5 | flat (within noise) |
| GWN | +0.0157 ± 0.0155* | 0/5 | improvement |
| Synth | +0.0127 ± 0.0119* | 0/5 | improvement |
| WikiDocs | -0.0232 ± 0.0026* | 5/5 | regression |

GWN (+0.0157\*, 5 moved val labels) and Synth (+0.0127\*, **zero** moved labels)
move on the frozen key with no metric component at all, so the 482 changed training
labels did change the model. But the same deltas under twin credit are
+0.0061 ± 0.0076 and +0.0080 ± 0.0086 -- within noise -- and the absolute movement
is 7 and 4 entities. This is a real but tiny effect on which twin gets named, not
on which place gets found.

### 4.3 `tools/twin_credit_eval.py` on the best checkpoint of each arm

Best seed by `_last5.exact_match_avg`: e14_no_cf seed101 (0.9245), e24_rstar
seed617 (0.9291). Both scored on the **original** labels.

| | source | N | strict EM | twin-credit EM | delta | twin swaps |
|---|---|---|---|---|---|---|
| e14 seed101 | Prodigy | 500 | 92.40% | 93.60% | +1.20 | 6 |
|  | TR | 271 | 89.30% | 90.41% | +1.11 | 3 |
|  | LGL | 946 | 89.53% | 89.64% | +0.11 | 1 |
|  | GWN | 460 | 92.61% | 93.91% | +1.30 | 6 |
|  | Synth | 299 | 94.98% | 96.32% | +1.34 | 4 |
|  | WikiDocs | 6362 | 92.58% | 93.89% | +1.30 | 83 |
|  | **macro** |  | **91.90%** | **92.96%** | +1.06 |  |
| e24 seed617 | Prodigy | 500 | 91.40% | 93.00% | +1.60 | 8 |
|  | TR | 271 | 91.88% | 91.88% | +0.00 | 0 |
|  | LGL | 946 | 87.74% | 87.95% | +0.21 | 2 |
|  | GWN | 460 | 94.35% | 95.43% | +1.09 | 5 |
|  | Synth | 299 | 98.66% | 99.00% | +0.33 | 1 |
|  | WikiDocs | 6362 | 90.44% | 94.23% | +3.79 | 241 |
|  | **macro** |  | **92.41%** | **93.58%** | +1.17 |  |

e24's WikiDocs twin swaps go 83 -> 241 against the original key while its
twin-credit EM *rises* (93.89% -> 94.23%): it names the other twin, and finds the
place just as often.

## 5. Recommendation

**Do not make the `_r2` pickles the default training data.**

1. On the frozen answer key the rewrite is **not significant** (macro EM
   +0.0034 ± 0.0083; acc@161 -0.0012 ± 0.0014), and it is a trade that buys small
   gains on five sources by giving up **-0.0232\*** on WikiDocs.
2. Under a granularity-blind metric it does **nothing at all**: twin-credit learning
   effect +0.0021 ± 0.0049 macro, -0.0001 ± 0.0015 on WikiDocs. The two arms find
   the same places and disagree only about which twin to name.
3. Adopting it would silently redefine the headline metric for every future arm --
   the -2.3 WikiDocs shift is a units change, not a result -- and break paired
   comparison with every arm already in `experiments/`.

**Why the e12 premise expired.** That analysis was run on
`tools/mordecai_2026-08-19.pt`, a pre-Wave-2 checkpoint that resolved A/P twins to
the populated place 96.9% of the time; the A-side WikiDocs labels really were a tax
on it. The current e14_no_cf model, with the `cue`/`shape`/`prom` feature blocks,
gets 74.6% of exactly those labels **right** -- it has learned the corpus's
convention. Wave 2 absorbed the A/P granularity problem as a modelling problem, so
there is no longer a label tax to refund. The predicted WikiDocs ceiling (+3.6) did
materialise -- as +0.0310\* on the rewritten key -- but with its sign reversed on
the frozen one.

**What to adopt instead: the metric.** `tools/twin_credit_eval.py` is worth keeping
as a secondary reported number. It is exactly invariant to the convention this arm
changed (every metric effect above is 0.0000), which makes it the right instrument
for granularity questions, and on the current best checkpoint it reads +1.06 macro
over strict EM (TR+LGL+GWN pooled +0.60). It also localises where granularity still
costs: LGL's twin-credit gap is +0.11, i.e. essentially none of LGL's remaining
error is A/P granularity -- consistent with finding 3, that LGL's budget is
wrong-admin1 disambiguation.

**Kept, harmless:** `--pickle-suffix` (default `""`, verified a no-op) and the
`_r2` pickles plus their compact caches on disk, so this can be re-tested cheaply
if a future recipe changes the premise again.
