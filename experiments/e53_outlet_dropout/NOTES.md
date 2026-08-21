# e53_outlet_dropout — teaching the outlet model to live without an outlet

Second campaign, Phase 2. The follow-up e50 said had to happen before the
outlet feature could ship. Base recipe is `e29_swa_ep15` plus the `outlet`
block; the only change is `--outlet-dropout`. Full write-up: the "e53 follow-up"
section of `experiments/campaign2/outlet_feature_report.md`.

**Verdict: ADOPTED at p=0.5. It fixes the blocker at no measurable cost. With
the outlet present, TLG-hard is +0.0281 over e29 (t = 8.01) and statistically
indistinguishable from e50 (−0.0019, t = −0.62). With the outlet withheld on
LGL — the condition that killed e50 at −0.0283 (t = −5.45) — d50 reads −0.0051
(t = −1.41, n.s.), i.e. back inside noise of the baseline. Guardrails all n.s.
p=0.3 is dominated: it matches on the present condition but leaves a residual
withheld regression of −0.0068 (t = −2.96) that still trips the criterion.**

## Why the arm existed

e50's outlet block is worth +0.0300 TLG-hard when the document's outlet is
known. But `tools/outlet_degradation.py` found that serving the e50 checkpoint
on an LGL document with the outlet *withheld* scored 0.8710 against the
baseline's 0.9027 — the model had learned to lean on the newsroom prior for
news-shaped documents and had nothing to fall back on. That is a realistic
serving condition (an unknown masthead, a bare text field), so e50 could not
ship as trained.

The reason the guardrails never saw it: GWN, Prodigy, Synth and WikiDocs carry
null outlets in *every* training example, so the model learned a perfectly good
no-outlet policy for **their** kind of document and never for LGL's. The fix is
to put news documents on both sides of the mask.

## The mechanism

`TrainData.set_outlet_dropout(p, epoch, seed)` overwrites the five outlet
columns with the exact null encoding a no-outlet source gets, for a random
subset of documents, redrawn every epoch. Three properties that mattered:

* **Whole documents, not entities.** Dropping one mention of an article and not
  its neighbour would leak the prior back through the document's other entities
  and would not resemble any serving condition. Verified: every entity of a
  document always agrees.
* **Deterministic in (document, epoch, seed).** The draw is
  `default_rng(doc_hash ^ seed*1000003 ^ epoch*65537)`, not the global RNG the
  shuffling DataLoader also draws from, so the campaign's bit-reproducibility
  rule holds for the dropout schedule too. Verified: same (epoch, seed) gives an
  identical mask twice; different epoch and different seed both differ.
* **Only documents that have an outlet** are eligible; the rest are already at
  the null encoding.

**No-op verified**: at the default `--outlet-dropout 0.0` all five e29 baseline
seeds still reproduce `experiments/e29_swa_ep15/seed*.json` md5-identical
(seed42 = `1d84e9529c77db0ef5b9c639d2dff96f`), and e50 arm seed42 reproduces
its TLG-hard 0.8990 / LGL-nc 0.9472 exactly.

## Commands

```
for S in 42 101 202 617 1848; do
  bash tools/run_e50.sh d50 $S     # --outlet-dropout 0.5
  bash tools/run_e50.sh d30 $S     # --outlet-dropout 0.3
done
uv run python tools/e53_aggregate.py            # conditions (a) and (c)
uv run python tools/outlet_conditions_eval.py   # condition (b)
```

## Results — 5 seeds {42, 101, 202, 617, 1848}, paired, t(4) > 2.776

### Condition (a) — outlet PRESENT at eval

| metric | e29 | e50 | **d50** | d30 | d50 vs e29 | d50 vs e50 |
|---|---|---|---|---|---|---|
| TLG-hard | 0.8676 | 0.8976 | **0.8956** | 0.8980 | **+0.0281, t 8.01 \*** | −0.0019, t −0.62 |
| LGL non-country EM | 0.8739 | 0.9440 | **0.9457** | 0.9452 | +0.0719, t 22.20 \* | +0.0018, t 1.20 |
| LGL novel-pair EM | 0.8181 | 0.9292 | **0.9301** | 0.9306 | +0.1120, t 13.55 \* | +0.0009, t 0.29 |
| novel-pair EM, all | 0.7731 | 0.8080 | **0.8010** | 0.8039 | +0.0279, t 10.42 \* | −0.0070, t −1.98 |
| macro EM, 6 sources | 0.9258 | 0.9380 | **0.9364** | 0.9381 | +0.0106, t 4.34 \* | −0.0016, t −1.36 |
| twin-credit macro | 0.9241 | 0.9371 | **0.9357** | 0.9377 | +0.0115, t 4.73 \* | −0.0014, t −0.77 |

**Dropout is free on the present condition.** Every "vs e50" column is
non-significant; the largest is novel-pair-all at −0.0070 (t = −1.98).

### Condition (b) — outlet WITHHELD on LGL (the blocker)

LGL held-out, 946 scored entities, one scorer for every arm and condition.

| arm | present | withheld | withheld − e29 baseline | |
|---|---|---|---|---|
| e29 baseline | 0.8981 | 0.8981 | (reference) | |
| e50 (no dropout) | 0.9584 | 0.8698 | **−0.0283 ± 0.0052, t = −5.45** | **\*** |
| **d50 (p=0.5)** | **0.9605** | **0.8930** | **−0.0051 ± 0.0036, t = −1.41** | **n.s.** |
| d30 (p=0.3) | 0.9607 | 0.8913 | −0.0068 ± 0.0023, t = −2.96 | \* |

Per seed (present / withheld):

| seed | e29 | e50 | d50 | d30 |
|---|---|---|---|---|
| 42 | 0.9027 | 0.9641 / 0.8710 | 0.9662 / 0.9059 | 0.9662 / 0.9006 |
| 101 | 0.8964 | 0.9545 / 0.8805 | 0.9567 / 0.8869 | 0.9545 / 0.8953 |
| 202 | 0.9027 | 0.9567 / 0.8562 | 0.9567 / 0.8911 | 0.9630 / 0.8922 |
| 617 | 0.8890 | 0.9598 / 0.8647 | 0.9630 / 0.8932 | 0.9609 / 0.8816 |
| 1848 | 0.8996 | 0.9567 / 0.8763 | 0.9598 / 0.8879 | 0.9588 / 0.8869 |

### Condition (c) — guardrails, all non-significant

| source | e29 | d50 | Δ | t |
|---|---|---|---|---|
| Prodigy | 0.9092 | 0.8976 | −0.0116 ± 0.0124 | −0.94 |
| TR | 0.9092 | 0.9166 | +0.0074 ± 0.0048 | 1.53 |
| GWN | 0.9317 | 0.9317 | +0.0000 ± 0.0027 | 0.00 |
| Synth | 0.9773 | 0.9826 | +0.0054 ± 0.0036 | 1.49 |
| WikiDocs | 0.9292 | 0.9293 | +0.0001 ± 0.0010 | 0.13 |

(d30's only significant guardrail movement is TR **+**0.0096, t = 2.98 — an
improvement, not a regression.)

## p=0.5 vs p=0.3

They are indistinguishable with the outlet present (TLG-hard 0.8956 vs 0.8980,
LGL-nc 0.9457 vs 0.9452 — all within noise of each other and of e50). They
differ only where the arm exists to differ: **withheld, p=0.5 reads −0.0051
(t = −1.41) and p=0.3 reads −0.0068 (t = −2.96)**. More dropout buys a better
fallback and costs nothing, so p=0.5 is the pick, and the monotonicity suggests
p=0.7 is worth one cheap probe if anyone wants the withheld number flat rather
than merely non-significant.

## ADOPT criterion, checked

> (b) restored to within noise of baseline AND (a) still significant at t(4) > 2.776

* **d50: (b) t = −1.41 (n.s.) ✓ and (a) t = 8.01 ✓ → ADOPT.**
* d30: (a) t = 10.08 ✓ but (b) t = −2.96, still significant ✗ → fails.

## Kept for reuse

* `tools/train.py --outlet-dropout` — verified no-op at its default.
* `mordecai3/torch_model.py: TrainData.set_outlet_dropout` — the pattern
  generalises to any future feature block that is present for some sources and
  absent for others, which is the structural situation the campaign keeps
  running into.
* `tools/outlet_conditions_eval.py` — scores any checkpoint under
  present/withheld on one denominator; this is the harness any optional-feature
  arm should be judged on from now on.
* `experiments/e53_outlet_dropout/conditions_lgl.json` (curated table) and
  `conditions_lgl_v2.json` (researched table).

Checkpoints were written to scratch and deleted; the `seed*.json` /
`seed*.metrics2.json` files and the two `conditions_*.json` are the record.
