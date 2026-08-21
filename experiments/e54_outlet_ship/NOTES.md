# e54_outlet_ship — the outlet block, merged into the mainline and re-run

Second campaign, Phase 2. Not a new idea: e50 built the `outlet` feature block
and e53 discharged its two pre-ship conditions, both in a worktree. This entry
is the **integration** — the block ported onto the Phase-0 mainline, the
mainline pickles re-enriched, and the ship arm re-trained on the *researched*
home table (e50/e53 trained on the curated one). Full write-up:
`experiments/campaign2/outlet_integration_report.md`.

**Verdict: SHIP CANDIDATE seed 42, named and staged. The merge is a proven
no-op with the block off — all five e29 seeds reproduce
`experiments/e29_swa_ep15/seed*.pt` AND `seed*.json` md5-identical on the
re-enriched pickles — and the arm is worth +0.0368 ± 0.0070 TLG-hard (t = 11.80)
over e29, better than e53's +0.0281 because the researched table resolves all
120 domains and gives TR-News 275 point-homes instead of ~144. Serving e2e EM
on the D2 denominator goes 66.99 → 67.95 with no outlet supplied and → 70.40
with outlets, and LGL with the outlet withheld is −0.0047 ± 0.0018 (t = −2.65,
n.s.) — inside the criterion but by less room than e53 had.**

## What changed from e53

| | e50 / e53 | e54 |
|---|---|---|
| tree | worktree, older base | mainline, post-Phase-0 |
| home table at **training** time | curated (94 point-homes, 5 domains unresolved) | **researched** (105 point-homes, 0 unresolved) |
| metric code | worktree `error_utils.py` | mainline (twin-credit denominator differs) |
| serving path | spec only (§8) | built: `geoparse_doc(..., outlet=)` etc. |
| checkpoints | scratch, deleted | kept here, one staged into `assets/` |

The recipe is otherwise e29_swa_ep15 + `--feature-blocks ",outlet"` +
`--outlet-dropout 0.5`, which is exactly e53's adopted d50.

## The identity gate

Run three times — after the code merge, after re-enriching the pickles in
place, and after the last edit — and once more across all five seeds:

* **checkpoints**: all five e29 seeds md5-match `experiments/e29_swa_ep15/seed*.pt`
  (seed42 `22785b60360b7c21edc1ca8b7261167e`).
* **metrics**: all five md5-match `seed*.json` (seed42 `1d84e9529c77db0ef5b9c639d2dff96f`).

`raw_data/pickled_es/*_enriched_compact.pkl` now carry 47 feature columns
instead of 42, and a recipe that does not name `outlet` reads byte-identical
features out of them. The rerun baselines live in `e29_ref/` and are the paired
reference below.

## Commands

```
# 1. attach the researched-table outlet block to the mainline pickles, in place
uv run python tools/enrich_pickles.py --outlet-only \
  --data-dir raw_data/pickled_es --out-dir raw_data/pickled_es \
  --outlet-table researched --outlet-home-cache experiments/e54_outlet_ship/outlet_homes.json

# 2. the ship arm and its paired baseline, five seeds each
for S in 42 101 202 617 1848; do bash tools/run_e54.sh $S d50; done
for S in 42 101 202 617 1848; do bash tools/run_e54.sh $S baseline; done   # -> e29_ref/

# 3. aggregate, withheld-outlet condition, end to end
uv run python tools/e54_aggregate.py --json-out experiments/e54_outlet_ship/aggregate.json
uv run python tools/outlet_conditions_eval.py --source lgl --data-dir raw_data/pickled_es \
  --arm "e29:prom,name,cue,sib,geo,shape:experiments/e29_swa_ep15/seed{seed}.pt" \
  --arm "e54:prom,name,cue,sib,geo,shape,outlet:experiments/e54_outlet_ship/seed{seed}.pt" \
  --baseline e29 --out experiments/e54_outlet_ship/conditions_lgl.json
uv run python tools/end_to_end_eval.py evaluate \
  --model-path experiments/e54_outlet_ship/seed42.pt \
  --feature-blocks "prom,name,cue,sib,geo,shape,outlet" --outlet-sources "lgl,tr" \
  --variants serving --out experiments/e54_outlet_ship/e2e/e54_seed42_lgltr.json
```

## Results — 5 seeds {42, 101, 202, 617, 1848}, paired, t(4) > 2.776

Both sides are `seed*.metrics2.json`, the Phase-0 standard suite, so one metric
implementation under one reserved-row convention scores both arms.

| metric | e29 | e54 | paired Δ | t | |
|---|---|---|---|---|---|
| **TLG-hard (primary)** | 0.8676 | **0.9043** | **+0.0368 ± 0.0070** | **11.80** | **\*** |
| LGL non-country EM | 0.8739 | 0.9452 | +0.0714 ± 0.0081 | 19.79 | \* |
| LGL EM | 0.8937 | 0.9533 | +0.0596 ± 0.0068 | 19.62 | \* |
| LGL novel-pair EM | 0.8181 | 0.9278 | +0.1097 ± 0.0155 | 15.84 | \* |
| LGL twin-credit | 0.8956 | 0.9545 | +0.0590 ± 0.0068 | 19.32 | \* |
| novel-pair EM, all sources (guardrail) | 0.7731 | 0.8040 | +0.0309 ± 0.0072 | 9.56 | \* |
| twin-credit macro (no Synth) | 0.9241 | 0.9405 | +0.0164 ± 0.0066 | 5.54 | \* |
| macro EM, 5 sources (Synth excluded, D4) | 0.9109 | 0.9271 | +0.0162 ± 0.0095 | 3.83 | \* |
| macro EM, 6 sources (continuity) | 0.9214 | 0.9359 | +0.0145 ± 0.0071 | 4.57 | \* |

TLG-hard's three ingredients:

| | e29 | e54 | Δ | t |
|---|---|---|---|---|
| LGL non-country | 0.8739 | 0.9452 | +0.0714 | 19.79 \* |
| TR non-country | 0.8810 | 0.9143 | **+0.0333** | **5.72 \*** |
| GWN non-country | 0.8479 | 0.8535 | +0.0056 | 1.18 |

**TR is the new news.** Under the curated table TR moved +0.0114 (n.s., e50)
and +0.0074 (n.s., e53). The researched table resolves all 35 TR domains, so
TR's coverage goes from 237 point / 337 country / **340 no home at all** to
275 / 639 / **0** — and the gain becomes significant. (LGL likewise: 2,790 /
338 / 117 → 3,107 / 138 / 0.) That was e50 §9's first "also worth doing", and
it landed for free.

### Guardrails — the four sources that cannot have an outlet are flat

| source | e29 | e54 | Δ | t | has outlets? |
|---|---|---|---|---|---|
| Prodigy | 0.9016 | 0.8944 | −0.0072 ± 0.0350 | −0.46 | no |
| GWN | 0.9283 | 0.9304 | +0.0022 ± 0.0049 | 1.00 | no |
| Synth | 0.9739 | 0.9799 | +0.0060 ± 0.0069 | 1.96 | no |
| WikiDocs | 0.9270 | 0.9276 | +0.0006 ± 0.0020 | 0.69 | no |
| TR | 0.9041 | 0.9299 | +0.0258 ± 0.0094 | 6.14 | yes |

The encoder report's tripwire — "if WikiDocs moves at all, the model is reading
the mask" — reads +0.0006 at t = 0.69.

### Per seed

| seed | TLG e29 | TLG e54 | Δ | LGL-nc e29 | LGL-nc e54 | Δ | novel-all | twin | macro-5 |
|---|---|---|---|---|---|---|---|---|---|
| **42** | 0.8597 | **0.9069** | +0.0471 | 0.8769 | 0.9472 | +0.0704 | **0.8042** | **0.9456** | **0.9326** |
| 101 | 0.8716 | 0.9014 | +0.0298 | 0.8744 | 0.9447 | +0.0704 | 0.8014 | 0.9340 | 0.9217 |
| 202 | 0.8641 | 0.8979 | +0.0338 | 0.8756 | 0.9485 | +0.0729 | 0.7956 | 0.9389 | 0.9254 |
| 617 | 0.8653 | 0.9057 | +0.0404 | 0.8656 | 0.9485 | +0.0829 | 0.8168 | 0.9421 | 0.9269 |
| 1848 | 0.8771 | 0.9098 | +0.0328 | 0.8769 | 0.9372 | +0.0603 | 0.8019 | 0.9419 | 0.9290 |

Every seed positive on both, no overlap with zero.

### Outlet withheld (the e53 blocker), LGL held-out, 946 scored entities

| arm | present | withheld | withheld − e29 | |
|---|---|---|---|---|
| e29 baseline | 0.8981 | 0.8981 | (reference) | |
| **e54 (d50, researched)** | **0.9609** | **0.8934** | **−0.0047 ± 0.0018, t = −2.65** | **n.s.** |
| *e53 d50 (curated), for reference* | 0.9605 | 0.8930 | −0.0051 ± 0.0036, t = −1.41 | n.s. |

Same point estimate as e53 to four decimals; the *t* is larger only because the
seed-to-seed spread is half as wide. It clears the criterion, and it is the one
number in this entry with little margin.

**p = 0.7 probe (`p07/`, 5 seeds, same setup): does not fix it, and p = 0.5
stands.** Withheld −0.0057 ± 0.0071 (t = −1.79) against d50's −0.0047, i.e.
marginally *worse* and statistically identical (paired d70 − d50 = −0.0011,
t = −0.53); still 4 of 5 seeds negative. Present-side TLG-hard 0.9026 vs
0.9043, paired −0.0017 (t = −0.61, n.s.) — no significant cost, but every
aggregate cell is worse-or-equal and the only significant movement anywhere is
a Synth regression (−0.0067, t = −3.16). The decision rule needed withheld ≈ 0
*and* no present-side cost; the first clause fails. **The e53 monotonicity
reading (d30 → d50 → 0) was wrong: the residual is a floor of ~5 entities out
of 946 that dropout cannot reach.** Report §11.

### End to end from raw text (D2 denominator, 260 held-out docs, seed 42)

| configuration | e2e EM | acc@161 | det R | det P | oracle-span EM |
|---|---|---|---|---|---|
| e29 seed42 (Phase-0 reference) | 66.99 | 69.19 | 78.22 | 75.50 | 83.45 |
| e54 seed42, **no outlet supplied** | **67.95** | 69.96 | 78.22 | 75.50 | 84.42 |
| e54 seed42, LGL outlets | 70.01 | 71.98 | 78.22 | 75.50 | 87.17 |
| e54 seed42, LGL + TR outlets | **70.40** | 72.36 | 78.22 | 75.50 | 87.41 |

Detection is identical in all four rows — same NER, same spans — so every point
is resolution. Emitted-location precision goes 77.08 → 82.32.

## Ship candidate

**seed 42.** It leads on novel-pair EM (0.8042), twin-credit (0.9456), macro-5
(0.9326), macro-6 (0.9394) and the withheld condition (0.8996), and is second
on TLG-hard by 0.0029 — a quarter of the seed sd. Staged as
`mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt` (+ sidecar, + the 120-row
`assets/outlet_homes.json`), packaged in `pyproject.toml`, **but not made the
default**: that is one line in `mordecai3/geoparse.py` and the owner's call.

## Kept for reuse

* `mordecai3/outlet_features.py` — the block; serving calls the same functions
  the enrichment does, so train/serve parity is by construction and
  `tests/test_outlet_features.py` polices it.
* `mordecai3/geoparse.py` — `outlet=` / `outlets=` on the document entry
  points, `normalize_outlet`, `lookup_outlet_home`, `read_outlet_homes`. The
  argument is inert on a checkpoint without the block, which is tested.
* `tools/enrich_pickles.py --outlet-only --outlet-table researched|curated`.
* `tools/train.py --outlet-dropout` (no-op at its default, verified 5/5) and
  `torch_model.TrainData.set_outlet_dropout`.
* `tools/run_e54.sh`, `tools/e54_aggregate.py` — the paired 5-seed harness for
  any arm scored against a rerun e29 reference.
* `tools/outlet_conditions_eval.py --arm name:blocks:pattern` — now points at
  arbitrary checkpoints, not just the e50 scratch dir.
* `tools/end_to_end_eval.py --feature-blocks --outlet-sources`.
* `experiments/e54_outlet_ship/` — five checkpoints + sidecars, `seed*.json`,
  `seed*.metrics2.json`, `aggregate.json`, `conditions_lgl.json`, `e2e/`,
  `outlet_homes.json`, and `e29_ref/` (the paired baseline).
