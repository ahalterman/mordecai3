# e50_outlet — the newspaper's own location as a candidate feature

Second campaign, Phase 2 ("cheap feature arms"); the arm
`experiments/campaign2/encoder_scoping_report.md` §5b/§6e specified as E44. Base
recipe is the ship recipe `e29_swa_ep15`; the only change is
`--feature-blocks "…,outlet"`. Full write-up:
`experiments/campaign2/outlet_feature_report.md`.

**Verdict: ADOPTED. TLG-hard +0.0300 ± 0.0039 (t = 7.61 against the t(4) = 2.776
criterion) — the largest single-block gain in either campaign. LGL non-country
EM +0.0701 (t = 18.3); LGL novel-pair EM, the number memorisation cannot move,
+0.1111 (t = 15.0). Every source without outlet metadata is flat. A
home-permutation control that keeps both mask channels bit-identical reproduces
none of it, so the gain is locality and not the corpus indicator the block also
unavoidably is. Two pre-ship conditions, neither changing the verdict: rebuild
the home table from an independent newspaper directory, and fix the missing
no-outlet fallback (e51, below).**

## Why the arm existed

Every LGL article carries a `<domain>` and 80.8% of LGL's linked toponyms fall
in their outlet's modal admin1. Local papers write about their own patch, and
that is a prior nothing else in the model can see: the sibling and anchor
geometry needs a *second* place name in the document to triangulate from, and
the errors this targets are exactly the ones where the article offers none —
`parispi.net` writing "Paris" with no other toponym, resolved to France three
times over. The ceiling was measured before the arm was built: 46.5% of LGL
errors have the gold in the outlet's home admin1 with the prediction elsewhere,
against 5.9% that would move the wrong way.

## The feature

Five columns in `mordecai3/outlet_features.py`, registered as block `outlet`
(appended last in `torch_model.FEATURE_BLOCKS`, so no existing column moves):
`has_outlet_home`, `log_km_to_outlet_home`, `outlet_same_adm1`,
`outlet_same_country`, `has_outlet_country`. Two mask channels because a home
can be a newsroom point (local paper) or only a country (Haaretz, CBC); wire
services get no home at all. The distance takes the sibling geometry's
`NO_ANCHOR_KM` sentinel when absent, never 0.0 — 0 km is the *best* value for a
distance and the placeholder row competes in the softmax. The four sources with
no outlet metadata (GWN, Prodigy, Synth, WikiDocs — 81% of held-out entities)
get the full null.

## Leak protocol — the whole point of the arm

The home table (`tools/outlet_home_table.py`, 120 domains) was written from the
**domain strings plus outside knowledge of the mastheads, and nothing else**: no
article text, no `<gaztag>`, no geonameid, no per-domain gold statistic from
either split. Values name places in *words*; `geocode_homes` turns them into
coordinates through the same GeoNames index the ranker retrieves from.

This mattered more than expected. **Only 1 of LGL's 26 held-out domains appears
in training at all**, so training-split statistics — permitted under the brief —
would have covered almost nothing. And a purely mechanical domain-token
geocoder, the other leak-proof option, is *actively wrong in the direction that
matters*: it returns Paris **France** for `parispi.net` (the Paris
Post-Intelligencer of Paris **Tennessee**) and Concord **California** for the
Concord **New Hampshire** Monitor. A local paper called Paris is evidence
against Paris, France, which is precisely what the feature exists to say.

`tools/outlet_leak_audit.py` — five mechanical checks, all pass. The decisive
one: the article↔entity join can use the gold id as a key, so the audit re-runs
it on mention strings alone and diffs the result. **On LGL the two agree on all
3,245 entities**, so a held-out LGL entity's outlet provably does not depend on
its label. (TR: 1 of 914 differs.) Also checked: no numeric ids in the table,
re-geocoding reproduces the cache exactly, held-out features are identical when
recomputed with the training half never loaded, and — by AST, not grep, because
these modules discuss the answer key at length in prose — the feature path reads
no label field.

**Residual risk, not closable by code**: 55.3% of the 94 point-homes needed
masthead knowledge rather than the domain string (the other 44.7% contain the
city name outright). That knowledge came from a language model and cannot be
*proved* free of LGL. Condition 1 below is the fix.

## Commands

```
# geocode the table, attach the block to the FROZEN enriched pickles (the other
# 33 features are copied through untouched -- that is what makes it one variable)
uv run python tools/enrich_pickles.py --outlet-only \
  --data-dir raw_data/pickled_es --out-dir $E50/pickled_es \
  --outlet-home-cache $E50/outlet_homes.json

for ARM in baseline arm; do for S in 42 101 202 617 1848; do
  bash tools/run_e50.sh $ARM $S      # e29 recipe; --data-dir $E50; blocks differ only
done; done

# control: every outlet wears another outlet's home, masks bit-identical
bash tools/build_e50_perm.sh
for S in 42 101 202 617 1848; do bash tools/run_e50.sh perm $S; done

uv run python tools/outlet_leak_audit.py
uv run python tools/outlet_aggregate.py
```

`tools/train.py` is **unmodified** — the block name flows through the existing
`--feature-blocks`, and the rebuilt pickles through the existing `--data-dir`.

**No-op verified at the strongest available standard**: all five baseline seeds
reproduce `experiments/e29_swa_ep15/seed*.json` **md5-identical** (seed42 =
`1d84e9529c77db0ef5b9c639d2dff96f`, the same hash e30's NOTES quotes). The
pickle rebuild changed nothing and appending a block shifted no column.

## Results — 5 seeds {42, 101, 202, 617, 1848}, paired, t(4) > 2.776

| metric | e29 | e50 | Δ | t | verdict |
|---|---|---|---|---|---|
| **TLG-hard (primary)** | 0.8676 | 0.8976 | **+0.0300 ± 0.0039** | **7.61** | **confirmed win** |
| LGL non-country EM | 0.8739 | 0.9440 | +0.0701 ± 0.0038 | 18.26 | confirmed win |
| LGL EM (frozen metric) | 0.8981 | 0.9584 | +0.0603 ± 0.0029 | 20.81 | confirmed win |
| LGL novel-pair EM | 0.8181 | 0.9292 | +0.1111 ± 0.0074 | 14.96 | confirmed win |
| LGL twin-credit | 0.8956 | 0.9541 | +0.0586 ± 0.0030 | 19.51 | confirmed win |
| novel-pair EM, all sources | 0.7731 | 0.8080 | +0.0349 ± 0.0044 | 8.00 | confirmed win |
| twin-credit macro (no Synth) | 0.9241 | 0.9371 | +0.0130 ± 0.0023 | 5.63 | confirmed win |
| macro EM, 6 sources (continuity) | 0.9258 | 0.9380 | +0.0122 ± 0.0024 | 5.00 | confirmed win |
| acc@161 km | 0.9659 | 0.9776 | +0.0117 ± 0.0008 | 14.27 | confirmed win |

TLG-hard's ingredients: LGL non-country +0.0701 (t 18.3\*), TR +0.0114 (t 1.47),
GWN +0.0085 (t 2.25). The primary metric moves because LGL moves; TR is only
46.7% covered and by national outlets whose concentration is far weaker.

### Guardrails — all non-significant

| source | Δ | t | has outlets? |
|---|---|---|---|
| Prodigy | −0.0036 ± 0.0100 | −0.36 | no |
| GWN | +0.0035 ± 0.0019 | 1.84 | no |
| Synth | +0.0027 ± 0.0032 | 0.83 | no |
| WikiDocs | +0.0018 ± 0.0012 | 1.50 | no |
| TR | +0.0089 ± 0.0052 | 1.71 | partly |

The encoder report's tripwire was "if WikiDocs moves at all, the model is
reading the mask". It moves +0.0018 at t = 1.50.

### The control that settles the mask question

Every outlet given **another outlet's home**, permuted within level. Verified on
the pickles: both mask channels and all 33 non-outlet columns bit-identical to
the real arm; only the three evidence columns change.

| contrast | TLG-hard | t | LGL non-country | t |
|---|---|---|---|---|
| permuted vs baseline | −0.0055 | −0.98 | **−0.0173** | **−5.24** |
| real vs permuted | +0.0355 | 9.96 | +0.0874 | 116.0 |

The mask is worth nothing, and a *wrong* locality prior significantly hurts LGL.
100% of the gain is the true article↔newsroom correspondence. This also
disposes of "five extra columns of free capacity".

### Mechanism (seed 42, held-out LGL, 92 baseline errors)

59.8% of baseline errors are "gold in the home admin1, prediction elsewhere";
5.4% are the mirror case. **The arm fixed 53 of those 55 fixable errors —
96.4%.** Net +58: 66 newly fixed, 8 newly broken, 6 of the 8 being the predicted
failure mode. It collects almost exactly the mass the ceiling predicted, at the
predicted cost. Marquee case, as forecast: `parispi.net` "Paris" 2968815 (FR) →
4647963 (TN), three times.

## The condition that blocks shipping, and e51

`tools/outlet_degradation.py`: serve the e50 checkpoint on held-out LGL with the
outlet **withheld** and it scores 0.8710 against the baseline checkpoint's
0.9027 — **−0.0317**. The model learned to lean on the prior for news-shaped
documents and has no fallback when it is missing. This is invisible in the
guardrails because GWN/Prodigy/Synth/WikiDocs had null outlets in *every*
training example, so a proper no-outlet policy was learned for them and never
for LGL.

**e51 = outlet dropout**: null the block for ~30% of LGL/TR documents during
training so the no-outlet policy is trained on news documents too. Cheap, and it
should be run before any ship decision.

## Pre-ship conditions

1. Rebuild the 120-row home table from a public newspaper directory instead of
   curator knowledge, and re-run (retires the §3.4 risk).
2. Run e51 (outlet dropout) and re-check the withheld-outlet number.

## Kept for reuse

* `mordecai3/outlet_features.py` — the block; the serving path calls the same
  functions the enrichment does, so train/serve parity is by construction.
* `tools/outlet_home_table.py` — the curated table and its geocoder; the
  provenance rules are in the module docstring and are the artifact under audit.
* `tools/outlet_align.py` — recovers article membership (and therefore any
  article-level metadata) from the pickles by `doc_tensor` hash plus a
  subsequence match. Generally useful: any future per-document feature needs it.
* `tools/outlet_leak_audit.py` — the audit pattern, reusable for the next
  external-knowledge feature.
* `tools/enrich_pickles.py --outlet-only` — attaches one block to already-enriched
  pickles and writes the compact cache, without re-deriving 33 features from
  Elasticsearch. This is how a one-variable feature contrast should be built.
* `tools/enrich_pickles.py --permute-homes` — the mask-vs-signal control.
* `experiments/e50_outlet/outlet_homes.json` — the 109 geocoded homes these runs
  used.

Checkpoints were written to scratch and deleted; the `seed*.json` /
`seed*.metrics2.json` files are the record.
