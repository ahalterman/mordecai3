# e60_docsplit — the e29 recipe under document-keyed splits

Campaign 2, Phase 0, item S2 (decision D1 follow-through). **Not comparable to
any other ledger entry**: this arm changes the held-out *set*, not the model.
Its numbers answer "what does the recipe score when no document straddles the
split", and nothing else. The current entity-level held-out sets stay frozen as
DEV (D1); the default split is unchanged.

## What "doc-keyed" means here

`tools/train.py --split-mode doc` (`split_by_doc`). The pickles carry no
document id, but every entity of a document shares its `doc_tensor`, so the
document is recovered by hashing that tensor — the same trick
`tools/end_to_end_eval.py::heldout_doc_indices` and the Wave-2b sibling
features use. Each document is then assigned by a hash bucket
(`int(md5[:8],16)/2**32 < 0.7` → train), so the assignment is independent of
order, stable across sources, seeds and reruns, and no document can be on both
sides.

The default (`--split-mode entity`) cuts each source's flat entity list at 70%.
That is a document boundary only because the entities are in document order —
and for Synth, which is shuffled before the split, it is not one at all (14% of
its held-out documents also appear in training,
`experiments/campaign2/data_quality_report.md`).

Held-out sizes, doc-keyed vs the frozen entity split:

| source | entity split | doc split |
|---|---|---|
| Prodigy | 500 | 524 |
| TR | 274 | 267 |
| LGL | 973 | 1,058 |
| GWN | 474 | 389 |
| Synth | 300 | 292 |
| WikiDocs | 6,456 | 6,275 |

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --mix-dim 512 \
    --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --epochs 15 --avg-params --avg-mode swa --seed $S --split-mode doc \
    --checkpoint-out experiments/e60_docsplit/seed$S.pt \
    --metrics-out experiments/e60_docsplit/seed$S.json > experiments/e60_docsplit/seed$S.log 2>&1
done
```

## Result (5 seeds, mean ± sd; **unpaired** — different held-out sets)

| metric | e29 (entity split, DEV) | e60 (doc split) |
|---|---|---|
| macro EM, 6 sources (frozen key) | 0.9258 ± 0.0042 | 0.9222 ± 0.0037 |
| macro EM, 5 sources (no Synth, unified) | 0.9109 ± 0.0068 | 0.9118 ± 0.0037 |
| **TLG-hard (unified)** | **0.8676 ± 0.0068** | **0.8525 ± 0.0086** |
| novel-pair EM | 0.7731 ± 0.0068 | 0.8033 ± 0.0056 |
| seen-pair EM | 0.9604 (seed101) | 0.9597 ± 0.0018 |
| unanswerable rate | 1.55% | 2.20% |
| abstention rate / precision | 1.36% / 56% | 1.60% / 79% |

Per-source EM moves in opposite directions: WikiDocs **up** (0.9277 → 0.9470),
TR **down** (0.8893 → 0.8783), LGL **down** (0.8964 → 0.8866), GWN up
(0.9261 → 0.9293). Twin-credit is not defined for this arm — the twin cache is
keyed to the DEV split, so `tools/train.py` does not load it here. Ignore the
`twin_credit*` fields in these `metrics2.json` files: seeds 42/101/202/617 were
run before the "no cache" case reported `nan` and carry a meaningless `0.0`;
seed 1848 carries `nan`, which is what the field means.

## Reading

1. **The recipe is not an artifact of the split.** Macro-of-six moves −0.0036,
   inside the seed spread of either arm.
2. **Novel-pair EM goes UP by 3 points, which is not the direction leakage
   would predict.** The doc-keyed split has *fewer* novel pairs (1,380 vs
   1,747 of a similar-sized held-out set), so the two novel sets are different
   populations, not a harder and an easier version of the same one. Re-keying
   the split does not fix answer-key memorisation — 81% of held-out pairs occur
   in training either way — because the memorisation is lexical, not
   documentary.
3. **TLG-hard falls 1.5 points**, and the unanswerable rate rises from 1.55% to
   2.20%: the doc-keyed held-out sets of TR and LGL are genuinely harder (more
   of their golds are unreachable). This is the honest reading of "the frozen
   sets are a little kind to us", and it is small.
4. **A doc-keyed split is not the test set.** It removes document overlap,
   which was already measured at −0.0012 EM. The thing that would actually
   move the scoreboard is a corpus with a different label-generating process,
   which is what D1's untouched modern-news TEST set is for.
