# e30_absw100 — train at the serving window, with out-of-window golds labelled "abstain"

Second campaign, calibration/abstention track (see
`experiments/campaign2/calibration_report.md`). Base recipe is the ship recipe
`e29_swa_ep15`; the only changes are `--window 100 --full-null-row`.

**Verdict: REJECTED as a training recipe. Confirmed win on abstention, confirmed
loss on accuracy, and the two cancel on the metric that combines them
(selective exact match at fixed coverage). Keep `e29_swa_ep15` and take the
abstention gain at decode time instead. The checkpoints are kept: if a
deployment's product need is specifically "tell me this mention is not
geolocatable", e30 is the better detector at a known cost of ~0.4-0.6 EM.**

## Why the arm existed

At serving (`Geoparser`, `max_choices=100`) 2.54% of held-out mentions have no
reachable answer: 1.55% whose gold is not in the candidate list at all, and
0.99% whose gold *is* retrievable but sits past row 100. Training runs at
window 500, where the second group does not exist — its golds are ordinary
in-window labels — so the model has never been trained to abstain on 39% of
the unanswerable traffic it meets in production. Training at the serving window
with `--full-null-row` maps those golds onto the reserved "no correct answer"
row, which is the class `TrainData.create_labels` already trains.

## Commands

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --mix-dim 512 \
    --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --epochs 15 --avg-params --avg-mode swa --seed $S \
    --window 100 --full-null-row \
    --checkpoint-out experiments/e30_absw100/seed$S.pt \
    --metrics-out experiments/e30_absw100/seed$S.json > experiments/e30_absw100/seed$S.log 2>&1
done
```

`--window` and `--abstain-weight` are new in `tools/train.py`. `--window`
separates the pickle filename (`--max-choices 500`, which is how the pickles
were built) from the number of rows the model scores; `--full-null-row` is
required with a smaller window, because otherwise `TrainData.create_labels`
indexes a length-`window` label array with a gold position from the full list.
**No-op verified the way `--pickle-suffix` was**: rerunning the e29 recipe with
the edited `train.py` and default flags reproduces
`experiments/e29_swa_ep15/seed42.json` byte for byte (md5
`1d84e9529c77db0ef5b9c639d2dff96f`).

## Reading the numbers

The arm's own `seedS.json` is evaluated at **window 100** and is therefore NOT
comparable to the ledger, whose `_last5` numbers are all window-500. Every
comparison below re-scores the final SWA checkpoints of both arms with
`tools/calibration_eval.py` at a common window. Paired per-seed deltas over
{42, 101, 202, 617, 1848}; `t(4)` with the auditor's 2.776 critical value, with
the campaign's older 2 SE rule shown alongside.

### Frozen key (window 500) — ledger continuity

| metric | e29 | e30 | Δ | 2 SE | t(4) | verdict |
|---|---|---|---|---|---|---|
| macro EM, 6 sources | 0.9258 | 0.9201 | −0.0057 | 0.0041 | −2.78 | confirmed loss |
| pooled EM, group (a) | 0.9259 | 0.9202 | −0.0056 | 0.0023 | −4.81 | confirmed loss |
| TLG-hard (TR+LGL+GWN, non-country golds, n=1219) | 0.8730 | 0.8690 | −0.0040 | 0.0030 | −2.69 | 2 SE only |
| EM over all held-out mentions | 0.9115 | 0.9060 | −0.0055 | 0.0023 | −4.81 | confirmed loss |

Per source (window 500, EM, paired Δ): WikiDocs −0.0052 (t −5.3), LGL −0.0076
(t −2.3), Prodigy −0.0184 (t −1.9), TR −0.0037, Synth −0.0007, GWN +0.0013.

### Serving key (window 100) — what the arm was built for

| metric | e29 | e30 | Δ | 2 SE | t(4) | verdict |
|---|---|---|---|---|---|---|
| **AUROC `p_reserved`, detects unanswerable** | 0.8582 | **0.9308** | **+0.0727** | 0.0193 | +7.52 | **confirmed win** |
| unanswerable caught by the flag (of 228) | 84.6 | **117.2** | +32.6 | 5.31 | +12.27 | **confirmed win** |
| flag precision | 0.8828 | **0.9021** | +0.0193 | 0.0092 | +4.21 | **confirmed win** |
| flag recall over all wrong answers | 0.1419 | 0.2122 | +0.0704 | 0.0092 | +15.30 | confirmed win |
| flag fire rate | 0.0151 | 0.0230 | +0.0078 | 0.0010 | +16.00 | confirmed |
| AUROC `p_pred_full`, wrong-answer | 0.8972 | 0.8919 | −0.0053 | 0.0102 | −1.05 | ns |
| AUROC combo | 0.9057 | 0.9151 | +0.0094 | 0.0075 | +2.50 | 2 SE only |
| pooled EM, group (a) answerable | 0.9296 | 0.9260 | −0.0035 | 0.0016 | −4.50 | confirmed loss |
| macro EM, 6 sources | 0.9335 | 0.9258 | −0.0077 | 0.0036 | −4.32 | confirmed loss |
| TLG-hard, answerable | 0.8915 | 0.8824 | −0.0092 | 0.0026 | −7.05 | confirmed loss |
| EM over all mentions | 0.9060 | 0.9025 | −0.0035 | 0.0015 | −4.50 | confirmed loss |
| **selective EM @ 90% coverage** | 0.9556 | 0.9535 | −0.0021 | 0.0021 | −1.95 | **ns** |
| **AURC (`p_pred_full`, lower better)** | 0.0197 | 0.0209 | +0.0013 | 0.0027 | +0.94 | **ns** |
| ECE, raw | 0.0230 | 0.0373 | +0.0142 | 0.0038 | +7.40 | confirmed loss |
| ECE, after LOSO temperature | 0.0238 | 0.0248 | +0.0010 | 0.0047 | +0.40 | ns |
| fitted temperature | 0.851 | 0.818 | −0.034 | 0.021 | −3.21 | confirmed |

## Verdict, in words

The mechanism worked exactly as designed: give the reserved row training
signal for "the gold is past the window" and it becomes a much better
unanswerable detector — +0.073 AUROC, catching 117 of the 228 unanswerable
serving mentions instead of 85, at *higher* precision (90.2% vs 88.3%). That
part is unambiguous and large relative to seed noise.

The cost was not in the pilot's single seed: ranking accuracy falls by a small
but statistically confirmed amount on every key (−0.0057 macro EM frozen,
−0.0035 answerable EM at the serving window, −0.0092 TLG-hard). The likely
mechanism is that a 100-row softmax gives each example one fifth as many
negatives, so the discriminative signal per step is weaker; the arm cannot be
rescued by tuning the abstention side, because it is the *ranking* side that
pays.

Since the two effects land on opposite sides of the same product metric, the
honest test is the one that contains both: **selective exact match at fixed
coverage, and AURC. On both, e30 is indistinguishable from e29 (t = −1.95 and
+0.94)**. The arm relocates error rather than removing it. It also worsens raw
calibration (more under-confident, T 0.818 vs 0.851), though not after scaling.

The pre-declared guard was "answerable EM must not fall by more than seed noise
(±0.010)": numerically it passes (−0.0035), but the paired test says the
regression is real and systematic, which is the stricter and more informative
reading — and the reason to reject.

## Kept for reuse

* Checkpoints + sidecars (`seed*.pt`, `seed*.pt.json`, which now record
  `train_window: 100`) — the best unanswerable detector produced so far.
* `tools/train.py --window` — the pickle-name/window decoupling is generally
  useful (it is also the cheapest way to measure any train/serve window
  mismatch) and is a verified no-op at its default.
