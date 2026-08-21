#!/bin/bash
# e54_outlet_ship: one training run of the ship-candidate arm.
#   usage: run_e54.sh <seed> [arm]
#
# The e29_swa_ep15 recipe plus the `outlet` feature block and per-document
# outlet dropout at p=0.5 (the e53 pick).  Unlike run_e50.sh this reads the
# MAINLINE pickles -- `raw_data/pickled_es` now carries the outlet block, which
# is a no-op for any recipe that does not name it (identity gate, see
# experiments/campaign2/outlet_integration_report.md §2).
set -u
REPO=/home/andy/projects/mordecai3
TWIN="$REPO/experiments/campaign2/twin_gold.json"

SEED=${1:?usage: run_e54.sh <seed> [arm]}
ARM=${2:-d50}
OUT="$REPO/experiments/e54_outlet_ship"
case "$ARM" in
  d50) BLOCKS="prom,name,cue,sib,geo,shape,outlet"; DROP="--outlet-dropout 0.5" ;;
  # The probe e53 s11.4 deferred twice: does more dropout make the withheld
  # condition genuinely flat, and what does it cost with the outlet present?
  d70) BLOCKS="prom,name,cue,sib,geo,shape,outlet"; DROP="--outlet-dropout 0.7"
       OUT="$OUT/p07" ;;
  arm) BLOCKS="prom,name,cue,sib,geo,shape,outlet"; DROP="" ;;
  # The paired reference: the e29 recipe on the SAME (now outlet-carrying)
  # pickles. It is the identity gate as well as the baseline -- its seed*.json
  # must stay md5-identical to experiments/e29_swa_ep15/seed*.json.
  baseline) BLOCKS="prom,name,cue,sib,geo,shape"; DROP=""; OUT="$OUT/e29_ref" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

mkdir -p "$OUT"
cd "$REPO" || exit 1

WANDB_MODE=offline uv run python tools/train.py train \
  --mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp \
  --label-smoothing 0.05 \
  --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
  --enriched --feature-blocks "$BLOCKS" $DROP \
  --epochs 15 --avg-params --avg-mode swa --seed "$SEED" \
  --twin-cache "$TWIN" \
  --checkpoint-out "$OUT/seed${SEED}.pt" \
  --metrics-out "$OUT/seed${SEED}.json" > "$OUT/seed${SEED}.log" 2>&1
echo "$ARM seed$SEED exit=$?"
