#!/usr/bin/env bash
# e57: the R1 abbreviation-normalisation composition grid.
#
# Every cell is run TWICE on the same code, same session, same documents:
# once with `--no-normalize-place-abbrevs` (the pre-e57 query, which must
# reproduce the published rows) and once with the flag on (the default).
#
#   e29_seed42  / serving             -> the Phase-0 reference, 66.99. GATE.
#   e29_seed101 / serving, head_gold  -> 67.66 / 77.69
#   e54_seed42  / head_gold, no outlet-> 78.55
#   e54_seed42  / +LGL,TR outlets     -> 70.39 (serving) / 80.85 (head). THE CELL.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=experiments/e57_r1_retrieval/e2e
mkdir -p "$OUT"
BASE="prom,name,cue,sib,geo,shape"

run () {  # name model feature_blocks outlet_sources variants abbrev_flag
  local tag="$1_$6"
  if [ -f "$OUT/$tag.json" ]; then echo "=== $tag (cached) ==="; return; fi
  echo "=== $tag ==="
  local flag="--normalize-place-abbrevs"
  [ "$6" = "off" ] && flag="--no-normalize-place-abbrevs"
  uv run python tools/end_to_end_eval.py evaluate \
    --model-path "$2" --feature-blocks "$3" --outlet-sources "$4" \
    --variants "$5" $flag --out "$OUT/$tag.json" 2>&1 | grep -E "e2e EM|documents,"
}

for F in off on; do
  run e29_seed42        experiments/e29_swa_ep15/seed42.pt                 "$BASE"        ""       serving             "$F"
  run e29_seed101       mordecai3/assets/mordecai_2026-08-20_seed101.pt    "$BASE"        ""       serving,head_gold   "$F"
  run e54_seed42_noout  mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt "$BASE,outlet" ""       head_gold           "$F"
  run e54_seed42_outlet mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt "$BASE,outlet" "lgl,tr" serving,head_gold   "$F"
done
