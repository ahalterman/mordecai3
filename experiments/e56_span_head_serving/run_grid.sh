#!/usr/bin/env bash
# The e56 combined grid: ranker x span detector x outlet, on the 260 held-out
# TR/LGL/GWN documents, D2 denominator.
#
#   ranker         e29 seed42 (the Phase-0 reference row, and the gate)
#                  e29 seed101 (the packaged default)
#                  e54 seed42 (the staged outlet ship candidate)
#   span detector  none (`serving`) | gold head | C_all head
#   outlets        none | LGL+TR supplied (only an e54 checkpoint reads them)
#
# `serving` on e29 seed42 must come back 0.6699 exactly: that is the
# byte-identity gate on the span-head integration.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=experiments/e56_span_head_serving/e2e
mkdir -p "$OUT"
V=serving,head_gold,head_all
BASE="prom,name,cue,sib,geo,shape"

run () {  # name model feature_blocks outlet_sources
  echo "=== $1 ==="
  uv run python tools/end_to_end_eval.py evaluate \
    --model-path "$2" --feature-blocks "$3" --outlet-sources "$4" \
    --variants "$V" --out "$OUT/$1.json" 2>&1 | grep -E "det R\(exact\)|D2 excludes|documents,"
}

run e29_seed42        experiments/e29_swa_ep15/seed42.pt                   "$BASE" ""
run e29_seed101       mordecai3/assets/mordecai_2026-08-20_seed101.pt      "$BASE" ""
run e54_seed42_noout  mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt   "$BASE,outlet" ""
run e54_seed42_outlet mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt   "$BASE,outlet" "lgl,tr"
