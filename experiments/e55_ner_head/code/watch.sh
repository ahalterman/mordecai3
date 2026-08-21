#!/bin/bash
# Emit one line per newly-finished arm in a results.json, then exit when the
# named process is gone.
RES="$1"
PAT="$2"
seen=""
while true; do
  cur=$(python3 - "$RES" <<'EOF'
import json,sys
try:
    r=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for k,v in sorted(r.items()):
    p=v['scores']['pooled']
    print(f"{k:22s} P {p['P']:.2f} R {p['R']:.2f} F1 {p['F1']:.2f} nest {p['R_nested']:.1f} demFP {p['fp_on_demonym']}")
EOF
)
  comm -13 <(echo "$seen") <(echo "$cur") 2>/dev/null
  seen="$cur"
  if ! pgrep -f "$PAT" > /dev/null; then
    echo "DONE: $PAT no longer running"
    break
  fi
  sleep 20
done
