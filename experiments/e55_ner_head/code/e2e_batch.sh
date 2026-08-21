#!/bin/bash
# End-to-end EM for several prediction files, one after another.
set -u
H=/tmp/claude-1000/-home-andy-projects-mordecai3/a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner_scale
cd /home/andy/projects/mordecai3
for f in "$@"; do
  echo "### $f"
  uv run python -u "$H/e2e.py" "$H/$f" 2>/dev/null | grep -E "^(tr|lgl|gwn):|^POOLED"
done
