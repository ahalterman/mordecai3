"""End-to-end EM for a scaling arm's predicted spans.

Thin wrapper around the pilot's `ner/e2e_tagger.py` (same ranker checkpoint,
same `tools/end_to_end_eval.score_examples` path, same D2 denominator) that
accepts an absolute predictions path.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), "ner")
sys.path.insert(0, PILOT)

import e2e_tagger  # noqa: E402

if __name__ == "__main__":
    src = os.path.abspath(sys.argv[1])
    name = os.path.basename(src)
    dst = os.path.join(PILOT, name)
    if os.path.abspath(dst) != src:
        shutil.copy(src, dst)
    e2e_tagger.main(name)
    out = os.path.join(PILOT, f"e2e_{name.replace('.json', '')}.json")
    if os.path.exists(out):
        shutil.copy(out, os.path.join(HERE, os.path.basename(out)))
        print("wrote", os.path.join(HERE, os.path.basename(out)))
