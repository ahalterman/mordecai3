"""Byte-identity gate: `span_detector=None` is the pre-change serving path.

The span-head integration adds a branch to `Geoparser.__init__` and one to
`Geoparser._geoparse_docs`. Both are guarded by `span_detector is not None`,
which defaults to None, so the claim is that a caller who does not ask for the
head gets bit-identical output. This checks it rather than asserting it: the
whole geoparse of the 260 held-out documents is dumped canonically and hashed,
and the hash is compared against the same run under a copy of the package with
the span-head hunks removed.

    # 1. determinism baseline, and the post-change hash
    uv run python experiments/e56_span_head_serving/identity_check.py
    # 2. the same, with a pre-change copy of the package first on sys.path
    PYTHONPATH=<pre_change_dir> uv run python .../identity_check.py

`--head NAME` hashes the head path instead, which is how the grid's detection
rows are confirmed to come from `Geoparser` and not only from the harness.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.append(REPO)
os.chdir(REPO)

import mordecai3  # noqa: E402
from mordecai3 import Geoparser  # noqa: E402
from tools.end_to_end_eval import heldout_doc_indices, read_corpus  # noqa: E402

MODEL = "mordecai3/assets/mordecai_2026-08-20_seed101.pt"
KEYS = ("search_name", "start_char", "end_char", "no_match", "p_no_match",
        "geonameid", "name", "country_code3", "feature_code", "lat", "lon",
        "score", "city_id", "city_name")


def main():
    head = None
    if "--head" in sys.argv:
        head = sys.argv[sys.argv.index("--head") + 1]
    print(f"package: {os.path.dirname(mordecai3.__file__)}")
    # Not passed at all when there is no head, so this script runs unchanged
    # against a copy of the package that predates the argument.
    geo = Geoparser(model_path=MODEL,
                    **({"span_detector": head} if head else {}))
    dump = []
    for src in ("tr", "lgl", "gwn"):
        articles = read_corpus(src)
        keep, _ = heldout_doc_indices(src, articles)
        texts = [articles[i]["text"] for i in sorted(keep)]
        for res in geo.geoparse_batch(texts):
            dump.append([[f"{k}={e.get(k)!r}" for k in KEYS]
                         for e in res["geolocated_ents"]])
    blob = json.dumps(dump, sort_keys=True).encode()
    n = sum(len(d) for d in dump)
    print(f"span_detector={head!r}  mentions {n}  "
          f"md5 {hashlib.md5(blob).hexdigest()}")


if __name__ == "__main__":
    main()
