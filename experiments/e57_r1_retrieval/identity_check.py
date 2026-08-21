"""e57 gate 1, at the byte level: `normalize_place_abbrevs=False` is inert.

e56 established a hash for the whole geoparse of the 260 held-out documents on
the default path: `f2656f881c0e85632ad9ec8235a0755d`, 2,159 mentions. R1 adds
two lines to `GeonamesService.build_name_search` behind a flag, so with the
flag off that hash has to come back unchanged -- not "the metrics match", the
same bytes. With the flag on it must NOT, or the rule is not firing.

Same script, same key list and same model as
`experiments/e56_span_head_serving/identity_check.py`, so the hashes are
directly comparable to that report's table.

    uv run python experiments/e57_r1_retrieval/identity_check.py
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

from mordecai3 import Geoparser  # noqa: E402
from tools.end_to_end_eval import heldout_doc_indices, read_corpus  # noqa: E402

MODEL = "mordecai3/assets/mordecai_2026-08-20_seed101.pt"
E56_HASH = "f2656f881c0e85632ad9ec8235a0755d"
KEYS = ("search_name", "start_char", "end_char", "no_match", "p_no_match",
        "geonameid", "name", "country_code3", "feature_code", "lat", "lon",
        "score", "city_id", "city_name")


def run(geo):
    dump = []
    for src in ("tr", "lgl", "gwn"):
        articles = read_corpus(src)
        keep, _ = heldout_doc_indices(src, articles)
        texts = [articles[i]["text"] for i in sorted(keep)]
        for res in geo.geoparse_batch(texts):
            dump.append([[f"{k}={e.get(k)!r}" for k in KEYS]
                         for e in res["geolocated_ents"]])
    blob = json.dumps(dump, sort_keys=True).encode()
    return sum(len(d) for d in dump), hashlib.md5(blob).hexdigest()


def main():
    geo = Geoparser(model_path=MODEL, normalize_place_abbrevs=False)
    n_off, h_off = run(geo)
    print(f"normalize_place_abbrevs=False  mentions {n_off}  md5 {h_off}")
    print(f"  e56 published                mentions 2159  md5 {E56_HASH}"
          f"   {'MATCH' if h_off == E56_HASH else 'DIFFERS'}")

    geo.geonames.normalize_place_abbrevs = True
    geo.geonames.clear_cache()
    n_on, h_on = run(geo)
    print(f"normalize_place_abbrevs=True   mentions {n_on}  md5 {h_on}"
          f"   {'INERT (bad)' if h_on == h_off else 'differs, as it must'}")


if __name__ == "__main__":
    main()
