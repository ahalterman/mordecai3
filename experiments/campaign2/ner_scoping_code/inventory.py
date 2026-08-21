"""Counts for the data inventory: golds, demonyms, nesting, span lengths."""
import json
from collections import Counter

D = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
     "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/data")


def load(src):
    with open(f"{D}/{src}_docs.json") as f:
        return json.load(f)["docs"]


def is_demonym(g):
    """D2: is this gold row a demonym / non-literal modifier?

    Frozen once as data. Union of two independent signals:
      * spaCy tags the whole gold span NORP and no token GPE/LOC
      * GeoWebNews' own annotation type says Non_Literal_Modifier
    """
    labs = set(g["spacy_labels"] or [])
    norp = ("NORP" in labs) and not (labs & {"GPE", "LOC", "EVENT_LOC"})
    gwn = g.get("gtype") == "Non_Literal_Modifier"
    return norp or gwn


if __name__ == "__main__":
    tot = Counter()
    for src in ["tr", "lgl", "gwn"]:
        docs = load(src)
        for split in ["train", "heldout"]:
            ds = [d for d in docs if d["heldout"] == (split == "heldout")]
            c = Counter()
            c["docs"] = len(ds)
            c["tokens"] = sum(d["n_tokens"] for d in ds)
            lens = Counter()
            for d in ds:
                for g in d["golds"]:
                    c["rows"] += 1
                    if not g["geonameid"]:
                        c["unlinked"] += 1
                        continue
                    c["linked"] += 1
                    if g["tok_start"] is None:
                        c["no_token_align"] += 1
                        continue
                    lens[g["tok_end"] - g["tok_start"]] += 1
                    dem = is_demonym(g)
                    if dem:
                        c["demonym"] += 1
                        continue
                    c["D2_gold"] += 1
                    if g["nested_in"]:
                        c["D2_nested"] += 1
                        c["nested_" + g["nested_in"]] += 1
                    labs = set(g["spacy_labels"] or [])
                    if labs & {"GPE", "LOC", "EVENT_LOC"}:
                        c["D2_spacy_geo"] += 1
            print(src, split, dict(c))
            print("   span token lengths:", dict(sorted(lens.items())))
            if split == "heldout":
                for k, v in c.items():
                    tot[k] += v
    print("HELD-OUT POOLED:", dict(tot))
