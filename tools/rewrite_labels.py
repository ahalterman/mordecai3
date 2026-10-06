"""Apply the guarded A/P label rewrite (`R**`) to the enriched pickles.

Background (e12 analysis, experiments/PLAN.md Wave 4 / e12_labels): a large
share of "granularity" errors are not errors at all.  A mention like *Paris* has
two co-located GeoNames entries with the same name -- the populated place
(`feature_class == "P"`) and the administrative unit that contains it
(`feature_class == "A"`) -- and the corpora disagree about which one is the
answer.  Every non-Wikipedia mention of *paris* in this training data is
annotated P-side; all 123 WikiDocs mentions are annotated A-side.  The model
already resolves such pairs to P 96.9% of the time, so the A-side labels are
mostly a tax.

This script moves A-side gold labels onto their P twin, under five guards that
were derived from the cases where an unguarded rule goes wrong.  It reads the
frozen `*_enriched.pkl` files and writes `*_enriched_r2.pkl` next to them, with
only `correct` and `correct_geonamesid` changed.  Both the training and the
held-out portion are rewritten: the point is to change the label convention,
and strict exact match against the original labels can still be computed from
the original pickles (see tools/twin_credit_eval.py).

    uv run python tools/rewrite_labels.py                  # write the pickles
    uv run python tools/rewrite_labels.py --dry-run        # counts only
    uv run python tools/rewrite_labels.py --verify         # re-read and diff
"""
import argparse
import math
import os
import pickle
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

MAX_RESULTS = 500
TRAIN_FRAC = 0.7
NULL_GEONAMEID = "NULL"

# Sources in the order train.py loads them, with the pickle stem(s) behind each.
SOURCES = [
    ("Prodigy", ["prodigy"]),
    ("TR", ["tr"]),
    ("LGL", ["lgl"]),
    ("GWN", ["gwn"]),
    ("Synth", ["syn_cities", "syn_caps"]),
    ("WikiDocs", ["wiki_docs"]),
]

#
#   Twin classes
#
# A twin class is a connected component over an entity's *model-visible*
# candidates: same normalized name, within AP_TWIN_DEGREES in both lat and lon,
# same country, and joined only across the A/P class boundary.  A component
# counts only if it has two or more members and spans both classes, so an
# A-P-P chain with slightly different coordinates lands in a single class.
AP_TWIN_DEGREES = 0.15

# Tokens dropped when building the name key.  An administrative unit is very
# often its settlement's name plus one of these ("Homs Governorate" ~ "Homs",
# "Qatana District" ~ "Qatana"), which plain string equality misses; matching on
# the stripped key roughly doubles twin coverage (e12 report table 1a).
STRIP_TOKENS = {
    "county", "counties", "province", "provincia", "provincie", "district",
    "districts", "governorate", "muhafazat", "state", "states", "region",
    "regione", "regional", "prefecture", "department", "departement",
    "municipality", "municipio", "oblast", "krai", "canton", "parish",
    "division", "territory", "metropolitan", "area", "borough", "township",
    "regency", "kabupaten", "shire", "voivodeship", "raion", "rayon",
    "commune", "arrondissement", "distrito", "departamento", "prefectura",
    "kreis", "landkreis", "amphoe", "changwat", "wilayah", "city", "town",
    "special", "administrative", "capital", "of", "the", "and", "xian",
    "shi", "ken", "si", "gun", "do", "autonomous", "subdistrict", "sub",
    "urban", "rural", "greater", "metro", "island", "islands",
}
_PUNCT = re.compile(r"[^a-z0-9 ]+")

# Same list tools/enrich_pickles.py uses for `mention_admin_cue`.
ADMIN_CUE_TOKENS = ["county", "province", "district", "governorate", "state",
                    "region", "prefecture", "department", "municipality",
                    "oblast", "canton", "parish", "division", "territory",
                    "metropolitan area", "borough", "township", "regency",
                    "shire", "voivodeship", "raion", "commune", "city"]
ADMIN_CUE_RE = re.compile(r"\b(?:{})\b".format("|".join(ADMIN_CUE_TOKENS)), re.I)

# Guard (iii): country and first-order admin units are never "the city".
# Without this, `Maryland` (ADM1) is relabelled to `Maryland City` (pop 8k) and
# `Oklahoma` to `Oklahoma City` -- both A-side in the human corpora.
STATE_CODES = {"PCLI", "PCL", "PCLD", "PCLS", "PCLF", "PCLIX", "TERR",
               "ADM1", "ADM1H"}
# Guard (v): an admin unit more than this many times the population of its P
# twin is a county, not a city.  Without it, LGL's bare county mentions
# (`Kanawha`, `Otoe`, `Braxton`, `Muscogee`) move to namesake pop-0 hamlets.
POP_RATIO = 3.0
LOG_POP_RATIO = math.log10(POP_RATIO)


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def strip_key(name):
    s = _PUNCT.sub(" ", deaccent(str(name)).lower().replace("-", " "))
    toks = [t for t in s.split() if t and t not in STRIP_TOKENS]
    if not toks:            # a name that is nothing but stopwords stays intact
        toks = s.split()
    return " ".join(toks)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def visible_candidates(entity):
    """(index, candidate) for the rows the model can actually score.

    Rows past `max_choices` are never seen, and when the candidate list is
    longer than `max_choices` the last in-window row is overwritten by the
    reserved "not present" slot, so it cannot be predicted either.
    """
    choices = entity["es_choices"]
    n = len(choices)
    n_eff = min(n, MAX_RESULTS)
    out = []
    for i in range(n_eff):
        if i == MAX_RESULTS - 1 and n > MAX_RESULTS:
            continue
        gid = choices[i].get("geonameid")
        if gid is None or str(gid).strip().upper() == NULL_GEONAMEID:
            continue
        out.append((i, choices[i]))
    return out


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def twin_classes(cands, radius=AP_TWIN_DEGREES, same_country=True):
    """index -> tuple(indices) for every candidate sitting in an A/P twin class.

    `cands` is the output of visible_candidates.
    """
    groups = defaultdict(list)
    info = {}
    for idx, c in cands:
        lat, lon = fnum(c.get("lat")), fnum(c.get("lon"))
        if lat is None or lon is None:
            continue
        rec = (idx, lat, lon, str(c.get("feature_class") or ""),
               str(c.get("country_code3") or ""))
        info[idx] = rec
        groups[strip_key(c.get("name") or "")].append(rec)
    uf = _UF([idx for idx, _ in cands])
    touched = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            ai, alat, alon, acls, acc = group[i]
            for j in range(i + 1, len(group)):
                bi, blat, blon, bcls, bcc = group[j]
                if {acls, bcls} != {"A", "P"}:
                    continue
                if abs(alat - blat) >= radius or abs(alon - blon) >= radius:
                    continue
                if same_country and acc != bcc:
                    continue
                uf.union(ai, bi)
                touched.add(ai)
                touched.add(bi)
    comps = defaultdict(list)
    for idx in touched:
        comps[uf.find(idx)].append(idx)
    out = {}
    for members in comps.values():
        if len(members) < 2:
            continue
        classes = {info[i][3] for i in members}
        if "A" not in classes or "P" not in classes:
            continue
        t = tuple(sorted(members))
        for i in members:
            out[i] = t
    return out


def _pop(c):
    return float(c.get("log_population") or 0.0)


def _gid_order(c):
    try:
        return -int(str(c.get("geonameid")))
    except (TypeError, ValueError):
        return 0


def gold_index(entity):
    return next((i for i, v in enumerate(entity.get("correct") or []) if v), None)


def rstar2_target(entity):
    """The index R** would move this entity's gold label to, or None.

    Guards, all of which must hold:
      (i)   the gold is an administrative unit (`feature_class == "A"`);
      (ii)  the mention string carries no admin cue word -- the
            "Aleppo Governorate" / "Jefferson County" exemption;
      (iii) the gold is not country- or first-order-admin level;
      (iv)  its twin class contains a populated place;
      (v)   the gold's population is not more than 3x the P twin's.
    """
    gi = gold_index(entity)
    if gi is None:
        return None
    cands = visible_candidates(entity)
    by_idx = dict(cands)
    gold = by_idx.get(gi)
    if gold is None:                                            # gold unreachable
        return None
    if str(gold.get("feature_class") or "") != "A":             # (i)
        return None
    if ADMIN_CUE_RE.search(str(entity.get("search_name") or "")):   # (ii)
        return None
    if str(gold.get("feature_code") or "") in STATE_CODES:      # (iii)
        return None
    klass = twin_classes(cands).get(gi)
    if not klass:
        return None
    p_side = [i for i in klass if str(by_idx[i].get("feature_class")) == "P"]
    if not p_side:                                              # (iv)
        return None
    target = max(p_side, key=lambda i: (_pop(by_idx[i]), _gid_order(by_idx[i])))
    if _pop(gold) > _pop(by_idx[target]) + LOG_POP_RATIO:       # (v)
        return None
    if str(by_idx[target].get("geonameid")) == str(gold.get("geonameid")):
        return None
    return target


def rewrite_entity(entity):
    """Apply R** in place.  Returns (old_gid, new_gid) or None."""
    target = rstar2_target(entity)
    if target is None:
        return None
    correct = entity["correct"]
    gi = gold_index(entity)
    old_gid = str(entity["es_choices"][gi].get("geonameid"))
    new_gid = str(entity["es_choices"][target].get("geonameid"))
    kind = type(correct[gi])
    correct[gi] = kind(0)
    correct[target] = kind(1)
    entity["correct_geonamesid"] = new_gid
    return old_gid, new_gid


#
#   Split bookkeeping: attribute each flip to the train or the held-out half
#
def split_membership(source, loaded):
    """{stem: {entity index: "train"|"val"}} matching tools/train.py.

    train.py drops malformed tensors, then splits positionally at train_frac.
    Synth additionally shuffles each of its two pickles with seed 617 and keeps
    the first 500 of each before splitting the concatenation, so most of those
    two files never enters a run at all.
    """
    if source == "Synth":
        order = []
        for stem in ("syn_cities", "syn_caps"):
            keep = [i for i, e in enumerate(loaded[stem]) if len(e["tensor"]) > 1]
            rng = random.Random(617)
            rng.shuffle(keep)
            order.extend((stem, i) for i in keep[0:500])
        cut = round(TRAIN_FRAC * len(order))
        out = {stem: {} for stem in loaded}
        for n, (stem, i) in enumerate(order):
            out[stem][i] = "train" if n < cut else "val"
        return out
    stem = list(loaded)[0]
    keep = [i for i, e in enumerate(loaded[stem]) if len(e["tensor"]) > 1]
    cut = round(TRAIN_FRAC * len(keep))
    return {stem: {i: ("train" if n < cut else "val") for n, i in enumerate(keep)}}


def pickle_path(data_dir, stem, suffix, max_results=MAX_RESULTS,
                limit_types="all_loc_types", fuzzy=0):
    return (f"{data_dir}/pickled_es/es_formatted_{stem}_{max_results}"
            f"_{limit_types}_fuzzy_{fuzzy}{suffix}.pkl")


def same_except_labels(a, b):
    """True when two entities agree on everything but `correct*`."""
    import numpy as np
    if set(a) != set(b):
        return False
    for k in a:
        if k in ("correct", "correct_geonamesid"):
            continue
        av, bv = a[k], b[k]
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            if not np.array_equal(av, bv):
                return False
        elif av != bv:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="raw_data")
    ap.add_argument("--max-results", type=int, default=MAX_RESULTS)
    ap.add_argument("--limit-types", default="all_loc_types")
    ap.add_argument("--fuzzy", type=int, default=0)
    ap.add_argument("--in-suffix", default="_enriched")
    ap.add_argument("--out-suffix", default="_enriched_r2")
    ap.add_argument("--dry-run", action="store_true",
                    help="count the rewrites, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="re-read what was written and diff it against the input")
    ap.add_argument("--examples", type=int, default=12,
                    help="how many top rewritten mention strings to print per source")
    args = ap.parse_args()

    def path(stem, suffix):
        return pickle_path(args.data_dir, stem, suffix, args.max_results,
                           args.limit_types, args.fuzzy)

    rows = []
    grand = Counter()
    for source, stems in SOURCES:
        loaded = {}
        for stem in stems:
            fn = path(stem, args.in_suffix)
            if not os.path.exists(fn):
                sys.exit(f"missing input pickle: {fn}")
            with open(fn, "rb") as f:
                loaded[stem] = pickle.load(f)
        member = split_membership(source, loaded)

        counts = Counter()
        names = Counter()
        example = {}
        for stem, data in loaded.items():
            where = member.get(stem, {})
            counts["entities"] += len(data)
            for i, entity in enumerate(data):
                res = rewrite_entity(entity)
                if res is None:
                    continue
                bucket = where.get(i, "unused")
                counts[bucket] += 1
                counts["all"] += 1
                key = str(entity.get("search_name") or "").strip().lower()
                if bucket in ("train", "val"):
                    names[key] += 1
                    example.setdefault(key, (entity, res))

        rows.append((source, counts["entities"], counts["train"], counts["val"],
                     counts["unused"], counts["all"]))
        grand.update(counts)

        if args.examples and names:
            print(f"\n{source}: top rewritten mentions "
                  f"({counts['train']} train / {counts['val']} val)")
            for key, n in names.most_common(args.examples):
                entity, (old_gid, new_gid) = example[key]
                byid = {str(c.get("geonameid")): c for c in entity["es_choices"]}
                o, t = byid.get(old_gid, {}), byid.get(new_gid, {})
                print(f"    {key:<28} x{n:<4} "
                      f"{o.get('name')} ({o.get('feature_code')}) -> "
                      f"{t.get('name')} ({t.get('feature_code')})")

        if not args.dry_run:
            for stem, data in loaded.items():
                out = path(stem, args.out_suffix)
                with open(out, "wb") as f:
                    pickle.dump(data, f, protocol=4)
                print(f"  wrote {os.path.basename(out)} "
                      f"({len(data)} entities, {os.path.getsize(out) / 1e9:.2f} GB)")
        del loaded

    print("\n| source | N entities | rewrites (train) | rewrites (val) | "
          "outside split | total |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print(f"| **all** | {grand['entities']} | {grand['train']} | {grand['val']} "
          f"| {grand['unused']} | {grand['all']} |")

    if args.verify and not args.dry_run:
        print("\nVerifying: every entity must agree on every key but `correct*`.")
        bad = 0
        checked = 0
        for source, stems in SOURCES:
            for stem in stems:
                with open(path(stem, args.in_suffix), "rb") as f:
                    orig = pickle.load(f)
                with open(path(stem, args.out_suffix), "rb") as f:
                    new = pickle.load(f)
                if len(orig) != len(new):
                    print(f"  {stem}: LENGTH MISMATCH {len(orig)} vs {len(new)}")
                    bad += 1
                    continue
                changed = 0
                for a, b in zip(orig, new):
                    checked += 1
                    if not same_except_labels(a, b):
                        bad += 1
                    if a["correct_geonamesid"] != b["correct_geonamesid"]:
                        changed += 1
                        n_true = sum(1 for v in b["correct"] if v)
                        if n_true != 1:
                            print(f"  {stem}: rewritten entity has {n_true} correct rows")
                            bad += 1
                            continue
                        pos = next(i for i, v in enumerate(b["correct"]) if v)
                        if str(b["es_choices"][pos]["geonameid"]) != b["correct_geonamesid"]:
                            print(f"  {stem}: correct[] and correct_geonamesid disagree")
                            bad += 1
                    elif a["correct"] != b["correct"]:
                        print(f"  {stem}: correct[] changed but the gold id did not")
                        bad += 1
                print(f"  {stem}: {len(orig)} entities, {changed} labels changed")
                del orig, new
        print(f"Verified {checked} entities; {bad} problems.")
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
