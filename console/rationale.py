"""Why the ranker preferred one candidate over the next.

The design handoff asks for a one-or-two-sentence `rationale` under each
disambiguation, and notes that the mock authors it by hand: "if Mordecai cannot
produce this, either omit the block or synthesize it from the ranker's feature
weights -- do not fabricate it in the UI."

It can produce it. `Geoparser.geoparse_doc(trim=False, top_k=k)` keeps the
enrichment features on every candidate, so the top-1 and top-2 of a mention
differ by a readable vector: population, name-match exactness, how many other
mentions in the document share an admin unit, how far the candidate sits from
the nearest already-anchored toponym. Reporting the largest of those
differences is a genuine account of the decision.

Two honesty constraints shape what this module will and will not say.

* **It reports feature differences, not causes.** The ranker is a neural model
  over ~40 features; nothing here is an attribution in the Shapley sense. The
  phrasing is therefore always comparative -- "chosen over X, which ..." -- and
  never claims the feature *caused* the choice. Getting this wrong would make
  the console lie confidently, which is worse than saying nothing.
* **It says nothing when there is nothing to say.** A mention with one
  candidate, or with a runner-up that differs on no feature worth a sentence,
  gets `None` and the UI drops the block.
"""

import math

# Each entry: (feature key, higher-is-better, template). The template is
# rendered with the winning candidate's advantage already established, so it
# only has to name the evidence.
#
# Ordered by how much a reader learns from it, not by model weight: the
# document-level signals go first because "it agrees with the rest of the
# document" is the thing a human cannot see at a glance, while population is
# last because it is already on screen.
FEATURE_PHRASES = [
    ("sib_adm1", True,
     "shares a first-order admin unit with {n_sib} other mention{s} in the document"),
    ("sib_adm2", True,
     "shares a second-order admin unit with other mentions"),
    ("sib_country", True,
     "sits in a country another mention names outright"),
    ("frac_anchors_150km", True,
     "lies within 150 km of {pct} of the document's already-anchored places"),
    ("frac_anchors_50km", True,
     "lies within 50 km of {pct} of the anchored places"),
    ("log_min_km_anchor", False,
     "is {km} from the nearest anchored toponym"),
    ("exact_name_match", True,
     "matches the mention string exactly"),
    ("exact_altname_match", True,
     "matches one of the record's alternate names exactly"),
    ("is_unique_exact_match", True,
     "is the only exact name match in the gazetteer"),
    ("mention_admin_cue", True,
     "is the administrative reading the mention's own wording asks for"),
    ("is_seat_any", True,
     "is an administrative seat"),
    ("is_max_pop", True,
     "is the most populous candidate with this name"),
    ("log_population", True,
     "is far larger by population"),
    ("ap_twin", True,
     "is the reading the AP-style pairing in the text implies"),
    ("min_dist", False,
     "is a closer string match to the mention"),
]

# A feature has to differ by more than this to be worth a clause. Most of these
# are 0/1 indicators, where any difference is total; the continuous ones
# (log_population, log_min_km_anchor) are on log scales where 0.35 is roughly a
# factor of 1.4 and below that the difference is not something to build a
# sentence on.
MIN_DELTA = 0.35

# At most this many clauses. Three reasons is an explanation; six is a dump.
MAX_CLAUSES = 2


def _fmt_km(log_km):
    """`log_min_km_anchor` is log1p(km); put it back into something readable."""
    km = math.expm1(log_km)
    if km < 1:
        return "under a kilometre"
    if km < 10:
        return f"{km:.1f} km"
    return f"{round(km):,} km"


def _clause(key, winner, loser, sib_count):
    """One rendered clause about how `winner` beats `loser` on `key`, or None."""
    if key not in winner or key not in loser:
        return None
    try:
        w, l = float(winner[key]), float(loser[key])
    except (TypeError, ValueError):
        return None
    if math.isnan(w) or math.isnan(l):
        return None

    for feature_key, higher_better, template in FEATURE_PHRASES:
        if feature_key != key:
            continue
        delta = (w - l) if higher_better else (l - w)
        if delta <= MIN_DELTA:
            return None
        return template.format(
            n_sib=sib_count,
            s="" if sib_count == 1 else "s",
            pct=f"{w:.0%}",
            km=_fmt_km(w),
        )
    return None


def explain(candidates, sib_count=0, placed=True):
    """A rationale sentence for a ranked candidate list, or None.

    Parameters
    ----------
    candidates : list of dicts
        The mention's candidates in rank order, as `top_k=` returns them, with
        `trim=False` so the enrichment features are still attached.
    sib_count : int
        How many other mentions the document resolved. Only used to make the
        "shares an admin unit with N other mentions" clause concrete.
    placed : bool
        Whether the model actually placed the mention. False changes the verb
        and adds a closing clause: the ranking among candidates still happened
        and is still worth showing, but nothing was "preferred" if the
        abstention row outscored every candidate, and saying so would describe
        a decision the model did not make.

    Returns
    -------
    str or None
    """
    if not candidates or len(candidates) < 2:
        return None
    winner, runner_up = candidates[0], candidates[1]

    clauses = []
    for key, _, _ in FEATURE_PHRASES:
        clause = _clause(key, winner, runner_up, sib_count)
        if clause and clause not in clauses:
            clauses.append(clause)
        if len(clauses) >= MAX_CLAUSES:
            break

    if not clauses:
        return None

    loser_name = runner_up.get("name") or "the runner-up"
    loser_code = runner_up.get("feature_code") or ""
    loser = f"{loser_name} ({loser_code})" if loser_code else loser_name
    margin = float(winner.get("score", 0)) - float(runner_up.get("score", 0))

    body = clauses[0] if len(clauses) == 1 else f"{clauses[0]}, and {clauses[1]}"
    if placed:
        return f"Preferred over {loser} by {margin:.2f}: this candidate {body}."
    return (f"Ranked above {loser} by {margin:.2f} because it {body} -- but the "
            f"model's \u201cno correct candidate\u201d row outscored them all, so "
            f"nothing was chosen.")
