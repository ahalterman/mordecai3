"""Mordecai's output -> the shape the console consumes.

The design handoff is explicit that Mordecai's internal field names must not
leak into the components (`API_CONTRACT.md`: "write a thin adapter ... do not
let Mordecai's internal field names leak"). This is that adapter, and it is the
only place in the console that knows what `p_no_match` or `es_choices` are.

Three of the contract's fields need more than a rename:

* **`review`** — the contract derives it from `top1 - top2 < review_gate`.
  Mordecai has a second and better signal: `p_no_match`, the model's calibrated
  probability that none of the candidates is right. Those flag different
  failures -- a close margin means "I cannot choose", a high `p_no_match` means
  "I do not think this is any of these" -- so both are computed and the reason
  is reported alongside the flag rather than collapsed into a bare boolean.
* **`rationale`** — generated from the real ranker features by `rationale.py`,
  not authored. Returns None rather than inventing a sentence.
* **`boundary`** — not in the original contract. Resolved administrative units
  get their geoBoundaries polygon (see `boundaries.py`); everything else stays
  a point, and the map draws it as one.

Character offsets pass through untouched. The contract warns that they must
index the exact string that was sent, and Mordecai's `start_char`/`end_char`
already do -- they come off the spaCy doc built from that string.
"""

import time

from rationale import explain

# Feature classes, for the one-line `display` string under each candidate.
FEATURE_CLASS_NAMES = {
    "A": "admin area", "P": "populated place", "H": "water", "L": "region",
    "R": "road", "S": "spot/facility", "T": "terrain", "U": "undersea",
    "V": "vegetation",
}

# Feature codes worth calling out by name in `display`, because they are the
# distinction the analyst is usually choosing between.
FEATURE_CODE_NAMES = {
    "PPLC": "national capital", "PPLA": "admin seat", "PPLA2": "2nd-order seat",
    "PPLA3": "3rd-order seat", "PPLA4": "4th-order seat", "PPL": "town",
    "PCLI": "country", "ADM1": "1st-order division", "ADM2": "2nd-order division",
    "ADM3": "3rd-order division", "ADMD": "admin division",
    "AIRP": "airport", "STM": "stream", "LK": "lake", "MT": "mountain",
    "RGN": "region", "ISL": "island", "PRSH": "parish",
}


def _human_population(pop):
    if not pop:
        return None
    if pop >= 1_000_000:
        return f"pop {pop / 1_000_000:.1f}M"
    if pop >= 1_000:
        return f"pop {pop / 1_000:.1f}K"
    return f"pop {pop}"


def _display(cand):
    """The pre-formatted second line under a candidate.

    `API_CONTRACT.md` asks for `FEATURE_CODE · country · qualifier · pop`, and
    is explicit that the adapter builds it rather than the component -- so the
    component never has to know what a feature code is.
    """
    parts = [cand.get("feature_code") or "?"]
    if cand.get("country_code3") and cand["country_code3"] != "NULL":
        parts.append(cand["country_code3"])
    qualifier = (FEATURE_CODE_NAMES.get(cand.get("feature_code"))
                 or FEATURE_CLASS_NAMES.get(cand.get("feature_class")))
    if qualifier:
        parts.append(qualifier)
    admin = cand.get("admin1_name")
    if admin and admin != "NULL" and admin.lower() not in (
            (cand.get("name") or "").lower(),):
        parts.append(admin)
    pop = _human_population(cand.get("population"))
    if pop:
        parts.append(pop)
    return " · ".join(parts)


def _candidate(cand):
    """One candidate, stripped to what the UI shows."""
    return {
        "geonameid": cand.get("geonameid"),
        "name": cand.get("name"),
        "feature_code": cand.get("feature_code"),
        "feature_class": cand.get("feature_class"),
        "country_code3": cand.get("country_code3"),
        "admin1": cand.get("admin1_name") or None,
        "admin2": cand.get("admin2_name") or None,
        "population": cand.get("population") or 0,
        "lat": cand.get("lat"),
        "lon": cand.get("lon"),
        "confidence": round(float(cand.get("score", 0.0)), 4),
        "display": _display(cand),
    }


def _review_flags(entity, candidates, review_gate):
    """Why (and whether) this entity wants a human.

    Returns `(review: bool, reasons: list[str], margin: float or None)`.
    Two independent signals, deliberately not collapsed:

    `margin` is the contract's own test -- the gap between the top two
    candidates. Small means the model could not choose.

    `p_no_match` is Mordecai's calibrated probability that no candidate is
    right, which is a different question and the one the ranker was actually
    calibrated on (calibration_report.md: AUROC 0.899 as a wrong-answer score).
    A mention can have a huge margin and still be flagged, because the model is
    confident about which candidate it would pick *and* confident that picking
    one is a mistake.
    """
    reasons = []
    margin = None
    if len(candidates) >= 2:
        margin = (float(candidates[0].get("score", 0.0))
                  - float(candidates[1].get("score", 0.0)))
        if margin < review_gate:
            reasons.append(f"top-2 margin {margin:.2f} < {review_gate:.2f}")

    p_no_match = float(entity.get("p_no_match", 0.0))
    if entity.get("no_match"):
        reasons.append(f"model declined to place it (p={p_no_match:.2f})")
    elif p_no_match >= 0.5:
        reasons.append(f"p(no correct candidate) {p_no_match:.2f}")

    return bool(reasons), reasons, margin


def to_console(result, *, doc_id, text, review_gate=0.40, top_k=5,
               boundary_store=None, timings=None):
    """A Mordecai `geoparse_doc` result -> the console's response body.

    Parameters
    ----------
    result : dict
        What `geoparse_doc(text, top_k=k, trim=False)` returned. `trim=False`
        matters: the rationale is built from the ranker features that `trim`
        strips.
    doc_id, text : str
        Echoed back. `text` is the exact string the offsets index.
    review_gate : float
    top_k : int
    boundary_store : BoundaryStore or None
        None disables the boundary layer; every place stays a point.
    timings : dict or None
        Stage timings in ms, measured by the caller.

    Returns
    -------
    dict
    """
    entities = []
    ents = result.get("geolocated_ents") or []

    # How many mentions the document actually placed, for the rationale's
    # "shares an admin unit with N other mentions" clause. Counted once here
    # rather than recomputed per entity.
    placed = sum(1 for e in ents if not e.get("no_match"))

    for i, ent in enumerate(ents):
        raw_candidates = ent.get("candidates") or []
        candidates = [_candidate(c) for c in raw_candidates[:top_k]]
        review, reasons, margin = _review_flags(ent, raw_candidates, review_gate)

        resolved = None
        boundary = None
        if not ent.get("no_match"):
            resolved = {
                "geonameid": ent.get("geonameid"),
                "name": ent.get("name"),
                "feature_code": ent.get("feature_code"),
                "feature_class": ent.get("feature_class"),
                "country_code3": ent.get("country_code3"),
                "admin1": ent.get("admin1_name") or None,
                "admin2": ent.get("admin2_name") or None,
                "population": ent.get("population") or 0,
                "lat": ent.get("lat"),
                "lon": ent.get("lon"),
                "confidence": round(float(ent.get("score", 0.0)), 4),
            }
            if boundary_store is not None:
                boundary = boundary_store.lookup(ent)

        entities.append({
            # Stable within the document and derived from position, so a re-run
            # of the same text produces the same keys -- which the contract
            # requires, since this id joins the span, the pin, the table row
            # and the candidate panel.
            "id": f"e{i}",
            "text": ent.get("search_name"),
            "start": ent.get("start_char"),
            "end": ent.get("end_char"),
            "label": ent.get("label") or "LOC",
            # Only the learned span detector produces a real per-span score;
            # spaCy's entity recogniser does not expose one, and inventing a
            # number here is exactly what the contract warns against.
            "ner_score": ent.get("span_score"),
            "resolved": resolved,
            "boundary": boundary,
            "review": review,
            "review_reasons": reasons,
            "margin": round(margin, 4) if margin is not None else None,
            "p_no_match": round(float(ent.get("p_no_match", 0.0)), 4),
            "rationale": explain(raw_candidates, sib_count=max(placed - 1, 0),
                                 placed=not ent.get("no_match")),
            "candidates": candidates,
        })

    return {
        "doc_id": doc_id,
        "text": text,
        "token_count": result.get("token_count"),
        "timing_ms": timings or {},
        "entities": entities,
        "stats": {
            "spans": len(entities),
            "resolved": sum(1 for e in entities if e["resolved"]),
            "flagged": sum(1 for e in entities if e["review"]),
            "with_boundary": sum(1 for e in entities if e["boundary"]),
        },
    }


class Stopwatch:
    """Stage timings for the telemetry strip, measured rather than jittered."""

    def __init__(self):
        self.marks = {}
        self._t0 = time.perf_counter()
        self._last = self._t0

    def mark(self, name):
        now = time.perf_counter()
        self.marks[name] = round((now - self._last) * 1000, 1)
        self._last = now
        return self

    def total(self):
        self.marks["total"] = round((time.perf_counter() - self._t0) * 1000, 1)
        return self.marks
