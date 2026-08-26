# API contract — what the UI needs from Mordecai

> The exact Mordecai 3/4 Python API is **not** pinned here. This document describes the
> **shape the UI consumes**. Write a thin adapter that calls Mordecai and emits this shape;
> do not let Mordecai's internal field names leak into the components.

## 1. Endpoint

```
POST /api/geoparse
Content-Type: application/json

{
  "doc_id": "TSW-4471",
  "text": "BAMAKO — Convoy movement was reported ...",
  "options": {
    "top_k": 5,
    "review_gate": 0.40,
    "country_prior": null,          // optional ISO3 hint
    "as_of": "1977-09-14",          // optional; enables the historical-alias layer
    "language": null                 // null = autodetect
  }
}
```

### Response

```jsonc
{
  "doc_id": "TSW-4471",
  "language": { "code": "fr", "confidence": 0.61 },
  "timing_ms": { "ner": 18.4, "rank": 9.2, "total": 31.7 },
  "token_count": 612,
  "entities": [
    {
      "id": "e0",                    // stable within the doc; used as the DOM/pin key
      "text": "Gao",                 // surface form, verbatim
      "start": 68,                   // character offsets into `text` (UTF-8 codepoints)
      "end": 71,
      "label": "GPE",                // GPE | LOC | FAC — from the NER pass
      "ner_score": 0.98,
      "resolved": {                  // null when nothing clears the floor
        "geonameid": 2456136,
        "name": "Gao",
        "feature_code": "P.PPLA",
        "country_code3": "MLI",
        "admin1": "Gao",
        "population": 86633,
        "lat": 16.2717,
        "lon": -0.0402,
        "confidence": 0.94
      },
      "review": false,               // true when top-1 - top-2 < review_gate
      "rationale": "Co-occurs with Ansongo (98 km) and Bourem (95 km); ...",
      "candidates": [                // ordered, length <= top_k
        {
          "geonameid": 2456136, "name": "Gao", "feature_code": "P.PPLA",
          "country_code3": "MLI", "admin1": "Gao", "population": 86633,
          "lat": 16.2717, "lon": -0.0402, "confidence": 0.94,
          "display": "P.PPLA · Mali · admin seat · pop 86.6K"
        }
      ]
    }
  ]
}
```

## 2. Fields the UI genuinely requires

| Field | Used by | Notes |
|---|---|---|
| `entities[].start/end` | Document pane | Drives the highlight segmentation. **Must be character offsets into the exact string that was sent**, or the highlights will drift. |
| `entities[].id` | Everything | The join key between text span, map pin, span table row, and candidate panel. Must be stable across a re-run of the same doc. |
| `resolved.lat/lon` | Map | WGS84 decimal degrees. |
| `resolved.confidence` | Pin label bar, candidate bars, footer stats | 0–1. |
| `review` | Amber vs. red highlight, "FLAGGED FOR REVIEW" count | If Mordecai does not emit this, derive it: `top1.confidence - top2.confidence < review_gate`. |
| `candidates[].display` | Disambiguate panel second line | Pre-formatted by the adapter, not the component. Format: `FEATURE_CODE · country · qualifier · pop`. |
| `rationale` | Disambiguate panel footer | Free text, one or two sentences. If Mordecai cannot produce this, either omit the block or synthesize it from the ranker's feature weights — **do not fabricate it in the UI**. |

## 3. Things the mock fakes that need real implementations

1. **`rationale`** — in the mock this is authored prose. Real version should come from the ranker
   (feature attributions, nearest co-resolved toponym + distance, governing-verb signal).
2. **Historical alias resolution** (`as_of`) — the 1977 document shows ABYSSINIA → Ethiopia and
   Massawa's 1977-vs-now country. Confirm whether the gazetteer layer supports date-scoped lookup;
   if not, cut the feature or back it with a static alias table.
3. **Off-frame candidate bearings** — the map draws edge arrows for candidates outside the current
   viewport (e.g. Niger the country vs. the Niger river). Requires candidate coordinates, which the
   contract above already provides.
4. **Telemetry strip** (throughput, p50, VRAM, queue depth) — mocked with jitter. Wire to real
   metrics or delete it; jittering fake numbers in a shipped product is worse than no numbers.
5. **Streaming parse** — the reveal animation is a client-side timer over an already-complete
   response. If the backend can stream per-entity results (SSE/WebSocket), the animation becomes
   real progress instead of theatre; that is the better implementation.

## 4. Corpus fixture

`fixtures/corpus.json` contains the three demo documents and their full expected responses in
the shape above. Use it to build the UI before the backend exists, and keep it as a fixture for
component tests.
