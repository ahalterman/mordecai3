# Geoparse Console

A single-screen analyst console for Mordecai. Text goes in, toponyms light up in
the document, markers land on the map, and the disambiguate panel shows *why* a
given "Gao" or "Ménaka" resolved the way it did.

Built from the design handoff in [`handoff/`](handoff/) (Claude Design), against
real Mordecai — no fixtures, no canned responses. The visual direction is a
costume; the data under it is honest, and this file is mostly about where the
line between those two sits.

```
uv sync --extra console --extra gpu --group train --group dev
./console/fetch_boundaries.sh        # ~1.25 GB, once
python console/build_boundaries.py   # -> 85 MB SQLite, ~55 s
python console/server.py             # http://127.0.0.1:8000
```

Elasticsearch with the GeoNames index must be up (`docker compose up`). Without
the boundary store the console still runs; every place is just a point.

---

## What is real, and what the mock faked

`handoff/API_CONTRACT.md` §3 lists five things the prototype invented. Here is
what happened to each.

| Mock | Status |
|---|---|
| `rationale` — authored prose | **Real.** Built from the ranker's own features. See below. |
| Telemetry — jittered numbers | **Real, and shorter.** Only measured values are sent; unmeasurable cells are absent rather than invented. |
| Historical `as_of` gazetteer | **Cut.** No date-scoped lookup exists. The 1977 document stays as an uppercase-OCR stress case and says so. |
| Off-frame candidate bearings | **Real.** Drawn from actual runner-up coordinates. |
| Streaming parse | **Not streamed, and not claimed to be.** See "the reveal" below. |

Two things the mock did not have, added here: **boundary polygons** for resolved
administrative units, and a **second review signal** the model already produces.

### `rationale`

`Geoparser.geoparse_doc(trim=False, top_k=k)` keeps ~40 enrichment features on
every candidate, so the top-1 and top-2 of a mention differ by a readable
vector. `rationale.py` reports the largest of those differences:

> Preferred over Bourem (PPLA2) by 0.52: this candidate is the most populous
> candidate with this name, and is far larger by population.

> Preferred over Labézanga (PPL) by 0.81: this candidate shares a first-order
> admin unit with 12 other mentions in the document.

Two constraints are wired into that module. It reports **feature differences,
not causes** — nothing here is a Shapley attribution, so the phrasing is always
comparative and never says a feature *caused* the choice. And it returns `None`
when there is nothing to say, rather than filling the block.

A mention the model **declined** to place is never described as "preferred
over" anything; it gets a different sentence saying the abstention row
outscored every candidate.

### Review flags

The contract derives `review` from `top1 − top2 < review_gate`. Mordecai also
emits `p_no_match` — the calibrated probability that *no* candidate is right,
which is the score the temperature was actually fit for (AUROC 0.899 as a
wrong-answer detector). Those catch different failures:

- small margin → *"I cannot choose between these"*
- high `p_no_match` → *"I do not think it is any of these"*

`RN17` in the wire document is the second kind: the model has a clear favourite
and is confident that picking it would be wrong. Both are computed and the
console names which one fired, instead of collapsing them into a boolean.

---

## Boundaries

The bonus feature: return the *shape* of an administrative unit, not a point.

Source is [geoBoundaries](https://www.geoboundaries.org/) CGAZ (gbOpen,
CC-BY 4.0) — global composites at ADM0/ADM1/ADM2, clipped to a consistent
border set. `build_boundaries.py` streams the three GeoJSON files (1.25 GB)
into SQLite, simplifying per level: **52,791 shapes in 85 MB**, built in about
55 seconds.

### The join is the hard part

CGAZ carries only `shapeGroup` (ISO3) and `shapeName`. There is no GeoNames id
in it and no admin code, so a name match is the obvious approach — and it loses
to transliteration (*Tillabéri* / *Tillaberi*), to renaming, and to the
"Province of X" / "X Region" / "X" family.

So the primary join is **geometric**: which polygon contains the coordinate the
ranker already committed to. No string heuristics, and cheap — a bbox prefilter
in SQL rejects essentially every wrong polygon before any geometry is parsed.

But point-in-polygon always answers, and that is the danger. Mali split
**Ménaka** out of Gao as a new region in 2016; GeoNames records Ménaka as an
ADM1, CGAZ still carries the older nine-region layout, and the polygon
containing Ménaka's coordinate is **Gao**. A confident, plausible, wrong
province.

So a hit is **confirmed by name** when the record names its own unit. That
rejects Ménaka/Gao and passes Cercle de Bourem ↔ Bourem, Tillabéri ↔ Tillaberi,
Idlib ↔ Idleb. When the two sources disagree, the answer is a point — never the
wrong shape.

### Calibration

`validate_boundaries.py` runs the join against every ADM1 and ADM2 record in the
GeoNames index. At ADM1 the similarity distribution is sharply bimodal:

```
      0.00-0.50     228 #####          <- different unit
      0.50-0.60     233 #####
      0.60-0.70      66 #
      0.70-0.75      26 #              <- the valley: 66 of 3,455 pairs
      0.75-0.80      19
      0.80-0.84      21
      0.84-0.88      48 #
      0.88-0.92      83 ##
      0.92-0.96      81 ##
      0.96-1.00    2650 ##############  <- same unit, spelled differently
```

The threshold sits in the valley, which is the point: sweeping it from 0.70 to
0.90 moves ADM1 coverage by 3.7 points. **A parameter you cannot get wrong is
worth more than a well-tuned one.**

ADM2 does not separate cleanly, and the report says so rather than tuning it
away — there, the two gazetteers genuinely disagree about what the units *are*.
`San Juan Talpa` vs `San Luis Talpa` is a correct rejection (different
municipalities); `Guigang Shi` vs `Guipingshi` is a wrong acceptance at 0.861
(different cities in Guangxi). No threshold fixes that.

| Level | GeoNames records | Contained by a CGAZ polygon | Confirmed → boundary |
|---|---|---|---|
| ADM1 | 3,885 | 88.9% | **73.4%** |
| ADM2 | 45,515 | 96.5% | **76.3%** |

ADM0 joins exactly on ISO3 — no geometry test, no name test, 100%.

Making the name comparison multilingual (`rayon`, `savivaldybė`, `wilayat`,
`kabupaten` …) was worth **+11.1 points at ADM2** and +0.6 at ADM1. Deliberately
*not* in that list: `city`, `urban`, `rural`, `metropolitan`, and the compass
points — they look like noise words and are the entire difference between
`Osh City` and `Osh Region`.

The console shows the match method and score in the RESOLVE and DISAMBIGUATE
panels, and draws a weak match **dashed**. GeoJSON export uses the polygon
wherever there is one.

Full report: [`BOUNDARY_JOIN_REPORT.txt`](BOUNDARY_JOIN_REPORT.txt).

---

## The reveal animation

Entities light up one at a time in document order, and the handoff suggests
driving that from a real per-entity stream if the backend can.

It cannot, honestly. `geoparse_batch` pools every mention in a document into a
**single** model forward pass — there is no partial result to stream, because
the whole document resolves at once. So this is a replay of a complete answer:
presentation, not progress. Set `pipeline.spanRevealMs: 0` to skip it;
`prefers-reduced-motion` skips it automatically.

---

## Layout of the code

```
console/
  server.py              FastAPI: /api/geoparse, /api/batch, /api/config, /api/telemetry
  adapter.py             Mordecai result -> the console's response shape
  rationale.py           top-1 vs top-2 feature deltas -> a sentence
  boundaries.py          GeoNames -> geoBoundaries join, against the SQLite store
  build_boundaries.py    CGAZ GeoJSON -> SQLite
  validate_boundaries.py threshold calibration + coverage report
  fetch_boundaries.sh    download the CGAZ composites
  test_console_ui.py     Playwright end-to-end against a running server
  console.config.json    the configuration surface (schema alongside)
  corpus.json            the three demo documents
  static/                index.html, console.css, console.js, geoscope.js, vendor/
  handoff/               the original design package, unmodified
```

Nothing is bundled or transpiled. `console.js` is one ES module and one state
object; `geoscope.js` is a custom element. d3, topojson and the Natural Earth
basemap are **vendored locally** — a demo that needs the internet is a demo that
fails in the one room without it.

### One inference thread

Every model call, construction included, runs on a single
`ThreadPoolExecutor(max_workers=1)`.

This is not a preference. spaCy on GPU installs CuPy as thinc's array backend,
and CuPy's current-device state is **thread-local** — a forward pass dispatched
to FastAPI's anonymous threadpool builds its index tensors on the CPU and dies
with *"Expected all tensors to be on the same device"*, while the identical code
works perfectly in a script. Pinning the thread fixes it, and it is the right
shape anyway: one GPU cannot run two forward passes faster than one, so
serialising costs nothing and makes `QUEUE` on the status bar a real number.

A warmup parse runs at startup. The first CUDA forward pass costs ~970 ms
against ~90 ms steady state, and that is exactly the parse an audience watches.

---

## Library changes this needed

Additive and backward compatible; see the commits on this branch.

- `geoparse_doc(..., top_k=k)` / `geoparse_batch(..., top_k=k)` attach each
  mention's top-k scored candidates under `candidates`, **one row per mention**
  — unlike `debug=True`, which changes the row count. Mentions the model
  declined carry the list too, so an abstention can be shown with its
  runners-up.
- Every mention carries the detector's own `label`: spaCy's `GPE`/`LOC`/`FAC`,
  `NESTED` from the gazetteer pass, or `SPAN` plus a `span_score` from the
  learned head. Only the learned head has a real per-span probability, so
  `ner_score` is null on the spaCy path rather than fabricated.
- `population` stays on candidate dicts. It is a scalar, it cannot reach the
  model (`ALL_FEATURE_KEYS` is a fixed list), and it is the single most useful
  number for a human choosing between two same-named candidates.
- `Geoparser.model_path` records the checkpoint that was loaded, so the title
  bar can name it instead of guessing.
- `SpanTagger.scored_spans()` keeps the head's probability. `spans()` is
  unchanged — it is the documented interface with tests on it.

---

## Configuration

`console.config.json` (JSON Schema alongside) drives palette, layout, grit,
navigation limits, top-k, review gate, reveal timings, export formats and the
boundary layer. Adding a palette needs no code change: the key becomes a valid
`theme.palette` value, and the map reads `--acc` / `--map-land` / `--map-tint`
off its host, so the terrain repaints with no other wiring.

`theme.chrome: "stripped"` hides every ornament — for screenshots, and for
people who find the costume distracting.

## Testing

```
python console/test_console_ui.py     # needs a server on :8077, or set CONSOLE_URL
```

25 checks against a real browser and a real backend: span rendering, offset
fidelity, marker and polygon rendering, bidirectional hover/click sync, all five
stage panels, GeoJSON validity, and that hover does not rebuild the document
DOM. **Any browser console error fails the run** — a silent `TypeError` in a
render function leaves a pane blank and looks like "no data", which is the one
failure a screenshot does not catch.

## Known rough edges

- **Type floor.** The design's smallest text is 8.5px, which its own handoff
  flags as below accessibility guidance. It is `--micro` in one place in
  `console.css`; raise it and re-check the layout before putting this in front
  of anyone who has to read it all day.
- **English only.** The ranker is English-trained and there is no language
  detection. The French wire document parses because its toponyms are toponyms,
  not because the model handles French.
- **≥1280px.** Below that the console scrolls sideways, by design.
- **ADM2 boundary precision**, as above.
