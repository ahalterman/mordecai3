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
./console/fetch_boundaries.sh               # ~1.25 GB, once
uv run python console/build_boundaries.py   # -> 85 MB SQLite, ~55 s
uv run python console/server.py             # http://127.0.0.1:8000
```

**`uv run`, not bare `python`.** Nothing here is importable from a system or
conda interpreter, and the failure is not a clean "module not found" — a conda
base environment with its own older FastAPI and Pydantic installed gets far
enough to raise `ImportError: cannot import name 'Undefined' from
'pydantic.fields'`, which reads like a dependency conflict in this project
rather than the wrong interpreter. Prefix every command in this file with
`uv run`, or activate `.venv` first.

Elasticsearch with the GeoNames index must be up. Without the boundary store
the console still runs; every place is just a point.

**Standing this up somewhere else?** [`DEPLOY.md`](DEPLOY.md) is the end-to-end
version — prerequisites, the GeoNames index, running it as a service, exposing
it safely, and a troubleshooting table. Note that the repository's
`compose.yaml` mounts the small test subset, not the full gazetteer.

On a headless box, `--listen` binds `0.0.0.0` and logs the address to open from
another machine. There is no authentication, so use it on a trusted network
only — put it behind an SSH tunnel (`ssh -L 8000:localhost:8000 host`) if the
network is not one.

```
uv run python console/server.py --listen --port 8077
uv run python console/server.py --host 192.168.0.233   # or bind one interface
```

---

## What is real, and what the mock faked

`handoff/API_CONTRACT.md` §3 lists five things the prototype invented. Here is
what happened to each.

| Mock | Status |
|---|---|
| `rationale` — authored prose | **Real.** Built from the ranker's own features. See below. |
| Telemetry — jittered numbers | **Real, and shorter.** Only measured values are sent; unmeasurable cells are absent rather than invented. |
| Historical `as_of` gazetteer | **Cut.** No date-scoped lookup exists. The 1977 document stays as an uppercase-OCR stress case and says so. |
| "SAT" basemap — procedural relief | **Both, and labelled.** The synthetic terrain is still there under **RELIEF**; **SATELLITE** is real NASA imagery. See below. |
| Off-frame candidate bearings | **Real, and on request.** Drawn from actual runner-up coordinates, but only while the pointer is on a candidate row. See "Candidate ghosts" below. |
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

## Mentions and places

A document that says "Ukraine" four times used to produce four markers on one
coordinate, four identical label plates stacked on each other, and four copies
of the same polygon compounding each other's fill. That is one piece of
information drawn four times, plus a legibility problem.

So the console distinguishes two things it had been conflating:

- a **mention** is one span of text. The candidate list, the margin, the
  rationale and the review flag are per mention, and they genuinely differ
  between them — the same record can be an easy call in one sentence and a
  close one in the next;
- a **place** is one gazetteer record. The map is per place: one marker, one
  polygon, one entry in the coordinate readout.

`placeKey` in `console.js` is the join, and it is a span's `data-place`
attribute in the DOM. The marker plate carries a `×4` tally so collapsing the
mentions does not quietly hide that the document leans on one place
repeatedly, and hovering or selecting any mention lights up its siblings in
the text as `link` — quieter than the selected span, loud enough to show that
one pin stands for all of them. The DISAMBIGUATE header names which one you
are reading: *MENTION 2 OF 4 AT THIS PLACE*.

The footer counts both, because they answer different questions: `RESOLVED` is
how much of the text was placed, `PLACES` is how many distinct things it was
placed on, and `BOUNDARIES` counts places rather than mentions so that it
agrees with the number of polygons visible beside it. `stats.places` and
`stats.places_with_boundary` come from the adapter, so the batch summary has
them too.

## Framing a country

The boundary layer looked intermittent, and the cause was not the join.

A shape's bounding box is measured on a map cut at the antimeridian, so a
country whose territory crosses that line comes back with a box spanning the
entire globe. Eight of the 218 ADM0 shapes in the store are in that state —
Russia, the United States, France, the United Kingdom, New Zealand, Fiji,
Kiribati, Antarctica — and four of those are among the most frequently
mentioned countries there are. The map fits its view to the union of the pins
and their boundary extents, so a single mention of Russia framed the whole
world, on which every polygon is a few pixels wide. Nothing had failed; it was
all being drawn at a size indistinguishable from absent.

`boundaries.py` now sends `focus_bbox` alongside `bbox`: the extent of the one
part of the shape that contains the coordinate the ranker committed to.
Metropolitan France, the Russian mainland, the lower 48. The remaining parts
are still drawn, they just do not get a vote on the framing. `bbox` is
unchanged and still the true extent.

The cost is stated in the docstring rather than hidden: an archipelago whose
extent was never pathological now frames tighter than it needs to, because
Indonesia's GeoNames centroid lands on Sulawesi. That is a worse frame in one
case against an unreadable one in eight, and the shape is drawn either way.

### A polygon may widen the frame; it may not take it over

`focus_bbox` fixed the pathological case and left a milder one. The ACLED
Ukraine report mentions Russia five times, so the frame picks up Russia's
mainland extent — which legitimately reaches 180°E — while every marker in the
document sits between 29°E and 100°E. More than half the map ended up east of
anything the document was about, and Kyiv, Kostiantynivka, Kramatorsk and
Sloviansk were squeezed into a few pixels of each other.

So the boundary contribution is capped. If the markers already span enough to
frame the map on their own (`map.fitMinPinSpanDeg`, 5°) and the boundaries
would blow that out by more than `map.fitMaxBlowUp` (2×), the fit uses the
markers plus 15% and lets the oversized polygon run off the edge.

Both guards matter. Without the span floor a document naming one country would
try to frame on a single point; without the ratio every document would lose the
polygon context that made the boundary layer worth adding. Measured over the
shipped corpus the three ops documents come in at ×1.0–×1.5 and the ACLED one
at ×3.0, so the threshold is not finely balanced.

A related defect fell out of the same frame: the graticule was drawn at a
fixed 2°, so a view of half the globe put ninety labelled parallels down the
left edge, where they merged into a solid bar. The configured value is the
finest spacing now rather than the only one, and the spacing steps up until
the frame carries a readable number of lines. The badge reports what was
actually drawn.

## Candidate ghosts

The active mention's runners-up used to be drawn permanently, as dashed
warn-coloured rings with off-frame ones clamped to the edge as bearings — the
handoff's design, faithfully ported, and wrong in practice. An unprompted ring
on a map reads as *"this place is in the document and something is wrong with
it"*, and on this map red already means "flagged for review", so the ghosts
were spending the one colour that had a job. The Sahel document drew rings over
Antarctic research stations, which is a candidate list, not a finding.

The mechanism was still worth keeping: that "Niger" the country beat "Niger"
the river is the disambiguate panel's whole argument, and a ranked list of
names does not convey the *distance* between the options. So it is drawn when
the reader asks for it — while the pointer is on a candidate row — in `--alt`
rather than warn, and with a leader line back to the place that won, which is
the part that carries the argument. `map.candidateGhosts` switches between
`hover` (the default), `always` (the old behaviour) and `off`.

## The paste box

It sits above the corpus rather than below it, at eleven rows of 11.5px with an
accent edge. It had been five rows of 10.5px at the bottom of the rail, under
three corpus cards — the smallest, lowest thing on the screen, and the first
thing anyone actually wants to use. `index.html` for the order, `.pastebox` in
`console.css` for the treatment.

---

## The reveal animation

Entities light up one at a time in document order, and the handoff suggests
driving that from a real per-entity stream if the backend can.

It cannot, honestly. `geoparse_batch` pools every mention in a document into a
**single** model forward pass — there is no partial result to stream, because
the whole document resolves at once. So this is a replay of a complete answer:
presentation, not progress. Set `pipeline.spanRevealMs: 0` to skip it;
`prefers-reduced-motion` skips it automatically.

Since it is presentation, its job is to make a fast thing *look* fast, and it
was doing the opposite: 320 ms per mention meant a 30 ms parse followed by four
seconds of theatre. It runs at 80 ms now, and `pipeline.revealBudgetMs` caps
the whole run at 1.4 s — past the point where one-at-a-time would overrun that,
mentions land in small groups instead of the steps stretching out, so a long
pasted article finishes in about the time a short one does. Pins still arrive
in document order, which is the part worth keeping.

What had put a floor under the step is worth knowing before tuning it further.
Each step changes only marker *state*, but `setScene` used to treat any change
to its pin key as structural and re-render the whole map, re-rasterising the
terrain's `feTurbulence` and `feDiffuseLighting` — by far the most expensive
thing on the screen — several times a second. `setScene` now separates which
places are on screen (refit, reset the view) from what the projection depends
on (refit, keep the view) from what only the markers depend on (repaint the
overlay). The reveal only ever moves the third.

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

### Export formats

The panel opens on **MORDECAI** (`raw` in the config): what `geoparse_doc`
actually returned, its field names and its structure, with every enrichment
feature still on every candidate because the console asks for `trim=False`. It
is large, and that is the point — it is the one a person can diff, script
against, or attach to a bug report. Nothing trims it on the way out. Single
documents only: one untrimmed result per document across a 500-document batch
is a response nobody asked for.

The other four are shaped for this console's own contract — flattened, renamed,
with the ranker's features stripped — and are a tab away. `export.defaultFormat`
picks which one opens; it used to be ignored in favour of a hard-coded GeoJSON.

## Two looks

The console ships two themes, switchable from the **LOOK** menu without a
reload, because the same engine gets shown to audiences who read the same
screen very differently.

- **Operations** — the original: dark, amber, wide tracking, scanlines, a
  radar sweep on the map, a UTC clock, and a vocabulary of INGEST /
  DISAMBIGUATE / FLAGGED FOR REVIEW.
- **Field brief** — warm paper, one teal accent, plain language, and nothing
  that idles. Built from the *Text geoparsing humanitarian* design handoff in
  `handoff/`.

**The layout is identical.** Same panes, same three drag seams, same map
component, same bidirectional text↔map selection. What differs is the register.

### Why a theme is more than a palette

The palette mechanism already existed and was not close to enough. The military
read of this screen lives in three places a colour swap does not reach:

1. **The words.** `BOOTING`, `DISAMBIGUATE`, `FLAGGED FOR REVIEW`, `PARSE`. A
   warm palette under those labels still reads as an ops console. So every
   user-facing string is now in `COPY` in `console.js`, and a theme overrides
   the subset it wants under `themes.<name>.copy`. Anything it leaves out falls
   through to the ops default.
2. **The physics.** Glow, `0 0 14px` halos, 8.5px labels, `.24em` tracking,
   three things that blink. The four tracking roles (`--tr-brand`,
   `--tr-label`, `--tr-title`, `--tr-btn`) and the two type floors (`--micro`,
   `--small`) are tokens for exactly this reason: the field theme resets the
   register of the whole screen by redefining six custom properties, and only
   then overrides the handful of rules that genuinely need it.
3. **The corpus.** The ops sample documents are a convoy movement report and a
   declassified 1977 cable. No palette makes those the right thing to put in
   front of a humanitarian audience, so each document in `corpus.json` carries a
   `themes` list and the rail shows the ones belonging to the current look. A
   document with no `themes` key belongs to every theme.

   The field document is a real one: ACLED's Ukraine war situation update for
   1–7 August 2026, as published on ReliefWeb, with the source URL under the
   title on screen. It is a better demo than anything authored would be —
   23 mentions over 12 distinct places, five of them close calls, and `Kyiv`
   resolving to the *oblast* in "the Kyiv region" but to the *city* in "hit Kyiv
   city", which is the disambiguation argument making itself.

### Entry points

`routes` in the config maps a URL to a look, and the server registers one route
per entry:

| URL | Opens |
|---|---|
| `/console` | Operations |
| `/demo` | Field brief |
| `/` | Whatever is remembered, else `theme.name` |

A link carries its audience, which beats a link plus an instruction to change a
setting — and it means a projector, a phone and a colleague's laptop all open
the same way, which `localStorage` alone cannot promise. Precedence is:
an explicit switch this session, then the URL, then the remembered preference,
then the config default. The URL outranks the preference because it was typed
on purpose and the preference was not. Switching looks from a routed URL
rewrites the path with `replaceState`, so a link copied mid-demo opens what was
on screen.

`routes["/demo"].lock` pins a URL to one look — the menu then offers that
theme's palettes and nothing else. It is a presentation choice, not a security
boundary: both consoles run the same engine over the same corpus file, and
neither holds anything the other does not. Routes are read once at startup, so
adding one needs a restart; the rest of the config is re-read per request.

### The menu bar

`LOOK` (theme + that theme's palettes), `CHROME` (ornaments, scanlines,
vignette, film grain, map sweep, pane hairline), `MAP` (relief / satellite /
vector, boundary polygons) and `PANES` (split / theater, reset sizes).

It is built from one declarative spec — `menuSpec()` in `console.js` — over
four generic item types: `radio`, `toggle`, `action`, `swatches`. Adding a
control is a row in that array and no CSS.

### Palettes are per theme

Each theme owns its own, under `themes.<name>.palettes`: amber, olive,
phosphor and graphite for ops; field and slate for the field look. A palette is
only meaningful inside the look it was drawn for, and having them in one flat
list invited picking an amber terrain under warm paper. Adding one still needs
no code — the key becomes selectable in the menu, and the map reads `--acc` /
`--map-land` / `--map-tint` off its host, so the terrain repaints with no other
wiring. Theme, palette, chrome flags and layout all persist in `localStorage`;
the config supplies the defaults.

### The calm map

`geo-scope` gained a `calm` option rather than the handoff's separate
`calmmap.js`. It keeps pan/zoom, the boundary polygons and the satellite
imagery — none of which the design's map has — and swaps what makes the ops map
read as a sensor: the relief filter drops from six octaves at surface scale 7 to
three at 2.2 and is laid *over* a flat land fill at 62% rather than being the
land; the sweep, the registration corners and the centre reticle go; and the
markers become dots and rounded chips instead of crosshairs on dark plates. The
scale bar stays, because it is the one piece of map chrome that is a fact about
the picture rather than a claim about the instrument.

## Configuration

`console.config.json` drives the themes and their palettes, layout, grit,
navigation limits, top-k, review gate, reveal timings, export formats, the
candidate-ghost mode, the boundary layer and the imagery source.
(`handoff/config/console.config.schema.json` is the design's schema and has
drifted — it predates `themes`, `boundaries`, `input`, `map.imagery` and the
`imagery` map mode. Nothing validates against it.)

`theme.chrome: "stripped"` hides every ornament — for screenshots, and for
people who find the costume distracting. The CHROME menu does the same thing
one ornament at a time.

### Satellite imagery

**SATELLITE** draws real tiles from [NASA EOSDIS
GIBS](https://nasa-gibs.github.io/gibs-api-docs/) — `BlueMarble_NextGeneration`
by default: MODIS-derived, cloud-cleared, public domain, **no API key**. The
alignment is exact rather than fitted. `d3.geoMercator` *is* web mercator, so
substituting the tile pyramid's pixel space into the projection collapses both
axes to one affine map — `screen = f·world + offset`, `f = 2πs/W`,
`offset = (tx − sπ, ty − sπ)`. There is no tile-specific projection and no
second opinion about where anything is; a marker does not move by a pixel
between RELIEF and SATELLITE, and the test asserts that.

Three things worth knowing:

- **It is the one layer that needs the network.** d3, topojson and the basemap
  are vendored because a demo that needs the internet fails in the one room
  without it; a world tileset cannot be. Four consecutive tile failures and the
  map falls back to the procedural relief, badges itself `RASTER UNREACHABLE`,
  and says so in the status bar. Selecting SATELLITE again retries.
- **Blue Marble stops at z8** (~500 m/px). Past that the last level is
  stretched, and the badge reads `OVER-ZOOMED` rather than letting a blurry
  frame pass for full resolution. For finer imagery, point `map.imagery` at a
  provider with deeper tiles — that generally means an API key.
  `VIIRS_SNPP_CorrectedReflectance_TrueColor` with
  `GoogleMapsCompatible_Level9` and `"time": "2026-08-24"` is the keyless
  half-step: yesterday's actual satellite pass, ~250 m, and clouds.
- **The tiles are graded into the palette** — `desaturate`, `brightness`, and a
  `--map-tint` multiply — because fixed-colour imagery under a themed UI
  otherwise reads as a foreign rectangle. `desaturate: 1, tint: 0, brightness:
  1` gives the untouched pixels. GIBS sends `Cache-Control: no-store`, so tiles
  are fetched once into object URLs and kept in an LRU rather than re-requested
  on every repaint.

These are adjustable from the screen itself, and all of them outlive the tab in
`localStorage` while the config keeps supplying the defaults:

- **Look, palette and ornaments.** In the menu bar — see *Two looks* above.
  Only the map needs telling when the palette changes; everything else is
  reading the same custom properties.
- **Pane sizes.** The three seams — rail, the column split, the row split — are
  drag handles. Double-click one to put it back. They move `--rail-w`,
  `--split-col` and `--split-row` on `:root`, which is the only place the
  layout proportions are written down.

## Testing

```
uv run python console/test_console_ui.py   # needs a server on :8077, or set CONSOLE_URL
```

73 checks against a real browser and a real backend: span rendering, offset
fidelity, marker and polygon rendering, bidirectional hover/click sync, all five
stage panels, GeoJSON validity, raw-export structure, that repeated mentions
collapse to one marker and one polygon, that a country with overseas territory
does not frame the globe, that candidate ghosts appear only while the pointer
is on a candidate row, that hover does not rebuild the document DOM, that
hovering a marker does not re-solve the label packing, that satellite tiles
register with the same projection as the markers and fall back to the relief
when the tile host is cut, that the palette swatches and the pane splitters do
what they say, and that switching to the field theme keeps every pane, applies
its palette, replaces the ops vocabulary, loads its own sample reports and
parses it for real, draws the calm relief and stops the sweep, that `/demo` and
`/console` open the look they name even when `localStorage` says otherwise, and
that an oversized country polygon does not take over the map frame. **Any browser console error fails the run** — a silent `TypeError` in a
render function leaves a pane blank and looks like "no data", which is the one
failure a screenshot does not catch.

## Known rough edges

- **Type floor.** The ops theme's smallest text is 8.5px, which its own handoff
  flags as below accessibility guidance. It is `--micro` in one place in
  `console.css`; raise it and re-check the layout before putting this in front
  of anyone who has to read it all day. The field theme already raises it to
  10px and `--small` to 11px.
- **No authentication.** Still none. See
  [ACCESS_AND_FEEDBACK.md](ACCESS_AND_FEEDBACK.md) and
  [DEPLOY.md](DEPLOY.md#step-8--letting-other-people-reach-it).
- **English only.** The ranker is English-trained and there is no language
  detection. The French wire document parses because its toponyms are toponyms,
  not because the model handles French.
- **One sample report under the field theme.** Swapping in the real ACLED
  report replaced the three authored ones, and with them the French and
  lowercase-transcript stress cases that the field look used to demonstrate.
  Add real equivalents to `corpus.json` with `"themes": ["field"]`.
- **≥1280px.** Below that the console scrolls sideways, by design.
- **ADM2 boundary precision**, as above.
- **Archipelago framing.** `focus_bbox` frames on the part of a shape holding
  the gazetteer centroid, so a document saying only "Indonesia" frames on
  Sulawesi with the rest of the archipelago spilling off the edges. See
  "Framing a country" for why that trade was taken.
