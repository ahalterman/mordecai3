# Handoff: Geoparse Console (Mordecai v4 demo)

## Overview

A single-screen analyst console for a **text geoparsing** engine. Text goes in, toponyms come
out, and the operator watches them light up in the document while the corresponding markers land
on a map — that sync is the whole point of the demo. Around that hero moment sits the full
pipeline (ingest → parse → disambiguate → resolve → export) and a candidate-ranking panel that
shows *why* a given "Gao" or "the Niger" resolved the way it did.

Built for two audiences at once: a conference/consulting demo of the library's new major version,
and a plausible sketch of what a real operator UI for it would be.

Visual direction is deliberate: mid-2000s techno-thriller / satellite-ops. Dark, monospaced,
bracketed, noisy. It is a costume over a real tool — the data model underneath is honest.

## About the design files

The files in `design/` are **design references created in HTML**. They are prototypes showing
intended look and behavior — not production code to lift. The task is to **recreate these designs
in the target codebase's environment** (React, Vue, Svelte, Python+HTMX, whatever the project
already uses) with its established patterns, state management, and build tooling. If no
environment exists yet, pick the most appropriate one and implement there.

Two exceptions worth reusing more literally:

- **`design/geoscope.js`** is a dependency-light custom element wrapping d3-geo + TopoJSON. The
  projection math, fit-to-pins logic, pan/zoom transform composition, and SVG terrain filters in
  it are the real thing and are tedious to re-derive. Port it, don't reinvent it.
- **`config/console.config.json`** is the config surface the user asked for. Treat it as the spec
  for what must be configurable.

`design/support.js` is the prototyping runtime that renders the `.dc.html` file. **Do not port
it.** It exists only so `design/Mordecai Geoparse Console.dc.html` opens in a browser for
reference.

## Fidelity

**High-fidelity.** Colors, type, spacing, motion timings, and copy are all final-intent. Recreate
pixel-close, then swap in the codebase's own primitives where they exist.

Caveat: the *data* is synthetic. Place names, coordinates, feature codes and GeoNames-style ids
are real or realistic; document text, confidences, rationales and telemetry numbers are authored.
See `API_CONTRACT.md` §3 for the list of things the mock fakes.

---

## Layout

One full-viewport shell, `100vh`, `display:flex; column`, `min-width: 980px`. Body is
`overflow-x:auto; overflow-y:hidden` — the console scrolls sideways below its minimum, never
vertically.

```
┌──────────────────────────────────────────────────────────────────────┐
│ TITLE BAR                                                    52px    │
├──────────┬───────────────────────────────────────────────────────────┤
│          │  ┌──────────────────┬──────────────────────────────────┐  │
│  RAIL    │  │                  │  MAP                  1.42fr     │  │
│  214px   │  │  DOCUMENT        ├──────────────────────────────────┤  │
│          │  │  1.02fr          │  STAGE PANEL          1fr        │  │
│          │  └──────────────────┴──────────────────────────────────┘  │
├──────────┴───────────────────────────────────────────────────────────┤
│ STATUS BAR                                                   32px    │
└──────────────────────────────────────────────────────────────────────┘
```

The workspace is a CSS grid with named areas — this is the mechanism behind the `layout` config
key, and the only place in the design that uses a stylesheet rule rather than an inline style:

```css
[data-el="main"]{
  display:grid; gap:1px; background:var(--edge); flex:1; min-height:0;
  grid-template-columns:1.02fr 1.3fr;
  grid-template-rows:1.42fr 1fr;
  grid-template-areas:"doc map" "doc stage";
}
:root[data-layout="theater"] [data-el="main"]{
  grid-template-columns:1.1fr 1fr;
  grid-template-rows:1.25fr 1fr;
  grid-template-areas:"map map" "doc stage";
}
```

`SPLIT` (default) puts the document full-height on the left with map over stage panel on the
right. `THEATER` promotes the map to a full-width band across the top with document and stage
panel below — for wall displays and for the "watch the pins land" moment specifically.

Every panel carries `min-width:0; min-height:0` so grid children can actually shrink.

---

## Screens / views

There is one screen. It has five *stage views* (the pipeline) and three *documents* (the corpus),
and those two axes are independent — the stage panel changes, the document pane and map do not.

### 1. Title bar — 52px, `background:var(--panel2)`, `border-bottom:1px solid var(--edge)`

| Region | Width | Contents |
|---|---|---|
| Lockup | 214px, right border | 14×14px `var(--acc)` square rotated 45°, `box-shadow:0 0 14px` accent @60%. Then "MORDECAI" — Barlow Condensed 700 / 20px / `letter-spacing:.22em` — over "GEOPARSE ENGINE" — JetBrains Mono 400 / 8.5px / `.2em` / `var(--dim)`. |
| Center | flex | Classification chip: JetBrains Mono 500 / 10px / `.28em`, color `var(--warn)`, 1px border in warn @55%, padding `4px 9px`. Then a dashed 1px rule (`repeating-linear-gradient(90deg, var(--edge) 0 5px, transparent 5px 11px)`). Then three stat pairs: `BUILD 4.0.0-rc.2`, `RANK mordecai-rank/v4`, `DEV cuda:0 41°C` — labels `var(--dim)`, values `var(--fg)`, 10px mono, gap 22px. |
| Clock | 172px, left border | UTC `HH:MM:SS` — JetBrains Mono 500 / 15px / `var(--acc)`, above `UTC · SESSION A7-1194` — 8.5px `.18em` `var(--dim)`. Right-aligned. |

### 2. Left rail — 214px, `var(--panel2)`, right border, `overflow:hidden`

Column flex. Middle section scrolls (`flex:1; min-height:0; overflow-y:auto`); the footer block
is `flex:none` and pinned. **This matters** — the RE-RUN control is the trigger for the demo's
hero moment and must never be the thing that gets clipped at short viewport heights.

- **`PIPELINE`** section label — mono 500 / 9.5px / `.24em` / `var(--dim)`, padding `13px 14px 9px`.
- **Five stage rows.** `display:flex; align-items:center; gap:9px; padding:8px 14px; cursor:pointer`.
  Left: two-digit index, mono 400 / 9px / `var(--dim)`, 16px wide. Center: name, Barlow Condensed
  600 / 13.5px / `.16em`. Right: meta string, mono 400 / 8.5px / `var(--dim)`. Far right: 6px status dot.
  - Inactive: `color:var(--fg); opacity:.68`, dot `var(--edge)`.
  - Active: `background: color-mix(in srgb, var(--acc) 13%, transparent)`, `color:var(--acc)`,
    `box-shadow: inset 2px 0 0 var(--acc)`, dot `var(--acc)`.
  - Rows: `01 INGEST / 3 SRC`, `02 PARSE / NER`, `03 DISAMBIGUATE / TOP-K 5`, `04 RESOLVE / GAZ`,
    `05 EXPORT / 4 FMT`.
- **1px `var(--edge)` divider**, then **`CORPUS`** label, then **three document cards**:
  `padding:9px 14px; border-left:2px solid`. Title line Barlow Condensed 600 / 12.5px / `.13em`
  with ellipsis, size right-aligned mono 8px `var(--dim)`; second line source string mono 8.5px
  `var(--dim)`. Selected: left border `var(--acc)`, background accent @10%, text `var(--acc)`.
  Unselected: transparent border, `opacity:.62`.
- **Footer (pinned, `border-top:1px solid var(--edge)`, padding `12px 14px 14px`, gap 9px):**
  - **SAT / VECTOR** segmented pair. Two equal halves, 1px gap over `var(--edge)`, 1px edge border.
    Each: `padding:7px`, Barlow Condensed 600 / 10px / `.2em`, centered. Active half `background:var(--acc); color:#08090a`;
    inactive `background:var(--panel); color:var(--dim)`.
  - **▶ RE-RUN PARSE** button. Full width, `padding:10px`, 1px `var(--acc)` border, background
    accent @12%, text `var(--acc)`, Barlow Condensed 600 / 12px / `.2em`, glyph `▶` at 9px, gap 9px.
    Hover inverts to solid accent on `#08090a`. Label becomes `PARSING…` while a run is in flight.
  - **Gazetteer footnote**, mono 400 / 8.5px / line-height 1.7 / `var(--dim)`:
    `GAZ · GEONAMES 2026.07` / `12.41M FEATURES · FAISS-IVF`.

### 3. Document pane — `grid-area:doc`, `var(--panel)`

- **Header** (`padding:12px 18px 11px`, bottom edge border): title Barlow Condensed 600 / 16px /
  `.15em` `var(--fg)`; meta line mono 400 / 9px / `.09em` `var(--dim)`. Right: a 5px `var(--acc)`
  square blinking on a 1.6s step animation plus `TOKENIZED` at mono 8.5px `.16em`.
- **Body** (`flex:1; overflow-y:auto; padding:22px 26px 26px`). Inner column
  `white-space:pre-wrap; text-wrap:pretty; max-width:70ch`.

  ⚠️ **Implementation trap:** because the body is `pre-wrap`, any whitespace *between* the
  rendered span elements becomes visible line breaks. Emit the segment spans with **no whitespace
  between them**.

  Type per document kind:
  | kind | family | size | line-height | tracking |
  |---|---|---|---|---|
  | `wire` | Source Serif 4 | 16.5px | 1.85 | .002em |
  | `social` | JetBrains Mono | 12.5px | 2.05 | — |
  | `archive` | JetBrains Mono | 12.5px | 2.15 | .05em |

- **Entity span states.** Each toponym is a span carrying `data-st`; everything else is plain text.
  ```css
  [data-eid]        { cursor:crosshair; border-radius:1px;
                      transition:background .16s, color .16s, box-shadow .2s }
  [data-st=pending] { opacity:.34; border-bottom:1px dotted var(--dim) }
  [data-st=scan]    { background:var(--acc); color:#08090a; font-weight:600;
                      box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 30%,transparent) }
  [data-st=ok]      { color:var(--acc); background:color-mix(in srgb,var(--acc) 9%,transparent);
                      box-shadow:inset 0 -1px 0 color-mix(in srgb,var(--acc) 55%,transparent) }
  [data-st=amb]     { color:var(--warn); background:color-mix(in srgb,var(--warn) 10%,transparent);
                      box-shadow:inset 0 -1px 0 var(--warn) }
  [data-st=sel]     { background:var(--acc); color:#08090a; font-weight:600;
                      box-shadow:0 0 0 3px color-mix(in srgb,var(--acc) 34%,transparent) }
  [data-st=ok]:hover, [data-st=amb]:hover { /* same as sel */ }
  ```
- **Footer** (30px, top edge border, `padding:0 18px`, mono 400 / 9px / `.1em` / `var(--dim)`, gap 16px):
  `SPANS n` · `RESOLVED n` (accent) · `AMBIGUOUS n` (warn) · dashed filler rule · `NN.N ms`.
- A 1px accent hairline sits at the pane's top edge, `opacity:.2`, breathing 0.15→0.5→0.15 over 5s.

### 4. Map — `grid-area:map`, `var(--map-sea)`, `position:relative; overflow:hidden`

The `<geo-scope>` element fills the panel absolutely. Everything else is HTML overlay with
`pointer-events:none` except the reset control.

- **Top gradient strip**, 34px, `linear-gradient(180deg, rgba(4,6,9,.92), rgba(4,6,9,0))`:
  region title Barlow Condensed 600 / 12px / `.24em` `var(--acc)`, then mode meta mono 9px `var(--dim)`
  (`EO COMPOSITE · SYNTHETIC RELIEF · MERCATOR` / `VECTOR OVERLAY · NE 110M · MERCATOR`).
- **Left badge stack** (`left:14px; top:44px`, gap 5px): mono 8.5px `.14em` `var(--dim)`,
  `background:rgba(4,6,9,.6)`, `padding:2px 6px`, `border-left:2px solid var(--acc)`.
  SAT → `BAND 4-3-2 PSEUDO`, `CLOUD 04%`, `GSD 12 M`. VECTOR → `GRATICULE 2°`, `ADMIN-0 MESH`, `NO RASTER`.
- **Right coordinate readout** (`right:12px; top:38px`): live lat / lon of the selected entity in
  `DD.DDDD° N` form, accent, mono 9px, `text-shadow:0 0 8px rgba(0,0,0,.9)`; third line
  `ALT nnn KM · WGS84` in `var(--dim)`.
- **Bottom-right nav strip** (`right:12px; bottom:12px`): `SCROLL ZOOM · DRAG PAN` hint in dim,
  a `×N.N` zoom chip (accent text, `rgba(4,6,9,.72)` fill, accent-40% border), and a `RESET`
  button (solid accent border, inverts on hover) — the only interactive overlay.

#### Inside `<geo-scope>` (see `design/geoscope.js`)

Shadow DOM, one `<svg>`, rebuilt as an innerHTML string on every committed change.

- **Geometry**: `world-atlas@2.0.2/countries-110m.json`, `topojson.feature` for land and
  `topojson.mesh(…, (a,b) => a !== b)` for internal borders. `d3.geoMercator()` fit to a
  `MultiPoint` of the current document's pins with padding `0.26 × min(W,H)`.
- **Pan/zoom** is hand-rolled (no `d3-zoom`) as a two-level transform so the expensive SVG filters
  are not re-rasterised during a gesture:
  - *committed* `(zk, zx, zy)` folded into the projection: `scale(s₀·zk)`,
    `translate([t₀x·zk + zx, t₀y·zk + zy])`.
  - *live* `(lk, lx, ly)` applied as a `transform` attribute on the `<g class="scene">` wrapper
    during the gesture only.
  - Wheel about point *p* with factor *f*: `lk *= f; lx = lx·f + p.x(1−f); ly = ly·f + p.y(1−f)`.
    Factor is `Math.exp(-deltaY × 0.0018)`. Commit is debounced 190ms.
  - Drag accumulates `movementX/Y` into `lx/ly` via window-level listeners; commits on mouseup if
    moved > 3px (which also suppresses the pin click).
  - Compose on commit: `zk' = zk·lk`, `zx' = zx·lk + lx`, `zy' = zy·lk + ly`, then full re-render.
  - Clamp `zk ∈ [0.6, 28]`. Double-click resets. Changing documents resets (keyed on the pin id list).
- **SAT terrain** is a `feTurbulence(fractalNoise, baseFrequency "0.0075 0.0105", 6 octaves)` fed
  into `feDiffuseLighting(surfaceScale 7, diffuseConstant 1.05, distantLight azimuth 308°
  elevation 46°)` with `lighting-color: var(--map-land)`, painted on a rect clipped to the land
  path, then multiplied with a `var(--map-tint)` rect at 42%. Sea gets its own lower-octave
  turbulence at 55%. A radial vignette closes it out. **This is why the palettes change the
  terrain colour** — `--map-land` is the light source, `--map-tint` the multiply.
- **VECTOR mode**: land as a 1.15px accent stroke with a `feGaussianBlur(2.2)` merge glow, dotted
  fill pattern inside the land clip, 2° graticule at 13% and 10° at 30%, borders as a dashed
  `var(--alt)` mesh.
- **Markers**: an 6/9px ring (dashed when `review`), four crosshair ticks, a 1.7px centre dot, a
  leader line up-right to a 13px-tall label plate (`#05060a` fill, 1px coloured border) holding the
  uppercase name at mono 600 / 8.5px / `.08em` and a 16×4px confidence bar. Active markers add a
  pulsing ring (r 6→26, opacity .85→0, 1.9s). Colours: `var(--alt)` resting, `var(--acc)` active,
  `var(--warn)` when flagged; `opacity` .25 pending / .72 resting / 1 active.
- **Off-frame candidates**: candidates outside the viewport are clamped to the frame edge and drawn
  as a rotated triangle plus `NAME · OFF-FRAME` in warn. In-frame ones get a dashed ghost ring.
- **Chrome**: corner brackets, centre crosshair with a dotted 30px ring, a scale bar whose label is
  computed by `geoPath`-inverting two screen points, a blinking `LIVE · EO/IR` badge, and a 9s
  linear sweep gradient (`translateX(-14% → 114%)`).

### 5. Stage panel — `grid-area:stage`, `var(--panel)`

36px header: stage name Barlow Condensed 600 / 13px / `.22em` `var(--acc)`, dashed filler rule,
stage meta mono 9px `.12em` `var(--dim)`. Body scrolls.

- **INGEST** — three source rows, `grid-template-columns:1fr 74px 62px 58px`, `padding:9px 10px`,
  `var(--panel2)` fill, mono 10px. Columns: title / adapter (`rss|social|ocr`) / size / state
  (`ACTIVE` accent, `CACHED` dim). Below: a dashed-border note block, mono 10px / 1.8:
  `NORMALIZE → NFKC · STRIP MARKUP · LANG-ID (fasttext) → fr 0.61 / en 0.39` and
  `SEGMENT → 14 SENTENCES · 612 TOKENS`.
- **PARSE** — a span table, `grid-template-columns:1fr 54px 52px 46px`, header row at mono 500 /
  8.5px / `.16em` dim (`SPAN | LABEL | CHAR | P`). Rows are clickable and select the entity; the
  active row takes accent @14% background and accent text. Label column in `var(--alt)`, char span
  in dim, probability right-aligned accent.
- **DISAMBIGUATE** *(default)* — the panel that carries the demo.
  - Heading: entity surface form, Barlow Condensed 600 / 19px / `.13em` `var(--fg)`; beside it
    mono 9px dim: `GPE · 13 SPANS IN DOC · REVIEW FLAG|AUTO-ACCEPT`.
  - Candidate rows, `padding:9px 11px`, `border-left:2px solid`. Rank `NN` in dim mono 9px; name
    Barlow Condensed 500 / 12.5px / `.11em`; confidence right, accent mono 9px. Second line: the
    `display` string (mono 8.5px dim, ellipsised) and an 82×4px bar over `var(--edge)`.
    Rank 1 gets an accent left border and accent @9% fill — or **warn** border and warn @9% fill
    when the entity is flagged. Ranks 2+ get `var(--edge)` border, `var(--panel2)` fill, dim bar.
  - Rationale block: `border-left:2px solid var(--acc)`, `var(--panel2)` fill, mono 10px / 1.75,
    prefixed `RATIONALE ▸ ` in accent.
- **RESOLVE** — key/value rows, `grid-template-columns:104px 1fr`, `padding:7px 10px`,
  `var(--panel2)`, mono 10px, keys dim / values fg. Keys: GEONAMEID, NAME, FEATURE CODE, ADMIN,
  POPULATION, COORDINATES, CONFIDENCE (suffixed `· ACCEPTED` or `· BELOW GATE`), SOURCE.
- **EXPORT** — a four-way segmented format switcher (same treatment as SAT/VECTOR), then a `<pre>`
  on `#06070a` with a 1px edge border, mono 9.5px / 1.65 in `var(--alt)`, `max-height:180px`,
  scrollable. Footer line: `n FEATURES · n FLAGGED FOR REVIEW` and a `WRITE ▸` button.

### 6. Status bar — 32px, `var(--panel2)`, top edge border

Left cell (fixed, right border): 6px blinking accent dot + state word (`READY` / `PARSING` /
`RESOLVED`) at mono 9px `.18em` accent. Centre: `HH:MM:SS.mmm` timestamp in `var(--alt)` then the
current log line in dim mono 9.5px, single-line ellipsis, trailed by a 1s-blinking `▊` caret.
Right cell (fixed, left border): `THROUGHPUT n DOC/S`, `P50 n ms`, `MEM n GB`, `QUEUE n`.

### 7. Grit overlays — three fixed layers over everything, `pointer-events:none`

1. **Scanlines**, z 40: `repeating-linear-gradient(0deg, rgba(0,0,0,.34) 0 1px, transparent 1px 3px)`,
   `mix-blend-mode:multiply`, `opacity: calc(var(--grit) * .5)`.
2. **Film noise**, z 41: an inline-SVG `feTurbulence(baseFrequency .85, 3 octaves)` data-URI tile
   at 160×160, `mix-blend-mode:overlay`, `opacity: var(--grit)`, with a 7s step flicker that spikes
   to 1.7× for one frame.
3. **Vignette**, z 42: `radial-gradient(ellipse 88% 78% at 50% 46%, transparent 42%, rgba(0,0,0,.55) 100%)`.

In a real implementation the noise tile should be a static PNG/WebP asset, not a data-URI filter —
it repaints on every composite otherwise.

---

## Interactions & behavior

### The hero moment — the parse run

Fires on load (after 900ms) and on every RE-RUN / document switch.

1. All entity spans → `pending` (34% opacity, dotted underline). All map markers → 25% opacity.
   Status word → `PARSING`. Parse time → `—`. Selection cleared.
2. After 320ms, step through entities **in document order**:
   - set that entity to `scan` (inverted accent block, 3px accent halo), select it — which drives
     the map to focus it, the coordinate readout to update, and the disambiguate panel to swap in
     its candidates;
   - push a log line: `rank.v4    <name> → top-1 0.94 · k=5 · 13ms`;
   - after **230ms**, settle it to `ok` or `amb`;
   - wait **90ms**, next entity.
3. On completion: status word → `RESOLVED`, parse time filled in, and selection lands on the
   **first flagged entity** — so the console comes to rest showing a disambiguation problem, not a
   trivial success.

Total ≈ 320ms + n × 320ms. For the 13-span wire document, about 4.5 seconds.

If the backend can stream per-entity results, drive this from the stream instead of a timer.

### Selection and hover — bidirectional sync

One `selected` id and one `hovered` id; the effective entity is `hovered ?? selected`. Everything
reads from that single value: the document highlight, the marker's active state, the coordinate
readout, the PARSE row highlight, and the whole DISAMBIGUATE / RESOLVE panel.

| Action | Effect |
|---|---|
| Hover a document span | sets `hovered` — marker grows + pulses, coordinates update, candidates swap. No panel navigation. |
| Click a document span | sets `selected`, clears `hovered`, switches the stage panel to DISAMBIGUATE. |
| Hover a map marker | sets `hovered` (component emits `pinhover`) — document span inverts to `sel`. |
| Click a map marker | sets `selected`, switches to DISAMBIGUATE (`pinclick`). Suppressed if the pointer moved > 3px, so drags don't select. |
| Click a PARSE table row | same as clicking the span. |
| Click a stage row | changes the stage panel only. |
| Click a corpus card | loads that document, resets zoom, re-runs the parse. |
| Wheel / drag on map | zoom about cursor / pan. Double-click or RESET restores the fit. |

Hover transitions are 160ms on background and color, 200ms on box-shadow. Nothing else animates on
interaction — the ambience is all idle-loop.

### Ambient motion (all idle, all CSS except the two intervals)

| Element | Motion |
|---|---|
| Map sweep | 9s linear, gradient band `translateX(-14% → 114%)` |
| Active marker | 1.9s ease-out pulse ring, r 6→26, opacity .85→0 |
| `LIVE · EO/IR` badge | 2.4s step blink, 1 → .35 at 88% |
| Status + header dots | 1.4–1.6s step blink |
| Log caret | 1s step blink |
| Document top hairline | 5s ease-in-out, opacity .15 ↔ .5 |
| Film noise | 7s step flicker, one-frame spike to 1.7× |
| Clock + telemetry | 1000ms interval; jittered ranges — throughput 10.6–12.2, p50 17.2–20.3, mem 6.0–6.4, queue 0–2, alt 405–420, temp 40–44 |
| Log line | 2600ms interval, random pick from an 8-line pool |

Respect `prefers-reduced-motion`: drop the sweep, pulse, flicker and blinks; keep the parse reveal
(it is information, not decoration) but consider collapsing it to a single step.

---

## State management

```ts
type Status = 'pending' | 'ok' | 'amb';

interface ConsoleState {
  docId:    string;                    // active corpus document
  stage:    'ingest'|'parse'|'dis'|'resolve'|'export';
  mapMode:  'sat' | 'wire';
  segments: Segment[];                 // derived from doc text + entity offsets
  status:   Record<string, Status>;    // entityId -> resolution status
  selected: string | null;
  hovered:  string | null;
  running:  boolean;
  zoom:     number;                    // mirrored from the map component
  format:   'geojson'|'jsonl'|'csv'|'wkt';
  telemetry: { clock; temp; tput; p50; mem; queue; alt };
  log:      { time: string; line: string };
}
```

Derivations (compute, don't store): `active = hovered ?? selected`; `spans = entities.length`;
`resolved = count(status === 'ok')`; `flagged = count(status === 'amb')`; segment `data-st` =
`active === id && settled ? 'sel' : status[id] ?? 'pending'`.

Data fetching: one `POST /api/geoparse` per document (see `API_CONTRACT.md`). Cache by `doc_id`;
RE-RUN replays the reveal animation from cache rather than re-hitting the backend, unless options
changed. The basemap TopoJSON is fetched once and shared across all map instances.

---

## Design tokens

Palettes live in `config/console.config.json` under `palettes` and are applied as CSS custom
properties on `:root`, switched with `document.documentElement.dataset.pal`. The map component
reads `--acc`, `--alt`, `--warn`, `--map-land`, `--map-tint`, `--map-sea` off its host via
`getComputedStyle`, so a palette change repaints the terrain with no other wiring.

| Token | amber (default) | olive | phosphor | graphite |
|---|---|---|---|---|
| `--acc` | `#f2a33c` | `#c8d074` | `#74ef8f` | `#e8e3d5` |
| `--alt` | `#86a6b8` | `#93a46e` | `#b7c46d` | `#8f9aa3` |
| `--warn` | `#e2452c` | `#e0692a` | `#ff6a2b` | `#d0512f` |
| `--map-land` | `#c2b391` | `#b9c085` | `#a6c49c` | `#a9a59a` |
| `--map-tint` | `#5d5124` | `#3d4c1c` | `#274526` | `#3a3a3c` |
| `--map-sea` | `#080b10` | `#0c1310` | `#040a07` | `#090a0c` |
| `--bg` | `#0a0a09` | `#0a0c08` | `#070a08` | `#0b0b0c` |
| `--panel` | `#101110` | `#11140e` | `#0d110e` | `#101113` |
| `--panel2` | `#0c0d0c` | `#0d0f0a` | `#090d0a` | `#0c0c0e` |
| `--edge` | `rgba(226,214,182,.13)` | `rgba(202,212,158,.14)` | `rgba(170,230,185,.13)` | `rgba(220,220,220,.12)` |
| `--fg` | `#cdc8b8` | `#cbceb1` | `#c1d6c4` | `#c7c6c1` |
| `--dim` | `#7a776c` | `#787c62` | `#6d7f71` | `#75767b` |

`#08090a` is the fixed "ink on accent" colour for inverted chips and buttons across all palettes.

**Typography** — three families, all Google Fonts:

| Role | Family | Weights | Usage |
|---|---|---|---|
| Chrome / labels / headings | Barlow Condensed | 400 500 600 700 | 10–20px, tracking `.11em`–`.24em`, mostly uppercase |
| Data / telemetry / code | JetBrains Mono | 300 400 500 700 | 8.5–15px, tracking `.04em`–`.28em` |
| Wire copy | Source Serif 4 | 400 600, italic | 16.5px / 1.85 |

Smallest text is **8.5px** (badges and column headers). That is deliberate for the aesthetic and
is below normal accessibility guidance — if this ships to real analysts, bump the floor to 11px
and re-check the layout, or offer a density toggle.

**Spacing** — 1px hairline gaps (grid gutters, list separators), then a 4 / 5 / 7 / 9 / 11 / 12 /
14 / 16 / 18 / 22 / 26px scale. **Radius** — 0 everywhere except a 1px softening on entity spans
and 50% on status dots. **Shadows** — no drop shadows; depth comes from `inset` borders,
`box-shadow: 0 0 0 3px` accent halos, and a single `0 0 14px` accent glow on the logo diamond.

---

## Configuration surface

`config/console.config.json` (+ JSON Schema alongside) is the file the user asked for. Keys:

| Key | Values | Effect |
|---|---|---|
| `theme.palette` | `amber` `olive` `phosphor` `graphite` | Sets `:root[data-pal]`. |
| `theme.grit` | 0–1 | `--grit`; scales scanline and noise opacity. `0` = clean. |
| `theme.chrome` | `full` `stripped` | `stripped` hides every `[data-orn]` ornament (dashed rules, badge stack, TOKENIZED chip, scanlines, noise) and mutes the classification banner to 35%. Use for screenshots and for people who find the costume distracting. |
| `theme.layout` | `split` `theater` | Grid template, above. |
| `theme.scanlines` / `vignette` / `flicker` | bool | Individual overlay kill switches. |
| `map.defaultMode` | `sat` `wire` | Initial basemap treatment. |
| `map.pan` / `zoom` / `minZoom` / `maxZoom` | | Navigation limits. |
| `map.fitPaddingRatio` | number | Padding as a fraction of `min(W,H)` when fitting to pins. |
| `pipeline.topK` | int | Candidates requested and rendered. |
| `pipeline.reviewGate` | 0–1 | Top-1 minus top-2 margin below which an entity is flagged. Drives the warn styling everywhere. |
| `pipeline.autorun` | bool | Whether the reveal animation plays on load. |
| `pipeline.spanRevealMs` / `spanGapMs` / `runStartDelayMs` | ms | Reveal choreography. Set reveal to 0 to disable the animation entirely. |
| `export.formats` / `defaultFormat` | | Which export tabs exist. |
| `brand.*` | strings | Product name, build string, classification banner text. |
| `palettes` | object | Full palette definitions — add your own here; the key becomes a valid `theme.palette` value with no code change. |

---

## Assets

No image, icon, or font files are bundled.

- **Fonts** — Barlow Condensed, JetBrains Mono, Source Serif 4, loaded from Google Fonts. Self-host
  for an offline/air-gapped build.
- **Basemap** — `world-atlas@2.0.2/countries-110m.json` (Natural Earth, public domain) from jsDelivr.
  **Vendor this file**; do not fetch a CDN at runtime in a deployed tool.
- **Libraries** — d3 7.9.0 and topojson-client 3.1.0, loaded with SRI hashes (kept in the HTML head).
  Only `geoMercator`, `geoPath`, `geoGraticule` and `topojson.feature/mesh` are used — a
  `d3-geo` + `topojson-client` install is enough; the full d3 bundle is a prototyping convenience.
- **Iconography** — none. Every glyph is a typed character (`▶ ▸ ▊ ◆ ×`) or drawn SVG primitive.
  If the codebase has an icon set, the segmented toggles and the run button are the natural places
  to use it.
- **Noise texture** — generated by an inline SVG filter. Replace with a static tile in production.

---

## Files

```
design_handoff_geoparse_console/
├── README.md                        ← this file
├── API_CONTRACT.md                  ← request/response shape + what the mock fakes
├── config/
│   ├── console.config.json          ← the configuration surface
│   └── console.config.schema.json   ← JSON Schema for it
├── fixtures/
│   └── corpus.json                  ← 3 documents, 28 entities, full expected responses
└── design/
    ├── Mordecai Geoparse Console.dc.html   ← the prototype; open in a browser
    ├── geoscope.js                          ← map component — port this one
    └── support.js                           ← prototyping runtime; DO NOT port
```

Open `design/Mordecai Geoparse Console.dc.html` directly in a browser (needs network access for
fonts, d3, and the basemap).

## Suggested build order

1. Shell + tokens + palette switching. Get the grid, both layouts, and all four palettes right
   with static content — it is the cheapest thing to get wrong.
2. Document pane with static offsets from `fixtures/corpus.json`. Nail the span states.
3. `geoscope` port. Real geometry, both modes, pan/zoom, markers. Independent of everything else.
4. Wire selection/hover both directions. This is the demo.
5. Stage panels, then the parse reveal, then grit and ambience last.
6. Swap the fixture for the real adapter.

## Open questions for the implementer

- Which Mordecai entrypoint backs `/api/geoparse`, and does it return per-candidate confidences or
  only a top-1? The disambiguate panel is meaningless without the ranked list.
- Can the ranker expose an attribution good enough to generate `rationale` honestly?
- Is date-scoped (historical) gazetteer lookup real, or should the 1977 document drop that claim?
- Should telemetry be wired to real metrics or removed?
- Target viewport: this is designed for ≥1280px wide. Is there a laptop or projector floor to hit?
