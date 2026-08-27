"""End-to-end check of the console against a running server.

Drives a real browser at a real backend: nothing here is mocked, so a pass
means the whole chain -- spaCy, the ranker, Elasticsearch, the boundary store,
the adapter, and the frontend -- agrees.

    python console/server.py &                  # or uvicorn, port 8000
    CONSOLE_URL=http://127.0.0.1:8000 python console/test_console_ui.py

Any browser console error fails the run. That is deliberate: a silent
`TypeError` in a render function leaves a pane blank and looks like "no data",
which is exactly the failure a screenshot does not catch.
"""

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get("CONSOLE_URL", "http://127.0.0.1:8077")
SHOTS = Path(os.environ.get("CONSOLE_SHOTS", "/tmp/console-shots"))

# Noise from the page that is not a defect. Kept explicit and short: a broad
# filter here would hide the errors this file exists to catch.
IGNORE_CONSOLE = (
    "favicon",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    # The imagery section below severs this host on purpose to exercise the
    # offline fallback; Chromium logs the aborted tile requests. What the
    # fallback actually does is asserted directly rather than by silence.
    "gibs.earthdata.nasa.gov",
)


class Checks:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def ok(self, condition, label, detail=""):
        if condition:
            self.passes += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append(f"{label} -- {detail}" if detail else label)
            print(f"  FAIL  {label}  {detail}")


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    c = Checks()
    errors = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 950})

        def open_menu(name):
            """Open one menu bar dropdown by its (theme-dependent) label.

            The labels are copy, so they change with the theme; the order does
            not. LOOK is first, CHROME second, MAP third, PANES fourth.
            """
            idx = {"look": 0, "chrome": 1, "map": 2, "panes": 3}[name]
            page.click(f"#menus .menu:nth-child({idx + 1}) .mtrig")
            page.wait_for_timeout(120)

        def note_console(m):
            # Chromium reports a blocked subresource as a bare "Failed to load
            # resource: net::ERR_FAILED" with the URL only in `location`, so
            # the filter has to look at both or the imagery section's
            # deliberate outage fails the run.
            where = (m.location or {}).get("url", "")
            if m.type == "error" and not any(
                    s in m.text or s in where for s in IGNORE_CONSOLE):
                errors.append(f"[{m.type}] {m.text} ({where})")

        page.on("console", note_console)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))

        print(f"\n-- loading {URL}")
        page.goto(URL, wait_until="networkidle", timeout=60_000)

        # The parse runs on load; wait for it to come to rest.
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        print("-- parse settled\n")

        # ---------------------------------------------------------- structure
        spans = page.query_selector_all("#doc-text [data-eid]")
        c.ok(len(spans) >= 10, "document renders entity spans",
             f"found {len(spans)}")

        states = {s.get_attribute("data-st") for s in spans}
        c.ok(not (states & {"pending", "scan"}),
             "every span settled after the run", f"states={sorted(states)}")
        c.ok(bool(states & {"ok", "sel"}), "spans reach a resolved state",
             f"states={sorted(states)}")

        # Offsets must reproduce the document text exactly, or highlights drift.
        doc_text = page.eval_on_selector("#doc-text", "el => el.textContent")
        api_text = page.evaluate(
            "fetch('/api/config').then(r=>r.json()).then(c=>c.corpus[0].text)")
        c.ok(doc_text == api_text,
             "rendered text is byte-identical to the source",
             f"{len(doc_text)} vs {len(api_text)} chars")

        # ------------------------------------------------------------ the map
        pins = page.evaluate(
            "document.querySelector('#scope').shadowRoot"
            ".querySelectorAll('[data-pid]').length")
        c.ok(pins >= 8, "map draws markers", f"found {pins}")

        polys = page.evaluate(
            "document.querySelector('#scope').shadowRoot"
            ".querySelectorAll('[data-boundary]').length")
        c.ok(polys >= 1, "map draws at least one boundary polygon",
             f"found {polys}")

        land = page.evaluate(
            "document.querySelector('#scope').shadowRoot"
            ".querySelector('svg').innerHTML.length")
        c.ok(land > 20_000, "basemap geometry rendered",
             f"svg innerHTML {land} chars")

        # ------------------------------------------------------ stage panels
        c.ok(page.query_selector("#stage-body .cand") is not None,
             "disambiguate panel shows candidates")
        c.ok(page.query_selector("#stage-body .note") is not None,
             "disambiguate panel shows a rationale or review note")

        # --------------------------------------------------- bidirectional sync
        # Addressed by selector, not by handle: the panes re-render, and a
        # handle held across a render is detached. (That churn was itself a
        # bug -- see the incremental paths in renderDoc/_renderOverlay.)
        eid = spans[3].get_attribute("data-eid")
        span_text = spans[3].text_content()
        page.click(f'#doc-text [data-eid="{eid}"]')
        page.wait_for_timeout(300)
        c.ok(page.eval_on_selector(f'#doc-text [data-eid="{eid}"]',
                                   "el => el.dataset.st") == "sel",
             "clicking a span selects it")
        heading = page.eval_on_selector("#stage-body h2", "el => el.textContent")
        c.ok(heading.strip() == span_text.strip(),
             "the disambiguate panel follows the selection",
             f"panel says {heading!r}, span was {span_text!r}")

        # The map's active marker must be the same *place*: this is the sync.
        # Markers are keyed on the gazetteer record, not on the span, because
        # four mentions of Ukraine are one marker; `data-place` on the span is
        # that key.
        place = page.eval_on_selector(f'#doc-text [data-eid="{eid}"]',
                                      "el => el.dataset.place")
        c.ok(page.evaluate(
            "id => { const sr = document.querySelector('#scope').shadowRoot;"
            "  const g = sr.querySelector(`[data-pid=\"${id}\"]`);"
            "  return g ? g.getAttribute('opacity') : null; }", place) == "1",
            "the map marker for the selected place goes active",
            f"place key was {place!r}")

        # ------------------------------------------------------- deduplication
        # Repeated mentions of one place are one marker and one polygon. The
        # corpus documents repeat plenty of names, so this is a real test on
        # them; the assertion is the invariant, not a fixed number.
        dedup = page.evaluate("""() => {
          const sr = document.querySelector('#scope').shadowRoot;
          const spans = [...document.querySelectorAll('#doc-text [data-place]')]
            .map(s => s.dataset.place).filter(Boolean);
          const pins = [...sr.querySelectorAll('[data-pid]')]
            .map(g => g.getAttribute('data-pid'));
          const bounds = [...sr.querySelectorAll('[data-boundary]')]
            .map(g => g.getAttribute('data-boundary'));
          return { places: [...new Set(spans)].length, mentions: spans.length,
                   pins: pins.length, uniquePins: [...new Set(pins)].length,
                   bounds: bounds.length, uniqueBounds: [...new Set(bounds)].length };
        }""")
        c.ok(dedup["pins"] == dedup["uniquePins"] == dedup["places"],
             "one marker per place, not per mention", str(dedup))
        c.ok(dedup["bounds"] == dedup["uniqueBounds"],
             "one boundary polygon per place", str(dedup))

        # Every other mention of the selected place is marked as linked to it,
        # which is what makes one marker legible as standing for several spans.
        kin = page.evaluate(
            "p => [...document.querySelectorAll('#doc-text [data-place]')]"
            "  .filter(s => s.dataset.place === p).map(s => s.dataset.st)", place)
        c.ok(kin.count("sel") == 1 and set(kin) <= {"sel", "link"},
             "the other mentions of the selected place light up with it",
             str(kin))

        # --------------------------------------------------- candidate ghosts
        # Rival candidates are drawn only while the pointer is on a candidate
        # row. An unprompted ring on the map reads as an alert, and warn is
        # already spoken for by the review flag.
        ghosts = ("() => (document.querySelector('#scope').shadowRoot.innerHTML"
                  "  .match(/OFF-FRAME|stroke-dasharray=\"2 2\"/g) || []).length")
        at_rest = page.evaluate(ghosts)
        rows = page.query_selector_all("#stage-body [data-cand]")
        summoned = at_rest
        if len(rows) > 1:
            rows[-1].hover()
            page.wait_for_timeout(250)
            summoned = page.evaluate(ghosts)
            page.mouse.move(4, 4)
            page.wait_for_timeout(250)
        c.ok(at_rest == 0, "no candidate ghosts on the resting map", f"{at_rest} drawn")
        c.ok(len(rows) < 2 or summoned > 0 or page.evaluate(ghosts) == 0,
             "hovering a candidate row draws it on the map",
             f"rest={at_rest} hover={summoned}")
        c.ok(page.evaluate(ghosts) == 0,
             "the ghost goes away when the pointer leaves the row")

        # Hovering a different span must move the highlight without rebuilding
        # the document -- the node identity is the assertion.
        other = spans[5].get_attribute("data-eid")
        page.evaluate("id => document.querySelector('#doc-text [data-eid=\"'+id+'\"]')"
                      ".__probe = 'marked'", other)
        page.hover(f'#doc-text [data-eid="{other}"]')
        page.wait_for_timeout(250)
        c.ok(page.eval_on_selector(f'#doc-text [data-eid="{other}"]',
                                   "el => el.dataset.st") == "sel",
             "hovering a span highlights it")
        c.ok(page.eval_on_selector(f'#doc-text [data-eid="{other}"]',
                                   "el => el.__probe") == "marked",
             "hover does not rebuild the document DOM")

        # ------------------------------------------------------- stage cycling
        for stage, probe in (("ingest", ".tbl"), ("parse", ".tbl .row"),
                             ("resolve", ".kv"), ("export", ".pre")):
            page.click(f'[data-stage="{stage}"]')
            page.wait_for_timeout(200)
            c.ok(page.query_selector(f"#stage-body {probe}") is not None,
                 f"{stage.upper()} panel renders")
            page.screenshot(path=str(SHOTS / f"stage-{stage}.png"))

        # The panel opens on Mordecai's own output, not this console's
        # reshaping of it -- `export.defaultFormat` decides, and it used to be
        # ignored in favour of a hard-coded geojson.
        page.click('[data-stage="export"]')
        page.wait_for_timeout(200)
        c.ok(page.eval_on_selector("#stage-body .seg4 button.on",
                                   "el => el.dataset.fmt") == "raw",
             "the export panel opens on the raw Mordecai result")

        # GeoJSON must still be parseable, and must use polygons where it has
        # them.
        page.click('[data-fmt="geojson"]')
        page.wait_for_timeout(200)
        gj = page.eval_on_selector("#stage-body .pre", "el => el.textContent")
        try:
            import json
            parsed = json.loads(gj)
            kinds = {f["geometry"]["type"] for f in parsed["features"]}
            c.ok(parsed["type"] == "FeatureCollection", "GeoJSON export is valid")
            c.ok(bool(kinds & {"Polygon", "MultiPolygon"}),
                 "GeoJSON export uses boundary polygons where available",
                 f"geometry types: {sorted(kinds)}")
        except Exception as exc:
            c.ok(False, "GeoJSON export is valid", str(exc))

        # ------------------------------------------------------------ map mode
        page.click('[data-mode="wire"]')
        page.wait_for_timeout(400)
        page.click('[data-stage="dis"]')
        page.wait_for_timeout(200)
        page.screenshot(path=str(SHOTS / "vector-mode.png"))
        c.ok(page.evaluate(
            "document.querySelector('#scope').shadowRoot"
            ".querySelectorAll('[data-pid]').length") >= 8,
            "markers survive a basemap mode switch")
        page.click('[data-mode="sat"]')
        page.wait_for_timeout(400)

        # ------------------------------------------------------------ imagery
        # The one layer that is not vendored. Tiles must land on the same
        # projection as everything else -- a marker may not move a pixel
        # between the procedural relief and the real thing.
        def pin_xy():
            return page.evaluate("""() => {
              const g = document.querySelector('#scope').shadowRoot
                          .querySelector('[data-pid]');
              const r = g.querySelector('circle[fill="transparent"]')
                         .getBoundingClientRect();
              return [r.x + r.width / 2, r.y + r.height / 2];
            }""")

        relief_xy = pin_xy()
        page.click('[data-mode="imagery"]')
        page.wait_for_timeout(3500)
        tiles = page.evaluate("""() => {
          const sc = document.querySelector('#scope');
          const im = [...sc.shadowRoot.querySelectorAll('image[data-tile]')];
          return { n: im.length, loaded: im.filter(i => i.getAttribute('href')).length,
                   z: sc.dataset.tilezoom };
        }""")
        c.ok(tiles["n"] > 0 and tiles["loaded"] == tiles["n"],
             "satellite tiles load", str(tiles))
        c.ok(max(abs(a - b) for a, b in zip(relief_xy, pin_xy())) < 1.5,
             "tiles register with the projection the markers use",
             f"{relief_xy} vs {pin_xy()}")
        badges = page.inner_text("#map-badges")
        c.ok("GIBS" in badges and f"Z{tiles['z']}" in badges,
             "the map credits the imagery source and names the tile level",
             badges.replace("\n", " · "))
        page.screenshot(path=str(SHOTS / "imagery.png"))

        # Cut the tile host and start over: the console must fall back to the
        # relief and say so, not sit on a black rectangle. Everything else here
        # is vendored precisely so a room with no network still works. The
        # reload matters -- tiles already fetched stay in the component's cache
        # and would answer without touching the network, which is right, and
        # would test nothing.
        page.route("**gibs.earthdata.nasa.gov/**", lambda r: r.abort())
        page.reload(wait_until="networkidle", timeout=60_000)
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        page.click('[data-mode="imagery"]')
        page.wait_for_timeout(3000)
        c.ok("RASTER UNREACHABLE" in page.inner_text("#map-badges"),
             "unreachable tiles fall back to the procedural relief",
             page.inner_text("#map-badges").replace("\n", " · "))
        c.ok(page.evaluate(
             "() => document.querySelector('#scope').shadowRoot"
             "  .querySelectorAll('image[data-tile]').length") == 0,
             "the fallback draws no empty tile frames")
        c.ok(page.evaluate("() => document.querySelector('#scope').shadowRoot"
                           "  .querySelectorAll('[data-pid]').length") > 0,
             "markers survive the fallback")

        # And re-selecting it retries, rather than latching off for the
        # session: the tiles that failed are dropped from the cache so the
        # retry actually goes back to the network.
        page.unroute("**gibs.earthdata.nasa.gov/**")
        page.click('[data-mode="sat"]')
        page.wait_for_timeout(200)
        page.click('[data-mode="imagery"]')
        page.wait_for_timeout(4000)
        c.ok(page.evaluate(
             "() => [...document.querySelector('#scope').shadowRoot"
             "  .querySelectorAll('image[data-tile]')]"
             "  .filter(i => i.getAttribute('href')).length") > 0,
             "selecting satellite again retries after a failure")
        page.click('[data-mode="sat"]')
        page.wait_for_timeout(300)

        # ------------------------------------------------------- pasted text
        page.fill("#paste", "Heavy fighting was reported in Aleppo and near "
                            "Idlib, in northern Syria, as well as in Homs.")
        page.click("#run")
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        pasted_spans = page.query_selector_all("#doc-text [data-eid]")
        c.ok(len(pasted_spans) >= 3, "pasted text parses",
             f"found {len(pasted_spans)} spans")
        page.screenshot(path=str(SHOTS / "pasted.png"), full_page=False)

        # ------------------------------------------- framing a scattered country
        # A shape's bounding box is measured on a map cut at the antimeridian,
        # so France's -- Wallis at 176W, Guadeloupe at 61W -- spans 350 degrees,
        # and Russia's, the United States', Fiji's and New Zealand's span the
        # full 360. Fitting the view to that shows the whole world, on which
        # every polygon is a few pixels wide and the boundary layer looks
        # broken. The server sends `focus_bbox` for this, and the fit uses it.
        #
        # Asserted through the graticule spacing, which is chosen from the
        # visible span -- about ten to fourteen lines across the frame -- and is
        # therefore a direct read on how much world is showing: a France and
        # Germany frame is ~20 degrees wide and steps to 2 or 5, while the
        # whole globe steps to 30.
        page.fill("#paste", "France and Germany issued a joint statement.")
        page.click("#run")
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        step = page.evaluate(
            "() => Number(document.querySelector('#scope').dataset.graticule)")
        c.ok(step <= 5, "a country with overseas territory does not frame the globe",
             f"graticule stepped to {step} degrees")
        page.screenshot(path=str(SHOTS / "scattered-country.png"))

        # ---------------------------------------------------------- raw export
        # The format that hands back what `geoparse_doc` actually returned,
        # untrimmed, rather than this console's own reshaping of it.
        page.click('[data-stage="export"]')
        page.wait_for_timeout(250)
        raw_tab = page.query_selector('[data-fmt="raw"]')
        c.ok(raw_tab is not None, "the export panel offers the raw result")
        if raw_tab:
            raw_tab.click()
            page.wait_for_timeout(300)
            try:
                doc = json.loads(page.inner_text(".pre"))
            except ValueError as exc:
                doc = None
                c.ok(False, "raw export is valid JSON", str(exc))
            if doc is not None:
                ents = doc.get("geolocated_ents") or []
                first = (ents[0].get("candidates") or [{}])[0] if ents else {}
                c.ok("geolocated_ents" in doc,
                     "raw export is Mordecai's own structure", str(list(doc))[:120])
                # The point of the format: `trim=False` leaves the ranker's
                # enrichment features on every candidate, and trimming them
                # here would remove the only reason to export it.
                c.ok(len(first) > 20,
                     "raw export keeps the ranker features on candidates",
                     f"{len(first)} keys on the first candidate")
        page.click('[data-stage="dis"]')
        page.wait_for_timeout(200)

        # ------------------------------------------------------------- corpus
        page.click('[data-doc="archive"]')
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        c.ok(len(page.query_selector_all("#doc-text [data-eid]")) >= 3,
             "the uppercase OCR document parses")
        page.screenshot(path=str(SHOTS / "archive.png"))

        page.click('[data-doc="wire"]')
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        page.screenshot(path=str(SHOTS / "console.png"))

        # -------------------------------------------------- marker stability
        # Hovering a marker must not re-solve the label packing. It used to:
        # the active pin sorted first and grew, so every plate on the map could
        # jump to a different corner -- and because the plates were hit targets
        # themselves, one landing under the pointer made a different pin active
        # and did it again. In a cluster the map never came to rest.
        plates = """() => {
          const sr = document.querySelector('#scope').shadowRoot;
          const out = {};
          sr.querySelectorAll('[data-pid]').forEach(g => {
            const t = g.querySelector('g[transform^="translate"]');
            out[g.getAttribute('data-pid')] = t ? t.getAttribute('transform') : '';
          });
          return out;
        }"""
        markers = page.evaluate("""() =>
          [...document.querySelector('#scope').shadowRoot
             .querySelectorAll('[data-pid]')].map(g => {
            const c = g.querySelector('circle[fill="transparent"]');
            const r = c.getBoundingClientRect();
            return { id: g.getAttribute('data-pid'),
                     x: r.x + r.width / 2, y: r.y + r.height / 2 };
          })""")
        c.ok(len(markers) >= 2 and all(m["id"] for m in markers),
             "every marker carries a hit disc", f"{len(markers)} markers")

        if len(markers) >= 2:
            snaps = []
            for m in markers[:3]:
                page.mouse.move(m["x"], m["y"])
                page.wait_for_timeout(180)
                snaps.append((m["id"], page.evaluate(plates)))
            moved = sorted({
                pid
                for (a_id, a), (b_id, b) in zip(snaps, snaps[1:])
                for pid in set(a) & set(b)
                # The pin being pointed at is allowed to gain a plate; nothing
                # else may move.
                if pid not in (a_id, b_id) and a[pid] != b[pid]
            })
            c.ok(not moved, "hovering a marker does not move other labels",
                 f"moved: {moved}")
            page.mouse.move(m["x"], m["y"] - 220)  # off the markers
            page.wait_for_timeout(120)

        # ---------------------------------------------------------- menu bar
        # The menus are built from a spec rather than written out, so this
        # checks the machinery -- a trigger opens exactly one dropdown, a click
        # elsewhere closes it -- rather than any particular menu's contents.
        triggers = page.query_selector_all("#menus .mtrig")
        c.ok(len(triggers) >= 3, "the menu bar offers its menus",
             f"{len(triggers)} menus")
        open_menu("look")
        c.ok(page.eval_on_selector_all(
             "#menus .menu.open", "els => els.length") == 1,
             "one menu opens at a time")
        page.screenshot(path=str(SHOTS / "menu-look.png"))
        page.mouse.click(900, 500)
        page.wait_for_timeout(150)
        c.ok(page.eval_on_selector_all(
             "#menus .menu.open", "els => els.length") == 0,
             "clicking away closes the menu")

        # ----------------------------------------------------------- palette
        # Every palette the current theme defines is selectable from the LOOK
        # menu, and the map repaints from the same custom properties the rest
        # of the screen uses.
        open_menu("look")
        swatches = page.query_selector_all("#palrow [data-pal]")
        c.ok(len(swatches) >= 2, "the menu offers the theme's palettes",
             f"{len(swatches)} swatches")
        if swatches:
            page.click('[data-pal="olive"]')
            page.wait_for_timeout(400)
            acc = page.evaluate(
                "getComputedStyle(document.documentElement)"
                "  .getPropertyValue('--acc').trim()")
            c.ok(page.evaluate("document.documentElement.dataset.pal") == "olive"
                 and acc.lower() == "#c8d074",
                 "the olive palette applies", f"--acc={acc}")
            c.ok(page.evaluate(
                 "document.querySelector('#scope').shadowRoot.innerHTML"
                 "  .toLowerCase().includes('#c8d074')"),
                 "the map repaints in the chosen palette")
            page.screenshot(path=str(SHOTS / "palette-olive.png"))
            open_menu("look")
            page.click('[data-pal="amber"]')
            page.wait_for_timeout(400)

        # ------------------------------------------------------------- themes
        # The whole point of the second look: same layout, same panes, same
        # markers -- different palette, different words, no ambient motion.
        # The words matter as much as the colours, so this asserts on both.
        before_panes = page.evaluate(
            "[...document.querySelectorAll('[data-el]')].map(e => e.dataset.el).join()")
        open_menu("look")
        page.click("#menus .mitem:has-text('Field brief')")
        page.wait_for_function(
            "document.documentElement.dataset.theme === 'field'", timeout=15000)
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'Done'",
            timeout=90000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(SHOTS / "theme-field.png"))

        c.ok(page.evaluate(
             "[...document.querySelectorAll('[data-el]')].map(e => e.dataset.el).join()")
             == before_panes,
             "the field theme keeps the same panes")
        bg = page.evaluate(
            "getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()")
        c.ok(bg.lower() == "#f2efe6", "the field palette applies", f"--bg={bg}")
        c.ok(page.evaluate("document.querySelector('#lab-pipeline').textContent")
             == "How it works",
             "the field theme relabels the rail")
        c.ok("BOOTING" not in page.content() and "DISAMBIGUATE" not in page.content(),
             "the ops vocabulary is gone from the field theme")
        c.ok(page.evaluate(
             "document.querySelector('#scope').shadowRoot.innerHTML"
             "  .includes('terrcalm')"),
             "the map draws its calm relief")
        c.ok(not page.evaluate(
             "document.querySelector('#scope').shadowRoot.innerHTML"
             "  .includes('class=\\'sweep\\'') ||"
             "document.querySelector('#scope').shadowRoot.innerHTML"
             "  .includes('class=\"sweep\"')"),
             "the field theme stops the map sweep")
        c.ok(page.evaluate(
             "getComputedStyle(document.querySelector('.scanlines')).display") == "none",
             "the field theme drops the scanlines")

        # The corpus is part of the theme, so switching looks must have loaded
        # a document that belongs to the new one -- and parsed it for real.
        c.ok(page.evaluate("document.querySelectorAll('#corpus .doc-card').length") == 1
             and "Kostiantynivka" in page.evaluate(
                 "document.querySelector('#doc-text').textContent"),
             "the field theme loads its own sample report")
        # The document repeats Ukraine and Russia several times each, which is
        # the case the mention/place split exists for: many spans, one marker.
        c.ok(page.evaluate("document.querySelectorAll('#doc-text [data-eid]').length")
             > page.evaluate(
                 "document.querySelector('#scope').shadowRoot"
                 "  .querySelectorAll('[data-pid]').length"),
             "repeated mentions collapse to fewer markers")
        c.ok(page.evaluate("document.querySelectorAll('#doc-text [data-eid]').length") > 3,
             "the field theme's report resolves place names")

        pals = page.query_selector_all("#palrow [data-pal]")
        open_menu("look")
        pals = page.query_selector_all("#palrow [data-pal]")
        c.ok(all(page.evaluate("e => e.dataset.pal", el) in ("field", "slate")
                 for el in pals) and len(pals) == 2,
             "the palettes offered are the field theme's own",
             ",".join(page.evaluate("e => e.dataset.pal", el) for el in pals))
        page.click('[data-pal="slate"]')
        page.wait_for_timeout(400)
        page.screenshot(path=str(SHOTS / "theme-field-slate.png"))
        open_menu("look")
        page.click('[data-pal="field"]')
        page.wait_for_timeout(300)

        # Back to ops, which the rest of the run assumes.
        open_menu("look")
        page.click("#menus .mitem:has-text('Operations')")
        page.wait_for_function(
            "document.documentElement.dataset.theme === 'ops'", timeout=15000)
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90000)
        c.ok(page.evaluate("document.querySelector('#lab-pipeline').textContent")
             == "PIPELINE",
             "switching back restores the ops vocabulary")
        page.screenshot(path=str(SHOTS / "theme-ops.png"))

        # --------------------------------------------------------- splitters
        width = "() => document.querySelector('[data-el=\"doc\"]').getBoundingClientRect().width"
        before = page.evaluate(width)
        box = page.query_selector("#gut-col").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 - 140,
                        box["y"] + box["height"] / 2, steps=10)
        page.mouse.up()
        page.wait_for_timeout(300)
        after = page.evaluate(width)
        c.ok(abs((before - after) - 140) < 12,
             "dragging the column seam resizes the document pane",
             f"{before:.0f} -> {after:.0f}")
        page.screenshot(path=str(SHOTS / "resized.png"))

        page.dblclick("#gut-col")
        page.wait_for_timeout(300)
        c.ok(abs(page.evaluate(width) - before) < 2,
             "double-clicking the seam restores the default split",
             f"{page.evaluate(width):.0f} vs {before:.0f}")

        # ------------------------------------------------------------ layouts
        # Driven through the menu rather than by setting the attribute, so the
        # menu action and the stylesheet are checked together.
        for layout, label in (("theater", "THEATER"), ("split", "SPLIT")):
            open_menu("panes")
            page.click(f"#menus .mitem:has-text('{label}')")
            page.wait_for_timeout(500)
            c.ok(page.evaluate("document.documentElement.dataset.layout") == layout,
                 f"the {layout} layout applies from the menu")
            page.screenshot(path=str(SHOTS / f"layout-{layout}.png"))
        # The body must never scroll sideways at the design's target width.
        overflow = page.evaluate(
            "document.body.scrollWidth - document.body.clientWidth")
        c.ok(overflow <= 0, "no horizontal overflow at 1600px",
             f"{overflow}px over")

        # ------------------------------------------------------- entry points
        # `/demo` and `/console` are the same document; the URL picks the look.
        # The check that matters is precedence: localStorage holds "ops" by now
        # -- the theme menu was used above -- and the URL has to beat it, or a
        # link handed to someone who has opened the console before lands them
        # on the wrong one.
        c.ok(page.evaluate("localStorage.getItem('mordecai.theme')") == "ops",
             "the remembered theme is the one the menu last chose")

        page.goto(f"{URL}/demo", wait_until="networkidle", timeout=60_000)
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'Done'",
            timeout=90_000)
        c.ok(page.evaluate("document.documentElement.dataset.theme") == "field",
             "/demo opens the field theme over the remembered one")

        # The report names Russia five times, whose polygon reaches 180E while
        # every marker sits between 29E and 100E. Fitting to the union puts
        # more than half the map east of anything the document mentions, so the
        # boundary contribution is capped -- see `_fitGeometry`. Measured as the
        # share of the frame's width lying east of the easternmost marker.
        dead = page.evaluate('''() => {
          const s = document.querySelector('#scope').shadowRoot;
          const xs = [...s.querySelectorAll('[data-pid] circle')]
            .map(c => +c.getAttribute('cx')).filter(Number.isFinite);
          const w = s.querySelector('svg').viewBox.baseVal.width;
          return xs.length ? (w - Math.max(...xs)) / w : 1;
        }''')
        c.ok(dead < 0.3, "an oversized polygon does not take over the frame",
             f"{dead:.0%} of the frame is east of every marker")
        c.ok("reliefweb.int" in page.evaluate(
             "document.querySelector('#doc-meta').textContent"),
             "/demo cites the report's source")
        page.screenshot(path=str(SHOTS / "route-demo.png"))

        # Switching looks from a routed URL rewrites the path, so a link copied
        # mid-demo opens what was on screen.
        open_menu("look")
        page.click("#menus .mitem:has-text('Operations')")
        page.wait_for_function(
            "document.documentElement.dataset.theme === 'ops'", timeout=15_000)
        c.ok(page.evaluate("location.pathname") == "/console",
             "switching looks rewrites the URL to match",
             page.evaluate("location.pathname"))

        page.goto(f"{URL}/console", wait_until="networkidle", timeout=60_000)
        page.wait_for_function(
            "document.querySelector('#statusword').textContent === 'RESOLVED'",
            timeout=90_000)
        c.ok(page.evaluate("document.documentElement.dataset.theme") == "ops",
             "/console opens the ops theme")

        c.ok(not errors, "no browser console errors",
             "; ".join(errors[:4]))

        browser.close()

    print(f"\n{c.passes} passed, {len(c.failures)} failed")
    for f in c.failures:
        print(f"  - {f}")
    print(f"screenshots in {SHOTS}")
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
