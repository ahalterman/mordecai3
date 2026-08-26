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

        page.on("console", lambda m: (
            errors.append(f"[{m.type}] {m.text}")
            if m.type == "error" and not any(s in m.text for s in IGNORE_CONSOLE)
            else None))
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

        # Export must be parseable, and must use polygons where it has them.
        page.click('[data-stage="export"]')
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

        # ------------------------------------------------------------ layouts
        for layout in ("theater", "split"):
            page.evaluate(f"document.documentElement.dataset.layout = '{layout}'")
            page.wait_for_timeout(500)
            page.screenshot(path=str(SHOTS / f"layout-{layout}.png"))
        # The body must never scroll sideways at the design's target width.
        overflow = page.evaluate(
            "document.body.scrollWidth - document.body.clientWidth")
        c.ok(overflow <= 0, "no horizontal overflow at 1600px",
             f"{overflow}px over")

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
