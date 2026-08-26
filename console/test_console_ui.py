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

        # The map's active marker must be the same entity: this is the sync.
        c.ok(page.evaluate(
            "id => { const sr = document.querySelector('#scope').shadowRoot;"
            "  const g = sr.querySelector(`[data-pid=\"${id}\"]`);"
            "  return g ? g.getAttribute('opacity') : null; }", eid) == "1",
            "the map marker for the selected entity goes active")

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
