/* geo-scope -- the map surface for the geoparse console.
 *
 * Ported from the design handoff's prototype, which the README asks to port
 * rather than reinvent: the Mercator fit, the two-level pan/zoom transform and
 * the SVG terrain filters are the real thing and tedious to re-derive. What is
 * different here:
 *
 *   - the basemap and d3/topojson are loaded from local vendored files, not a
 *     CDN. A demo that needs the internet is a demo that fails in the one room
 *     without it;
 *   - a boundary layer. Mordecai resolves administrative units to polygons via
 *     geoBoundaries, and drawing a province as a point when its shape is known
 *     throws away the answer. Polygons render under the pins so the marker
 *     stays the click target;
 *   - a boundary whose gazetteer join was uncertain is drawn dashed and
 *     labelled with its match score, rather than drawn as if it were certain;
 *   - navigation limits, graticule spacing, sweep period and fit padding come
 *     from the console config instead of being baked in;
 *   - `prefers-reduced-motion` drops the sweep, the pulse and the blink.
 *
 * Events out: `pinclick`, `pinhover`, `viewchange` (detail = zoom factor).
 */
(function () {
  const DEFAULTS = {
    atlas: '/vendor/countries-110m.json',
    minZoom: 0.6,
    maxZoom: 28,
    fitPaddingRatio: 0.26,
    graticuleStepDeg: 2,
    graticuleMajorStepDeg: 10,
    sweepSeconds: 9,
    pan: true,
    zoom: true,
    boundaryFillOpacity: 0.16,
    boundaryStrokeWidth: 1.1,
    weakMatchBelow: 0.95,
  };

  let atlasPromise = null;

  function loadAtlas(url) {
    if (!atlasPromise) {
      atlasPromise = fetch(url)
        .then(r => {
          if (!r.ok) throw new Error(`basemap ${url}: HTTP ${r.status}`);
          return r.json();
        })
        .then(t => ({
          land: window.topojson.feature(t, t.objects.countries),
          borders: window.topojson.mesh(t, t.objects.countries, (a, b) => a !== b),
        }));
    }
    return atlasPromise;
  }

  function waitLibs() {
    return new Promise((res, rej) => {
      let waited = 0;
      const tick = () => {
        if (window.d3 && window.topojson) return res();
        // Fail loudly rather than spinning forever: a missing vendor file is a
        // deployment mistake, and a map stuck on "ACQUIRING TILESET" hides it.
        if ((waited += 60) > 10000) return rej(new Error('d3 / topojson never loaded'));
        setTimeout(tick, 60);
      };
      tick();
    });
  }

  const esc = s => String(s).replace(/[<>&"]/g, c => (
    { '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));

  const reduceMotion = () =>
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  class GeoScope extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this._pins = [];
      this._ghosts = [];
      this._mode = 'sat';
      this._active = null;
      this._w = 0; this._h = 0;
      this._ready = false;
      this._seed = 11;
      // committed transform
      this._zk = 1; this._zx = 0; this._zy = 0;
      // live (in-gesture) transform, folded into the committed one on release
      this._lk = 1; this._lx = 0; this._ly = 0;
      this._view = '';   // which places -- changing it resets pan and zoom
      this._fit = '';    // what the projection depends on
      this._paint = '';  // what the markers depend on
      this._gkey = '';
      this._moved = 0;
      this._painted = false;
      this._last = null;
      this._opts = { ...DEFAULTS };
    }

    configure(opts) {
      this._opts = { ...this._opts, ...(opts || {}) };
      if (this._ready) this.render();
    }

    connectedCallback() {
      const sweep = this._opts.sweepSeconds;
      this.shadowRoot.innerHTML = `<style>
        :host{display:block;width:100%;height:100%;position:relative;overflow:hidden;background:#06070a;
          cursor:grab;font-family:'JetBrains Mono',ui-monospace,monospace;contain:layout paint}
        svg{position:absolute;inset:0;width:100%;height:100%;display:block}
        .lbl{font:600 8.5px 'JetBrains Mono',ui-monospace,monospace;letter-spacing:.08em}
        .sweep{animation:sw ${sweep}s linear infinite}
        @keyframes sw{0%{transform:translateX(-14%)}100%{transform:translateX(114%)}}
        .pulse{animation:pl 1.9s ease-out infinite}
        @keyframes pl{0%{r:6;opacity:.85}100%{r:26;opacity:0}}
        .blink{animation:bk 2.4s steps(1,end) infinite}
        @keyframes bk{0%,88%{opacity:1}89%,100%{opacity:.35}}
        @media (prefers-reduced-motion: reduce){
          .sweep{display:none}
          .pulse,.blink{animation:none}
        }
        .boot{position:absolute;inset:0;display:grid;place-items:center;color:var(--acc,#f0a13c);
          font:600 10px 'JetBrains Mono',monospace;letter-spacing:.28em;background:#06070a;
          text-align:center;padding:0 20px;line-height:2}
      </style><div class="boot">LOADING BASEMAP…</div>`;

      this._ro = new ResizeObserver(() => {
        const r = this.getBoundingClientRect();
        if (Math.abs(r.width - this._w) > 1 || Math.abs(r.height - this._h) > 1) {
          this._w = r.width; this._h = r.height; this.render();
        }
      });
      this._ro.observe(this);
      this._bindNav();

      waitLibs()
        .then(() => loadAtlas(this._opts.atlas))
        .then(a => {
          this._atlas = a; this._ready = true;
          const r = this.getBoundingClientRect();
          this._w = r.width; this._h = r.height;
          this.render();
        })
        .catch(err => {
          const boot = this.shadowRoot.querySelector('.boot');
          if (boot) boot.textContent = `BASEMAP UNAVAILABLE — ${err.message}`;
          console.error('geo-scope:', err);
        });
    }

    disconnectedCallback() { if (this._ro) this._ro.disconnect(); }

    // ------------------------------------------------------------ transform

    _applyLive() {
      const g = this.shadowRoot.querySelector('.scene');
      if (g) {
        g.setAttribute('transform',
          `translate(${this._lx.toFixed(2)},${this._ly.toFixed(2)}) scale(${this._lk.toFixed(4)})`);
      }
    }

    _commit() {
      this._zk *= this._lk;
      this._zx = this._zx * this._lk + this._lx;
      this._zy = this._zy * this._lk + this._ly;
      this._lk = 1; this._lx = 0; this._ly = 0;
      this.dispatchEvent(new CustomEvent('viewchange',
        { detail: this._zk, bubbles: true, composed: true }));
      this.render();
    }

    resetView() {
      this._zk = 1; this._zx = 0; this._zy = 0;
      this._lk = 1; this._lx = 0; this._ly = 0;
      this.dispatchEvent(new CustomEvent('viewchange',
        { detail: 1, bubbles: true, composed: true }));
      this.render();
    }

    _bindNav() {
      this.addEventListener('wheel', ev => {
        if (!this._opts.zoom) return;
        ev.preventDefault();
        const r = this.getBoundingClientRect();
        const px = ev.clientX - r.left, py = ev.clientY - r.top;
        const f = Math.exp(-ev.deltaY * 0.0018);
        const nk = this._zk * this._lk * f;
        if (nk < this._opts.minZoom || nk > this._opts.maxZoom) return;
        // Zoom about the cursor: scale, then shift so the point under the
        // pointer stays under the pointer.
        this._lk *= f;
        this._lx = this._lx * f + px * (1 - f);
        this._ly = this._ly * f + py * (1 - f);
        this._applyLive();
        clearTimeout(this._wt);
        // Debounced, because committing re-rasterises the terrain filters.
        this._wt = setTimeout(() => this._commit(), 190);
      }, { passive: false });

      this.addEventListener('mousedown', ev => {
        if (ev.button !== 0 || !this._opts.pan) return;
        ev.preventDefault();
        this._moved = 0;
        this.style.cursor = 'grabbing';
        const mv = e => {
          this._moved += Math.abs(e.movementX) + Math.abs(e.movementY);
          this._lx += e.movementX; this._ly += e.movementY;
          this._applyLive();
        };
        const up = () => {
          window.removeEventListener('mousemove', mv);
          window.removeEventListener('mouseup', up);
          this.style.cursor = 'grab';
          // Only a real drag commits -- and the same threshold suppresses the
          // click, so dragging across a pin does not select it.
          if (this._moved > 3) this._commit();
        };
        window.addEventListener('mousemove', mv);
        window.addEventListener('mouseup', up);
      });

      this.addEventListener('dblclick', () => this.resetView());
    }

    // ---------------------------------------------------------------- scene

    /** Take a new scene, and repaint no more of it than actually changed.
     *
     * Three levels, because they cost wildly different amounts. A full
     * `render()` re-rasterises feTurbulence and feDiffuseLighting across the
     * whole terrain, which is by far the most expensive thing on this screen
     * and is unaffected by anything below the projection:
     *
     *   `view`  -- which places are on screen. Refit, and reset any pan and
     *              zoom: panning around and then switching documents should
     *              not leave you lost.
     *   `fit`   -- the above plus which places carry a polygon, since a
     *              boundary's extent feeds the projection. Refit, but keep
     *              the view where the reader put it.
     *   `paint` -- everything else the markers depend on: statuses, mention
     *              tallies, ghosts, the active place. Repaint the overlay
     *              groups and leave the terrain alone.
     *
     * The reveal animation only ever moves `paint`: it flips pins from
     * pending to resolved, one after another, at a rate the terrain could not
     * possibly keep up with. Before this split it forced a full re-rasterise
     * per step, which is what put a ceiling on how fast the reveal could run
     * -- so this is the change that lets it be quick.
     */
    setScene(o) {
      let refit = false, repaint = false;
      if (o.pins) {
        const view = o.pins.map(p => p.id).join('|');
        const fit = o.pins.map(p => `${p.id}:${p.boundary ? 'b' : ''}`).join('|');
        const paint = o.pins.map(p => `${p.id}:${p.status}:${p.count || 1}`).join('|');
        if (view !== this._view) { this._zk = 1; this._zx = 0; this._zy = 0; }
        if (fit !== this._fit) refit = true;
        else if (paint !== this._paint) repaint = true;
        this._view = view; this._fit = fit; this._paint = paint;
        this._pins = o.pins;
      }
      if ('ghosts' in o) {
        const gk = (o.ghosts || []).map(g => `${g.lat},${g.lon}`).join('|');
        if (gk !== this._gkey) { this._gkey = gk; repaint = true; }
        this._ghosts = o.ghosts || [];
      }
      // The basemap and the terrain seed are the terrain, so these do need it.
      if (o.mode && o.mode !== this._mode) { this._mode = o.mode; refit = true; }
      if (o.seed && o.seed !== this._seed) { this._seed = o.seed; refit = true; }
      if ('active' in o && o.active !== this._active) {
        this._active = o.active;
        repaint = true;
      }
      if (refit || !this._painted) this.render();
      else if (repaint) this._renderOverlay();
    }

    refresh() { this.render(); }

    _theme() {
      const cs = getComputedStyle(this);
      const g = (n, f) => (cs.getPropertyValue(n) || '').trim() || f;
      return {
        acc: g('--acc', '#f0a13c'),
        alt: g('--alt', '#7fa8bd'),
        warn: g('--warn', '#e2452c'),
        land: g('--map-land', '#b9ad8e'),
        sea: g('--map-sea', '#0a0d12'),
        tint: g('--map-tint', '#5d5124'),
      };
    }

    /** Everything the view must contain: the pins, plus any boundary extents.
     *
     * Fitting to points alone crops a country polygon at the first zoom level,
     * because a country's marker sits at its centroid while its shape runs
     * hundreds of kilometres past the frame.
     *
     * `focus_bbox` rather than `bbox`, and the difference is the whole reason
     * boundaries looked broken: a shape's true bounding box is measured on a
     * map cut at the antimeridian, so Russia's, the United States', Fiji's and
     * New Zealand's all come back spanning the entire globe, and France's and
     * the United Kingdom's nearly do on the strength of their overseas
     * territories. Fitting to that shows the world, on which every polygon is
     * a handful of pixels -- which is indistinguishable from the boundary
     * layer not working. The server sends the extent of the part of the shape
     * the place's own coordinate sits in alongside; see `_focus_bbox` in
     * `boundaries.py`. Falls back to `bbox` for a store built before that
     * field existed.
     */
    _fitGeometry(pins) {
      const coords = [];
      pins.forEach(p => {
        if (p.lat == null || p.lon == null) return;
        coords.push([p.lon, p.lat]);
        const bd = p.boundary;
        const b = bd && (bd.focus_bbox || bd.bbox);
        if (b) coords.push([b[0], b[1]], [b[2], b[3]]);
      });
      return coords.length ? { type: 'MultiPoint', coordinates: coords } : null;
    }

    /** Graticule spacing for the frame currently on screen.
     *
     * The configured value is the *finest* spacing, not the only one. Drawn
     * literally at 2 degrees, a frame showing half the globe -- which any
     * document spanning two continents produces -- puts ninety labelled
     * parallels down its left edge, where they merge into a solid bar. So step
     * up a ladder until the wider axis carries a readable number of lines.
     * Never finer than the config asks for, so the look at the default fit is
     * still the designed one.
     */
    _graticuleStep(proj, W, H, finest) {
      const sw = proj.invert([0, H]), ne = proj.invert([W, 0]);
      if (!sw || !ne) return finest;
      const span = Math.max(Math.abs(ne[0] - sw[0]), Math.abs(ne[1] - sw[1]));
      const LADDER = [1, 2, 5, 10, 15, 20, 30];
      return LADDER.find(v => v >= finest && span / v <= 14)
        || LADDER[LADDER.length - 1];
    }

    render() {
      if (!this._ready || !this._w || !this._h) return;
      const d3 = window.d3, W = this._w, H = this._h, t = this._theme();
      const o = this._opts;
      const pins = this._pins.filter(p => p.lat != null);

      const proj = d3.geoMercator();
      const pad = Math.min(W, H) * o.fitPaddingRatio;
      const fit = this._fitGeometry(pins);
      proj.fitExtent([[pad, pad], [W - pad, H - pad]], fit || { type: 'Sphere' });

      // Fold the committed pan/zoom into the projection, so the expensive
      // filters rasterise once per committed change rather than per frame.
      const bs0 = proj.scale(), bt0 = proj.translate();
      proj.scale(bs0 * this._zk)
        .translate([bt0[0] * this._zk + this._zx, bt0[1] * this._zk + this._zy]);

      const path = d3.geoPath(proj);
      const landD = path(this._atlas.land) || '';
      const bordD = path(this._atlas.borders) || '';
      const step = this._graticuleStep(proj, W, H, o.graticuleStepDeg);
      const majStep = Math.max(o.graticuleMajorStepDeg, step * 5);
      // So the console's badge can name the spacing actually drawn rather
      // than the one sitting in the config file.
      this.dataset.graticule = String(step);
      const gratD = path(d3.geoGraticule().step([step, step])()) || '';
      const gratMajD = path(d3.geoGraticule().step([majStep, majStep])()) || '';
      const wire = this._mode === 'wire';
      const uid = 'g' + this._seed;

      const defs = `
      <defs>
        <clipPath id="${uid}land"><path d="${landD}"/></clipPath>
        <filter id="${uid}terr" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.0075 0.0105" numOctaves="6" seed="${this._seed}" result="n"/>
          <feDiffuseLighting in="n" lighting-color="${t.land}" surfaceScale="7" diffuseConstant="1.05" result="l">
            <feDistantLight azimuth="308" elevation="46"/>
          </feDiffuseLighting>
        </filter>
        <filter id="${uid}sea" x="-10%" y="-10%" width="120%" height="120%" color-interpolation-filters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.004 0.012" numOctaves="3" seed="${this._seed + 4}" result="n"/>
          <feDiffuseLighting in="n" lighting-color="#2b3d4c" surfaceScale="3" diffuseConstant="0.75" result="l">
            <feDistantLight azimuth="300" elevation="60"/>
          </feDiffuseLighting>
        </filter>
        <filter id="${uid}glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <pattern id="${uid}dots" width="7" height="7" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="0.65" fill="${t.acc}" opacity="0.5"/>
        </pattern>
        <pattern id="${uid}hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="${t.acc}" stroke-width="0.7" opacity="0.35"/>
        </pattern>
        <linearGradient id="${uid}sw" x1="0" x2="1">
          <stop offset="0" stop-color="${t.acc}" stop-opacity="0"/>
          <stop offset="0.72" stop-color="${t.acc}" stop-opacity="0.16"/>
          <stop offset="1" stop-color="${t.acc}" stop-opacity="0"/>
        </linearGradient>
        <radialGradient id="${uid}vig" cx="50%" cy="46%" r="72%">
          <stop offset="0.5" stop-color="#000" stop-opacity="0"/>
          <stop offset="1" stop-color="#000" stop-opacity="0.72"/>
        </radialGradient>
      </defs>`;

      const base = wire ? `
        <rect width="${W}" height="${H}" fill="#05070a"/>
        <g clip-path="url(#${uid}land)">
          <rect width="${W}" height="${H}" fill="url(#${uid}dots)" opacity="0.5"/>
        </g>
        <path d="${gratD}" fill="none" stroke="${t.acc}" stroke-width="0.35" opacity="0.13"/>
        <path d="${gratMajD}" fill="none" stroke="${t.acc}" stroke-width="0.6" opacity="0.3"/>
        <path d="${landD}" fill="none" stroke="${t.acc}" stroke-width="1.15" opacity="0.9" filter="url(#${uid}glow)"/>
        <path d="${bordD}" fill="none" stroke="${t.alt}" stroke-width="0.7" opacity="0.55" stroke-dasharray="3 2.5"/>
      ` : `
        <rect width="${W}" height="${H}" fill="${t.sea}"/>
        <rect width="${W}" height="${H}" filter="url(#${uid}sea)" opacity="0.55"/>
        <g clip-path="url(#${uid}land)">
          <rect x="-40" y="-40" width="${W + 80}" height="${H + 80}" filter="url(#${uid}terr)"/>
          <rect width="${W}" height="${H}" fill="${t.tint}" opacity="0.42" style="mix-blend-mode:multiply"/>
        </g>
        <path d="${landD}" fill="none" stroke="#0b0d0e" stroke-width="1.6" opacity="0.65"/>
        <path d="${gratD}" fill="none" stroke="${t.acc}" stroke-width="0.3" opacity="0.07"/>
        <path d="${gratMajD}" fill="none" stroke="${t.acc}" stroke-width="0.55" opacity="0.18"/>
        <path d="${bordD}" fill="none" stroke="${t.warn}" stroke-width="0.9" opacity="0.42" stroke-dasharray="5 3"/>
        <rect width="${W}" height="${H}" fill="url(#${uid}vig)"/>
      `;

      const boundarySvg = this._renderBoundaries(pins, path, t, o);

      // graticule tick labels along the frame
      let ticks = '';
      const sw = proj.invert([0, H]), ne = proj.invert([W, 0]);
      if (sw && ne) {
        const [w0, s0] = sw, [e0, n0] = ne;
        for (let lon = Math.ceil(w0 / step) * step; lon < e0; lon += step) {
          const x = proj([lon, (n0 + s0) / 2])[0];
          if (x > 26 && x < W - 26) {
            ticks += `<text class="lbl" x="${x.toFixed(1)}" y="${H - 7}" fill="${t.acc}" opacity="0.45" text-anchor="middle">${lon > 0 ? lon.toFixed(0) + 'E' : (lon < 0 ? (-lon).toFixed(0) + 'W' : '0')}</text>`;
          }
        }
        for (let lat = Math.ceil(s0 / step) * step; lat < n0; lat += step) {
          const y = proj([(w0 + e0) / 2, lat])[1];
          if (y > 24 && y < H - 24) {
            ticks += `<text class="lbl" x="7" y="${(y + 3).toFixed(1)}" fill="${t.acc}" opacity="0.45">${lat > 0 ? lat.toFixed(0) + 'N' : (-lat).toFixed(0) + 'S'}</text>`;
          }
        }
      }

      const pinSvg = this._renderPins(pins, proj, t);
      const ghostSvg = this._renderGhosts(proj, t, W, H);
      const chrome = this._renderChrome(proj, t, W, H, wire, uid);

      this.shadowRoot.querySelector('.boot')?.remove();
      let svg = this.shadowRoot.querySelector('svg');
      if (!svg) {
        svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.shadowRoot.appendChild(svg);
        svg.addEventListener('click', e => {
          if (this._moved > 3) return;
          const g = e.target.closest('[data-pid]');
          if (g) {
            this.dispatchEvent(new CustomEvent('pinclick',
              { detail: g.getAttribute('data-pid'), bubbles: true, composed: true }));
          }
        });
        svg.addEventListener('mousemove', e => {
          const g = e.target.closest('[data-pid]');
          const id = g ? g.getAttribute('data-pid') : null;
          // Only fire on change: this runs on every pointer move over the map.
          if (id !== this._lastHover) {
            this._lastHover = id;
            this.dispatchEvent(new CustomEvent('pinhover',
              { detail: id, bubbles: true, composed: true }));
          }
        });
      }
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.innerHTML = defs
        + '<g class="scene">' + base
        + '<g class="overlay">' + boundarySvg + ghostSvg + pinSvg + '</g>'
        + '</g>' + ticks + chrome;

      // Kept so `_renderOverlay` can repaint the markers without re-deriving
      // the projection, which is the whole saving.
      this._last = { path, proj, t, o, W, H };
      this._painted = true;
    }

    /** Repaint only the markers, boundaries and ghosts.
     *
     * Everything under `.scene` outside `.overlay` -- terrain, graticule,
     * borders, vignette -- depends on the projection and the palette, neither
     * of which a hover changes.
     */
    _renderOverlay() {
      const g = this.shadowRoot.querySelector('.overlay');
      if (!g || !this._last) { this.render(); return; }
      const { path, proj, t, o } = this._last;
      const pins = this._pins.filter(p => p.lat != null);
      g.innerHTML = this._renderBoundaries(pins, path, t, o)
        + this._renderGhosts(proj, t, this._w, this._h)
        + this._renderPins(pins, proj, t);
    }

    /** Administrative polygons, under the pins.
     *
     * Drawn before the markers so the marker stays on top and remains the
     * click target -- a filled ADM0 covering the frame would otherwise swallow
     * every pointer event, which is why the group is `pointer-events:none`.
     */
    _renderBoundaries(pins, path, t, o) {
      let out = '';
      pins.forEach(p => {
        const b = p.boundary;
        if (!b || !b.geometry) return;
        const d = path(b.geometry);
        if (!d) return;
        const on = p.id === this._active;
        const col = p.status === 'ambiguous' ? t.warn : (on ? t.acc : t.alt);
        // A join the gazetteers only half agreed on is drawn dashed and
        // labelled with the score, rather than drawn as though it were
        // certain. `name_score` is null when there was nothing to compare --
        // an ADM0 keyed on the country code -- which is not a weak match.
        const weak = b.name_score != null && b.name_score < o.weakMatchBelow;
        // The satellite terrain is a busy mid-tone, so a resting polygon needs
        // more than a token wash to read at all; the wire basemap is nearly
        // black and needs much less.
        const rest = this._mode === 'wire' ? 0.45 : 0.75;
        const fillOp = on ? o.boundaryFillOpacity : o.boundaryFillOpacity * rest;
        out += `<g data-boundary="${esc(p.id)}" style="pointer-events:none">
          <path d="${d}" fill="${col}" fill-opacity="${fillOp.toFixed(3)}"
                stroke="${col}" stroke-width="${(on ? o.boundaryStrokeWidth * 1.5 : o.boundaryStrokeWidth).toFixed(2)}"
                stroke-opacity="${on ? 0.95 : 0.68}"
                ${weak ? 'stroke-dasharray="6 4"' : ''}/>`;
        if (on && weak) {
          const c = path.centroid(b.geometry);
          if (c && isFinite(c[0])) {
            out += `<text class="lbl" x="${c[0].toFixed(1)}" y="${c[1].toFixed(1)}"
                     fill="${t.warn}" text-anchor="middle" opacity="0.9"
                     >ADM${b.level} ${esc(String(b.name).toUpperCase())} · MATCH ${b.name_score.toFixed(2)}</text>`;
          }
        }
        out += '</g>';
      });
      return out;
    }

    /** Markers, with label placement that does not pile up.
     *
     * A Sahel document resolves a dozen places inside a few degrees, and the
     * handoff's fixed up-and-right label offset turns that into an unreadable
     * stack. Labels are placed greedily instead: the active pin first so its
     * label is never the one dropped, then the rest by confidence, each taking
     * the first of four corner positions that does not overlap an already
     * placed label. A pin whose label will not fit anywhere keeps its marker
     * and loses only the plate -- the marker is the click target and the
     * hover already names the place.
     */
    _renderPins(pins, proj, t) {
      const placed = [];
      const hits = (a, b) => !(a.x + a.w < b.x || b.x + b.w < a.x
                            || a.y + a.h < b.y || b.y + b.h < a.y);

      // Active first, then most confident, so the labels that survive a
      // crowded frame are the ones worth reading.
      const order = pins.map((p, i) => [p, i]).sort((A, B) => {
        const a = A[0], b = B[0];
        if ((a.id === this._active) !== (b.id === this._active)) {
          return a.id === this._active ? -1 : 1;
        }
        return (b.conf ?? 0) - (a.conf ?? 0);
      });

      const byIndex = [];
      for (const [p, i] of order) {
        const xy = proj([p.lon, p.lat]);
        if (!xy) { byIndex[i] = ''; continue; }
        const [x, y] = xy;
        const on = p.id === this._active;
        const amb = p.status === 'ambiguous';
        const col = amb ? t.warn : (on ? t.acc : t.alt);
        const op = on ? 1 : (p.status === 'pending' ? 0.25 : 0.72);
        const r = on ? 9 : 6;
        // 5.9px per character is JetBrains Mono's advance at 8.5px with a
        // little slack for the fallback face, plus 5px of left padding and
        // 24px reserved for the confidence bar -- which the label text ran
        // under at the handoff's 5.4/26.
        // A pin can stand for several mentions of the same place; the plate
        // says how many, so that collapsing them does not quietly hide that
        // the document leans on this one repeatedly.
        const tally = p.count > 1 ? `×${p.count}` : '';
        const lw = Math.max(38, ((p.label || '').length + tally.length * 1.2) * 5.9 + 34);
        const pulse = (on && !reduceMotion())
          ? `<circle cx="${x}" cy="${y}" r="6" fill="none" stroke="${col}" stroke-width="1.2" class="pulse"/>` : '';

        // Four corners, in preference order: the handoff's up-right first.
        const spots = [
          { lx: x + r + 13, ly: y - 22, tx: x + r + 13, ty: y - 11 },
          { lx: x + r + 13, ly: y + 9,  tx: x + r + 13, ty: y + 9 },
          { lx: x - r - 13 - lw, ly: y - 22, tx: x - r - 13, ty: y - 11 },
          { lx: x - r - 13 - lw, ly: y + 9,  tx: x - r - 13, ty: y + 9 },
        ];
        // 2px of slack, so labels that merely touch still both render.
        const spot = spots.find(sp => !placed.some(
          q => hits({ x: sp.lx - 2, y: sp.ly - 2, w: lw + 4, h: 17 }, q)));
        if (spot) placed.push({ x: spot.lx, y: spot.ly, w: lw, h: 13 });

        const label = spot ? `
          <line x1="${spot.tx > x ? x + r + 4 : x - r - 4}" y1="${y}" x2="${spot.tx}" y2="${spot.ty}" stroke="${col}" stroke-width="0.7" opacity="0.8"/>
          <g transform="translate(${spot.lx.toFixed(1)},${spot.ly.toFixed(1)})">
            <rect width="${lw}" height="13" fill="#05060a" opacity="${on ? 0.92 : 0.7}" stroke="${col}" stroke-width="${on ? 0.9 : 0.5}" stroke-opacity="0.8"/>
            <text class="lbl" x="5" y="9.2" fill="${col}">${esc((p.label || '').toUpperCase())}${
              tally ? `<tspan opacity="0.65"> ${tally}</tspan>` : ''}</text>
            ${p.conf != null ? `<rect x="${lw - 20}" y="4.5" width="16" height="4" fill="none" stroke="${col}" stroke-width="0.5" opacity="0.7"/>
            <rect x="${lw - 19.4}" y="5.1" width="${(14.8 * p.conf).toFixed(1)}" height="2.8" fill="${col}" opacity="0.85"/>` : ''}
          </g>` : '';

        byIndex[i] = `<g data-pid="${esc(p.id)}" style="cursor:crosshair" opacity="${op}">
          ${pulse}
          <circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${col}" stroke-width="${on ? 1.3 : 0.9}" ${amb ? 'stroke-dasharray="2.5 2"' : ''}/>
          <line x1="${x - r - 5}" y1="${y}" x2="${x - r + 1}" y2="${y}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x + r - 1}" y1="${y}" x2="${x + r + 5}" y2="${y}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x}" y1="${y - r - 5}" x2="${x}" y2="${y - r + 1}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x}" y1="${y + r - 1}" x2="${x}" y2="${y + r + 5}" stroke="${col}" stroke-width="0.9"/>
          <circle cx="${x}" cy="${y}" r="1.7" fill="${col}"/>
          ${label}
        </g>`;
      }
      // Emitted in document order so the active marker is not buried, then
      // re-appended below -- SVG paints last-on-top.
      const activeIdx = pins.findIndex(p => p.id === this._active);
      return byIndex.filter((_, i) => i !== activeIdx).join('')
        + (activeIdx >= 0 ? byIndex[activeIdx] : '');
    }

    /** Rival candidates: in-frame as dashed ghost rings, off-frame as edge bearings.
     *
     * Drawn in `--alt`, not `--warn`. These are places the ranker considered
     * and did not choose, which is a different statement from "flagged for
     * review" -- and warn is the colour that already carries that meaning, on
     * the pins and on the polygons both. `console.js` decides whether there is
     * anything here to draw at all; see `ghostsFor`.
     */
    _renderGhosts(proj, t, W, H) {
      let out = '';
      this._ghosts.forEach(g => {
        const xy = proj([g.lon, g.lat]); if (!xy) return;
        let [x, y] = xy;
        // A leader from the place that won to the one being pointed at. The
        // distance between them is the argument the ranked list cannot make
        // -- "Niger the country, not Niger the river" is a statement about
        // geography -- and without it a lone ring inside a cluster of markers
        // is invisible, which is what a candidate in a capital city looks
        // like next to the capital's own pin.
        const from = g.from && proj(g.from);
        if (from) {
          out += `<line x1="${from[0].toFixed(1)}" y1="${from[1].toFixed(1)}"
                        x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"
                        stroke="${t.alt}" stroke-width="0.9" opacity="0.5"
                        stroke-dasharray="4 3" style="pointer-events:none"/>`;
        }
        const off = x < 14 || x > W - 14 || y < 14 || y > H - 14;
        if (off) {
          const cx = W / 2, cy = H / 2, dx = x - cx, dy = y - cy;
          const s = Math.min(Math.abs((W / 2 - 20) / (dx || 1e-6)),
                             Math.abs((H / 2 - 20) / (dy || 1e-6)));
          x = cx + dx * s; y = cy + dy * s;
          const ang = Math.atan2(dy, dx) * 180 / Math.PI;
          const flip = x > W - 150;
          out += `<g transform="translate(${x.toFixed(1)},${y.toFixed(1)})" opacity="0.8" style="pointer-events:none">
            <g transform="rotate(${ang.toFixed(1)})"><path d="M0,-4 L7,0 L0,4 Z" fill="${t.alt}" opacity="0.85"/></g>
            <text class="lbl" x="${flip ? -12 : 12}" y="3.5" fill="${t.alt}" text-anchor="${flip ? 'end' : 'start'}">${esc(String(g.label).toUpperCase())} · OFF-FRAME</text>
          </g>`;
        } else {
          // Bigger and brighter than it used to be, because it is summoned
          // now rather than permanent: it has one moment to be found.
          out += `<g opacity="0.95" style="pointer-events:none">
            <circle cx="${x}" cy="${y}" r="11" fill="none" stroke="${t.alt}" stroke-width="1.2" stroke-dasharray="3 2.5"/>
            <circle cx="${x}" cy="${y}" r="1.8" fill="${t.alt}"/>
            <rect x="${x + 14}" y="${y - 7}" width="${String(g.label).length * 5.9 + 10}" height="13"
                  fill="#05060a" opacity="0.85" stroke="${t.alt}" stroke-width="0.5" stroke-opacity="0.8"/>
            <text class="lbl" x="${x + 19}" y="${y + 2.2}" fill="${t.alt}">${esc(String(g.label).toUpperCase())}</text>
          </g>`;
        }
      });
      return out;
    }

    _renderChrome(proj, t, W, H, wire, uid) {
      // Scale bar length, from inverting two screen points 100px apart.
      let km = 0;
      const a = proj.invert([20, H - 30]), b = proj.invert([120, H - 30]);
      if (a && b) {
        const R = 6371, rad = Math.PI / 180;
        const cosd = Math.sin(a[1] * rad) * Math.sin(b[1] * rad)
          + Math.cos(a[1] * rad) * Math.cos(b[1] * rad) * Math.cos((b[0] - a[0]) * rad);
        km = Math.round(Math.acos(Math.min(1, Math.max(-1, cosd))) * R / 10) * 10;
      }
      return `
        <g opacity="0.55">
          <path d="M14,14 L14,34 M14,14 L34,14" fill="none" stroke="${t.acc}" stroke-width="1"/>
          <path d="M${W - 14},14 L${W - 14},34 M${W - 14},14 L${W - 34},14" fill="none" stroke="${t.acc}" stroke-width="1"/>
          <path d="M14,${H - 14} L14,${H - 34} M14,${H - 14} L34,${H - 14}" fill="none" stroke="${t.acc}" stroke-width="1"/>
          <path d="M${W - 14},${H - 14} L${W - 14},${H - 34} M${W - 14},${H - 14} L${W - 34},${H - 14}" fill="none" stroke="${t.acc}" stroke-width="1"/>
        </g>
        <g opacity="0.28">
          <line x1="${W / 2}" y1="${H / 2 - 16}" x2="${W / 2}" y2="${H / 2 + 16}" stroke="${t.acc}" stroke-width="0.7"/>
          <line x1="${W / 2 - 16}" y1="${H / 2}" x2="${W / 2 + 16}" y2="${H / 2}" stroke="${t.acc}" stroke-width="0.7"/>
          <circle cx="${W / 2}" cy="${H / 2}" r="30" fill="none" stroke="${t.acc}" stroke-width="0.5" stroke-dasharray="1 5"/>
        </g>
        <g transform="translate(20,${H - 34})">
          <line x1="0" y1="0" x2="100" y2="0" stroke="${t.acc}" stroke-width="1" opacity="0.7"/>
          <line x1="0" y1="-4" x2="0" y2="4" stroke="${t.acc}" stroke-width="1" opacity="0.7"/>
          <line x1="100" y1="-4" x2="100" y2="4" stroke="${t.acc}" stroke-width="1" opacity="0.7"/>
          <text class="lbl" x="50" y="-8" fill="${t.acc}" opacity="0.75" text-anchor="middle">${km} KM</text>
        </g>
        <g class="blink" transform="translate(${W - 116},${H - 26})">
          <circle cx="0" cy="-3" r="2.4" fill="${t.warn}"/>
          <text class="lbl" x="9" y="0" fill="${t.acc}" opacity="0.8">${wire ? 'VECTOR · NE 110M' : 'RELIEF · SYNTHETIC'}</text>
        </g>
        <rect class="sweep" x="${-W * 0.16}" y="0" width="${W * 0.3}" height="${H}" fill="url(#${uid}sw)" style="pointer-events:none"/>
      `;
    }
  }

  if (!customElements.get('geo-scope')) customElements.define('geo-scope', GeoScope);
})();
