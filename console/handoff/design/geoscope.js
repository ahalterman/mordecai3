/* geo-scope — gritty satellite / wireframe map surface for the geoparse console.
   Real geometry: Natural Earth 110m via world-atlas + d3-geo. No hand-drawn coastlines. */
(function () {
  const ATLAS = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json';
  let atlasPromise = null;

  function loadAtlas() {
    if (!atlasPromise) {
      atlasPromise = fetch(ATLAS).then(r => r.json()).then(t => ({
        land: window.topojson.feature(t, t.objects.countries),
        borders: window.topojson.mesh(t, t.objects.countries, (a, b) => a !== b)
      }));
    }
    return atlasPromise;
  }

  function waitLibs() {
    return new Promise(res => {
      const tick = () => (window.d3 && window.topojson) ? res() : setTimeout(tick, 60);
      tick();
    });
  }

  const esc = s => String(s).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));

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
      this._zk = 1; this._zx = 0; this._zy = 0;
      this._lk = 1; this._lx = 0; this._ly = 0;
      this._key = '';
      this._moved = 0;
    }

    connectedCallback() {
      this.shadowRoot.innerHTML = `<style>
        :host{display:block;width:100%;height:100%;position:relative;overflow:hidden;background:#06070a;
          cursor:grab;font-family:'JetBrains Mono',ui-monospace,monospace;contain:layout paint}
        svg{position:absolute;inset:0;width:100%;height:100%;display:block}
        .lbl{font:600 8.5px 'JetBrains Mono',ui-monospace,monospace;letter-spacing:.08em}
        .sweep{animation:sw 9s linear infinite}
        @keyframes sw{0%{transform:translateX(-14%)}100%{transform:translateX(114%)}}
        .pulse{animation:pl 1.9s ease-out infinite}
        @keyframes pl{0%{r:6;opacity:.85}100%{r:26;opacity:0}}
        .blink{animation:bk 2.4s steps(1,end) infinite}
        @keyframes bk{0%,88%{opacity:1}89%,100%{opacity:.35}}
        .boot{position:absolute;inset:0;display:grid;place-items:center;color:var(--acc,#f0a13c);
          font:600 10px 'JetBrains Mono',monospace;letter-spacing:.28em;background:#06070a}
      </style><div class="boot">ACQUIRING TILESET…</div>`;
      this._ro = new ResizeObserver(() => {
        const r = this.getBoundingClientRect();
        if (Math.abs(r.width - this._w) > 1 || Math.abs(r.height - this._h) > 1) {
          this._w = r.width; this._h = r.height; this.render();
        }
      });
      this._ro.observe(this);
      this._bindNav();
      waitLibs().then(loadAtlas).then(a => {
        this._atlas = a; this._ready = true;
        const r = this.getBoundingClientRect();
        this._w = r.width; this._h = r.height;
        this.render();
      });
    }

    disconnectedCallback() { if (this._ro) this._ro.disconnect(); }

    _applyLive() {
      const g = this.shadowRoot.querySelector('.scene');
      if (g) g.setAttribute('transform', `translate(${this._lx.toFixed(2)},${this._ly.toFixed(2)}) scale(${this._lk.toFixed(4)})`);
    }

    _commit() {
      this._zk *= this._lk;
      this._zx = this._zx * this._lk + this._lx;
      this._zy = this._zy * this._lk + this._ly;
      this._lk = 1; this._lx = 0; this._ly = 0;
      this.dispatchEvent(new CustomEvent('viewchange', { detail: this._zk, bubbles: true, composed: true }));
      this.render();
    }

    resetView() {
      this._zk = 1; this._zx = 0; this._zy = 0; this._lk = 1; this._lx = 0; this._ly = 0;
      this.dispatchEvent(new CustomEvent('viewchange', { detail: 1, bubbles: true, composed: true }));
      this.render();
    }

    _bindNav() {
      this.addEventListener('wheel', ev => {
        ev.preventDefault();
        const r = this.getBoundingClientRect();
        const px = ev.clientX - r.left, py = ev.clientY - r.top;
        const f = Math.exp(-ev.deltaY * 0.0018);
        const nk = this._zk * this._lk * f;
        if (nk < 0.6 || nk > 28) return;
        this._lk *= f;
        this._lx = this._lx * f + px * (1 - f);
        this._ly = this._ly * f + py * (1 - f);
        this._applyLive();
        clearTimeout(this._wt);
        this._wt = setTimeout(() => this._commit(), 190);
      }, { passive: false });

      this.addEventListener('mousedown', ev => {
        if (ev.button !== 0) return;
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
          if (this._moved > 3) this._commit();
        };
        window.addEventListener('mousemove', mv);
        window.addEventListener('mouseup', up);
      });

      this.addEventListener('dblclick', () => this.resetView());
    }

    setScene(o) {
      if (o.pins) {
        const k = o.pins.map(p => p.id).join('|');
        if (k !== this._key) { this._key = k; this._zk = 1; this._zx = 0; this._zy = 0; }
      }
      if (o.pins) this._pins = o.pins;
      if ('ghosts' in o) this._ghosts = o.ghosts || [];
      if (o.mode) this._mode = o.mode;
      if ('active' in o) this._active = o.active;
      if (o.seed) this._seed = o.seed;
      this.render();
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
        tint: g('--map-tint', '#5d5124')
      };
    }

    render() {
      if (!this._ready || !this._w || !this._h) return;
      const d3 = window.d3, W = this._w, H = this._h, t = this._theme();
      const pins = this._pins.filter(p => p.lat != null);
      const proj = d3.geoMercator();
      const pad = Math.min(W, H) * 0.26;
      if (pins.length) {
        proj.fitExtent([[pad, pad], [W - pad, H - pad]],
          { type: 'MultiPoint', coordinates: pins.map(p => [p.lon, p.lat]) });
      } else {
        proj.fitExtent([[pad, pad], [W - pad, H - pad]], { type: 'Sphere' });
      }
      const bs0 = proj.scale(), bt0 = proj.translate();
      proj.scale(bs0 * this._zk).translate([bt0[0] * this._zk + this._zx, bt0[1] * this._zk + this._zy]);
      const path = d3.geoPath(proj);
      const landD = path(this._atlas.land) || '';
      const bordD = path(this._atlas.borders) || '';
      const gratD = path(d3.geoGraticule().step([2, 2])()) || '';
      const gratMajD = path(d3.geoGraticule().step([10, 10])()) || '';
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

      // graticule tick labels along the frame
      let ticks = '';
      const [w0, s0] = proj.invert([0, H]), [e0, n0] = proj.invert([W, 0]);
      for (let lon = Math.ceil(w0 / 2) * 2; lon < e0; lon += 2) {
        const x = proj([lon, (n0 + s0) / 2])[0];
        if (x > 26 && x < W - 26) ticks += `<text class="lbl" x="${x.toFixed(1)}" y="${H - 7}" fill="${t.acc}" opacity="0.45" text-anchor="middle">${lon > 0 ? lon + 'E' : (lon < 0 ? -lon + 'W' : '0')}</text>`;
      }
      for (let lat = Math.ceil(s0 / 2) * 2; lat < n0; lat += 2) {
        const y = proj([(w0 + e0) / 2, lat])[1];
        if (y > 24 && y < H - 24) ticks += `<text class="lbl" x="7" y="${(y + 3).toFixed(1)}" fill="${t.acc}" opacity="0.45">${lat > 0 ? lat + 'N' : -lat + 'S'}</text>`;
      }

      // pins
      let pinSvg = '';
      pins.forEach((p, i) => {
        const xy = proj([p.lon, p.lat]); if (!xy) return;
        const [x, y] = xy;
        const on = p.id === this._active;
        const amb = p.status === 'ambiguous';
        const col = amb ? t.warn : (on ? t.acc : t.alt);
        const op = on ? 1 : (p.status === 'pending' ? 0.25 : 0.72);
        const r = on ? 9 : 6;
        const lw = Math.max(38, (p.label || '').length * 5.4 + 26);
        pinSvg += `<g data-pid="${esc(p.id)}" style="cursor:crosshair" opacity="${op}">
          ${on ? `<circle cx="${x}" cy="${y}" r="6" fill="none" stroke="${col}" stroke-width="1.2" class="pulse"/>` : ''}
          <circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${col}" stroke-width="${on ? 1.3 : 0.9}" ${amb ? 'stroke-dasharray="2.5 2"' : ''}/>
          <line x1="${x - r - 5}" y1="${y}" x2="${x - r + 1}" y2="${y}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x + r - 1}" y1="${y}" x2="${x + r + 5}" y2="${y}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x}" y1="${y - r - 5}" x2="${x}" y2="${y - r + 1}" stroke="${col}" stroke-width="0.9"/>
          <line x1="${x}" y1="${y + r - 1}" x2="${x}" y2="${y + r + 5}" stroke="${col}" stroke-width="0.9"/>
          <circle cx="${x}" cy="${y}" r="1.7" fill="${col}"/>
          <line x1="${x + r + 4}" y1="${y}" x2="${x + r + 13}" y2="${y - 11}" stroke="${col}" stroke-width="0.7" opacity="0.8"/>
          <g transform="translate(${(x + r + 13).toFixed(1)},${(y - 22).toFixed(1)})">
            <rect width="${lw}" height="13" fill="#05060a" opacity="${on ? 0.92 : 0.7}" stroke="${col}" stroke-width="${on ? 0.9 : 0.5}" stroke-opacity="0.8"/>
            <text class="lbl" x="5" y="9.2" fill="${col}">${esc((p.label || '').toUpperCase())}</text>
            ${p.conf != null ? `<rect x="${lw - 20}" y="4.5" width="16" height="4" fill="none" stroke="${col}" stroke-width="0.5" opacity="0.7"/>
            <rect x="${lw - 19.4}" y="5.1" width="${(14.8 * p.conf).toFixed(1)}" height="2.8" fill="${col}" opacity="0.85"/>` : ''}
          </g>
        </g>`;
      });

      // off-screen candidate bearings
      let ghostSvg = '';
      this._ghosts.forEach(g => {
        const xy = proj([g.lon, g.lat]); if (!xy) return;
        let [x, y] = xy;
        const off = x < 14 || x > W - 14 || y < 14 || y > H - 14;
        if (off) {
          const cx = W / 2, cy = H / 2, dx = x - cx, dy = y - cy;
          const s = Math.min(Math.abs((W / 2 - 20) / (dx || 1e-6)), Math.abs((H / 2 - 20) / (dy || 1e-6)));
          x = cx + dx * s; y = cy + dy * s;
          const ang = Math.atan2(dy, dx) * 180 / Math.PI;
          const flip = x > W - 150;
          ghostSvg += `<g transform="translate(${x.toFixed(1)},${y.toFixed(1)})" opacity="0.8">
            <g transform="rotate(${ang.toFixed(1)})"><path d="M0,-4 L7,0 L0,4 Z" fill="${t.warn}" opacity="0.85"/></g>
            <text class="lbl" x="${flip ? -12 : 12}" y="3.5" fill="${t.warn}" text-anchor="${flip ? 'end' : 'start'}">${esc(g.label.toUpperCase())} · OFF-FRAME</text>
          </g>`;
        } else {
          ghostSvg += `<g opacity="0.75">
            <circle cx="${x}" cy="${y}" r="7" fill="none" stroke="${t.warn}" stroke-width="0.8" stroke-dasharray="2 2"/>
            <circle cx="${x}" cy="${y}" r="1.4" fill="${t.warn}"/>
            <text class="lbl" x="${x + 11}" y="${y + 3}" fill="${t.warn}">${esc(g.label.toUpperCase())}</text>
          </g>`;
        }
      });

      // scale bar
      const km = (() => {
        const a = proj.invert([20, H - 30]), b = proj.invert([120, H - 30]);
        const R = 6371, rad = Math.PI / 180;
        const d = Math.acos(Math.min(1, Math.sin(a[1] * rad) * Math.sin(b[1] * rad) + Math.cos(a[1] * rad) * Math.cos(b[1] * rad) * Math.cos((b[0] - a[0]) * rad))) * R;
        return Math.round(d / 10) * 10;
      })();

      const chrome = `
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
        <g class="blink" transform="translate(${W - 108},${H - 26})">
          <circle cx="0" cy="-3" r="2.4" fill="${t.warn}"/>
          <text class="lbl" x="9" y="0" fill="${t.acc}" opacity="0.8">LIVE · ${wire ? 'VECTOR' : 'EO/IR'}</text>
        </g>
        <rect class="sweep" x="${-W * 0.16}" y="0" width="${W * 0.3}" height="${H}" fill="url(#${uid}sw)" style="pointer-events:none"/>
      `;

      this.shadowRoot.querySelector('.boot')?.remove();
      let svg = this.shadowRoot.querySelector('svg');
      if (!svg) {
        svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.shadowRoot.appendChild(svg);
        svg.addEventListener('click', e => {
          if (this._moved > 3) return;
          const g = e.target.closest('[data-pid]');
          if (g) this.dispatchEvent(new CustomEvent('pinclick', { detail: g.getAttribute('data-pid'), bubbles: true, composed: true }));
        });
        svg.addEventListener('mousemove', e => {
          const g = e.target.closest('[data-pid]');
          this.dispatchEvent(new CustomEvent('pinhover', { detail: g ? g.getAttribute('data-pid') : null, bubbles: true, composed: true }));
        });
      }
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.innerHTML = defs + '<g class="scene">' + base + ghostSvg + pinSvg + '</g>' + ticks + chrome;
    }
  }

  if (!customElements.get('geo-scope')) customElements.define('geo-scope', GeoScope);
})();
