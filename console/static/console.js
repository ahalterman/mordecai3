/* Mordecai geoparse console -- application state and rendering.
 *
 * One state object, one render pass per change. No framework: the whole screen
 * is a handful of panes that each read the same state, and a dependency that
 * needs a build step is a dependency that can break the morning of a talk.
 *
 * The one rule worth stating: `active = hovered ?? selected`, and *everything*
 * -- the document highlight, the map's active marker, the coordinate readout,
 * the parse table row, the disambiguate panel -- derives from that single
 * value. The bidirectional sync between text and map is not wiring between
 * them; it is both of them reading the same field.
 */

const $ = sel => document.querySelector(sel);

const STAGE_KEYS = ['ingest', 'parse', 'dis', 'resolve', 'export'];

/* Every user-facing string, in the ops register.
 *
 * A theme in console.config.json overrides the subset it wants under `copy`
 * and inherits the rest from here, so a third look is a config edit rather
 * than a hunt through this file for capital letters. The reason the whole
 * table exists: the military register of this console lives in its words at
 * least as much as in its colours -- BOOTING, DISAMBIGUATE, FLAGGED FOR
 * REVIEW -- and a warm palette under those labels still reads as an ops
 * screen. Two levels deep, which is as far as the grouping goes; `mergeCopy`
 * assumes exactly that.
 */
const COPY = {
  brandName: null,             // null -> brand.productName from the config
  brandSub: null,
  railPipeline: 'PIPELINE',
  railYourText: 'YOUR TEXT',
  railCorpus: 'CORPUS',
  pastePlaceholder: 'Paste any text here, then press PARSE below.',
  uploadLabel: '\u25b8 UPLOAD .txt / .jsonl',
  fileHint: 'Blank-line separated, or JSONL with a <code>text</code> field.',
  pastedTitle: 'PASTED TEXT',
  pastedSource: 'USER INPUT',
  pastedMeta: '{n} CHARS',
  batchMeta: 'BATCH {n} DOCS',
  run: 'PARSE',
  runAgain: 'RE-RUN PARSE',
  running: 'PARSING\u2026',
  batching: 'BATCH\u2026',
  tokenized: 'TOKENIZED',
  resetLabel: 'RESET',
  download: 'DOWNLOAD \u25b8',
  navHint: 'SCROLL ZOOM \u00b7 DRAG PAN',
  gazPrefix: 'GAZ \u00b7 GEONAMES',
  boundaryPrefix: 'BOUNDARIES',
  boundaryNone: 'NOT LOADED',
  boundaryShapes: 'SHAPES',
  stages: { ingest: 'INGEST', parse: 'PARSE', dis: 'DISAMBIGUATE',
            resolve: 'RESOLVE', export: 'EXPORT' },
  stageMeta: { docs: 'DOC', spans: 'SPANS', topk: 'TOP-K', fmt: 'FMT' },
  status: { booting: 'BOOTING', ready: 'READY', parsing: 'PARSING',
            resolved: 'RESOLVED', error: 'ERROR', offline: 'OFFLINE' },
  stats: { build: 'BUILD', model: 'MODEL', spans: 'SPANS', device: 'DEVICE' },
  foot: { spans: 'SPANS', resolved: 'RESOLVED', places: 'PLACES',
          flagged: 'FLAGGED', boundaries: 'BOUNDARIES',
          parsing: 'PARSING\u2026', none: 'NO SPANS' },
  notes: { rationale: 'RATIONALE \u25b8 ', review: 'REVIEW \u25b8 ',
           boundary: 'BOUNDARY \u25b8 ', batch: 'BATCH \u25b8 ',
           note: 'NOTE \u25b8 ', nomatch: 'NO MATCH \u25b8 ' },
  verdict: { flagged: 'FLAGGED FOR REVIEW', ok: 'AUTO-ACCEPT' },
  tbl: { source: 'SOURCE', adapter: 'ADAPTER', size: 'SIZE', state: 'STATE',
         active: 'ACTIVE', idle: 'IDLE', span: 'SPAN', label: 'LABEL',
         char: 'CHAR', pspan: 'P(SPAN)', presolve: 'P(RESOLVE)' },
  kv: { geonameid: 'GEONAMEID', name: 'NAME', feature: 'FEATURE CODE',
        admin1: 'ADMIN 1', admin2: 'ADMIN 2', country: 'COUNTRY',
        population: 'POPULATION', coordinates: 'COORDINATES',
        confidence: 'CONFIDENCE', pnomatch: 'P(NO MATCH)',
        geometry: 'GEOMETRY', geomsource: 'GEOM SOURCE',
        accepted: 'ACCEPTED', flagged: 'FLAGGED', pointonly: 'POINT ONLY' },
  empty: { span: 'Select a span to see its candidates.',
           nospans: 'No spans yet.', nocands: 'No gazetteer candidates.' },
  mentionOf: 'MENTION {n} OF {total} AT THIS PLACE',
  onlyMention: 'ONLY MENTION OF THIS PLACE',
  inCorpus: '{n} IN CORPUS',
  margin: 'MARGIN',
  coordPolygon: 'ADM{level} POLYGON',
  coordPoint: 'POINT',
  mapProjection: 'MERCATOR',
  mapGraticule: 'GRATICULE',
  mapBorders: 'ADMIN-0 MESH',
  mapDatum: 'WGS84',
  exportFeatures: 'FEATURES',
  noMatchMeta: 'NO MATCH',
  noMatchBody: 'The model declined to place “{text}”. Its calibrated probability '
    + 'that no candidate is correct is {p}. The candidates it rejected are '
    + 'under {stage}.',
  mapRelief: 'SYNTHETIC RELIEF',
  mapVector: 'VECTOR OVERLAY \u00b7 NE 110M',
  mapNoRaster: 'NO RASTER SOURCE',
  mapUnreachable: 'RASTER UNREACHABLE',
  menu: { look: 'LOOK', palette: 'PALETTE', chrome: 'CHROME', map: 'MAP',
          panes: 'PANES', layoutSplit: 'SPLIT', layoutTheater: 'THEATER',
          resetPanes: 'RESET PANE SIZES', ornaments: 'ORNAMENTS',
          scanlines: 'SCANLINES', vignette: 'VIGNETTE', grain: 'FILM GRAIN',
          sweep: 'MAP SWEEP', hairline: 'PANE HAIRLINE',
          boundaries: 'BOUNDARY POLYGONS',
          modeSat: 'RELIEF', modeImagery: 'SATELLITE', modeWire: 'VECTOR' },
};

/* The active theme's copy and chrome flags. Both are replaced wholesale by
 * `applyTheme`; nothing else writes them. */
let T = COPY;
let CH = {};

/** Uppercase, unless the theme says the screen does not shout.
 *  For values that come from data -- a place name, a document title -- rather
 *  than from COPY, which is already written in each theme's register. */
const UP = s => CH.upper === false ? String(s ?? '') : String(s ?? '').toUpperCase();

/** Two-level merge: a theme overriding `copy.foot.spans` must not lose
 *  `copy.foot.places`, which a spread at the top level would do. */
function mergeCopy(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) {
    out[k] = (v && typeof v === 'object' && !Array.isArray(v))
      ? { ...(base[k] || {}), ...v } : v;
  }
  return out;
}

const fill = (tpl, vars) =>
  String(tpl).replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));

const S = {
  config: null,
  corpus: [],
  backend: null,
  docId: null,          // corpus doc_id, or '__paste__'
  doc: null,            // the active document record {title, meta, kind, text}
  result: null,         // the last /api/geoparse response
  stage: 'dis',
  mapMode: 'sat',
  theme: null,          // key into config.themes -- the whole look
  route: null,          // config.routes entry for this URL, if any
  chrome: {},           // the active theme's ornament flags, menu-editable
  boundaries: null,     // menu override of boundaries.enabled, or null
  selected: null,
  hovered: null,
  running: false,
  imageryDown: false,   // tiles could not be reached; the map fell back
  settled: false,       // false while the reveal animation is stepping
  revealed: new Set(),  // entity ids already stepped past
  scanning: null,       // the entity mid-reveal
  zoom: 1,
  format: 'raw',        // overwritten from export.defaultFormat at boot
  palette: null,        // palette within the theme; each theme owns its own
  ghostCand: null,      // index of the candidate row under the pointer
  batch: null,          // last /api/batch summary
  log: { time: '', line: 'idle' },
};

const active = () => S.hovered ?? S.selected;
const entities = () => (S.result && S.result.entities) || [];

/* Indexed rather than scanned: `spanState` asks for the active entity once
 * per span, and `renderDoc` runs it for every span on every pointer move.
 * Linear lookups made that quadratic in the length of the document, which a
 * three-mention demo never notices and a pasted article does. */
let _index = { of: null, map: new Map() };

function entityById(id) {
  if (_index.of !== S.result) {
    _index = { of: S.result, map: new Map(entities().map(e => [e.id, e])) };
  }
  return _index.map.get(id) || null;
}

/* ── mentions and places ──────────────────────────────────────────────────── */
/* A *mention* is one span of text. A *place* is one gazetteer record. They are
 * not the same thing and the console had been treating them as if they were: a
 * document that says "Ukraine" four times produced four pins on one coordinate
 * with four identical labels stacked on top of each other, four copies of the
 * same polygon compounding each other's fill, and a map that looked like it had
 * found four things when it had found one.
 *
 * So the map is keyed on places from here down, and the panels stay keyed on
 * mentions -- because the candidate list, the margin and the rationale are
 * about *this* occurrence of the word and genuinely differ between them: the
 * same record can be an easy call in one sentence and a close one in the next.
 * The two are joined by `placeKey`, which is what lets one pin light up all
 * four spans and one span light up its pin.
 */
const placeKey = e => (e && e.resolved) ? `p${e.resolved.geonameid}` : null;
const activePlace = () => placeKey(entityById(active()));

let _places = { of: null, list: [] };

function places() {
  if (_places.of === S.result) return _places.list;
  const byKey = new Map();
  for (const e of entities()) {
    const k = placeKey(e);
    if (!k) continue;                       // a mention the model declined
    let p = byKey.get(k);
    if (!p) byKey.set(k, p = { key: k, resolved: e.resolved,
                               boundary: e.boundary, mentions: [] });
    p.mentions.push(e);
  }
  _places = { of: S.result, list: [...byKey.values()] };
  return _places.list;
}

const placeOf = key => places().find(p => p.key === key) || null;

/** Which mention a pin stands for.
 *
 * A pin can stand for several mentions and the panels below it show one. If
 * the mention already selected is one of them, keep it: hovering the pin you
 * are already reading about must not jump the panel to a different sentence.
 */
function mentionForPlace(key) {
  const p = key && placeOf(key);
  if (!p) return null;
  return (p.mentions.find(m => m.id === S.selected) || p.mentions[0]).id;
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  let cfg;
  try {
    cfg = await (await fetch('/api/config')).json();
  } catch (err) {
    setStatus(COPY.status.offline, `backend unreachable: ${err.message}`);
    return;
  }
  S.config = cfg.config || {};
  S.corpus = cfg.corpus || [];
  S.backend = cfg.backend || {};

  applyTheme();
  applyBrand();
  applyStaticCopy();
  renderMenus();
  restoreSplits();
  configureScope();
  renderStages();
  renderCorpus();
  bindControls();
  startClock();
  pollTelemetry();

  S.mapMode = (S.config.map && S.config.map.defaultMode) || 'sat';
  S.stage = (S.config.pipeline && S.config.pipeline.defaultStage) || 'dis';
  // This had been hard-coded to geojson, so `export.defaultFormat` in the
  // config was quietly doing nothing.
  const fmts = (S.config.export || {}).formats || [];
  const wanted = (S.config.export || {}).defaultFormat;
  S.format = fmts.includes(wanted) ? wanted : (fmts[0] || 'raw');
  syncModeButtons();

  const shown = corpus();
  if (shown.length) {
    selectDoc(shown[0].doc_id,
              (S.config.pipeline && S.config.pipeline.autorun) !== false);
  } else {
    setStatus(T.status.ready, 'no corpus configured — paste text to parse');
    renderAll();
  }
}

/** Which sample reports belong to the current look.
 *
 * The ops corpus opens on a convoy movement report filed from the Sahel and
 * a declassified 1977 cable. Those documents are the first thing anyone reads
 * on this screen, and no palette makes them the right thing to put in front
 * of a humanitarian audience -- so the corpus is part of the theme. A
 * document with no `themes` key belongs to every theme.
 */
function corpus() {
  return S.corpus.filter(d => !d.themes || d.themes.includes(S.theme));
}

const themeDef = () => (S.config.themes || {})[S.theme] || {};

/* Every token any palette in any theme sets. Switching themes has to clear
 * these before writing the new ones: a token the ops palettes define and the
 * field ones do not would otherwise survive the switch as a stale inline
 * style on :root, which is the kind of bug that only shows up on the second
 * toggle. */
const paletteKeys = () => [...new Set(
  Object.values(S.config.themes || {})
    .flatMap(th => Object.values(th.palettes || {}))
    .flatMap(pal => Object.keys(pal)))];

/** Install a theme: its palette, its copy, its ornament flags, its fonts.
 *
 * Everything downstream reads `T`, `CH`, and the custom properties this sets;
 * nothing else needs to know which theme is on. The map is the one exception,
 * and only because it paints into a shadow root -- see `setTheme`.
 */
/** The `routes` entry for the URL this page was opened at, if any.
 *
 * `/demo` and `/console` are the same document; what differs is which theme
 * comes up. Handing someone a link that lands on the right console beats
 * handing them one plus an instruction to change a setting -- and it means a
 * projector, a phone and a colleague's laptop all open the same way, which
 * `localStorage` on its own cannot promise.
 */
function routeFor(path) {
  const r = (S.config.routes || {})[path || location.pathname];
  return (r && (S.config.themes || {})[r.theme]) ? r : null;
}

/** The URL that opens a given theme directly, if one is configured. */
const slugFor = name => Object.entries(S.config.routes || {})
  .find(([, r]) => r.theme === name)?.[0] || null;

function applyTheme(name) {
  const themes = S.config.themes || {};
  const names = Object.keys(themes);
  S.route = routeFor();
  // Precedence: an explicit switch this session, then the URL, then the
  // remembered preference, then the config. The URL outranks the preference
  // because it was typed on purpose and the preference was not.
  const want = name || (S.route && S.route.theme)
    || store('theme') || (S.config.theme || {}).name;
  S.theme = themes[want] ? want : names[0];
  const th = themeDef();

  const pals = th.palettes || {};
  const savedPal = store(`pal.${S.theme}`);
  S.palette = pals[savedPal] ? savedPal
    : (pals[th.defaultPalette] ? th.defaultPalette : Object.keys(pals)[0]);

  T = mergeCopy(COPY, th.copy || {});
  // The menu writes into S.chrome, and those choices are the user's -- but
  // they are per-theme, because "scanlines off" is meaningless under a theme
  // that has no scanlines to begin with.
  let saved = {};
  try { saved = JSON.parse(store(`chrome.${S.theme}`) || '{}'); } catch { /* ignore */ }
  S.chrome = { upper: true, grit: 0.55, ...(th.chrome || {}), ...saved };
  CH = S.chrome;

  const root = document.documentElement;
  for (const k of paletteKeys()) root.style.removeProperty(k);
  applyPalette();

  const f = th.fonts || {};
  if (f.ui) root.style.setProperty('--cond', `'${f.ui}',system-ui,sans-serif`);
  if (f.mono) root.style.setProperty('--mono', `'${f.mono}',ui-monospace,monospace`);
  if (f.serif) root.style.setProperty('--serif', `'${f.serif}',Georgia,serif`);

  root.dataset.theme = S.theme;
  root.dataset.layout = store('layout') || (S.config.theme || {}).layout || 'split';
  root.dataset.chrome = (S.config.theme || {}).chrome || 'full';
  applyChrome();
}

/** The ornament flags, as attributes the stylesheet and the map can see. */
function applyChrome() {
  const root = document.documentElement;
  root.style.setProperty('--grit', CH.grit == null ? 0.55 : CH.grit);
  root.dataset.scanlines = CH.scanlines === false ? 'off' : 'on';
  root.dataset.vignette = CH.vignette === false ? 'off' : 'on';
  root.dataset.noise = CH.flicker === false ? 'off' : 'on';
  root.dataset.clock = CH.clock === false ? 'off' : 'on';
  root.dataset.hairline = CH.hairline === false ? 'off' : 'on';
  root.dataset.chrome = CH.ornaments === false ? 'stripped' : 'full';
}

function applyPalette() {
  const pal = (themeDef().palettes || {})[S.palette] || {};
  for (const [k, v] of Object.entries(pal)) {
    document.documentElement.style.setProperty(k, v);
  }
  document.documentElement.dataset.pal = S.palette;
}

/** Switch the whole look.
 *
 * The rebuild list is long because almost every string on the screen comes
 * from `T` and almost every cached render key is keyed on content rather than
 * on theme -- `renderStage` in particular would otherwise decide nothing had
 * changed and leave the old words in place. Clearing the two `data-built`
 * keys is what makes the switch total.
 */
function setTheme(name) {
  if (!(S.config.themes || {})[name] || name === S.theme) return;
  const wasShowing = S.docId;
  applyTheme(name);
  store('theme', name);
  // Keep the address bar honest, so the link someone copies mid-demo opens
  // what they were looking at. replaceState rather than pushState: switching a
  // theme is not somewhere to go Back to.
  const slug = slugFor(name);
  if (slug && slug !== location.pathname) {
    history.replaceState({}, '', slug + location.search + location.hash);
    S.route = routeFor(slug);
  }

  const el = $('#stage-body'); if (el) el.dataset.built = '';
  const dt = $('#doc-text'); if (dt) dt.dataset.built = '';

  applyBrand();
  applyStaticCopy();
  renderMenus();
  renderStages();
  renderCorpus();
  configureScope();          // the calm variant is a scope option
  $('#scope').refresh();
  syncModeButtons();
  syncRunLabel();

  // The corpus is part of the theme, so the document on screen may not belong
  // to the theme being switched to. Load that theme's first report instead --
  // but never throw away text the visitor pasted themselves.
  const shown = corpus();
  const stillThere = shown.some(d => d.doc_id === wasShowing);
  if (!stillThere && S.docId !== '__paste__' && shown.length) {
    selectDoc(shown[0].doc_id,
              (S.config.pipeline && S.config.pipeline.autorun) !== false);
  } else {
    renderAll();
  }
  pushLog(`look ${name}`);
}

function setPalette(name) {
  if (!(themeDef().palettes || {})[name]) return;
  S.palette = name;
  store(`pal.${S.theme}`, name);
  applyPalette();
  renderMenus();
  // The map reads --acc / --map-land / --map-tint off its host through
  // getComputedStyle at paint time, so it needs to be told to paint again;
  // nothing else on the screen does.
  $('#scope').refresh();
  pushLog(`palette ${name}`);
}

function setChrome(key, on) {
  S.chrome = { ...S.chrome, [key]: on };
  CH = S.chrome;
  const keep = {};
  for (const k of ['scanlines', 'vignette', 'flicker', 'sweep', 'hairline',
                   'ornaments', 'upper', 'clock']) {
    if (k in S.chrome) keep[k] = S.chrome[k];
  }
  store(`chrome.${S.theme}`, JSON.stringify(keep));
  applyChrome();
  if (key === 'sweep') { configureScope(); $('#scope').refresh(); }
  if (key === 'upper') { renderAll(); renderStages(); renderCorpus(); }
  renderMenus();
}

/* ── menu bar ─────────────────────────────────────────────────────────────── */

/* One declarative spec, one renderer. The point of the bar is that the next
 * thing worth turning on and off is a row in this array and nothing else --
 * which is why the item types are generic (`radio`, `toggle`, `action`,
 * `swatches`) rather than one bespoke control per feature.
 */
function menuSpec() {
  const themes = S.config.themes || {};
  const pals = themeDef().palettes || {};
  const layout = document.documentElement.dataset.layout;
  const modes = [['sat', T.menu.modeSat], ['imagery', T.menu.modeImagery],
                 ['wire', T.menu.modeWire]];
  const imageryOn = (S.config.map || {}).imagery
    && (S.config.map || {}).imagery.enabled !== false;

  // `routes["/demo"].lock` pins a URL to one look: the menu then offers that
  // theme's palettes and nothing else. It is a presentation choice, not a
  // security boundary -- both consoles run the same engine over the same
  // corpus file, and neither holds anything the other does not.
  const offered = (S.route && S.route.lock) ? { [S.theme]: themes[S.theme] } : themes;
  const lookItems = Object.keys(offered).length > 1
    ? [...Object.entries(offered).map(([k, th]) => ({
         type: 'radio', label: th.label || k, hint: th.hint,
         on: k === S.theme, act: () => setTheme(k),
       })),
       { type: 'sep' }, { type: 'head', label: T.menu.palette }]
    : [{ type: 'head', label: T.menu.palette }];

  return [
    { label: T.menu.look, items: [...lookItems, { type: 'swatches', pals }] },
    { label: T.menu.chrome, items: [
      { type: 'toggle', label: T.menu.ornaments,
        on: CH.ornaments !== false, act: v => setChrome('ornaments', v) },
      { type: 'toggle', label: T.menu.scanlines,
        on: CH.scanlines !== false, act: v => setChrome('scanlines', v) },
      { type: 'toggle', label: T.menu.vignette,
        on: CH.vignette !== false, act: v => setChrome('vignette', v) },
      { type: 'toggle', label: T.menu.grain,
        on: CH.flicker !== false, act: v => setChrome('flicker', v) },
      { type: 'toggle', label: T.menu.sweep,
        on: CH.sweep !== false, act: v => setChrome('sweep', v) },
      { type: 'toggle', label: T.menu.hairline,
        on: CH.hairline !== false, act: v => setChrome('hairline', v) },
    ] },
    { label: T.menu.map, items: [
      ...modes.map(([m, label]) => ({
        type: 'radio', label,
        on: S.mapMode === m,
        disabled: m === 'imagery' && !imageryOn,
        act: () => setMapMode(m),
      })),
      { type: 'sep' },
      { type: 'toggle', label: T.menu.boundaries,
        on: boundariesOn(), act: v => { S.boundaries = v; renderMap(); renderMenus(); } },
    ] },
    { label: T.menu.panes, items: [
      { type: 'radio', label: T.menu.layoutSplit, on: layout !== 'theater',
        act: () => setLayout('split') },
      { type: 'radio', label: T.menu.layoutTheater, on: layout === 'theater',
        act: () => setLayout('theater') },
      { type: 'sep' },
      { type: 'action', label: T.menu.resetPanes, act: () => resetSplits() },
    ] },
  ];
}

const boundariesOn = () => S.boundaries != null
  ? S.boundaries : (S.config.boundaries || {}).enabled !== false;

function setLayout(name) {
  document.documentElement.dataset.layout = name;
  store('layout', name);
  renderMenus();
  pushLog(`layout ${name}`);
}

function renderMenus() {
  const bar = $('#menus');
  if (!bar) return;
  const open = bar.querySelector('.menu.open');
  const openAt = open ? Number(open.dataset.mi) : -1;

  bar.innerHTML = menuSpec().map((m, mi) => `
    <div class="menu${mi === openAt ? ' open' : ''}" data-mi="${mi}">
      <button class="mtrig" aria-haspopup="true"
              aria-expanded="${mi === openAt}">${esc(m.label)}</button>
      <div class="mdrop" role="menu">${m.items.map((it, ii) =>
        menuItemHtml(it, mi, ii)).join('')}</div>
    </div>`).join('');

  bar.querySelectorAll('.mtrig').forEach(btn => {
    const menu = btn.parentElement;
    btn.onclick = ev => {
      ev.stopPropagation();
      const wasOpen = menu.classList.contains('open');
      bar.querySelectorAll('.menu').forEach(m => m.classList.remove('open'));
      menu.classList.toggle('open', !wasOpen);
      btn.setAttribute('aria-expanded', String(!wasOpen));
    };
    // Once one menu is open, sliding along the bar should walk between them
    // rather than needing a click per menu -- the behaviour of every menu bar
    // this is imitating.
    btn.onmouseenter = () => {
      if (!bar.querySelector('.menu.open')) return;
      bar.querySelectorAll('.menu').forEach(m => m.classList.remove('open'));
      menu.classList.add('open');
    };
  });

  bar.querySelectorAll('[data-act]').forEach(el => {
    const [mi, ii] = el.dataset.act.split(':').map(Number);
    el.onclick = ev => {
      ev.stopPropagation();
      const it = menuSpec()[mi].items[ii];
      if (it.disabled) return;
      closeMenus();
      // A toggle passes the value it is moving to; the others take no
      // argument and ignore it.
      it.act(it.type === 'toggle' ? !it.on : undefined);
    };
  });
  bar.querySelectorAll('[data-palpick]').forEach(el => {
    el.onclick = ev => { ev.stopPropagation(); closeMenus(); setPalette(el.dataset.palpick); };
  });
}

function menuItemHtml(it, mi, ii) {
  const at = `${mi}:${ii}`;
  if (it.type === 'sep') return '<div class="msep"></div>';
  if (it.type === 'head') return `<div class="mhead">${esc(it.label)}</div>`;
  if (it.type === 'swatches') {
    const names = Object.keys(it.pals);
    return `<div class="palrow" id="palrow">${names.map(n => {
      const pal = it.pals[n];
      const bg = `linear-gradient(90deg,${pal['--acc']} 0 50%,${pal['--map-land']} 50% 100%)`;
      return `<button data-pal="${esc(n)}" data-palpick="${esc(n)}"
                class="${n === S.palette ? 'on' : ''}" title="${esc(UP(n))}"
                aria-label="${esc(n)} palette"
                style="background:${esc(bg)}"></button>`;
    }).join('')}</div>`;
  }
  const mark = it.type === 'toggle' ? (it.on ? '\u2611' : '\u2610')
    : it.type === 'radio' ? (it.on ? '\u25cf' : '\u25cb') : '\u00a0';
  return `<button class="mitem${it.on ? ' on' : ''}${it.disabled ? ' off' : ''}"
            role="menuitem" data-act="${at}"${it.disabled ? ' disabled' : ''}>
            <span class="mmark">${mark}</span>
            <span class="mlab">${esc(it.label)}</span>
            ${it.hint ? `<span class="mhint">${esc(it.hint)}</span>` : ''}
          </button>`;
}

function closeMenus() {
  document.querySelectorAll('#menus .menu.open')
    .forEach(m => { m.classList.remove('open'); m.querySelector('.mtrig')
      .setAttribute('aria-expanded', 'false'); });
}

/** The strings that live in index.html rather than in a render function. */
function applyStaticCopy() {
  const set = (sel, val) => { const el = $(sel); if (el) el.textContent = val; };
  set('#lab-pipeline', T.railPipeline);
  set('#lab-yourtext', T.railYourText);
  set('#lab-corpus', T.railCorpus);
  set('#filebtn', T.uploadLabel);
  set('#navhint', T.navHint);
  set('#reset', T.resetLabel);
  const ta = $('#paste'); if (ta) ta.placeholder = T.pastePlaceholder;
  const fh = $('#filehint'); if (fh) fh.innerHTML = T.fileHint;
  const tok = $('#tokchip');
  if (tok) {
    tok.querySelector('.toklab').textContent = T.tokenized;
    tok.style.display = T.tokenized ? '' : 'none';
  }
  syncRunLabel();
}

function syncRunLabel() {
  const el = $('#runlabel');
  if (el) el.textContent = S.running ? T.running : (S.result ? T.runAgain : T.run);
}

function applyBrand() {
  const b = S.config.brand || {};
  $('#brand-name').textContent = T.brandName || b.productName || 'MORDECAI';
  $('#brand-sub').textContent = T.brandSub || b.productSubtitle || 'GEOPARSE ENGINE';
  const badge = $('#demobadge');
  if (badge) {
    badge.textContent = b.demoBadge || '';
    badge.style.display = (b.showDemoBadge && b.demoBadge) ? '' : 'none';
  }

  // The design's title bar carried a classification chip and an invented
  // device readout. These slots hold what the backend actually reports
  // instead; a made-up number in a status bar is worse than an empty slot.
  const be = S.backend || {};
  const pairs = [
    [T.stats.build, b.build || '—'],
    [T.stats.model, be.model || '—'],
    [T.stats.spans, be.span_detector || '—'],
    [T.stats.device, UP(be.device || '—')],
  ];
  $('#title-stats').innerHTML = pairs
    .map(([k, v]) => `<span><b>${k}</b>${esc(v)}</span>`).join('');

  const bd = be.boundaries || {};
  $('#gazfoot').innerHTML = `${esc(T.gazPrefix)}<br>${esc(T.boundaryPrefix)} · `
    + (bd.available ? `${fmtInt(bd.shapes)} ${esc(T.boundaryShapes)}`
                    : esc(T.boundaryNone));
}

function configureScope() {
  const m = S.config.map || {};
  const bd = S.config.boundaries || {};
  $('#scope').configure({
    atlas: (m.basemap && m.basemap.path) || '/vendor/countries-110m.json',
    minZoom: m.minZoom, maxZoom: m.maxZoom,
    fitPaddingRatio: m.fitPaddingRatio,
    fitMaxBlowUp: m.fitMaxBlowUp,
    fitMinPinSpanDeg: m.fitMinPinSpanDeg,
    graticuleStepDeg: m.graticuleStepDeg,
    graticuleMajorStepDeg: m.graticuleMajorStepDeg,
    sweepSeconds: m.sweepSeconds,
    pan: m.pan !== false, zoom: m.zoom !== false,
    calm: !!(themeDef().map || {}).calm,
    sweep: CH.sweep !== false,
    upper: CH.upper !== false,
    boundaryFillOpacity: bd.fillOpacity,
    boundaryStrokeWidth: bd.strokeWidth,
    weakMatchBelow: bd.weakMatchBelow,
    imagery: (m.imagery && m.imagery.enabled !== false) ? m.imagery : null,
  });

  // The one layer on this map that is not vendored, so it is the one that can
  // be missing. The component falls back to the procedural relief on its own;
  // this says so out loud rather than letting the switch look like a no-op.
  $('#scope').addEventListener('imagerystate', e => {
    if (e.detail.ok === false) {
      S.imageryDown = true;
      setStatus('READY', 'satellite tiles unreachable — falling back to '
        + 'procedural relief (the rest of the console is offline-capable)');
      renderMap();
    }
  });
  if (!(m.imagery && m.imagery.enabled !== false)) {
    const b = $('#mapmode').querySelector('[data-mode="imagery"]');
    if (b) { b.disabled = true; b.title = 'map.imagery is disabled in the config'; }
  }
}

/* ── rail ─────────────────────────────────────────────────────────────────── */

function renderStages() {
  $('#stages').innerHTML = STAGE_KEYS.map((key, i) =>
    `<div class="stage-row${key === S.stage ? ' on' : ''}" data-stage="${key}">
       <span class="idx">${String(i + 1).padStart(2, '0')}</span>
       <span class="nm">${esc(T.stages[key] || key)}</span>
       <span class="mt" data-stagemeta="${key}"></span>
       <span class="sd"></span>
     </div>`).join('');
  $('#stages').querySelectorAll('[data-stage]').forEach(el => {
    el.onclick = () => { S.stage = el.dataset.stage; renderStages(); renderStage(); };
  });
  updateStageMeta();
}

function updateStageMeta() {
  const n = entities().length;
  const k = (S.config.pipeline || {}).topK || 5;
  const m = T.stageMeta;
  const meta = {
    ingest: corpus().length ? `${corpus().length} ${m.docs}` : '—',
    parse: n ? `${n} ${m.spans}` : '—',
    dis: `${m.topk} ${k}`,
    resolve: S.result ? `${S.result.stats.resolved}/${n}` : '—',
    export: `${((S.config.export || {}).formats || []).length} ${m.fmt}`,
  };
  for (const [key, val] of Object.entries(meta)) {
    const el = document.querySelector(`[data-stagemeta="${key}"]`);
    if (el) el.textContent = val;
  }
}

function renderCorpus() {
  $('#corpus').innerHTML = corpus().map(d =>
    `<div class="doc-card${d.doc_id === S.docId ? ' on' : ''}" data-doc="${esc(d.doc_id)}">
       <div class="t"><b>${esc(UP(d.title))}</b><i>${fmtBytes(d.text.length)}</i></div>
       <div class="s">${esc(UP(d.source))}</div>
     </div>`).join('');
  $('#corpus').querySelectorAll('[data-doc]').forEach(el => {
    el.onclick = () => selectDoc(el.dataset.doc, true);
  });
}

function bindControls() {
  $('#run').onclick = () => {
    const pasted = $('#paste').value.trim();
    if (pasted) {
      S.docId = '__paste__';
      S.doc = { doc_id: '__paste__', title: T.pastedTitle, kind: 'paste',
                source: T.pastedSource,
                meta: fill(T.pastedMeta, { n: fmtInt(pasted.length) }),
                region: '', text: pasted };
      renderCorpus();
    }
    runParse();
  };

  $('#mapmode').querySelectorAll('[data-mode]').forEach(el => {
    el.onclick = () => setMapMode(el.dataset.mode);
  });

  // A click anywhere else closes an open menu; Escape does too. Both are what
  // a menu bar is expected to do, and without them the only way out of a
  // dropdown is to hit its trigger again.
  document.addEventListener('click', () => closeMenus());
  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape') closeMenus();
  });

  bindSplitters();

  $('#reset').onclick = () => $('#scope').resetView();
  $('#scope').addEventListener('viewchange', e => {
    S.zoom = e.detail;
    $('#zoomchip').textContent = '×' + S.zoom.toFixed(1);
    // Zooming changes which tile level is on screen, and the badges name it.
    // setScene is a no-op when nothing else moved, so this is cheap.
    if (S.mapMode === 'imagery') renderMap();
  });
  // The map speaks in places, the rest of the console speaks in mentions.
  $('#scope').addEventListener('pinhover', e => setHover(mentionForPlace(e.detail)));
  $('#scope').addEventListener('pinclick', e => select(mentionForPlace(e.detail)));

  $('#upload').onchange = ev => {
    const file = ev.target.files && ev.target.files[0];
    if (file) runBatch(file);
    ev.target.value = '';
  };

  // Keyboard: step through entities without hunting for spans.
  document.addEventListener('keydown', ev => {
    if (ev.target.tagName === 'TEXTAREA') return;
    const ids = entities().map(e => e.id);
    if (!ids.length) return;
    const i = ids.indexOf(S.selected);
    if (ev.key === 'ArrowDown' || ev.key === 'j') {
      ev.preventDefault(); select(ids[Math.min(i + 1, ids.length - 1)] || ids[0]);
    } else if (ev.key === 'ArrowUp' || ev.key === 'k') {
      ev.preventDefault(); select(ids[Math.max(i - 1, 0)]);
    }
  });
}

/* ── resizable panes ──────────────────────────────────────────────────────── */

/* Three seams, one mechanism. Each handle owns one custom property; dragging
 * writes it to :root as an inline style, which is what the grid and the
 * handles themselves are laid out from, so there is no second copy of the
 * geometry to keep in step. The clamps are what stop a drag from producing a
 * pane too narrow to hold its own header.
 */
const SPLITS = {
  'gut-rail': { prop: '--rail-w',   axis: 'x', unit: 'px', min: 158, max: 460,
                box: () => $('[data-el="shell"]') },
  'gut-col':  { prop: '--split-col', axis: 'x', unit: '%', min: 16, max: 80,
                box: () => $('[data-el="main"]') },
  'gut-row':  { prop: '--split-row', axis: 'y', unit: '%', min: 16, max: 84,
                box: () => $('[data-el="main"]') },
};

function bindSplitters() {
  for (const [id, sp] of Object.entries(SPLITS)) {
    const el = document.getElementById(id);
    if (!el) continue;

    el.addEventListener('pointerdown', ev => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      el.setPointerCapture(ev.pointerId);
      el.classList.add('drag');
      document.body.classList.add('resizing');
      document.body.classList.toggle('rowdrag', sp.axis === 'y');

      const move = e => {
        const r = sp.box().getBoundingClientRect();
        const raw = sp.unit === 'px'
          ? e.clientX - r.left
          : (sp.axis === 'x' ? (e.clientX - r.left) / r.width
                             : (e.clientY - r.top) / r.height) * 100;
        const v = Math.max(sp.min, Math.min(sp.max, raw));
        document.documentElement.style.setProperty(
          sp.prop, sp.unit === 'px' ? `${Math.round(v)}px` : `${v.toFixed(2)}%`);
      };
      const up = () => {
        el.removeEventListener('pointermove', move);
        el.removeEventListener('pointerup', up);
        el.classList.remove('drag');
        document.body.classList.remove('resizing', 'rowdrag');
        store('split', JSON.stringify(readSplits()));
      };
      el.addEventListener('pointermove', move);
      el.addEventListener('pointerup', up);
    });

    // A dragged pane is easy to get wrong and annoying to nudge back by hand.
    el.addEventListener('dblclick', () => {
      document.documentElement.style.removeProperty(sp.prop);
      store('split', JSON.stringify(readSplits()));
      pushLog(`reset ${sp.prop.replace('--', '')}`);
    });
  }
}

const readSplits = () => Object.fromEntries(
  Object.values(SPLITS)
    .map(sp => [sp.prop, document.documentElement.style.getPropertyValue(sp.prop)])
    .filter(([, v]) => v));

function resetSplits() {
  for (const sp of Object.values(SPLITS)) {
    document.documentElement.style.removeProperty(sp.prop);
  }
  store('split', '{}');
  pushLog('reset pane sizes');
}

function restoreSplits() {
  let saved;
  try { saved = JSON.parse(store('split') || '{}'); } catch { return; }
  const known = new Set(Object.values(SPLITS).map(sp => sp.prop));
  for (const [k, v] of Object.entries(saved)) {
    // Only properties this build still recognises, and only values that look
    // like the lengths they are meant to be -- localStorage is user-writable
    // and this ends up in a style attribute.
    if (known.has(k) && /^\d+(\.\d+)?(px|%)$/.test(String(v))) {
      document.documentElement.style.setProperty(k, v);
    }
  }
}

function setMapMode(mode) {
  S.mapMode = mode;
  if (mode === 'imagery') S.imageryDown = false;   // selecting it retries
  syncModeButtons(); renderMap(); renderMenus();
}

function syncModeButtons() {
  const label = { sat: T.menu.modeSat, imagery: T.menu.modeImagery,
                  wire: T.menu.modeWire };
  $('#mapmode').querySelectorAll('[data-mode]').forEach(el => {
    el.classList.toggle('on', el.dataset.mode === S.mapMode);
    el.textContent = label[el.dataset.mode] || el.dataset.mode;
  });
}

/* ── parse run ────────────────────────────────────────────────────────────── */

function selectDoc(docId, run) {
  const d = S.corpus.find(x => x.doc_id === docId);
  if (!d) return;
  S.docId = docId; S.doc = d;
  S.result = null; S.selected = null; S.hovered = null;
  S.revealed.clear(); S.settled = false;
  renderCorpus();
  renderAll();
  if (run) runParse();
}

async function runParse() {
  if (!S.doc || S.running) return;
  const p = S.config.pipeline || {};
  S.running = true; S.settled = false; S.revealed.clear();
  S.selected = null; S.hovered = null; S.scanning = null;
  $('#run').disabled = true; syncRunLabel();
  setStatus(T.status.parsing,
            `${S.doc.title.toLowerCase()} — ${S.doc.text.length} chars`);
  renderAll();

  const t0 = performance.now();
  try {
    const res = await fetch('/api/geoparse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id: S.docId, text: S.doc.text,
        options: { top_k: p.topK || 5, review_gate: p.reviewGate ?? 0.4 },
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 160)}`);
    S.result = await res.json();
  } catch (err) {
    S.running = false;
    $('#run').disabled = false; syncRunLabel();
    setStatus(T.status.error, String(err.message || err));
    renderAll();
    return;
  }

  const wall = Math.round(performance.now() - t0);
  setStatus(T.status.parsing, `${entities().length} spans in ${wall} ms round-trip`);
  await reveal();

  S.running = false; S.settled = true;
  $('#run').disabled = false; syncRunLabel();

  // Come to rest on a disambiguation problem rather than a trivial success --
  // but on a real one. A mention the model *declined* is flagged too, and its
  // panel is a list of candidates it rejected, which is the least informative
  // thing the console can be showing when someone first looks at it. Prefer a
  // flagged mention that actually resolved: that is a genuine "these two are
  // both plausible" case, which is what the panel is for.
  const ents = entities();
  const rest = ents.find(e => e.review && e.resolved)
    || ents.find(e => e.review)
    || ents[0];
  if (rest) { S.selected = rest.id; S.stage = 'dis'; renderStages(); }

  const st = S.result.stats;
  setStatus(T.status.resolved,
    `${st.resolved}/${st.spans} ${T.foot.resolved} · ${st.flagged} ${T.foot.flagged} · `
    + `${st.places ?? '?'} ${T.foot.places} · `
    + `${st.places_with_boundary ?? st.with_boundary} ${T.foot.boundaries} · `
    + `${S.result.timing_ms.total} ms`);
  renderAll();
}

/** Step the entities in document order, so the pins land one at a time.
 *
 * The handoff notes this could be driven by a real per-entity stream. It
 * cannot be here, honestly: `geoparse_batch` pools every mention in a document
 * into a single model forward pass, so there is no partial result to stream --
 * the whole document resolves at once. This is a replay of a complete answer,
 * and it is presentation rather than progress. Set `pipeline.spanRevealMs` to
 * 0 to skip it.
 */
async function reveal() {
  const p = S.config.pipeline || {};
  const stepMs = p.spanRevealMs ?? 60;
  const gapMs = p.spanGapMs ?? 20;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ents = entities();

  if (!stepMs || reduce) {
    ents.forEach(e => S.revealed.add(e.id));
    renderAll();
    return;
  }

  // A parse that takes 30 ms should not be followed by four seconds of
  // theatre; the animation is supposed to read as speed, and past a couple of
  // seconds it reads as waiting instead. So the per-entity step is fast, and
  // a whole-run budget caps what a long document can spend: past the point
  // where one-at-a-time would overrun it, entities land in small groups
  // rather than the steps stretching out. The pins still arrive in document
  // order, which is the part worth keeping.
  const budget = p.revealBudgetMs ?? 1400;
  const perStep = stepMs + gapMs;
  const steps = Math.max(1, Math.min(ents.length, Math.floor(budget / perStep)));
  const size = Math.ceil(ents.length / steps);

  await sleep(p.runStartDelayMs ?? 80);
  for (let i = 0; i < ents.length; i += size) {
    const group = ents.slice(i, i + size);
    group.forEach(e => S.revealed.add(e.id));
    // `scanning` is a single-entity state and the head of the group is the
    // honest one to point at: it is the entity the log line names.
    S.scanning = group[0].id; S.selected = group[0].id;
    for (const e of group) {
      pushLog(`rank  ${e.text} → ${e.resolved
        ? `${e.resolved.name} ${e.resolved.confidence.toFixed(2)}`
        : 'no match'} · k=${e.candidates.length}`);
    }
    renderAll();
    await sleep(stepMs);
    S.scanning = null;
    renderAll();
    await sleep(gapMs);
  }
}

async function runBatch(file) {
  const p = S.config.pipeline || {};
  S.running = true;
  $('#run').disabled = true; $('#runlabel').textContent = T.batching;
  setStatus(T.status.parsing, `batch: ${file.name} (${fmtBytes(file.size)})`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch(
      `/api/batch?top_k=${p.topK || 5}&review_gate=${p.reviewGate ?? 0.4}`,
      { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 160)}`);
    S.batch = await res.json();
  } catch (err) {
    S.running = false;
    $('#run').disabled = false; syncRunLabel();
    setStatus(T.status.error, String(err.message || err));
    return;
  }
  S.running = false;
  $('#run').disabled = false; syncRunLabel();

  // Show the first document of the batch in the main panes, and the whole
  // batch's totals under INGEST.
  const first = S.batch.documents[0];
  if (first) {
    S.docId = first.doc_id;
    S.doc = { doc_id: first.doc_id, title: UP(first.doc_id),
              kind: 'paste', source: S.batch.filename,
              meta: fill(T.batchMeta, { n: S.batch.stats.documents }), region: '',
              text: first.text };
    S.result = first;
    S.revealed = new Set(first.entities.map(e => e.id));
    S.settled = true;
    const flagged = first.entities.find(e => e.review);
    S.selected = (flagged || first.entities[0] || {}).id || null;
  }
  const st = S.batch.stats;
  S.stage = 'ingest'; renderStages();
  setStatus(T.status.resolved,
    `batch ${st.documents} docs · ${st.spans} ${T.foot.spans} · `
    + `${st.resolved} ${T.foot.resolved} · ${st.flagged} ${T.foot.flagged} · `
    + `${S.batch.timing_ms.total} ms`);
  renderCorpus();
  renderAll();
}

/* ── selection ────────────────────────────────────────────────────────────── */

function select(id) {
  if (!id) return;
  S.selected = id; S.hovered = null; S.ghostCand = null;
  if (S.stage !== 'dis' && S.stage !== 'resolve') { S.stage = 'dis'; renderStages(); }
  renderAll();
}

function setHover(id) {
  if (S.hovered === id) return;
  // The candidate rows are about to be rebuilt under the pointer, so their
  // mouseleave will never fire; drop the ghost they summoned here instead.
  S.hovered = id; S.ghostCand = null;
  renderAll();
}

/* ── render ───────────────────────────────────────────────────────────────── */

function renderAll() {
  renderDoc();
  renderMap();
  renderStage();
  updateStageMeta();
}

/** The document text with entity spans woven in.
 *
 * Built from character offsets, which the backend guarantees index this exact
 * string. Trap from the handoff worth keeping in mind: the container is
 * `pre-wrap`, so any whitespace *between* the emitted elements shows up as a
 * line break -- the segments below are joined with nothing.
 */
function renderDoc() {
  const el = $('#doc-text');
  if (!S.doc) { el.textContent = ''; return; }
  $('#doc-title').textContent = UP(S.doc.title || '');
  $('#doc-meta').textContent = UP(S.doc.meta || '');
  el.dataset.kind = S.doc.kind || 'paste';

  const text = S.doc.text;
  const ents = entities();
  if (!ents.length) {
    el.textContent = text;
    el.dataset.built = '';
    $('#doc-foot').innerHTML =
      `<span>${esc(S.running ? T.foot.parsing : T.foot.none)}</span>`;
    return;
  }

  // Hover changes `active`, which changes only which spans are highlighted --
  // the segmentation is identical. Rebuilding innerHTML for that would
  // re-create every node on every mouse move, which throws away any text the
  // user has selected and is pure waste. Mutate the attributes instead and
  // only rebuild when the document or the entity set actually changed.
  const key = `${S.docId}|${S.result.timing_ms.total}|${ents.length}`;
  if (el.dataset.built === key) {
    ents.forEach(e => {
      const span = el.querySelector(`[data-eid="${cssEsc(e.id)}"]`);
      if (span) span.dataset.st = spanState(e);
    });
    updateDocFoot();
    return;
  }

  const parts = [];
  let cursor = 0;
  for (const e of ents) {
    if (e.start > cursor) parts.push(escText(text.slice(cursor, e.start)));
    parts.push(
      // `data-place` is the join to the map: several spans carry the same one
      // and it is the id of the single marker that stands for them.
      `<span data-eid="${esc(e.id)}" data-place="${esc(placeKey(e) || '')}" `
      + `data-st="${spanState(e)}" `
      + `title="${esc(spanTitle(e))}">${escText(text.slice(e.start, e.end))}</span>`);
    cursor = e.end;
  }
  if (cursor < text.length) parts.push(escText(text.slice(cursor)));
  el.innerHTML = parts.join('');   // no whitespace between segments
  el.dataset.built = key;

  el.querySelectorAll('[data-eid]').forEach(span => {
    span.onclick = () => select(span.dataset.eid);
    span.onmouseenter = () => setHover(span.dataset.eid);
    span.onmouseleave = () => setHover(null);
  });
  updateDocFoot();
}

function updateDocFoot() {
  const st = S.result.stats;
  const f = T.foot;
  $('#doc-foot').innerHTML =
    `<span>${st.spans} ${esc(f.spans)}</span>`
    + `<span class="ok">${st.resolved} ${esc(f.resolved)}</span>`
    + `<span>${st.places ?? places().length} ${esc(f.places)}</span>`
    + `<span class="amb">${st.flagged} ${esc(f.flagged)}</span>`
    // Places, not mentions: this sits beside the map and has to agree with
    // the number of polygons drawn on it.
    + `<span>${st.places_with_boundary ?? st.with_boundary} ${esc(f.boundaries)}</span>`
    + `<span class="rule" data-orn></span>`
    + `<span>${S.result.timing_ms.total} ms · ${S.result.token_count} tok</span>`;
}

function spanState(e) {
  if (S.scanning === e.id) return 'scan';
  if (!S.revealed.has(e.id)) return 'pending';
  if (active() === e.id) return 'sel';
  // Another mention of the same place is active. This is the other half of
  // deduplicating the map: one pin now stands for several spans, so it has to
  // be able to point back at all of them.
  const ap = activePlace();
  if (ap && placeKey(e) === ap) return 'link';
  return e.review ? 'amb' : 'ok';
}

function spanTitle(e) {
  if (!e.resolved) return `${e.text} — no match (p=${e.p_no_match})`;
  const r = e.resolved;
  return `${r.name} · ${r.feature_code} · ${r.country_code3} · ${r.confidence}`;
}

function renderMap() {
  const scope = $('#scope');
  const ents = entities();
  const drawBoundaries = boundariesOn();

  // One pin per place, not per mention. See the note above `placeKey`.
  const pins = places().map(p => ({
    id: p.key,
    lat: p.resolved.lat, lon: p.resolved.lon,
    label: p.resolved.name,
    // Mentions of one record can score differently -- the sentences around
    // them differ -- so the marker carries the best of them and the panel
    // carries the one being read.
    conf: Math.max(...p.mentions.map(m => m.resolved.confidence)),
    count: p.mentions.length,
    // A pin appears once its first mention has been revealed, and reads as
    // flagged if any mention of it is: a place worth a second look in one
    // sentence is worth a second look on the map.
    status: !p.mentions.some(m => S.revealed.has(m.id)) ? 'pending'
      : (p.mentions.some(m => m.review) ? 'ambiguous' : 'ok'),
    boundary: drawBoundaries ? p.boundary : null,
  }));

  const cur = entityById(active());
  scope.setScene({ pins, ghosts: ghostsFor(cur),
                   mode: S.mapMode, active: activePlace() });

  $('#map-region').textContent = UP((S.doc && S.doc.region) || regionOf(ents));
  const im = (S.config.map || {}).imagery || {};
  const live = S.mapMode === 'imagery' && !S.imageryDown;
  $('#map-modemeta').textContent =
    live ? `${im.source || 'RASTER'} · ${T.mapProjection}`
    : `${S.mapMode === 'wire' ? T.mapVector : T.mapRelief} · ${T.mapProjection}`;
  // Every badge here is a claim about the map, so each one has to be true of
  // the pixels actually on screen -- including that the imagery is being
  // stretched past the resolution the layer publishes.
  const badges = live
    ? [im.attribution || 'RASTER IMAGERY',
       `${esc(im.layer || '')} · Z${scope.dataset.tilezoom || '?'}`,
       scope.dataset.tileoverzoom ? `${im.resolution || ''} · OVER-ZOOMED`
                                  : (im.resolution || 'NATIVE')]
    : S.mapMode === 'wire'
    ? [`${T.mapGraticule} ${scope.dataset.graticule
         || (S.config.map || {}).graticuleStepDeg || 2}°`,
       T.mapBorders, T.mapNoRaster]
    : [T.mapRelief,
       S.imageryDown ? T.mapUnreachable : T.mapNoRaster, T.mapDatum];
  $('#map-badges').innerHTML = badges.map(b => `<span>${b}</span>`).join('');

  if (cur && cur.resolved) {
    const r = cur.resolved;
    const b = cur.boundary;
    $('#map-coord').innerHTML =
      `${Math.abs(r.lat).toFixed(4)}° ${r.lat >= 0 ? 'N' : 'S'}<br>`
      + `${Math.abs(r.lon).toFixed(4)}° ${r.lon >= 0 ? 'E' : 'W'}<br>`
      + `<span class="sub">${esc(r.feature_code)} · ${esc(r.country_code3)}`
      + (b ? ` · ${esc(fill(T.coordPolygon, { level: b.level }))}`
           : ` · ${esc(T.coordPoint)}`) + `</span>`;
  } else {
    $('#map-coord').innerHTML = '';
  }
}

/** The rival candidates to draw on the map, if any.
 *
 * These used to be drawn permanently: every runner-up of the active mention,
 * as dashed warn-coloured rings, with off-frame ones clamped to the edge as
 * bearings. Faithful to the design handoff, and wrong in practice -- an
 * unprompted ring on a map reads as "this place is in the document and
 * something is wrong with it", and on this map red already means "flagged for
 * review", so the ghosts were quietly spending the one colour that had a job.
 * The Sahel corpus document drew rings over Antarctic research stations,
 * which is a candidate list, not a finding.
 *
 * The mechanism is still worth having: that "Niger" the country beat "Niger"
 * the river is the disambiguate panel's whole argument, and a ranked list of
 * names does not convey *distance* between the options. So it is drawn when
 * the reader asks for it -- while the pointer is on a candidate row -- and at
 * no other time. `map.candidateGhosts` switches between that, the old
 * always-on behaviour, and off.
 */
function ghostsFor(cur) {
  const mode = (S.config.map || {}).candidateGhosts || 'hover';
  if (!cur || !S.settled || mode === 'off') return [];
  // The winner already has a pin; a second ring on the same coordinate says
  // nothing.
  const rival = c => c.lat != null
    && (!cur.resolved || c.geonameid !== cur.resolved.geonameid);
  const ghost = (c, i) => ({
    lat: c.lat, lon: c.lon,
    label: `${String(i + 1).padStart(2, '0')} ${c.name}`,
    // Where the mention actually resolved, so the map can draw the leader
    // between the two and show how far apart the options were.
    from: cur.resolved ? [cur.resolved.lon, cur.resolved.lat] : null,
  });
  if (mode === 'always') {
    return cur.candidates.map(ghost).filter((_, i) => rival(cur.candidates[i]));
  }
  const c = S.ghostCand != null ? cur.candidates[S.ghostCand] : null;
  return (c && rival(c)) ? [ghost(c, S.ghostCand)] : [];
}

function regionOf(ents) {
  const countries = [...new Set(ents.filter(e => e.resolved)
    .map(e => e.resolved.country_code3))];
  if (!countries.length) return '';
  return countries.slice(0, 4).join(' · ') + (countries.length > 4 ? ' …' : '');
}

/* ── stage panel ──────────────────────────────────────────────────────────── */

function renderStage() {
  // Same reasoning as renderDoc: a hover that lands on the same entity must
  // not rebuild the panel underneath the pointer.
  const key = `${S.stage}|${S.docId}|${active()}|${S.format}|${S.settled}`
    + `|${entities().length}|${S.batch ? S.batch.filename : ''}`;
  if ($('#stage-body').dataset.built === key) return;
  $('#stage-body').dataset.built = key;

  $('#stage-title').textContent = T.stages[S.stage] || '';
  const body = $('#stage-body');
  const fns = { ingest: stageIngest, parse: stageParse, dis: stageDis,
                resolve: stageResolve, export: stageExport };
  const out = (fns[S.stage] || (() => ({ meta: '', html: '' })))();
  $('#stage-meta').textContent = out.meta || '';
  body.innerHTML = out.html;
  if (out.after) out.after(body);
}

function stageIngest() {
  const rows = corpus().map(d =>
    `<div class="row" data-doc="${esc(d.doc_id)}">
       <span>${esc(UP(d.title))}</span><span class="dim">${esc(UP(d.adapter || '—'))}</span>
       <span class="dim">${fmtBytes(d.text.length)}</span>
       <span class="${d.doc_id === S.docId ? 'num' : 'dim'}">${
         esc(d.doc_id === S.docId ? T.tbl.active : T.tbl.idle)}</span>
     </div>`).join('');

  let batchHtml = '';
  if (S.batch) {
    const st = S.batch.stats;
    batchHtml = `<div class="note"><b>${esc(T.notes.batch)}</b>${esc(S.batch.filename)} — `
      + `${st.documents} documents, ${st.spans} spans, ${st.resolved} resolved, `
      + `${st.flagged} flagged, in ${S.batch.timing_ms.total} ms total. `
      + `Run through <code>geoparse_batch</code>: one spaCy pass, pooled Elasticsearch `
      + `lookups, one model forward pass for the whole file.</div>`;
  }

  const d = S.doc;
  const note = d && d.note
    ? `<div class="note warn"><b>${esc(T.notes.note)}</b>${esc(d.note)}</div>` : '';

  return {
    meta: fill(T.inCorpus, { n: corpus().length }),
    html: `<div class="tbl" style="grid-template-columns:1fr 74px 62px 58px">
             <div class="hd"><span>${esc(T.tbl.source)}</span><span>${esc(T.tbl.adapter)}</span
               ><span>${esc(T.tbl.size)}</span><span>${esc(T.tbl.state)}</span></div>
             ${rows}
           </div>${note}${batchHtml}`,
    after: body => body.querySelectorAll('[data-doc]').forEach(el => {
      el.onclick = () => selectDoc(el.dataset.doc, true);
    }),
  };
}

function stageParse() {
  const ents = entities();
  if (!ents.length) return { meta: '—', html: `<div class="empty">${esc(T.empty.nospans)}</div>` };
  const rows = ents.map(e =>
    `<div class="row${active() === e.id ? ' on' : ''}" data-eid="${esc(e.id)}">
       <span>${esc(e.text)}</span>
       <span class="lab">${esc(e.label)}</span>
       <span class="dim">${e.start}–${e.end}</span>
       <span class="num">${e.ner_score != null ? e.ner_score.toFixed(2)
                          : (e.resolved ? e.resolved.confidence.toFixed(2) : '—')}</span>
     </div>`).join('');
  // `ner_score` exists only on the learned span-detector path; spaCy's entity
  // recogniser exposes no per-span probability, so the column falls back to
  // the resolution confidence and the header says which it is showing.
  const scoreCol = ents.some(e => e.ner_score != null) ? T.tbl.pspan : T.tbl.presolve;
  return {
    meta: UP(`${ents.length} ${T.foot.spans} · ${S.backend.span_detector || ''}`),
    html: `<div class="tbl" style="grid-template-columns:1fr 62px 62px 56px">
             <div class="hd"><span>${esc(T.tbl.span)}</span><span>${esc(T.tbl.label)}</span
               ><span>${esc(T.tbl.char)}</span><span>${esc(scoreCol)}</span></div>
             ${rows}
           </div>`,
    after: body => body.querySelectorAll('[data-eid]').forEach(el => {
      el.onclick = () => select(el.dataset.eid);
      el.onmouseenter = () => setHover(el.dataset.eid);
      el.onmouseleave = () => setHover(null);
    }),
  };
}

function stageDis() {
  const e = entityById(active());
  if (!e) return { meta: '—', html: `<div class="empty">${esc(T.empty.span)}</div>` };

  // Mentions of the same *record*, not of the same string: "Ukraine" and
  // "Ukrainian officials" are one place, and "Georgia" twice may well be two.
  // This is the count the single pin on the map now stands for.
  const sibs = placeOf(placeKey(e));
  const kin = sibs ? sibs.mentions : [e];
  const nth = kin.indexOf(e) + 1;

  const cands = e.candidates.map((c, i) =>
    `<div class="cand${i === 0 ? ' top' : ''}${i === 0 && e.review ? ' flag' : ''}"
          data-cand="${i}">
       <div class="l1">
         <span class="rk">${String(i + 1).padStart(2, '0')}</span>
         <span class="nm">${esc(c.name)}</span>
         <span class="cf">${c.confidence.toFixed(3)}</span>
       </div>
       <div class="l2">
         <span class="ds">${esc(c.display)}</span>
         <span class="bar"><i style="width:${(c.confidence * 100).toFixed(1)}%"></i></span>
       </div>
     </div>`).join('') || `<div class="empty">${esc(T.empty.nocands)}</div>`;

  let notes = '';
  if (e.rationale) {
    notes += `<div class="note"><b>${esc(T.notes.rationale)}</b>${esc(e.rationale)}</div>`;
  }
  if (e.review) {
    notes += `<div class="note warn"><b>${esc(T.notes.review)}</b>`
      + `${esc(e.review_reasons.join(' · '))}</div>`;
  }
  if (e.boundary) {
    const b = e.boundary;
    const weak = b.name_score != null
      && b.name_score < ((S.config.boundaries || {}).weakMatchBelow ?? 0.95);
    notes += `<div class="note${weak ? ' warn' : ''}"><b>${esc(T.notes.boundary)}</b>`
      + `ADM${b.level} “${esc(b.name)}” (${esc(b.iso3)}), matched by ${esc(b.match)}`
      + (b.name_score != null ? `, name agreement ${b.name_score.toFixed(2)}` : '')
      + (weak ? '. Drawn dashed: the two gazetteers only partly agree on this unit.' : '.')
      + `</div>`;
  }

  return {
    meta: `${T.margin} ${e.margin != null ? e.margin.toFixed(2) : '—'} · `
      + `${T.kv.pnomatch} ${e.p_no_match.toFixed(2)}`,
    html: `<div class="dis-head">
             <h2>${esc(e.text)}</h2>
             <span class="sub">${esc(UP(e.label))} · ${esc(kin.length > 1
               ? fill(T.mentionOf, { n: nth, total: kin.length })
               : T.onlyMention)} ·
               <span class="${e.review ? 'flag' : ''}">${
                 esc(e.review ? T.verdict.flagged : T.verdict.ok)}</span>
             </span>
           </div>${cands}${notes}`,
    after: body => {
      if (((S.config.map || {}).candidateGhosts || 'hover') !== 'hover') return;
      // Only the map repaints: rebuilding this panel under the pointer would
      // destroy the row the pointer is on, and its mouseleave with it.
      body.querySelectorAll('[data-cand]').forEach(el => {
        el.onmouseenter = () => { S.ghostCand = Number(el.dataset.cand); renderMap(); };
        el.onmouseleave = () => { S.ghostCand = null; renderMap(); };
      });
    },
  };
}

function stageResolve() {
  const e = entityById(active());
  if (!e) return { meta: '—', html: `<div class="empty">${esc(T.empty.span)}</div>` };
  if (!e.resolved) {
    return {
      meta: T.noMatchMeta,
      html: `<div class="note warn"><b>${esc(T.notes.nomatch)}</b>`
        + esc(fill(T.noMatchBody, { text: e.text,
                                    p: e.p_no_match.toFixed(2),
                                    stage: T.stages.dis }))
        + `</div>`,
    };
  }
  const r = e.resolved, b = e.boundary;
  const k = T.kv;
  const rows = [
    [k.geonameid, r.geonameid, ''],
    [k.name, r.name, ''],
    [k.feature, `${r.feature_code} (${r.feature_class})`, ''],
    [k.admin1, r.admin1 || '—', ''],
    [k.admin2, r.admin2 || '—', ''],
    [k.country, r.country_code3, ''],
    [k.population, r.population ? fmtInt(r.population) : '—', ''],
    [k.coordinates, `${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}`, ''],
    [k.confidence, `${r.confidence.toFixed(4)} · ${e.review ? k.flagged : k.accepted}`,
      e.review ? 'warn' : 'acc'],
    [k.pnomatch, e.p_no_match.toFixed(4), ''],
    [k.geometry, b ? `${fill(T.coordPolygon, { level: b.level })} · ${b.name}`
                   : k.pointonly, b ? 'acc' : ''],
    [k.geomsource, b ? `geoBoundaries CGAZ · ${b.match}` : '—', ''],
  ];
  return {
    meta: `GeoNames · ${r.country_code3}`,
    html: `<div class="kv">${rows.map(([k, v, cls]) =>
      `<span class="k">${k}</span><span class="v ${cls}">${esc(String(v))}</span>`).join('')}</div>`,
  };
}

// `raw` is the format's key everywhere else -- the config, the download
// extension map, the UI test -- but "RAW" on a button says nothing about
// whose shape it is, and it is the default now.
const FMT_LABEL = { raw: 'MORDECAI' };

function stageExport() {
  const formats = (S.config.export || {}).formats
    || ['raw', 'geojson', 'jsonl', 'csv', 'wkt'];
  const text = buildExport(S.format);
  return {
    meta: S.format === 'raw'
      ? `geoparse_doc · ${fmtBytes(text.length)}`
      : `${entities().filter(e => e.resolved).length} ${T.exportFeatures}`,
    html: `<div class="seg4">${formats.map(f =>
             `<button data-fmt="${f}" class="${f === S.format ? 'on' : ''}">${FMT_LABEL[f] || f.toUpperCase()}</button>`).join('')}</div>
           <pre class="pre">${escText(text)}</pre>
           <div class="exportfoot">
             <span>${entities().filter(e => e.resolved).length} ${esc(T.exportFeatures)} ·
                   ${entities().filter(e => e.review).length} ${esc(T.foot.flagged)} ·
                   ${entities().filter(e => e.boundary).length} ${esc(T.foot.boundaries)}</span>
             <button class="writebtn" id="download">${esc(T.download)}</button>
           </div>`,
    after: body => {
      body.querySelectorAll('[data-fmt]').forEach(el => {
        el.onclick = () => { S.format = el.dataset.fmt; renderStage(); };
      });
      body.querySelector('#download').onclick = () => downloadExport();
    },
  };
}

/** Export the result. GeoJSON uses the boundary polygon where there is one --
 *  a province exported as a point loses the thing the boundary layer added. */
function buildExport(fmt) {
  // Mordecai's own result, exactly as `geoparse_doc` returned it: its field
  // names, its structure, and -- because the console asks for `trim=False` --
  // every enrichment feature on every candidate. Every other format here is
  // reshaped for this console's contract, which is the wrong thing to hand
  // someone who wants to diff the output, script against it, or attach it to
  // a bug report. It is large, and that is the point; nothing trims it on the
  // way out. Placed above the guard below because a document where the model
  // declined everything is exactly when you want to see the raw result.
  if (fmt === 'raw') {
    return (S.result && S.result.raw)
      ? JSON.stringify(S.result.raw, null, 2)
      : '(this backend did not return the raw result)';
  }

  const ents = entities().filter(e => e.resolved);
  if (!ents.length) return '(nothing resolved yet)';

  if (fmt === 'geojson') {
    return JSON.stringify({
      type: 'FeatureCollection',
      crs: { type: 'name', properties: { name: (S.config.export || {}).crs } },
      features: ents.map(e => ({
        type: 'Feature',
        geometry: e.boundary ? e.boundary.geometry
          : { type: 'Point', coordinates: [e.resolved.lon, e.resolved.lat] },
        properties: {
          mention: e.text, start: e.start, end: e.end, label: e.label,
          geonameid: e.resolved.geonameid, name: e.resolved.name,
          feature_code: e.resolved.feature_code,
          country_code3: e.resolved.country_code3,
          admin1: e.resolved.admin1, population: e.resolved.population,
          confidence: e.resolved.confidence, p_no_match: e.p_no_match,
          review: e.review,
          geometry_source: e.boundary
            ? `geoBoundaries CGAZ ADM${e.boundary.level} (${e.boundary.match})`
            : 'GeoNames point',
        },
      })),
    }, null, 1);
  }

  if (fmt === 'jsonl') {
    return ents.map(e => JSON.stringify({
      doc_id: S.result.doc_id, mention: e.text, start: e.start, end: e.end,
      geonameid: e.resolved.geonameid, name: e.resolved.name,
      lat: e.resolved.lat, lon: e.resolved.lon,
      feature_code: e.resolved.feature_code, country: e.resolved.country_code3,
      confidence: e.resolved.confidence, review: e.review,
      boundary: e.boundary ? `ADM${e.boundary.level}` : null,
    })).join('\n');
  }

  if (fmt === 'csv') {
    const head = 'doc_id,mention,start,end,geonameid,name,feature_code,country,lat,lon,confidence,review,boundary';
    const q = v => {
      const s = String(v ?? '');
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [head, ...ents.map(e => [
      S.result.doc_id, e.text, e.start, e.end, e.resolved.geonameid, e.resolved.name,
      e.resolved.feature_code, e.resolved.country_code3,
      e.resolved.lat, e.resolved.lon, e.resolved.confidence, e.review,
      e.boundary ? `ADM${e.boundary.level}` : '',
    ].map(q).join(','))].join('\n');
  }

  // WKT. Boundaries are omitted deliberately: a simplified ADM2 ring is
  // thousands of characters and unreadable in a preview pane, so this format
  // is points only and says so.
  return '# points only; use GeoJSON for boundary polygons\n'
    + ents.map(e => `POINT (${e.resolved.lon} ${e.resolved.lat})\t${e.text}\t${e.resolved.name}`)
        .join('\n');
}

function downloadExport() {
  const ext = { geojson: 'geojson', jsonl: 'jsonl', csv: 'csv', wkt: 'wkt',
                raw: 'json' }[S.format] || 'txt';
  const blob = new Blob([buildExport(S.format)], { type: 'application/octet-stream' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  // `geoparse.json` would not say which of the five formats it is.
  const stem = S.format === 'raw' ? 'mordecai' : S.format;
  a.download = `${S.result.doc_id || 'geoparse'}.${stem}.${ext}`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── status bar ───────────────────────────────────────────────────────────── */

function setStatus(word, line) {
  $('#statusword').textContent = word;
  pushLog(line);
}

function pushLog(line) {
  const now = new Date();
  $('#logtime').textContent = now.toISOString().slice(11, 23);
  $('#logline').textContent = line;
}

function startClock() {
  // A UTC clock ticking in the corner is an ops-room signal, not information
  // this console has any use for -- the theme decides whether it is there.
  if (CH.clock === false) { $('#clock').textContent = ''; $('#clocksub').textContent = ''; }
  const tick = () => {
    if (CH.clock === false) { $('#clock').textContent = ''; $('#clocksub').textContent = ''; return; }
    $('#clock').textContent = new Date().toISOString().slice(11, 19);
    $('#clocksub').textContent = 'UTC';
  };
  tick();
  setInterval(tick, (S.config.telemetry || {}).clockIntervalMs || 1000);
}

/** Poll the backend for telemetry.
 *
 * Only fields the backend actually sends are rendered. The design's strip had
 * throughput, p50, VRAM and queue depth jittered in the client; its own
 * handoff note calls that worse than showing nothing, so an absent
 * measurement here is an absent cell.
 */
async function pollTelemetry() {
  const period = (S.config.telemetry || {}).pollIntervalMs || 4000;
  const tick = async () => {
    try {
      const t = await (await fetch('/api/telemetry')).json();
      const cells = [];
      if (t.p50_ms != null) cells.push(['P50', `${t.p50_ms} ms`]);
      if (t.last_ms != null) cells.push(['LAST', `${t.last_ms} ms`]);
      if (t.docs_parsed != null) cells.push(['DOCS', fmtInt(t.docs_parsed)]);
      if (t.vram_gb != null) cells.push(['VRAM', `${t.vram_gb} / ${t.vram_total_gb} GB`]);
      if (t.queue != null) cells.push(['QUEUE', t.queue]);
      if (t.gazetteer) cells.push(['ES', t.gazetteer.split(' · ').pop().toUpperCase()]);
      $('#telemetry').innerHTML = cells
        .map(([k, v]) => `<span>${k}<b>${esc(String(v))}</b></span>`).join('');
    } catch {
      $('#telemetry').innerHTML = '<span>TELEMETRY<b>UNREACHABLE</b></span>';
    }
  };
  await tick();
  setInterval(tick, period);
}

/* ── helpers ──────────────────────────────────────────────────────────────── */

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* localStorage, minus the ways it throws. Private windows and hardened
 * browser profiles make the whole object unreachable, and the console must
 * still come up -- these are preferences, not state. */
function store(key, val) {
  try {
    if (val === undefined) return localStorage.getItem(`mordecai.${key}`);
    localStorage.setItem(`mordecai.${key}`, val);
  } catch { /* preferences simply do not persist here */ }
  return null;
}
const esc = s => String(s ?? '').replace(/[<>&"]/g,
  c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));
// Text nodes only need the three; quotes are legitimate document content.
const escText = s => String(s ?? '').replace(/[<>&]/g,
  c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));
// Entity ids are `e<n>`, so this is belt and braces -- but a selector built by
// string concatenation is a bug waiting for the first id that is not.
const cssEsc = s => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s);
const fmtInt = n => Number(n).toLocaleString('en-US');
const fmtBytes = n => n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`;

boot();
