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

const STAGES = [
  { key: 'ingest',   name: 'INGEST',        meta: 'SOURCE' },
  { key: 'parse',    name: 'PARSE',         meta: 'SPANS' },
  { key: 'dis',      name: 'DISAMBIGUATE',  meta: 'TOP-K' },
  { key: 'resolve',  name: 'RESOLVE',       meta: 'GAZ' },
  { key: 'export',   name: 'EXPORT',        meta: 'FORMATS' },
];

const S = {
  config: null,
  corpus: [],
  backend: null,
  docId: null,          // corpus doc_id, or '__paste__'
  doc: null,            // the active document record {title, meta, kind, text}
  result: null,         // the last /api/geoparse response
  stage: 'dis',
  mapMode: 'sat',
  selected: null,
  hovered: null,
  running: false,
  settled: false,       // false while the reveal animation is stepping
  revealed: new Set(),  // entity ids already stepped past
  scanning: null,       // the entity mid-reveal
  zoom: 1,
  format: 'geojson',
  batch: null,          // last /api/batch summary
  log: { time: '', line: 'idle' },
};

const active = () => S.hovered ?? S.selected;
const entities = () => (S.result && S.result.entities) || [];
const entityById = id => entities().find(e => e.id === id) || null;

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  let cfg;
  try {
    cfg = await (await fetch('/api/config')).json();
  } catch (err) {
    setStatus('OFFLINE', `backend unreachable: ${err.message}`);
    return;
  }
  S.config = cfg.config || {};
  S.corpus = cfg.corpus || [];
  S.backend = cfg.backend || {};

  applyTheme();
  applyBrand();
  configureScope();
  renderStages();
  renderCorpus();
  bindControls();
  startClock();
  pollTelemetry();

  S.mapMode = (S.config.map && S.config.map.defaultMode) || 'sat';
  S.stage = (S.config.pipeline && S.config.pipeline.defaultStage) || 'dis';
  syncModeButtons();

  if (S.corpus.length) {
    selectDoc(S.corpus[0].doc_id,
              (S.config.pipeline && S.config.pipeline.autorun) !== false);
  } else {
    setStatus('READY', 'no corpus configured — paste text to parse');
    renderAll();
  }
}

function applyTheme() {
  const t = S.config.theme || {};
  const pal = (S.config.palettes || {})[t.palette || 'amber'];
  if (pal) {
    for (const [k, v] of Object.entries(pal)) {
      document.documentElement.style.setProperty(k, v);
    }
  }
  const root = document.documentElement;
  root.dataset.pal = t.palette || 'amber';
  root.dataset.layout = t.layout || 'split';
  root.dataset.chrome = t.chrome || 'full';
  root.style.setProperty('--grit', t.grit == null ? 0.55 : t.grit);
  if (t.scanlines === false) root.dataset.scanlines = 'off';
  if (t.vignette === false) root.dataset.vignette = 'off';
  if (t.flicker === false) root.dataset.noise = 'off';
}

function applyBrand() {
  const b = S.config.brand || {};
  $('#brand-name').textContent = b.productName || 'MORDECAI';
  $('#brand-sub').textContent = b.productSubtitle || 'GEOPARSE ENGINE';

  // The design's title bar carried a classification chip and an invented
  // device readout. These slots hold what the backend actually reports
  // instead; a made-up number in a status bar is worse than an empty slot.
  const be = S.backend || {};
  const pairs = [
    ['BUILD', b.build || '—'],
    ['MODEL', be.model || '—'],
    ['SPANS', be.span_detector || '—'],
    ['DEVICE', (be.device || '—').toUpperCase()],
  ];
  $('#title-stats').innerHTML = pairs
    .map(([k, v]) => `<span><b>${k}</b>${esc(v)}</span>`).join('');

  const bd = be.boundaries || {};
  $('#gazfoot').innerHTML = bd.available
    ? `GAZ · GEONAMES<br>BOUNDARIES · ${fmtInt(bd.shapes)} SHAPES`
    : 'GAZ · GEONAMES<br>BOUNDARIES · NOT LOADED';
}

function configureScope() {
  const m = S.config.map || {};
  const bd = S.config.boundaries || {};
  $('#scope').configure({
    atlas: (m.basemap && m.basemap.path) || '/vendor/countries-110m.json',
    minZoom: m.minZoom, maxZoom: m.maxZoom,
    fitPaddingRatio: m.fitPaddingRatio,
    graticuleStepDeg: m.graticuleStepDeg,
    graticuleMajorStepDeg: m.graticuleMajorStepDeg,
    sweepSeconds: m.sweepSeconds,
    pan: m.pan !== false, zoom: m.zoom !== false,
    boundaryFillOpacity: bd.fillOpacity,
    boundaryStrokeWidth: bd.strokeWidth,
    weakMatchBelow: bd.weakMatchBelow,
  });
}

/* ── rail ─────────────────────────────────────────────────────────────────── */

function renderStages() {
  $('#stages').innerHTML = STAGES.map((s, i) =>
    `<div class="stage-row${s.key === S.stage ? ' on' : ''}" data-stage="${s.key}">
       <span class="idx">${String(i + 1).padStart(2, '0')}</span>
       <span class="nm">${s.name}</span>
       <span class="mt" data-stagemeta="${s.key}"></span>
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
  const meta = {
    ingest: S.corpus.length ? `${S.corpus.length} DOC` : '—',
    parse: n ? `${n} SPANS` : '—',
    dis: `TOP-K ${k}`,
    resolve: S.result ? `${S.result.stats.resolved}/${n}` : '—',
    export: `${((S.config.export || {}).formats || []).length} FMT`,
  };
  for (const [key, val] of Object.entries(meta)) {
    const el = document.querySelector(`[data-stagemeta="${key}"]`);
    if (el) el.textContent = val;
  }
}

function renderCorpus() {
  $('#corpus').innerHTML = S.corpus.map(d =>
    `<div class="doc-card${d.doc_id === S.docId ? ' on' : ''}" data-doc="${esc(d.doc_id)}">
       <div class="t"><b>${esc(d.title)}</b><i>${fmtBytes(d.text.length)}</i></div>
       <div class="s">${esc(d.source)}</div>
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
      S.doc = { doc_id: '__paste__', title: 'PASTED TEXT', kind: 'paste',
                source: 'USER INPUT', meta: `${pasted.length} CHARS`,
                region: '', text: pasted };
      renderCorpus();
    }
    runParse();
  };

  $('#mapmode').querySelectorAll('[data-mode]').forEach(el => {
    el.onclick = () => { S.mapMode = el.dataset.mode; syncModeButtons(); renderMap(); };
  });

  $('#reset').onclick = () => $('#scope').resetView();
  $('#scope').addEventListener('viewchange', e => {
    S.zoom = e.detail;
    $('#zoomchip').textContent = '×' + S.zoom.toFixed(1);
  });
  $('#scope').addEventListener('pinhover', e => setHover(e.detail));
  $('#scope').addEventListener('pinclick', e => select(e.detail));

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

function syncModeButtons() {
  $('#mapmode').querySelectorAll('[data-mode]').forEach(el =>
    el.classList.toggle('on', el.dataset.mode === S.mapMode));
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
  $('#run').disabled = true; $('#runlabel').textContent = 'PARSING…';
  setStatus('PARSING', `${S.doc.title.toLowerCase()} — ${S.doc.text.length} chars`);
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
    $('#run').disabled = false; $('#runlabel').textContent = 'PARSE';
    setStatus('ERROR', String(err.message || err));
    renderAll();
    return;
  }

  const wall = Math.round(performance.now() - t0);
  setStatus('PARSING', `${entities().length} spans in ${wall} ms round-trip`);
  await reveal();

  S.running = false; S.settled = true;
  $('#run').disabled = false; $('#runlabel').textContent = 'RE-RUN PARSE';

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
  setStatus('RESOLVED',
    `${st.resolved}/${st.spans} resolved · ${st.flagged} flagged · `
    + `${st.with_boundary} with boundary · ${S.result.timing_ms.total} ms`);
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
  const stepMs = p.spanRevealMs ?? 230;
  const gapMs = p.spanGapMs ?? 90;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (!stepMs || reduce) {
    entities().forEach(e => S.revealed.add(e.id));
    renderAll();
    return;
  }

  await sleep(p.runStartDelayMs ?? 320);
  for (const e of entities()) {
    S.scanning = e.id; S.selected = e.id;
    pushLog(`rank  ${e.text} → ${e.resolved
      ? `${e.resolved.name} ${e.resolved.confidence.toFixed(2)}`
      : 'no match'} · k=${e.candidates.length}`);
    renderAll();
    await sleep(stepMs);
    S.scanning = null; S.revealed.add(e.id);
    renderAll();
    await sleep(gapMs);
  }
}

async function runBatch(file) {
  const p = S.config.pipeline || {};
  S.running = true;
  $('#run').disabled = true; $('#runlabel').textContent = 'BATCH…';
  setStatus('PARSING', `batch: ${file.name} (${fmtBytes(file.size)})`);
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
    $('#run').disabled = false; $('#runlabel').textContent = 'PARSE';
    setStatus('ERROR', String(err.message || err));
    return;
  }
  S.running = false;
  $('#run').disabled = false; $('#runlabel').textContent = 'PARSE';

  // Show the first document of the batch in the main panes, and the whole
  // batch's totals under INGEST.
  const first = S.batch.documents[0];
  if (first) {
    S.docId = first.doc_id;
    S.doc = { doc_id: first.doc_id, title: first.doc_id.toUpperCase(),
              kind: 'paste', source: S.batch.filename,
              meta: `BATCH ${S.batch.stats.documents} DOCS`, region: '',
              text: first.text };
    S.result = first;
    S.revealed = new Set(first.entities.map(e => e.id));
    S.settled = true;
    const flagged = first.entities.find(e => e.review);
    S.selected = (flagged || first.entities[0] || {}).id || null;
  }
  const st = S.batch.stats;
  S.stage = 'ingest'; renderStages();
  setStatus('RESOLVED',
    `batch ${st.documents} docs · ${st.spans} spans · ${st.resolved} resolved · `
    + `${st.flagged} flagged · ${S.batch.timing_ms.total} ms`);
  renderCorpus();
  renderAll();
}

/* ── selection ────────────────────────────────────────────────────────────── */

function select(id) {
  if (!id) return;
  S.selected = id; S.hovered = null;
  if (S.stage !== 'dis' && S.stage !== 'resolve') { S.stage = 'dis'; renderStages(); }
  renderAll();
}

function setHover(id) {
  if (S.hovered === id) return;
  S.hovered = id;
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
  $('#doc-title').textContent = S.doc.title || '';
  $('#doc-meta').textContent = S.doc.meta || '';
  el.dataset.kind = S.doc.kind || 'paste';

  const text = S.doc.text;
  const ents = entities();
  if (!ents.length) {
    el.textContent = text;
    el.dataset.built = '';
    $('#doc-foot').innerHTML = S.running
      ? '<span>PARSING…</span>' : '<span>NO SPANS</span>';
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
      `<span data-eid="${esc(e.id)}" data-st="${spanState(e)}" `
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
  $('#doc-foot').innerHTML =
    `<span>SPANS ${st.spans}</span>`
    + `<span class="ok">RESOLVED ${st.resolved}</span>`
    + `<span class="amb">FLAGGED ${st.flagged}</span>`
    + `<span>BOUNDARIES ${st.with_boundary}</span>`
    + `<span class="rule" data-orn></span>`
    + `<span>${S.result.timing_ms.total} ms · ${S.result.token_count} tok</span>`;
}

function spanState(e) {
  if (S.scanning === e.id) return 'scan';
  if (!S.revealed.has(e.id)) return 'pending';
  if (active() === e.id) return 'sel';
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
  const pins = ents
    .filter(e => e.resolved)
    .map(e => ({
      id: e.id,
      lat: e.resolved.lat, lon: e.resolved.lon,
      label: e.resolved.name,
      conf: e.resolved.confidence,
      status: !S.revealed.has(e.id) ? 'pending'
        : (e.review ? 'ambiguous' : 'ok'),
      boundary: (S.config.boundaries || {}).enabled === false ? null : e.boundary,
    }));

  // The runners-up of the active entity, so the map shows what it was chosen
  // *against* -- the whole point of the disambiguate view.
  const cur = entityById(active());
  const ghosts = (cur && S.settled)
    ? cur.candidates.slice(1)
        .filter(c => c.lat != null)
        .map(c => ({ lat: c.lat, lon: c.lon, label: c.name }))
    : [];

  scope.setScene({ pins, ghosts, mode: S.mapMode, active: active() });

  $('#map-region').textContent = (S.doc && S.doc.region) || regionOf(ents);
  $('#map-modemeta').textContent = S.mapMode === 'sat'
    ? 'SYNTHETIC RELIEF · MERCATOR'
    : 'VECTOR OVERLAY · NE 110M · MERCATOR';
  $('#map-badges').innerHTML = (S.mapMode === 'sat'
    ? ['RELIEF · PROCEDURAL', 'NO RASTER SOURCE', 'WGS84']
    : ['GRATICULE 2°', 'ADMIN-0 MESH', 'NO RASTER'])
    .map(b => `<span>${b}</span>`).join('');

  if (cur && cur.resolved) {
    const r = cur.resolved;
    const b = cur.boundary;
    $('#map-coord').innerHTML =
      `${Math.abs(r.lat).toFixed(4)}° ${r.lat >= 0 ? 'N' : 'S'}<br>`
      + `${Math.abs(r.lon).toFixed(4)}° ${r.lon >= 0 ? 'E' : 'W'}<br>`
      + `<span class="sub">${esc(r.feature_code)} · ${esc(r.country_code3)}`
      + (b ? ` · ADM${b.level} POLYGON` : ' · POINT') + `</span>`;
  } else {
    $('#map-coord').innerHTML = '';
  }
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

  const stage = STAGES.find(s => s.key === S.stage);
  $('#stage-title').textContent = stage ? stage.name : '';
  const body = $('#stage-body');
  const fns = { ingest: stageIngest, parse: stageParse, dis: stageDis,
                resolve: stageResolve, export: stageExport };
  const out = (fns[S.stage] || (() => ({ meta: '', html: '' })))();
  $('#stage-meta').textContent = out.meta || '';
  body.innerHTML = out.html;
  if (out.after) out.after(body);
}

function stageIngest() {
  const rows = S.corpus.map(d =>
    `<div class="row" data-doc="${esc(d.doc_id)}">
       <span>${esc(d.title)}</span><span class="dim">${esc(d.adapter || '—')}</span>
       <span class="dim">${fmtBytes(d.text.length)}</span>
       <span class="${d.doc_id === S.docId ? 'num' : 'dim'}">${d.doc_id === S.docId ? 'ACTIVE' : 'IDLE'}</span>
     </div>`).join('');

  let batchHtml = '';
  if (S.batch) {
    const st = S.batch.stats;
    batchHtml = `<div class="note"><b>BATCH ▸ </b>${esc(S.batch.filename)} — `
      + `${st.documents} documents, ${st.spans} spans, ${st.resolved} resolved, `
      + `${st.flagged} flagged, in ${S.batch.timing_ms.total} ms total. `
      + `Run through <code>geoparse_batch</code>: one spaCy pass, pooled Elasticsearch `
      + `lookups, one model forward pass for the whole file.</div>`;
  }

  const d = S.doc;
  const note = d && d.note ? `<div class="note warn"><b>NOTE ▸ </b>${esc(d.note)}</div>` : '';

  return {
    meta: `${S.corpus.length} IN CORPUS`,
    html: `<div class="tbl" style="grid-template-columns:1fr 74px 62px 58px">
             <div class="hd"><span>SOURCE</span><span>ADAPTER</span><span>SIZE</span><span>STATE</span></div>
             ${rows}
           </div>${note}${batchHtml}`,
    after: body => body.querySelectorAll('[data-doc]').forEach(el => {
      el.onclick = () => selectDoc(el.dataset.doc, true);
    }),
  };
}

function stageParse() {
  const ents = entities();
  if (!ents.length) return { meta: '—', html: '<div class="empty">No spans yet.</div>' };
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
  const scoreCol = ents.some(e => e.ner_score != null) ? 'P(SPAN)' : 'P(RESOLVE)';
  return {
    meta: `${ents.length} SPANS · ${S.backend.span_detector || ''}`.toUpperCase(),
    html: `<div class="tbl" style="grid-template-columns:1fr 62px 62px 56px">
             <div class="hd"><span>SPAN</span><span>LABEL</span><span>CHAR</span><span>${scoreCol}</span></div>
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
  if (!e) return { meta: '—', html: '<div class="empty">Select a span to see its candidates.</div>' };

  const sameText = entities().filter(x => x.text === e.text).length;
  const cands = e.candidates.map((c, i) =>
    `<div class="cand${i === 0 ? ' top' : ''}${i === 0 && e.review ? ' flag' : ''}">
       <div class="l1">
         <span class="rk">${String(i + 1).padStart(2, '0')}</span>
         <span class="nm">${esc(c.name)}</span>
         <span class="cf">${c.confidence.toFixed(3)}</span>
       </div>
       <div class="l2">
         <span class="ds">${esc(c.display)}</span>
         <span class="bar"><i style="width:${(c.confidence * 100).toFixed(1)}%"></i></span>
       </div>
     </div>`).join('') || '<div class="empty">No gazetteer candidates.</div>';

  let notes = '';
  if (e.rationale) {
    notes += `<div class="note"><b>RATIONALE ▸ </b>${esc(e.rationale)}</div>`;
  }
  if (e.review) {
    notes += `<div class="note warn"><b>REVIEW ▸ </b>${esc(e.review_reasons.join(' · '))}</div>`;
  }
  if (e.boundary) {
    const b = e.boundary;
    const weak = b.name_score != null
      && b.name_score < ((S.config.boundaries || {}).weakMatchBelow ?? 0.95);
    notes += `<div class="note${weak ? ' warn' : ''}"><b>BOUNDARY ▸ </b>`
      + `ADM${b.level} “${esc(b.name)}” (${esc(b.iso3)}), matched by ${esc(b.match)}`
      + (b.name_score != null ? `, name agreement ${b.name_score.toFixed(2)}` : '')
      + (weak ? '. Drawn dashed: the two gazetteers only partly agree on this unit.' : '.')
      + `</div>`;
  }

  return {
    meta: `MARGIN ${e.margin != null ? e.margin.toFixed(2) : '—'} · P(NO MATCH) ${e.p_no_match.toFixed(2)}`,
    html: `<div class="dis-head">
             <h2>${esc(e.text)}</h2>
             <span class="sub">${esc(e.label)} · ${sameText} MENTION${sameText === 1 ? '' : 'S'} IN DOC ·
               <span class="${e.review ? 'flag' : ''}">${e.review ? 'FLAGGED FOR REVIEW' : 'AUTO-ACCEPT'}</span>
             </span>
           </div>${cands}${notes}`,
  };
}

function stageResolve() {
  const e = entityById(active());
  if (!e) return { meta: '—', html: '<div class="empty">Select a span.</div>' };
  if (!e.resolved) {
    return {
      meta: 'NO MATCH',
      html: `<div class="note warn"><b>NO MATCH ▸ </b>The model declined to place
              “${esc(e.text)}”. Its calibrated probability that no candidate is
              correct is ${e.p_no_match.toFixed(2)}. The candidates it rejected are
              under DISAMBIGUATE.</div>`,
    };
  }
  const r = e.resolved, b = e.boundary;
  const rows = [
    ['GEONAMEID', r.geonameid, ''],
    ['NAME', r.name, ''],
    ['FEATURE CODE', `${r.feature_code} (${r.feature_class})`, ''],
    ['ADMIN 1', r.admin1 || '—', ''],
    ['ADMIN 2', r.admin2 || '—', ''],
    ['COUNTRY', r.country_code3, ''],
    ['POPULATION', r.population ? fmtInt(r.population) : '—', ''],
    ['COORDINATES', `${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}`, ''],
    ['CONFIDENCE', `${r.confidence.toFixed(4)} · ${e.review ? 'FLAGGED' : 'ACCEPTED'}`,
      e.review ? 'warn' : 'acc'],
    ['P(NO MATCH)', e.p_no_match.toFixed(4), ''],
    ['GEOMETRY', b ? `ADM${b.level} POLYGON · ${b.name}` : 'POINT ONLY', b ? 'acc' : ''],
    ['GEOM SOURCE', b ? `geoBoundaries CGAZ · ${b.match}` : '—', ''],
  ];
  return {
    meta: `GEONAMES · ${r.country_code3}`,
    html: `<div class="kv">${rows.map(([k, v, cls]) =>
      `<span class="k">${k}</span><span class="v ${cls}">${esc(String(v))}</span>`).join('')}</div>`,
  };
}

function stageExport() {
  const formats = (S.config.export || {}).formats || ['geojson', 'jsonl', 'csv', 'wkt'];
  const text = buildExport(S.format);
  return {
    meta: `${entities().filter(e => e.resolved).length} FEATURES`,
    html: `<div class="seg4">${formats.map(f =>
             `<button data-fmt="${f}" class="${f === S.format ? 'on' : ''}">${f.toUpperCase()}</button>`).join('')}</div>
           <pre class="pre">${escText(text)}</pre>
           <div class="exportfoot">
             <span>${entities().filter(e => e.resolved).length} FEATURES ·
                   ${entities().filter(e => e.review).length} FLAGGED ·
                   ${entities().filter(e => e.boundary).length} WITH BOUNDARY</span>
             <button class="writebtn" id="download">DOWNLOAD ▸</button>
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
  const ext = { geojson: 'geojson', jsonl: 'jsonl', csv: 'csv', wkt: 'wkt' }[S.format];
  const blob = new Blob([buildExport(S.format)], { type: 'application/octet-stream' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${S.result.doc_id || 'geoparse'}.${ext}`;
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
  const tick = () => {
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
