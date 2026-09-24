# Standing the console up on a new server

End to end, from a clean machine to a URL someone else can open. Written to be
followed by a person who has never seen this repository.

Roughly 45 minutes of your attention, most of it waiting on the GeoNames index.

`README.md` in this directory explains what the console *is* and why it makes
the choices it does. This file is only about getting it running.

---

## What you are standing up

Three pieces, and only the first is optional in any sense:

| Piece | What it is | Size on disk |
|---|---|---|
| **Elasticsearch + GeoNames** | the gazetteer. 12.6M place records | 1.9 GB index |
| **The Python environment** | spaCy, torch, the ranker, FastAPI | 7.9 GB with CUDA, ~3 GB without |
| **geoBoundaries** | administrative polygons. Optional | 1.25 GB download → 85 MB |

The model checkpoints are committed to the repository, so a `git clone` gets
them; there is nothing to download from a model host and no API key anywhere in
this stack. Everything runs locally.

**Elasticsearch is not optional and is the long pole.** Without it the console
exits at startup with `ElasticsearchConnectionError` rather than serving a
half-working page. Budget 30–40 minutes for the index the first time, most of it
unattended. Everything else is minutes.

**A GPU is not required.** `spacy.prefer_gpu()` falls back to CPU with no error.
Measured on this repository's own hardware, a single 39-token document with 11
mentions takes **30 ms on CPU** against roughly the same on a 4090 — for one
document at a time, which is what a demo does, the GPU buys nothing. It earns
its keep in `geoparse_batch` over thousands of documents. Skip `--extra gpu` on
a CPU-only box and the environment is less than half the size.

---

## Step 1 — push the branch (on your machine)

The console lives on the `ui-console` branch and has not been merged.

```bash
cd ~/projects/mordecai3
git status                 # should be clean
git push origin-ssh ui-console
```

This repository has two remotes for the same GitHub repo: `origin` over HTTPS
(which will ask for a personal access token) and `origin-ssh` over SSH. Use
`origin-ssh` unless you have a token to hand.

Then either open a pull request:

```bash
gh pr create --base main --head ui-console \
  --title "Geoparse console" --body "Single-screen analyst UI over the geoparser."
```

…or, if you would rather the demo just sit on `main` so the clone below needs no
`--branch` flag:

```bash
git checkout main && git merge ui-console && git push origin-ssh main
```

If you merge, drop the `--branch ui-console` from the clone in step 3.

### Is the repository public?

`ahalterman/mordecai3` is public, so the person standing this up needs no
credentials to clone it. If you make it private, they will need a deploy key or
a token, and `git clone https://…` will fail with a 404 rather than a
permissions error — which is confusing enough to be worth saying out loud.

---

## Step 2 — prerequisites (on the new server)

Ubuntu-flavoured; adjust to taste.

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip

# Docker, for Elasticsearch
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"      # then log out and back in

# uv, which manages the Python environment and the Python itself
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

You need **20 GB free** for a comfortable install with CUDA, 10 GB without.
Check before you start: `df -h .`

You do **not** need to install Python. `uv` fetches the interpreter the project
pins. If the machine already has conda, see the warning in step 4.

---

## Step 3 — clone and build the environment

```bash
git clone --branch ui-console https://github.com/ahalterman/mordecai3.git
cd mordecai3

# On a GPU box:
uv sync --extra console --extra gpu --group dev

# On a CPU-only box:
uv sync --extra console --group dev
```

`--group dev` pulls in the spaCy transformer model and Playwright, which the
verification step uses. Add `--group train` too if the machine will also be used
for training — but if you do, **always pass the full set of flags every time**.

### If `uv sync` dies compiling `curated_tokenizers`

```
Cython.Compiler.Errors.CompileError: curated_tokenizers/_bbpe.pyx
TypeError: 'NoneType' object is unsliceable
```

`curated-tokenizers` publishes wheels only up to **cp312**. On Python 3.13
there is nothing to download, so uv builds it from the sdist — and its Cython
sources do not compile under Cython 3.1 or later, which is what PyPI serves
now. It arrives indirectly: `en_core_web_trf` → `spacy-curated-transformers` →
`curated-tokenizers`, and `console/server.py` calls
`spacy.load("en_core_web_trf")`, so it cannot be dropped.

**The fix is already in `pyproject.toml`** — `[tool.uv]
build-constraint-dependencies = ["cython<3.1"]`, which pins the *build*
environment only and installs nothing at runtime. If you hit this, your clone
predates it: `git pull` and re-run `uv sync`. Expect one ~30 s compile.

This is a trap worth understanding rather than pattern-matching, because it
fails asymmetrically: a machine that built the wheel before Cython 3.1 shipped
keeps working from uv's cache forever. It passes on the laptop and fails on the
fresh server, which makes it look like something about the server is wrong.
Nothing is; the laptop is just holding a wheel that can no longer be produced.

Two alternatives if the constraint is not enough:

- **The build now fails in `gcc` instead of Cython.** The machine has no C
  toolchain. `sudo apt install build-essential`.
- **You would rather not compile at all.** Python 3.12 has published manylinux
  wheels for the whole chain: `uv sync -p 3.12 --extra console --group dev`.
  `requires-python` is `>=3.10`, so this is supported — it just means the server
  runs a different Python from your laptop.
A later `uv sync` with fewer flags prunes whatever the missing ones provided,
silently, and the next thing you run fails with a missing module.

Then fetch the browser Playwright drives, for step 7:

```bash
uv run playwright install chromium
```

Everything from here on is prefixed `uv run`. That is not decoration — see
step 4's warning.

---

## Step 4 — Elasticsearch with the GeoNames index

This is the part that takes time.

The index is not in this repository and cannot be — it is 1.9 GB of gazetteer.
It is built by a separate project:

> **https://github.com/openeventdata/es-geonames**

Follow its README. In outline: it downloads the GeoNames `allCountries` dump
(~400 MB), starts an Elasticsearch 7.10 container, and bulk-loads it into an
index called `geonames`. Expect 30–40 minutes, mostly unattended.

When it finishes you should be able to run this and see roughly 12.5M documents:

```bash
curl -s localhost:9200/_cat/indices?v
# health status index    ... docs.count  store.size
# yellow open   geonames ...   12571784       1.9gb
```

**That number is the check that matters.** A `geonames` index with far fewer
documents is the reduced test subset, not the real gazetteer, and the console
will start happily and then fail to resolve most place names — which looks like
a bad model rather than missing data.

Once built, the index is just a directory. To move it to another machine, stop
the container, copy the data directory, and point a new container at it:

```bash
docker run -d --name es_geonames \
  -p 127.0.0.1:9200:9200 \
  -e "discovery.type=single-node" \
  -v /path/to/geonames_index:/usr/share/elasticsearch/data \
  elasticsearch:7.10.1
```

> **Ignore `compose.yaml` in the repository root for this purpose.** It mounts
> `./tests/es_data/geonames_index`, which is the small Netherlands-only test
> subset used by the test suite and is gitignored — on a fresh clone it is an
> empty directory. `docker compose up` will give you an Elasticsearch with no
> gazetteer in it.

### If the machine has conda

Every command in this file is prefixed `uv run` for a reason. Bare `python` on a
conda machine is conda's base interpreter, which may have its own older FastAPI
and Pydantic installed. Those get far enough to fail *inside* FastAPI with

```
ImportError: cannot import name 'Undefined' from 'pydantic.fields'
```

which reads like a dependency conflict in this project and is not — it is the
wrong interpreter. `console/server.py` now detects this and says so, but the
other scripts do not. Use `uv run`, or activate `.venv` first.

---

## Step 5 — administrative boundaries (recommended)

Optional. Without it the console runs and every resolved place is a point; with
it, countries and provinces are drawn as their actual shapes, which is the
single most demo-able thing on the screen.

### Where it lives

One SQLite file, and the path is **fixed**:

```
<repo root>/data/geoboundaries/boundaries.sqlite      85 MB (88,395,776 bytes)
```

`console/boundaries.py` resolves it from its own location
(`Path(__file__).resolve().parent.parent / "data" / "geoboundaries"`), so it
does not depend on your working directory when you launch the server — but
there is **no environment variable and no config key** to point it somewhere
else. If you want it on another disk, symlink `data/geoboundaries` there.

What is inside: 52,791 simplified administrative polygons — 218 countries
(ADM0), 3,224 first-order units (ADM1), 49,349 second-order units (ADM2) —
each with its own bounding box, plus a `meta` table recording the source and
the per-level simplification tolerances. Built from the geoBoundaries CGAZ
composites, **CC-BY 4.0**; the console carries the attribution in the RESOLVE
panel and in the GeoJSON export properties, and it needs to keep doing so.

### ⚠ It is not in git

`.gitignore` excludes `data/geoboundaries/` — the store and the 1.25 GB of
GeoJSON it is built from. **`git clone` on the new server will not bring it**,
and nothing will complain: the console starts happily without it, every place
renders as a point, and the only sign is the rail footer reading
`BOUNDARIES · NOT LOADED` instead of `BOUNDARIES · 52,791 SHAPES`. That is the
one failure in this whole document that looks like success.

### Getting it onto the server

Two options. **Copying is almost always right** — 85 MB over the wire against
1.25 GB downloaded and ~55 s of CPU, and it guarantees the two machines are
drawing the same shapes.

```bash
# On the server: the directory has to exist first.
ssh you@server 'mkdir -p ~/mordecai3/data/geoboundaries'

# From your machine. rsync over scp: it resumes, which matters at 85 MB
# on a hotel connection.
rsync -avP data/geoboundaries/boundaries.sqlite \
      you@server:~/mordecai3/data/geoboundaries/
```

`scp data/geoboundaries/boundaries.sqlite you@server:~/mordecai3/data/geoboundaries/`
does the same thing without the resume.

It is a single self-contained file. The store is written in `journal_mode=delete`
and opened read-only (`file:…?mode=ro`), so there are no `-wal` or `-shm`
sidecars to remember, nothing to quiesce before copying, and the file can be
`chmod 444` on the server if you like. It only has to be *readable* by whatever
user runs the service — if you are running it as a systemd unit under a
different user than you rsynced as, check that.

**Rebuilding on the server** instead is only worth it if the link to your
laptop is worse than the server's link to GitHub, or if you want a fresher CGAZ
release than the one you built from:

```bash
./console/fetch_boundaries.sh               # 1.25 GB, once
uv run python console/build_boundaries.py   # -> 85 MB SQLite, ~55 s
rm data/geoboundaries/geoBoundariesCGAZ_ADM*.geojson    # reclaim the 1.25 GB
```

This needs the `console` extra for `ijson` and `shapely`, which the Step 3
`uv sync` line already installs. The three GeoJSON files are inputs only —
nothing reads them at runtime, and the console needs no network at all once the
store exists.

### Verify the transfer

Checksum both ends. This is the copy of the store in this repository as of
2026-08-27:

```bash
sha256sum data/geoboundaries/boundaries.sqlite
# c0b5427624bd24280c8288f172bc8409898171b668a6f47fb78a36be160cbe7e
```

A truncated SQLite file frequently still *opens*, so also ask SQLite itself,
and count the rows:

```bash
ssh you@server 'cd ~/mordecai3 && sqlite3 data/geoboundaries/boundaries.sqlite \
  "PRAGMA integrity_check; SELECT level, count(*) FROM shapes GROUP BY level;"'
# ok
# 0|218
# 1|3224
# 2|49349
```

No `sqlite3` binary on the server? The Python is already there:

```bash
uv run python -c "import sqlite3;c=sqlite3.connect('data/geoboundaries/boundaries.sqlite');\
print(c.execute('PRAGMA integrity_check').fetchone()[0],\
      c.execute('SELECT count(*) FROM shapes').fetchone()[0])"
# ok 52791
```

Then confirm the running console actually picked it up — this is the check that
matters, because it exercises the same code path the demo does:

```bash
curl -s localhost:8000/api/config \
  | python -c "import json,sys; print(json.load(sys.stdin)['backend']['boundaries'])"
# {'available': True, 'shapes': 52791, 'source': 'geoBoundaries CGAZ (gbOpen, CC-BY 4.0)'}
```

(`backend.boundaries`, not the top-level `config.boundaries` — that one is the
*display* settings for the layer and says `enabled: true` whether or not the
store is actually there.)

On screen: the rail footer reads `BOUNDARIES · 52,791 SHAPES`, and parsing the
sample report draws filled polygons rather than bare markers.

### When you would rebuild it

Rarely. The store is a pure function of the CGAZ release and the tolerances in
`TOLERANCE_DEG`, so rebuild only if you want newer geoBoundaries data or want
to change how aggressively the polygons are simplified. In particular
`focus_bbox` — the extent the map frames on, which is what stops a mention of
Russia from framing the whole globe — is computed per query in `boundaries.py`,
not stored, so improvements there never need a rebuild.

---

## Step 6 — run it

```bash
uv run python console/server.py                    # http://127.0.0.1:8000
uv run python console/server.py --listen           # 0.0.0.0, for a remote box
uv run python console/server.py --listen --port 8077
```

`--listen` logs the address to open from another machine:

```
WARNING console | listening on all interfaces -- the console is reachable at
                  http://192.168.0.233:8000 by anyone on this network, and it
                  has no authentication
```

Startup takes about five seconds and ends with `console ready`. A warmup parse
runs during that, deliberately: the first CUDA forward pass costs ~970 ms
against ~90 ms steady state, and that is exactly the parse an audience watches.

To keep it running after you disconnect:

```bash
# quick and dirty
setsid nohup uv run python console/server.py --listen > console.log 2>&1 < /dev/null &

# or properly, as a user service
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/mordecai-console.service <<'EOF'
[Unit]
Description=Mordecai geoparse console
After=network.target

[Service]
WorkingDirectory=%h/mordecai3
ExecStart=%h/.local/bin/uv run python console/server.py --listen
Restart=on-failure

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now mordecai-console
loginctl enable-linger "$USER"     # so it survives logout
journalctl --user -u mordecai-console -f
```

---

## Step 7 — verify it actually works

Do not trust a page that loads. The failure that matters here is a console that
starts, renders, and resolves nothing because the gazetteer is the test subset.

```bash
CONSOLE_URL=http://127.0.0.1:8000 uv run python console/test_console_ui.py
```

35 checks against a real browser and the real backend: span rendering, offset
fidelity, markers and polygons, the text/map sync, every stage panel, GeoJSON
validity, and that repeated mentions collapse to one marker. **Any browser
console error fails the run.** You want `35 passed, 0 failed`.

A ten-second smoke test without Playwright:

```bash
curl -s -X POST localhost:8000/api/geoparse \
  -H 'Content-Type: application/json' \
  -d '{"doc_id":"t","text":"Fighting near Gao and Ansongo in Mali."}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["stats"]); \
      print([(e["text"], e["resolved"] and e["resolved"]["name"]) for e in d["entities"]])'
```

Expect three spans, three resolved, and Mali carrying a boundary.

---

## Step 8 — letting other people reach it

**The console has no authentication.** None. Anyone who can reach the port can
use it, and `--listen` puts it in front of the whole network.

For a demo to a few people, the safest option is not to expose it at all — have
them tunnel:

```bash
ssh -L 8000:localhost:8000 you@server      # then open http://localhost:8000
```

Run the server without `--listen` in that case; the tunnel reaches localhost.

On a trusted LAN, `--listen` is fine. `--host` binds a single interface if the
machine is on more than one network:

```bash
uv run python console/server.py --host 192.168.0.233
```

For anything reachable from the internet, put it behind a reverse proxy that
terminates TLS and asks for a password. Do not skip the password: the endpoint
runs a transformer on arbitrary text a stranger supplies, which is a
denial-of-service surface even if you do not mind the compute bill.

```nginx
server {
    listen 443 ssl;
    server_name demo.example.org;
    # ssl_certificate ... ;

    location / {
        auth_basic "demo";
        auth_basic_user_file /etc/nginx/.htpasswd;   # htpasswd -c … demo
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;   # a large batch upload can take a while
    }
}
```

Add a rate limit on the two expensive endpoints while you are in there — the
password is shared and will get forwarded, and `/api/geoparse` is a transformer
forward pass per request. The full block, plus the argument for one credential
per audience rather than one for everybody, is in
**[ACCESS_AND_FEEDBACK.md](ACCESS_AND_FEEDBACK.md)**. That file also sets out
what per-person invite links would take, and what a thumbs-up/down record has
to carry to be worth anything as evaluation data — which is the reason to
consider invite links at all.

---

## Making it yours

`console/console.config.json` changes the demo without touching code: the two
themes and their palettes, which ornaments show, top-k, the review gate, reveal
timings, export formats, the boundary layer.

**Each look has its own URL.** `/console` opens the ops console and `/demo` the
field one, so you can hand an audience a link that lands on the right screen
instead of a link plus an instruction. `/` opens whichever was last used in that
browser, falling back to `theme.name` in the config. The LOOK menu still
switches between them live, and switching rewrites the path so a link copied
mid-demo opens what was on screen.

The sample documents are part of the theme: `/console` opens on a convoy
movement report, `/demo` on ACLED's Ukraine situation update as published on
ReliefWeb. See *Two looks* and *Entry points* in [README.md](README.md).

The routes come from `routes` in the config and are registered at **startup** —
adding or renaming one needs a restart, unlike the rest of the config, which is
re-read on every request.

Two worth knowing before a talk:

- `theme.chrome: "stripped"` removes every ornament — the scanlines, the grit,
  the corner brackets. Good for screenshots and for people who find the costume
  distracting.
- `pipeline.spanRevealMs: 0` turns off the sequential reveal entirely, so
  results appear the instant they arrive.

`console/corpus.json` holds the three documents in the sidebar. Replace them
with something from your own domain: it is `{"documents": [ … ]}`, and each
entry is `{doc_id, title, source, adapter, kind, meta, region, text}`. `kind`
picks the typeface the document is set in (`wire`, `social`, `archive`,
`paste`); the rest is labelling. Restart the server to pick up changes.

The layout wants **1280px or wider**. Below that it scrolls sideways, by design.

---

## When it does not work

| Symptom | Cause |
|---|---|
| `ImportError: cannot import name 'Undefined' from 'pydantic.fields'` | Bare `python` picked up conda or system Python. Use `uv run`. |
| `ElasticsearchConnectionError: Could not connect to Elasticsearch.` and the server exits | Elasticsearch is not running or is on another host. `docker ps`, start the container, or set `ES_HOST`. |
| Starts fine, but almost nothing resolves | The `geonames` index is the test subset. `curl -s localhost:9200/_cat/indices?v` — you want ~12.5M docs, not thousands. |
| Every place is a point, no polygons ever | No boundary store. The log says so at startup. Step 5. |
| `BOUNDARIES · NOT LOADED` in the sidebar | Same. |
| Map is blank, `BASEMAP UNAVAILABLE` | The vendored d3/basemap files did not come through. Check `console/static/vendor/` has three files. |
| First parse takes a second, the rest are fast | Expected — that is the CUDA warmup, and it is spent at startup rather than on stage. |
| `ModuleNotFoundError: playwright` when verifying | `uv sync` was run without `--group dev`. Re-sync with every flag you used before. |
| `uv sync` dies in `CompileError: curated_tokenizers/_bbpe.pyx` | See below — it needs a `git pull`, not a workaround. |
| Reachable from your laptop but not a colleague's | `--listen` binds all interfaces; a firewall is in the way. `sudo ufw allow 8000`. |
| Works locally, refuses through the proxy | Increase `proxy_read_timeout`; a batch upload of 500 documents outlives nginx's 60 s default. |

---

## What it costs

Disk figures and the two timings marked *measured* come from this repository's
development machine (RTX 4090, NVMe). The rest depend on your network and are
marked as such — they are there for planning, not as promises.

| | Time | Disk |
|---|---|---|
| `uv sync` with CUDA | network-bound; several minutes | **7.9 GB** |
| `uv sync` without CUDA | network-bound; a few minutes | ~3 GB |
| GeoNames index build | 30–40 min per `es-geonames`; unmeasured here | **1.9 GB** |
| geoBoundaries download | network-bound | **1.25 GB**, deletable after the build |
| geoBoundaries build | **55 s** (measured) | **85 MB** |
| Server startup | **~5 s** (measured) | — |
| One document, 11 mentions | **30 ms** CPU, ~30 ms GPU (measured) | — |

Of that 7.9 GB environment, 478 MB is the spaCy transformer and most of the
rest is torch and the CUDA runtime.
