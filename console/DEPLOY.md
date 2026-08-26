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

```bash
./console/fetch_boundaries.sh               # 1.25 GB, once
uv run python console/build_boundaries.py   # -> 85 MB SQLite, ~55 s
```

The three downloaded GeoJSON files are only inputs. Once
`data/geoboundaries/boundaries.sqlite` exists you can delete them and reclaim
the 1.25 GB:

```bash
rm data/geoboundaries/geoBoundariesCGAZ_ADM*.geojson
```

The SQLite file is portable — build it once and copy it between servers rather
than downloading 1.25 GB again.

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

---

## Making it yours

`console/console.config.json` changes the demo without touching code: palette,
which ornaments show, top-k, the review gate, reveal timings, export formats,
the boundary layer.

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
