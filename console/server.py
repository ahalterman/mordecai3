"""FastAPI backend for the geoparse console.

One process holds one spaCy pipeline, one `Geoparser` and one boundary store,
and serves the static frontend beside the API. That co-location is deliberate:
the console is a demo you run on a laptop next to a projector, and a two-service
deployment is one more thing to go wrong ten minutes before a talk.

    uv run uvicorn console.server:app --port 8000
    # or: python console/server.py

Endpoints
---------
GET  /api/config     the console config, the corpus list, and what the backend
                     actually is -- model, device, gazetteer, boundary store
POST /api/geoparse   one document -> the console response shape
POST /api/batch      many documents (uploaded file) -> per-document summaries
GET  /api/telemetry  real numbers for the status bar, or nothing

Everything the status bar shows is measured. The design handoff's telemetry
strip was mocked with jitter, and its own note says jittering fake numbers in a
shipped product is worse than no numbers -- so the fields that cannot be
measured here (queue depth, p50 across a fleet) are absent rather than
invented, and the frontend omits what it is not sent.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# `python console/server.py` puts this directory first on sys.path, which is
# what lets `import adapter` work; running it as a module does not, so add it
# explicitly either way.
CONSOLE_DIR = Path(__file__).resolve().parent
if str(CONSOLE_DIR) not in sys.path:
    sys.path.insert(0, str(CONSOLE_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adapter import Stopwatch, to_console
from boundaries import get_store

logging.basicConfig(
    level=os.environ.get("MORDECAI_LOG", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("console")

REPO_ROOT = CONSOLE_DIR.parent
STATIC_DIR = CONSOLE_DIR / "static"
CONFIG_PATH = CONSOLE_DIR / "console.config.json"
CORPUS_PATH = CONSOLE_DIR / "corpus.json"

# Longest document accepted in one request. Well above any pasted article and
# well below the point where a single spaCy transformer pass becomes a problem.
MAX_TEXT_CHARS = 200_000
MAX_BATCH_DOCS = 500


class Options(BaseModel):
    top_k: int = Field(5, ge=1, le=25)
    review_gate: float = Field(0.40, ge=0.0, le=1.0)
    known_country: str | None = None
    # The masthead a document came from. Worth +7.2 exact match on held-out LGL
    # when the checkpoint carries the outlet block, and free when it does not.
    outlet: str | None = None


class GeoparseRequest(BaseModel):
    doc_id: str = "adhoc"
    text: str
    options: Options = Field(default_factory=Options)


class Engine:
    """The loaded models, built once at startup."""

    def __init__(self):
        import spacy
        from mordecai3 import Geoparser
        from mordecai3.mordecai_utilities import spacy_doc_setup
        from spacy.tokens import Token

        try:
            Token.set_extension("tensor", default=False)
        except ValueError:
            pass
        spacy_doc_setup()

        t0 = time.perf_counter()
        use_gpu = spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("token_tensors")
        logger.info("spaCy loaded in %.1fs (gpu=%s)",
                    time.perf_counter() - t0, use_gpu)

        t0 = time.perf_counter()
        # The packaged default checkpoint and its config sidecar; the sidecar is
        # what tells the Geoparser which enrichment feature blocks to compute.
        self.geo = Geoparser(
            geo_asset_path=(REPO_ROOT / "mordecai3" / "assets"),
            hosts=[os.environ.get("ES_HOST", "localhost")],
            nlp=self.nlp,
            debug=False,
        )
        logger.info("geoparser loaded in %.1fs", time.perf_counter() - t0)

        self.boundaries = get_store()
        self.use_gpu = bool(use_gpu)
        # The Geoparser does not keep the checkpoint path it loaded, so name it
        # from the packaged default the same way it resolves one.
        self.model_name = os.environ.get("MORDECAI_MODEL_NAME") or next(
            (p.name for p in sorted(
                (REPO_ROOT / "mordecai3" / "assets").glob("mordecai_*.pt"),
                reverse=True)), "packaged default")
        self.started = time.time()
        # Rolling record of real work done, for the status bar.
        self.docs_parsed = 0
        self.recent_ms = []

        # The first CUDA forward pass pays for kernel autotuning and context
        # setup -- ~880 ms against ~90 ms steady state on this box. That is
        # exactly the parse an audience watches, so spend it at startup
        # instead. The counters are reset afterwards so the warmup does not
        # land in the telemetry as if it were work.
        try:
            t0 = time.perf_counter()
            self.parse("Fighting was reported near Gao and Ansongo in Mali.",
                       "__warmup__", Options())
            logger.info("warmup pass in %.0f ms", (time.perf_counter() - t0) * 1000)
        except Exception:
            logger.warning("warmup pass failed; first request will be slower",
                           exc_info=True)
        self.docs_parsed = 0
        self.recent_ms = []

    # -------------------------------------------------------------- parsing

    def parse(self, text, doc_id, options: Options):
        watch = Stopwatch()
        doc = self.nlp(text)
        watch.mark("ner")

        result = self.geo.geoparse_doc(
            doc,
            top_k=options.top_k,
            # trim=False keeps the ranker features on the candidates, which is
            # what `rationale.py` reads to explain the choice. The adapter never
            # forwards them.
            trim=False,
            known_country=options.known_country,
            outlet=options.outlet,
        )
        watch.mark("rank")

        payload = to_console(
            result, doc_id=doc_id, text=text,
            review_gate=options.review_gate, top_k=options.top_k,
            boundary_store=self.boundaries if self.boundaries.available else None,
        )
        watch.mark("boundaries")
        payload["timing_ms"] = watch.total()
        payload["token_count"] = len(doc)

        self.docs_parsed += 1
        self.recent_ms.append(payload["timing_ms"]["total"])
        del self.recent_ms[:-50]
        return payload

    def parse_many(self, texts, ids, options: Options):
        """Batched path.

        `geoparse_batch` does its own `nlp.pipe` batching, threads the ES
        lookups across the whole chunk and pools every entity into a single
        model forward pass -- so it is handed the raw strings, not pre-piped
        docs, and the ner/rank split a single document reports is not available
        here.
        """
        watch = Stopwatch()
        results = self.geo.geoparse_batch(
            texts, top_k=options.top_k, trim=False,
            known_country=options.known_country)
        watch.mark("parse")

        store = self.boundaries if self.boundaries.available else None
        out = [to_console(r, doc_id=i, text=t, review_gate=options.review_gate,
                          top_k=options.top_k, boundary_store=store)
               for r, i, t in zip(results, ids, texts)]
        watch.mark("boundaries")
        timings = watch.total()
        self.docs_parsed += len(texts)
        return out, timings

    # ------------------------------------------------------------ telemetry

    def telemetry(self):
        """Only what can actually be measured."""
        out = {"docs_parsed": self.docs_parsed,
               "uptime_s": round(time.time() - self.started)}
        if self.recent_ms:
            ordered = sorted(self.recent_ms)
            out["p50_ms"] = ordered[len(ordered) // 2]
            out["last_ms"] = self.recent_ms[-1]
        try:
            import torch
            if torch.cuda.is_available():
                out["device"] = torch.cuda.get_device_name(0)
                out["vram_gb"] = round(
                    torch.cuda.memory_reserved(0) / 1e9, 2)
                out["vram_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            else:
                out["device"] = "cpu"
        except Exception:
            out["device"] = "cpu"
        try:
            health = self.geo.geonames.conn.cluster.health()
            out["gazetteer"] = f"geonames · {health['status']}"
        except Exception:
            out["gazetteer"] = "geonames · unreachable"
        return out

    def describe(self):
        """What this backend actually is, for the title bar."""
        return {
            "model": self.model_name,
            "device": "cuda" if self.use_gpu else "cpu",
            "boundaries": {
                "available": self.boundaries.available,
                "shapes": self.boundaries.shape_count,
                "source": self.boundaries.meta.get("source"),
            },
            "span_detector": getattr(self.geo, "span_detector", None) or "spacy-ents",
        }


ENGINE: Engine | None = None

# Every model call -- construction included -- runs on this one thread.
#
# Two reasons, and the first is not optional. spaCy on GPU installs CuPy as
# thinc's array backend, and CuPy's current-device state is *thread-local*: a
# forward pass that starts on a different thread from the one that set the
# backend up builds its index tensors on the CPU and dies with "Expected all
# tensors to be on the same device". FastAPI runs `def` endpoints on an
# anonymous threadpool, so without this the GPU path fails on the first
# request while working perfectly in a script.
#
# The second reason is that it is the right shape anyway: one GPU cannot run
# two forward passes faster than one, so serialising them costs nothing and
# makes `queue` on the status bar a real measurement.
INFERENCE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mordecai")
_QUEUED = 0
_QUEUE_LOCK = threading.Lock()


async def _run(fn, *args):
    """Run `fn` on the inference thread, tracking how deep the queue got."""
    global _QUEUED
    with _QUEUE_LOCK:
        _QUEUED += 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(INFERENCE, fn, *args)
    finally:
        with _QUEUE_LOCK:
            _QUEUED -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE

    def build():
        global ENGINE
        ENGINE = Engine()

    # Constructed on the inference thread too, so the CuPy backend is set up on
    # the same thread that will later use it.
    await asyncio.get_running_loop().run_in_executor(INFERENCE, build)
    logger.info("console ready")
    yield
    INFERENCE.shutdown(wait=False)


app = FastAPI(title="Mordecai Geoparse Console", lifespan=lifespan)


def _engine() -> Engine:
    if ENGINE is None:
        raise HTTPException(503, "engine still loading")
    return ENGINE


def _load_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


@app.get("/api/config")
def get_config():   # no model call, so it can answer on the event loop
    return {
        "config": _load_json(CONFIG_PATH, {}),
        "corpus": _load_json(CORPUS_PATH, {"documents": []})["documents"],
        "backend": _engine().describe(),
    }


@app.post("/api/geoparse")
async def geoparse(req: GeoparseRequest):
    text = req.text
    if not text.strip():
        raise HTTPException(400, "empty text")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text longer than {MAX_TEXT_CHARS} characters")
    engine = _engine()
    try:
        return await _run(engine.parse, text, req.doc_id, req.options)
    except Exception:
        logger.exception("geoparse failed for doc_id=%s", req.doc_id)
        raise HTTPException(500, "geoparse failed -- see server log")


@app.post("/api/batch")
async def batch(file: UploadFile = File(...), top_k: int = 5,
                review_gate: float = 0.40):
    """Geoparse an uploaded file.

    Accepts JSONL (one object per line with a `text` field), or plain text with
    documents separated by blank lines. Runs through `geoparse_batch`, which
    shares the transformer pass and the ES lookups across the whole file -- the
    reason the INGEST stage exists at all rather than being decoration.
    """
    raw = (await file.read()).decode("utf-8", errors="replace")
    name = file.filename or "upload"

    docs = []
    if name.endswith((".jsonl", ".ndjson")):
        for n, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise HTTPException(400, f"{name}: line {n+1} is not valid JSON")
            text = obj.get("text") or obj.get("body") or ""
            if text.strip():
                docs.append((str(obj.get("doc_id") or obj.get("id") or f"L{n+1}"),
                             text))
    else:
        for n, chunk in enumerate(raw.split("\n\n")):
            if chunk.strip():
                docs.append((f"{name}#{n+1}", chunk.strip()))

    if not docs:
        raise HTTPException(400, f"{name}: no documents found")
    if len(docs) > MAX_BATCH_DOCS:
        raise HTTPException(413, f"{len(docs)} documents; limit is {MAX_BATCH_DOCS}")

    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    options = Options(top_k=top_k, review_gate=review_gate)
    try:
        results, timings = await _run(_engine().parse_many, texts, ids, options)
    except Exception:
        logger.exception("batch failed for %s", name)
        raise HTTPException(500, "batch geoparse failed -- see server log")

    return {"filename": name,
            "documents": results,
            "timing_ms": timings,
            "stats": {
                "documents": len(results),
                "spans": sum(r["stats"]["spans"] for r in results),
                "resolved": sum(r["stats"]["resolved"] for r in results),
                "flagged": sum(r["stats"]["flagged"] for r in results),
            }}


@app.get("/api/telemetry")
def telemetry():
    # Reads counters and asks Elasticsearch for cluster health; touches no
    # model, so it stays responsive while a parse is in flight.
    out = _engine().telemetry()
    out["queue"] = _QUEUED
    return out


@app.get("/healthz")
def healthz():
    return {"ok": ENGINE is not None}


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("console.server:app" if __package__ else "server:app",
                host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")),
                reload=False)
