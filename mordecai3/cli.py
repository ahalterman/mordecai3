"""The `mordecai3` command: build the GeoNames index and check a setup."""

import os
from pathlib import Path

import typer

app = typer.Typer(help="Mordecai 3 geoparser utilities.", no_args_is_help=True)
index_app = typer.Typer(help="Build and inspect the GeoNames Elasticsearch index.",
                        no_args_is_help=True)
app.add_typer(index_app, name="index")

DEFAULT_ES_URL = os.environ.get("MORDECAI_ES_URL", "http://localhost:9200")
ES_URL = typer.Option(DEFAULT_ES_URL, "--es-url",
                      help="Elasticsearch URL (env: MORDECAI_ES_URL).")
DATA_DIR = typer.Option(Path("geonames_data"), "--data-dir",
                        help="Where the GeoNames dump is downloaded to / read from.")


@index_app.command("build")
def index_build(es_url: str = ES_URL, data_dir: Path = DATA_DIR,
                skip_download: bool = typer.Option(
                    False, help="Reuse a gazetteer already in --data-dir.")):
    """Download GeoNames and (re)build the `geonames` index from scratch.

    Deletes and recreates only the `geonames` index; other indices on the node
    are left alone. Needs ~4 GB of disk for the dump plus ~2 GB for the index.
    """
    from . import index_builder as ib
    es = ib.connect(es_url)
    if not es.ping():
        typer.secho(f"No Elasticsearch at {es_url}. Start one with:\n\n"
                    "    docker run -d -p 127.0.0.1:9200:9200 -e discovery.type=single-node "
                    "-v $PWD/geonames_index:/usr/share/elasticsearch/data elasticsearch:7.10.1",
                    fg="red", err=True)
        raise typer.Exit(1)
    if not skip_download:
        ib.download(data_dir)
    before = ib.count(es)
    ib.recreate(es)
    ib.load(es, data_dir)
    after = ib.count(es)
    meta = ib.stamp(es, data_dir, after)
    typer.echo(f"geonames: {before if before is not None else 'no index'} -> {after} documents")
    typer.echo(f"provenance: {meta}")


@index_app.command("download")
def index_download(data_dir: Path = DATA_DIR):
    """Only fetch the GeoNames dump (no Elasticsearch needed)."""
    from . import index_builder as ib
    ib.download(data_dir)


@index_app.command("status")
def index_status(es_url: str = ES_URL):
    """Document count and build provenance of the `geonames` index."""
    from . import index_builder as ib
    es = ib.connect(es_url)
    if not es.ping():
        typer.secho(f"No Elasticsearch at {es_url}.", fg="red", err=True)
        raise typer.Exit(1)
    n = ib.count(es)
    if n is None:
        typer.secho("No 'geonames' index. Build one with `mordecai3 index build`.",
                    fg="red", err=True)
        raise typer.Exit(1)
    typer.echo(f"documents: {n:,}")
    meta = ib.provenance(es)
    typer.echo(f"provenance: {meta or 'none recorded (built before provenance stamping)'}")


@app.command("check")
def check(es_url: str = ES_URL):
    """Check that everything Mordecai needs at runtime is in place."""
    ok = True

    def report(passed, label, detail=""):
        nonlocal ok
        ok &= passed
        mark = typer.style("ok  " if passed else "FAIL", fg="green" if passed else "red")
        typer.echo(f"[{mark}] {label}{': ' + detail if detail else ''}")

    from . import __version__
    typer.echo(f"mordecai3 {__version__}")

    try:
        import spacy
        spacy.util.get_package_path("en_core_web_trf")
        report(True, "spaCy model en_core_web_trf")
    except Exception:
        report(False, "spaCy model en_core_web_trf",
               "python -m spacy download en_core_web_trf")

    try:
        import torch
        report(True, "torch", f"{torch.__version__}, "
               f"{'CUDA available' if torch.cuda.is_available() else 'CPU only'}")
    except Exception as e:
        report(False, "torch", str(e))

    from . import index_builder as ib
    es = ib.connect(es_url)
    if not es.ping():
        report(False, f"Elasticsearch at {es_url}", "not reachable")
    else:
        version = es.info()["version"]["number"]
        report(True, f"Elasticsearch at {es_url}", version)
        n = ib.count(es)
        if n is None:
            report(False, "geonames index", "missing -- `mordecai3 index build`")
        else:
            report(n > 10_000_000, "geonames index", f"{n:,} documents"
                   + ("" if n > 10_000_000 else " (a test or partial index?)"))
            meta = ib.provenance(es) or {}
            if meta.get("dump_date"):
                typer.echo(f"       GeoNames dump of {meta['dump_date']}")
    typer.echo("\nUsing Mordecai in research? `mordecai3 cite` prints the citation.")
    raise typer.Exit(0 if ok else 1)


CITATION = """@article{halterman2023mordecai,
  title={Mordecai 3: A neural geoparser and event geocoder},
  author={Halterman, Andrew},
  journal={arXiv preprint arXiv:2303.13675},
  year={2023}
}"""


@app.command("cite")
def cite():
    """Print the BibTeX citation for Mordecai 3."""
    typer.echo(CITATION)


@app.command("app")
def streamlit_app():
    """Launch the Streamlit demo page."""
    from . import run_streamlit_app
    run_streamlit_app()


def main():
    app()
