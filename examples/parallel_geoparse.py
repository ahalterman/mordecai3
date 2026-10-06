"""
Parallel geoparsing across multiple worker processes.

This sets up separate workers, each of which have their own spaCy model and Mordecai model. 
This uses way more memory, since everything gets duplicated per worker, but if you're running on a 
24GB card, it's not so bad.

Usage:
    python parallel_geoparse.py --input texts.jsonl --output results.jsonl --workers 4
    python parallel_geoparse.py --input texts.jsonl --output results.jsonl --workers 2 --device cpu

Input format (JSONL, one JSON object per line):
    {"text": "The fighting in Aleppo, Syria continued today..."}
    {"text": "President Biden spoke from the White House."}

Output format (JSONL, one JSON object per line, same order as input):
    {"doc_text": "...", "geolocated_ents": [...], ...}
"""

import argparse
import json
import multiprocessing as mp
import sys
import time


# ---------------------------------------------------------------------------
# Worker functions (run in child processes)
# ---------------------------------------------------------------------------

def _worker_init(device, es_hosts):
    """Each worker process loads its own Geoparser instance."""
    global _geo
    from mordecai3 import Geoparser
    _geo = Geoparser(hosts=es_hosts, device=device)


def _worker_fn(texts):
    """Process a chunk of texts using the worker's Geoparser."""
    return _geo.geoparse_batch(texts, show_progress=False)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def parallel_geoparse(texts, n_workers=4, device="cuda:0", es_hosts=None):
    """
    Split texts across worker processes, each with its own Geoparser.

    Parameters
    ----------
    texts : list of str
        Documents to geoparse.
    n_workers : int
        Number of worker processes.
    device : str
        PyTorch device for each worker (e.g. "cuda:0", "cpu").
    es_hosts : list of str or None
        Elasticsearch hosts. Defaults to ["localhost"].

    Returns
    -------
    list of dict
        One result dict per input document, in order.
    """
    if es_hosts is None:
        es_hosts = ["localhost"]

    # Split texts into n_workers roughly equal chunks
    chunk_size = max(1, len(texts) // n_workers + (1 if len(texts) % n_workers else 0))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    ctx = mp.get_context("spawn")  # required for CUDA compatibility
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(device, es_hosts)) as pool:
        chunk_results = pool.map(_worker_fn, chunks)

    # Flatten results maintaining original order
    return [r for chunk in chunk_results for r in chunk]


def main():
    parser = argparse.ArgumentParser(
        description="Parallel geoparsing across multiple Geoparser workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL file (each line: {\"text\": \"...\"})")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for results")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes (default: 4)")
    parser.add_argument("--device", default="cuda:0",
                        help="PyTorch device (default: cuda:0)")
    parser.add_argument("--es-host", default="localhost",
                        help="Elasticsearch host (default: localhost)")
    parser.add_argument("--text-field", default="text",
                        help="JSON field containing the text (default: text)")
    args = parser.parse_args()

    # Read input
    texts = []
    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.text_field not in obj:
                print(f"Warning: line {line_num} missing '{args.text_field}' field, skipping",
                      file=sys.stderr)
                continue
            texts.append(obj[args.text_field])

    if not texts:
        print("No texts found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(texts)} documents")
    print(f"Workers: {args.workers}, Device: {args.device}, ES: {args.es_host}")

    t0 = time.perf_counter()
    results = parallel_geoparse(
        texts,
        n_workers=args.workers,
        device=args.device,
        es_hosts=[args.es_host],
    )
    elapsed = time.perf_counter() - t0

    # Write output
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_ents = sum(len(r.get("geolocated_ents", [])) for r in results)
    print(f"Done: {len(results)} documents, {n_ents} entities, {elapsed:.1f}s "
          f"({len(results) / elapsed:.1f} docs/s)")


if __name__ == "__main__":
    main()
