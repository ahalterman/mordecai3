"""Turn the scraped Wikipedia dumps into document-level training data.

The scrape produced one JSONL row per *linked place mention*, each carrying its
own copy of the surrounding text. Three files, three different schemas:

    wiki_training_data_battles.jsonl    text / sent_text, start_char is into `text`
    wiki_training_data_protest.jsonl    same
    wiki_training_data_sents_disasters.jsonl  text_doc / text_sent, offsets for both

`wiki_training_data_sents.jsonl` -- the file the training pipeline has been
using -- is 2,160 of the 2,506 battles rows, flattened to one sentence per
document. That is 9% of the 23,668 annotated mentions available, and the
sentence framing costs more than the volume does: `_add_cross_entity_counts`
builds its adm1/country overlap features from the other place names *in the
same training example*, so a one-sentence example gets almost none of the
co-occurrence signal the model sees at inference time on a whole article.

This script:

  * merges the three files and groups mentions back onto the document they came
    from, so a document carries all of its labelled mentions at once;
  * strips the leftover wiki markup (''italic'', '''bold''', ==headings==) and
    remaps every offset, so the annotated span still lines up with the text;
  * splits documents longer than --max-chars on paragraph boundaries, which
    puts wiki documents in the same length range as the TR-News and LGL
    articles the model also trains on (median ~1.5k chars, p90 ~4k);
  * shuffles by article title, so a positional train/test split in
    `train.py:load_es_data` neither straddles an article nor puts every
    disaster document in the test half.

Output is one JSON object per document chunk:

    {"title", "category", "chunk", "text",
     "toponyms": [{"phrase", "start", "end", "geonamesid", "wikidata", "wiki_page"}]}
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

import typer

# The raw files and where each one keeps its document text and offsets. The
# three category files are one scrape; `wiki_big_sample` is a later, wider one
# (`geo_wiki/geo_wiki_parse.py`) over ~4,000 articles, already frequency-
# subsampled so that common country names do not swamp everything else. They
# overlap on ~1,200 documents, which the (title, text, span) grouping collapses.
SOURCES = [
    ("battles", "wiki_training_data_battles.jsonl", "text", "start_char", "end_char"),
    ("protest", "wiki_training_data_protest.jsonl", "text", "start_char", "end_char"),
    ("disasters", "wiki_training_data_sents_disasters.jsonl", "text_doc",
     "start_char_doc", "end_char_doc"),
    ("sampled", "wiki_big_sample.jsonl", "text", "start_char_doc", "end_char_doc"),
]

# Every markup form still present in the scraped text. Anything matched here is
# deleted outright, so each pattern must only ever cover markup characters:
# `'{2,}` cannot hit an English apostrophe, and the heading rule is anchored to
# a line so `x == y` in running text is left alone.
MARKUP = [
    re.compile(r"'{2,}"),                     # ''italic'' and '''bold'''
    re.compile(r"^\s*=+|=+\s*$", re.MULTILINE),  # ==Section headings==
]


def strip_markup(text, spans):
    """Delete markup from `text`, remapping `spans` onto the cleaned string.

    `spans` is a list of (start, end) into the original text. Returns the
    cleaned text and the remapped spans. A span that overlaps deleted markup is
    returned as None so the caller can drop it rather than silently shifting it
    onto the wrong characters.
    """
    deleted = bytearray(len(text))
    for pat in MARKUP:
        for m in pat.finditer(text):
            for i in range(m.start(), m.end()):
                deleted[i] = 1

    if not any(deleted):
        return text, list(spans)

    # new_index[i] is where original char i lands in the cleaned string.
    new_index = [0] * (len(text) + 1)
    out = []
    n = 0
    for i, ch in enumerate(text):
        new_index[i] = n
        if not deleted[i]:
            out.append(ch)
            n += 1
    new_index[len(text)] = n

    remapped = []
    for start, end in spans:
        if any(deleted[start:end]):
            remapped.append(None)
        else:
            remapped.append((new_index[start], new_index[end]))
    return "".join(out), remapped


def split_paragraphs(text, max_chars):
    """Split `text` into chunks of at most ~max_chars, preferring blank lines.

    Returns a list of (offset, chunk_text). Offsets are into `text`, so a
    toponym's span moves by a simple subtraction. A paragraph that is itself
    longer than max_chars is broken at the last newline or space before the
    limit; only 1-2 documents in the corpus hit that path.
    """
    if len(text) <= max_chars:
        return [(0, text)]

    # Keep the separators so offsets stay exact.
    paras = [(m.start(), m.group()) for m in re.finditer(r"[^\n]*\n*", text)
             if m.group()]

    chunks = []
    cur_start, cur = None, []
    cur_len = 0
    for offset, para in paras:
        while len(para) > max_chars:
            cut = para.rfind("\n", 0, max_chars)
            if cut <= 0:
                cut = para.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head, para = para[:cut], para[cut:]
            if cur:
                chunks.append((cur_start, "".join(cur)))
                cur, cur_len, cur_start = [], 0, None
            chunks.append((offset, head))
            offset += cut
        if cur and cur_len + len(para) > max_chars:
            chunks.append((cur_start, "".join(cur)))
            cur, cur_len, cur_start = [], 0, None
        if not cur:
            cur_start = offset
        cur.append(para)
        cur_len += len(para)
    if cur:
        chunks.append((cur_start, "".join(cur)))
    return chunks


def load_raw(wiki_dir, include):
    """Group every annotated mention onto its document text."""
    docs = defaultdict(lambda: {"title": None, "category": None, "mentions": []})
    counts = Counter()
    for category, fn, text_key, start_key, end_key in SOURCES:
        if category not in include:
            continue
        path = os.path.join(wiki_dir, fn)
        if not os.path.exists(path):
            counts[f"{category}: file not found, skipped"] += 1
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                text = row[text_key]
                start, end = row[start_key], row[end_key]
                # The scrape is internally consistent -- every row in every file
                # satisfies this -- but the check is cheap and the failure mode
                # (silently training on the wrong span) is not detectable later.
                if text[start:end] != row["ent_text"]:
                    counts[f"{category}: offset mismatch"] += 1
                    continue
                key = (row["title"], text)
                d = docs[key]
                d["title"] = row["title"]
                d["category"] = row.get("category") or category
                d["mentions"].append({
                    "phrase": row["ent_text"],
                    "start": start,
                    "end": end,
                    # ~6% of mentions were never linked to a geonames id. They
                    # are kept, not dropped: the formatter still hands them to
                    # Elasticsearch so they feed the document's adm1/country
                    # overlap features, and only then discards them as unlabelled.
                    "geonamesid": (str(row["correct_geonamesid"])
                                   if row["correct_geonamesid"] is not None else None),
                    "wikidata": row.get("geo_wikidata"),
                    "wiki_page": row.get("geo_wiki_page"),
                })
                counts[f"{category}: mentions"] += 1
    return docs, counts


app = typer.Typer(add_completion=False)


@app.command()
def main(wiki_dir: str = typer.Argument(..., help="directory holding the raw wiki jsonl files"),
         out_file: str = typer.Argument(..., help="where to write the prepared jsonl"),
         max_chars: int = 4000,
         seed: int = 617,
         include: str = "battles,protest,disasters,sampled"):
    include = {i.strip() for i in include.split(",") if i.strip()}
    docs, counts = load_raw(wiki_dir, include)
    print(f"{len(docs)} unique documents")

    records = []
    dropped = Counter()
    for (title, text), d in docs.items():
        mentions = d["mentions"]
        # Duplicate rows exist across files; a mention is identified by its span.
        by_span = {}
        for m in mentions:
            by_span.setdefault((m["start"], m["end"]), m)
        mentions = sorted(by_span.values(), key=lambda m: m["start"])

        clean, spans = strip_markup(text, [(m["start"], m["end"]) for m in mentions])
        kept = []
        for m, span in zip(mentions, spans):
            if span is None:
                dropped["span overlapped markup"] += 1
                continue
            m = dict(m, start=span[0], end=span[1])
            if clean[m["start"]:m["end"]] != m["phrase"]:
                dropped["remap mismatch"] += 1
                continue
            kept.append(m)

        for n, (offset, chunk_text) in enumerate(split_paragraphs(clean, max_chars)):
            lo, hi = offset, offset + len(chunk_text)
            in_chunk = [dict(m, start=m["start"] - lo, end=m["end"] - lo)
                        for m in kept if m["start"] >= lo and m["end"] <= hi]
            if not in_chunk:
                dropped["chunk with no toponyms"] += 1
                continue
            if not chunk_text.strip():
                dropped["blank chunk"] += 1
                continue
            records.append({"title": title, "category": d["category"], "chunk": n,
                            "text": chunk_text, "toponyms": in_chunk})

    # Shuffle whole articles, not documents: `load_es_data` splits train/test by
    # position, so entities from one article must not land on both sides of it,
    # and the three categories have to be interleaved or the test half is one
    # category.
    by_title = defaultdict(list)
    for r in records:
        by_title[r["title"]].append(r)
    titles = sorted(by_title)
    random.Random(seed).shuffle(titles)
    records = [r for t in titles for r in by_title[t]]

    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_top = sum(len(r["toponyms"]) for r in records)
    n_labelled = sum(1 for r in records for t in r["toponyms"] if t["geonamesid"])
    lens = sorted(len(r["text"]) for r in records)
    print(f"wrote {len(records)} documents, {n_top} toponyms "
          f"({n_labelled} with a geonames id) to {out_file}")
    print(f"  documents from {len(titles)} articles; "
          f"chars median {lens[len(lens)//2]}, max {lens[-1]}")
    print(f"  toponyms/document mean {n_top/len(records):.1f}")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    for k, v in dropped.most_common():
        print(f"  dropped -- {k}: {v}")


if __name__ == "__main__":
    app()
