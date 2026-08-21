# Integration spec — place-span head into `mordecai3/geoparse.py`

Staged, not applied: another agent is editing `geoparse.py` in the main tree.
Everything here lives in the worktree at
`.claude/worktrees/agent-a4c25ff5ea1f68094/` and in the session scratchpad
`scratchpad/ner_scale/ship/`.

Artifacts in this directory:

| file | what |
|---|---|
| `span_head.py` | the self-contained inference module (numpy + torch + a spaCy Doc) |
| `span_head_gold_42.pt` | N1 as scoped: gold TR/LGL/GWN only, threshold 0.5 |
| `span_head_C_all_42.pt` | N2's combined arm: + WikiDocsFull + silver nested + demonym negatives (w10), threshold 0.3 |
| `test_span_head.py` | parity test: the module reproduces the harness's held-out detection numbers |

Each checkpoint is a dict of `state_dict`, `threshold`, `max_span`, `arm`,
`seed` — 1.6 MB. Both reproduce their training-harness row **exactly** under
`test_span_head.py`, on the D2 denominator (2,084 golds, 260 held-out docs):

| checkpoint | det P | det R | det F1 | nested det R | demonym FP | predictions |
|---|---|---|---|---|---|---|
| `span_head_gold_42.pt` | 85.83 | 89.49 | **87.62** | 67.9 | 81 | 2,173 |
| `span_head_C_all_42.pt` | 83.95 | 88.87 | 86.34 | **76.5** | **56** | 2,206 |

Which to take is a product decision, and the report's §3f/§6 make the case:
`gold` is the best detection F1, `C_all` trades 1.3 F1 for +8.6 nested recall
and 25 fewer demonym false positives — the two things D2 and the nested-miss
class actually care about. Both are 5-seed-verified recipes; seed 42 is shipped
here for reproducibility, not because it is the best seed.

---

## 1. What it replaces

Three code paths in `geoparse.py` become one call.

| today | after |
|---|---|
| `doc_to_ex_expanded(doc, geo_labels=GEO_LABELS, ...)` — the **label filter**: emit every spaCy entity whose label is in `("GPE","LOC","EVENT_LOC","FAC")` | `tagger.doc_to_ex(doc)` — emit every token span the head scores above threshold |
| `trim_span_tokens` — the **span trimmer**: strip a leading determiner / trailing possessive from spaCy's span | gone. The head is trained on gold character offsets, so it emits the trimmed span directly. `KEEP_LEADING_THE` and its "The Hague" special case go with it |
| `nested_gazetteer_spans` — the **nested-gazetteer pass** (opt-in, one ES round trip per candidate string) | gone. The head emits overlapping spans, so a toponym inside an ORG comes out beside its host with no gazetteer lookup and no extra ES traffic |

`guess_in_rel`, `doc_tensor`, `locs_tensor` and `sent` are computed exactly as
`doc_to_ex_expanded` computes them — `span_head.SpanTagger.doc_to_ex` is a
copy of that function's body with the span source swapped. spaCy NER still runs:
`locs_tensor` needs `CONTEXT_LABELS` entities and `guess_in_rel` needs
`doc.ents` and the parse.

## 2. Call sites

`Geoparser._geoparse_docs` (and whatever else calls `doc_to_ex_expanded`):

```python
# construction
self.span_tagger = None
if span_head_path:
    from mordecai3.span_head import SpanTagger
    self.span_tagger = SpanTagger.load(span_head_path, device=self.device)

# extraction
if self.span_tagger is not None:
    doc_ex = self.span_tagger.doc_to_ex(doc, context_labels=CONTEXT_LABELS)
else:
    doc_ex = doc_to_ex_expanded(doc, geo_labels=self.geo_labels,
                                context_labels=CONTEXT_LABELS,
                                trim_spans=self.trim_spans)
    if self.nested_gazetteer_pass:
        doc_ex = doc_ex + nested_gazetteer_spans(doc, doc_ex, self.geonames)
```

Keep the old branch behind the flag for one release: it is the only way to
reproduce every frozen number in the ledger, and `tools/end_to_end_eval.py`'s
`ship` / `serving` variants depend on it.

## 3. Packaging

* checkpoint → `mordecai3/assets/span_head_<date>.pt` (2.0 MB), listed in
  `pyproject.toml`'s package data beside the ranker checkpoint;
* module → `mordecai3/span_head.py`;
* `Geoparser.__init__` gains `span_head_path=None`. When the asset is present
  and the caller passes nothing, default it on only after the TEST-corpus gate
  below.

The threshold travels **inside** the checkpoint (it is chosen on the training
corpora's dev split, and a caller who overrides it is off the measured
operating point). `SpanTagger.load(..., threshold=x)` exists for sweeps.

## 4. Cost

The head is 0.5 M parameters over token vectors the pipeline has already
computed for the ranker, so there is no second transformer pass. The pilot
measured **+2.70 ms/doc** (~+5% of pipeline wall clock); the scaled head is the
same architecture and the same candidate enumeration, so the number carries.
Against that, `nested_gazetteer_pass=True` costs ~19% of throughput and one ES
round trip per document, and it is what the head replaces.

Candidate count is `O(n_tokens x max_span)`; the enumeration is one batched
matmul, so long documents are not a problem, but a caller feeding 100 kB of
text in one `Doc` will allocate ~8 spans per token.

## 5. Gate before it is defaulted on

Unchanged from the scoping report's N1 gate, and it now matters more, not less:
every number in this track is on **held-out documents of the corpora the head
trained on** (TR-News / LGL / GeoWebNews). The head learns those corpora's span
conventions. Reproduce the detection and e2e numbers on D1's untouched
modern-news TEST corpus before the flag flips to on by default.

## 6. Two behaviours that change for callers

1. **Overlapping spans are emitted.** `Geoparser` output can now contain
   "Pittsburgh" and "University of Pittsburgh" at overlapping character
   offsets. Anything downstream that assumes a flat, non-overlapping entity
   list needs to say which it wants. Containment suppression was measured in
   the pilot (§5a) and is worth ~+0.1 F1, i.e. not worth adding for accuracy —
   but it may be worth adding as an output option for consumers that need
   flatness.
2. **NORP is never emitted, and now that is learned rather than filtered.**
   The label filter guaranteed it structurally; the head is trained on demonym
   spans as negatives and gets it right most of the time. The residual
   demonym false-positive count is a first-class metric in the report and it is
   not zero. A caller that must never see a demonym should keep a NORP-span
   veto as a post-filter.
