"""Place-span detection head over the frozen `en_core_web_trf` tensors.

Self-contained: the only runtime dependencies are numpy, torch and a spaCy Doc
whose tokens carry the `._.tensor` extension that `mordecai3.mordecai_utilities
.spacy_doc_setup` already installs. No second transformer forward pass -- the
768-d per-token vectors the pipeline computes for the ranker are reused, which
is why this costs ~5 ms/doc against the 1.7 ms of the label-filter path, and
less than the 8.7 ms of the `nested_gazetteer_spans` pass it replaces
(experiments/e55_ner_head/NOTES.md, re-measured end to end in
experiments/campaign2/span_head_serving_report.md).

Drop-in for the three things it replaces in `mordecai3/geoparse.py`:

    Geoparser(span_detector="all")      # the supported way in
    tagger = load_span_tagger("all")    # or, standalone:
    ex = tagger.doc_to_ex(doc)          # instead of doc_to_ex_expanded(...)

`doc_to_ex` returns exactly the dicts `doc_to_ex_expanded` returns
(`search_name`, `tensor`, `doc_tensor`, `locs_tensor`, `sent`, `in_rel`,
`start_char`, `end_char`), so everything downstream -- `add_es_data_batch`,
`ProductionData`, the ranker, `_resolve_results` -- is untouched.

The head emits arbitrary overlapping spans, so a toponym nested inside an
ORG/FAC span ("Pittsburgh" in "University of Pittsburgh") comes out alongside
its host without a gazetteer pass, and demonym spans are trained as negatives
rather than filtered by label.
"""
from __future__ import annotations

import logging
import os
from importlib import resources

import numpy as np
import torch
import torch.nn as nn

__all__ = ["SpanHead", "SpanTagger", "SPAN_HEAD_ASSETS", "resolve_span_head",
           "load_span_tagger"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MAX_SPAN = 8
# Labels whose tokens are pooled into `locs_tensor`; must stay equal to
# mordecai3.geoparse.CONTEXT_LABELS.
CONTEXT_LABELS = ("GPE", "LOC")

# The two packaged heads, both seed 42 of a 5-seed-verified recipe
# (experiments/campaign2/ner_head_scaling_report.md). Detection on the D2
# denominator (2,084 golds, 260 held-out TR/LGL/GWN documents):
#
#   "gold"  det F1 87.62, nested det R 67.9, demonym FPs 81, threshold 0.5
#           -- N1 as scoped: trained on TR/LGL/GWN gold toponyms only.
#   "all"   det F1 86.34, nested det R 76.5, demonym FPs 56, threshold 0.3
#           -- the combined arm: + WikiDocsFull anchors, silver nested spans
#           and 10x-weighted demonym negatives.
#
# Which one to serve is measured end to end in
# experiments/campaign2/span_head_serving_report.md: "gold" wins by ~0.7 exact
# match in every ranker condition, "all" is the one to take if demonym false
# positives matter more than pooled EM.
#
# STAGED, NOT DEFAULT. `Geoparser(span_detector=...)` is worth +10.0 to +10.6
# end-to-end exact match and -7.2 ms/doc, but every number is on held-out
# DOCUMENTS of the corpora the head trained on. The N1 gate -- reproduce
# detection and e2e on D1's untouched modern-news TEST corpus -- is open, and it
# is the only thing between this and flipping the default. Owner's call.
SPAN_HEAD_ASSETS = {
    "gold": "assets/span_head_2026-08-20_gold.pt",
    "all": "assets/span_head_2026-08-20_all.pt",
}


class SpanHead(nn.Module):
    """0.5 M-parameter span classifier over frozen 768-d token vectors."""

    def __init__(self, dim=768, hid=256, width=32, max_span=DEFAULT_MAX_SPAN,
                 dropout=0.2):
        super().__init__()
        self.max_span = max_span
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU(),
                                  nn.LayerNorm(hid), nn.Dropout(dropout))
        self.width = nn.Embedding(max_span + 1, width)
        self.out = nn.Sequential(nn.Linear(3 * hid + width, hid), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hid, 1))

    def forward(self, h, spans):
        """h: (T, 768) token vectors. spans: (S, 2) long, [start, end)."""
        p = self.proj(h)
        cs = torch.cat([torch.zeros(1, p.shape[1], device=p.device,
                                    dtype=p.dtype), p.cumsum(0)], 0)
        s, e = spans[:, 0], spans[:, 1]
        mean = (cs[e] - cs[s]) / (e - s).unsqueeze(1).to(p.dtype)
        rep = torch.cat([p[s], p[e - 1], mean, self.width(e - s)], -1)
        return self.out(rep).squeeze(-1)


class SpanTagger:
    """Inference wrapper: spaCy Doc -> place spans -> geoparser entity dicts."""

    def __init__(self, head: SpanHead, threshold: float, device=None):
        self.head = head.eval()
        self.threshold = float(threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head.to(self.device)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path, device=None, threshold=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        head = SpanHead(max_span=ckpt.get("max_span", DEFAULT_MAX_SPAN))
        head.load_state_dict(ckpt["state_dict"])
        return cls(head, threshold if threshold is not None
                   else ckpt["threshold"], device=device)

    # ---------------------------------------------------------- enumeration
    def _candidates(self, n_tokens, sent_starts):
        """Every span of <= max_span tokens that stays inside one sentence."""
        bounds = sorted(set([0] + [s for s in sent_starts if 0 < s < n_tokens]))
        bounds = bounds + [n_tokens]
        out = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            for s in range(lo, hi):
                for e in range(s + 1, min(s + self.head.max_span, hi) + 1):
                    out.append((s, e))
        return np.asarray(out, dtype="int64").reshape(-1, 2)

    # ------------------------------------------------------------- scoring
    @torch.no_grad()
    def span_scores(self, doc):
        """[(start_char, end_char, probability)] for every candidate span."""
        n = len(doc)
        if n == 0:
            return []
        spans = self._candidates(n, [t.i for t in doc if t.is_sent_start])
        if not len(spans):
            return []
        h = torch.from_numpy(
            np.vstack([t._.tensor for t in doc]).astype("float32")
        ).to(self.device)
        sp = torch.from_numpy(spans).to(self.device)
        p = torch.sigmoid(self.head(h, sp)).float().cpu().numpy()
        out = []
        for (s, e), q in zip(spans, p):
            out.append((doc[int(s)].idx,
                        doc[int(e) - 1].idx + len(doc[int(e) - 1].text),
                        float(q)))
        return out

    def spans(self, doc, threshold=None):
        """Character spans above threshold, de-duplicated and sorted."""
        thr = self.threshold if threshold is None else threshold
        return sorted({(lo, hi) for lo, hi, q in self.span_scores(doc)
                       if q >= thr})

    def scored_spans(self, doc, threshold=None):
        """`spans()`, but each entry keeps the head's probability.

        The same span can be scored more than once (overlapping windows), and
        the max is kept: the head's confidence in a span is the best evidence
        it found for it, not the last. Separate from `spans()` rather than a
        change to it, because that one is the documented interface.
        """
        thr = self.threshold if threshold is None else threshold
        best = {}
        for lo, hi, q in self.span_scores(doc):
            if q >= thr and q > best.get((lo, hi), 0.0):
                best[(lo, hi)] = q
        return sorted((lo, hi, q) for (lo, hi), q in best.items())

    # ------------------------------------------------------ geoparser shim
    def doc_to_ex(self, doc, context_labels=CONTEXT_LABELS, threshold=None):
        """`doc_to_ex_expanded`'s output shape, from the head's spans.

        Deliberately identical to the reference implementation in every field
        except which spans are emitted:

        * `doc_tensor` is the mean token tensor of the whole document;
        * `locs_tensor` is the mean over `context_labels` entity tokens that
          are not part of this mention (zeros if there are none);
        * `in_rel` is `guess_in_rel` over the mention's own tokens;
        * `sent` is the sentence text of the mention's first token.

        Plus two fields the reference implementation has no source for:
        `label` (always "SPAN") and `span_score`, the head's probability.

        `guess_in_rel` is imported lazily so this module can be used without
        the rest of the package (tests, offline scoring).
        """
        from .geoparse import guess_in_rel

        picked = self.scored_spans(doc, threshold)   # (lo, hi, probability)
        if not picked:
            return []
        doc_tensor = np.mean(np.vstack([t._.tensor for t in doc]), axis=0)
        ctx = [t for e in doc.ents if e.label_ in context_labels for t in e]
        data = []
        for lo, hi, span_score in picked:
            own = [t for t in doc if t.idx >= lo and t.idx + len(t.text) <= hi]
            if not own:
                continue
            own_i = {t.i for t in own}
            other = [t for t in ctx if t.i not in own_i]
            tensor = np.mean(np.vstack([t._.tensor for t in own]), axis=0)
            data.append({
                "search_name": doc.text[lo:hi],
                # The head has no notion of GPE vs LOC vs FAC -- it decides
                # only whether a span is a place -- so it reports its own name
                # and the probability behind the call, which the label-based
                # detectors have no equivalent of.
                "label": "SPAN",
                "span_score": span_score,
                "tensor": tensor,
                "doc_tensor": doc_tensor,
                "locs_tensor": (np.mean(np.vstack([t._.tensor for t in other]),
                                        axis=0)
                                if other else np.zeros(len(tensor))),
                "sent": own[0].sent.text,
                "in_rel": guess_in_rel(doc[own[0].i:own[-1].i + 1]),
                "start_char": lo,
                "end_char": hi})
        return data


def resolve_span_head(span_detector):
    """A `span_detector=` argument -> a checkpoint path.

    Accepts a packaged name ("gold" / "all", see `SPAN_HEAD_ASSETS`) or a
    filesystem path to any checkpoint `SpanTagger.load` can read. None is not
    accepted here: the caller decides what "no head" means.
    """
    if span_detector is None:
        raise ValueError("span_detector=None has no checkpoint")
    key = str(span_detector)
    if key in SPAN_HEAD_ASSETS:
        path = resources.files("mordecai3") / SPAN_HEAD_ASSETS[key]
        if not path.is_file():
            raise FileNotFoundError(
                f"span_detector={key!r} needs the packaged asset "
                f"{SPAN_HEAD_ASSETS[key]}, which is not installed")
        return path
    if os.path.exists(key):
        return span_detector
    raise ValueError(
        f"span_detector={span_detector!r} is neither one of "
        f"{sorted(SPAN_HEAD_ASSETS)} nor an existing path")


def load_span_tagger(span_detector, device=None, threshold=None):
    """`SpanTagger` for a packaged name or a checkpoint path."""
    path = resolve_span_head(span_detector)
    tagger = SpanTagger.load(path, device=device, threshold=threshold)
    logger.info(f"Span detection head {span_detector} loaded from {path} "
                f"(threshold {tagger.threshold}, "
                f"max_span {tagger.head.max_span})")
    return tagger
