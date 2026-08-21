# Campaign 3 — plan (drafted 2026-08-20, end of campaign 2)

Campaign 2 closed with the serving stack at **82.29 e2e EM** (D2 denominator,
staged flips applied) against an oracle-span ceiling of 89.05, from 67.66 at
the start of the day. What remains is qualitatively different work: the easy
levers are pulled, and the residual error is split between one index-config
line, annotation debt, and question marks about how well the whole edifice
transfers off the three dev corpora. Campaign 3 should be the campaign of
**generalisation and downstream fit**, not leaderboard motion on TR/LGL/GWN.

Reference reading: `experiments/campaign2/SYNTHESIS.md` (execution section),
`ner_retrain_scoping_report.md`, `ner_head_scaling_report.md`,
`span_head_serving_report.md`, `r1_retrieval_report.md`,
`gazetteer_hygiene_report.md`.

## New resource: OntoNotes 5.0 (Andy is moving it over from another server)

What it unlocks — and, importantly, what it does not:

**Unlocks.**
- **The N4 shared encoder becomes trainable from first principles.** Campaign
  2 distilled spaCy's OntoNotes knowledge (bit-exact roberta extraction);
  with the real corpus we can fine-tune any encoder on OntoNotes NER + our
  gold place spans jointly, and emit spans and ranker tensors from one
  forward pass. e56 measured that this would be *cheaper* than today's
  pipeline; the scoping probes say place supervision fixes the feature-class
  signal that stock encoders lacked. This merges the old e43 "typed encoder"
  item and retires spaCy from the serving path entirely.
- **Demonym negatives at corpus scale.** e55 found demonym negatives the one
  label source that paid, using only our ~300 NORP spans. OntoNotes has
  thousands of NORP annotations across genres.
- **Genre breadth.** OntoNotes spans newswire, broadcast, web text, and
  telephone conversation — the closest thing we have to "messier text"
  supervision (see Track C).

**Does NOT unlock.**
- **Nested toponyms.** OntoNotes NER is flat. The e55 finding stands: the
  binding constraint on the span head is in-domain, *complete*, *nested*
  gold, and no downloadable corpus supplies it. That is an annotation
  project (Track B), and OntoNotes does not substitute for it.
- More of the same flat news NER signal — e55 showed that axis is saturated
  from ~2,500 documents.

## Tracks

### A. N4 shared encoder (flagship candidate)
One fine-tuned encoder (roberta-base to start; ModernBERT-base is already in
the HF cache if long-context matters) trained on OntoNotes NER + our 5,034
gold place spans (+ Track B output when it lands), replacing BOTH
en_core_web_trf calls: span detection and the ranker's token tensors.
- Gate 1 (cheap): does the shared encoder's span F1 match the frozen-head
  87.6 on dev and the LOCO folds?
- Gate 2 (the real one): retrain the ranker on the new tensors, 5 seeds —
  does TLG-hard hold or improve? The mention-slot signal is the risk e56's
  probes de-risked but never end-to-end tested.
- Gate 3: latency. The pitch is one forward pass instead of two; verify on
  CPU too (today: ~185 ms/doc CPU — deployment-relevant).

### B. Annotation sprint: nested gold + the TEST corpus (D1 debt)
One annotation effort, two payoffs. Write the span convention FIRST — a
one-page document deciding, with examples, exactly what a place span is:
nested toponyms inside ORG/FAC names, adjectival forms ("British troops":
D2 says no; "the Turkish border": the border is a place — decide),
prepositional constructions ("outside Aleppo", "north of Kandahar"),
coordination ("New York and the outskirts of Newark"), vague regions
("eastern Ukraine", "the Sahel"). Then annotate modern news (2024–2026,
non-US-local sources deliberately over-represented) to that convention.
- Half becomes the frozen TEST corpus (the publication blocker; LOCO
  narrowed it but only new-family data closes it).
- Half becomes nested-gold training data — the one label source e55 proved
  the head is starved of (+0.7 F1 / +4.5 nested recall per doubling before
  the wiki plateau; gold is the axis that was never saturated).
- Costing note: e55's marginal-value measurements mean even ~500 carefully
  chosen documents beat the 20,302 wiki docs that bought nothing.

### C. Robustness to messier extractions (Andy's ask)
The dev corpora are clean newswire with clean golds. Downstream users feed
the system event-coder output, scraped text, and constructions like
"outside Aleppo" or "in New York and the outskirts of Newark". Three parts:
1. **Inventory reality.** Sample actual downstream inputs (NGEC-style event
   pipelines, scraped/OCR news, social media if in scope) and build a small
   "messy set" — not for training, for *looking*. What fraction of real
   mentions are prepositional/modified/coordinated/lowercased? We currently
   have no number.
2. **Decide the output contract for modified locations.** For "outside
   Aleppo", is the right answer Aleppo-the-city (with the modifier's
   semantics lost), Aleppo plus a spatial-relation flag, or abstention?
   Event-data users probably want *toponym + relation tag* — the resolver
   resolves Aleppo, and a light classifier over the preceding tokens tags
   {in, near, outside, north-of, between, en-route}. This is a new,
   separable model surface; it should NOT be jammed into span policy.
3. **Stress evals, cheap to build:** lowercase/ALL-CAPS variants of the dev
   set, headline-register text, coordination-heavy sentences. The span head
   has never seen any of these; spaCy had. Measure before assuming.

### D. Retrieval: the index rebuild (largest single measured lever)
Per the e52 spec: replace the `alt_name_length` sort with a relevance sort
(61% of residual retrieval misses are golds returned past rank 100 —
"Paris, Ontario" is unreachable at any window today), add abbreviations as
alt-names on the 66 ADM1 docs, stored `is_historical`/`superseded_by` so
sorting demotes defunct rows (do NOT delete them — 9 held-out golds are
defunct rows), whitespace-stripped variants (the Maunakea fix). Then
**retrain** — candidate sets shift, so features shift; this is a
train-affecting change, 5 seeds, both denominators (`em_cond` AND `em_all`,
the e52 anti-gaming rule).

### E. Small, sized, cheap
- Adjectival/demonym→place alias table on the R1 query hook (14 residual
  misses: "British" ×6 etc. — these are non-demonym GOLDS found as
  adjectival spans, so it's not a D2 violation).
- Multi-seed span head: the shipped head is one seed at the top of a ±1.4
  e2e spread. Retrain 5, pick honestly or logit-average.
- ES latency engineering (caching/batching): 57–65% of serving latency,
  untouched all campaign.

## Fresh-eyes audit — where campaign 2 may have overfit to this data
For a new reviewer (or a fresh Fable session): each of these is a place
where the campaign optimised what we could measure rather than what users
need. None invalidates the results; all deserve adversarial reading.

1. **Everything was tuned on TR/LGL/GWN.** TLG-hard, the head's training
   spans, the LOCO folds, the calibration thresholds, the error censuses —
   same three corpora throughout. LOCO says out-of-family the head is worth
   −0.6..+2.9 EM, not +10.5; every headline should be read with that
   asterisk until the TEST corpus exists.
2. **D2 removed demonyms from the task.** Right for metric honesty; but
   event-coding users may *want* "Syrian forces"→Syria. Consider demonym
   resolution as a separate opt-in output surface (with the OntoNotes NORP
   data Track A brings) rather than a deleted capability.
3. **The outlet feature assumes the caller knows the outlet.** The table is
   120 LGL-era domains; production needs a directory-scale source, and the
   feature's value on non-local-news (wire copy, international outlets) is
   unmeasured. Also: the leak audit's residual risk was retired by web
   research, but a fresh eye should re-check the *TR* outlet joins, which
   drove the surprise e54 gain.
4. **The demonym-FP metric is partly one corpus's annotation policy** (GWN
   contributes essentially all of it). Don't optimise that column further
   without deciding whether GWN's convention is ours.
5. **The A/P (city-vs-admin-unit) convention is learned, not chosen.**
   Twin-credit hides it in eval; serving users still receive whichever
   convention the training data voted. A documented, configurable
   granularity preference would serve users better than a metric patch.
6. **Calibration artefacts are dev-fit and regime-specific**: T=1.106 and
   the p≥0.5 knee hold only under span_detector="gold"; p_no_match is no
   longer a standalone unanswerability detector (AUROC 0.797). Any
   downstream consumer of these fields needs migration notes, not just a
   changelog line.
7. **`wrong_place` — 64% of residual TLG-hard errors — was never
   decomposed.** The censuses chased the fixable tails. The single biggest
   unexplored bucket in the system.
8. **WikiDocs numbers partly measure memorisation** (86% pair overlap) and
   doc-keyed splits don't fix it (leakage is lexical). Any future claim
   sliced on WikiDocs needs the novel-pair guardrail beside it.
9. **Output schema drift**: overlapping spans are now a real output
   category (24/2,013), score semantics changed at the temperature refit,
   and the R1 rule rewrites queries invisibly. An API-level CHANGELOG for
   downstream users should accompany the flips.
10. **All latency numbers are RTX-4090 numbers** except one CPU row.
    Deployment reality may be CPU-only.

## Suggested opening moves
1. Land the campaign-2 flips (owner decision) + write the downstream
   migration notes (item 9).
2. Start Track B's convention doc immediately — it gates both the TEST
   corpus and Track C's output contract, and it's a thinking task, not a
   compute task.
3. Track D index rebuild in parallel (pure engineering, biggest measured
   win remaining).
4. Track A once OntoNotes is on disk and Track B's first batch exists.
