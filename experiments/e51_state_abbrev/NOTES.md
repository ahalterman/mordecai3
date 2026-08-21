# e51_state_abbrev — US state-abbreviation aliases for `sib_adm1`

Campaign-2 Phase 2 (cheap feature arms). Full sizing report:
`experiments/campaign2/state_abbrev_report.md`.

**Verdict: REJECTED AT THE SIZING GATE — no arm was built, no seed was
trained.** The ceiling on the arm's own mechanism is **+0.0009 TLG-hard**
(+0.0004 on the 5-seed ensemble), against a pre-declared kill threshold of
~0.003 and seed noise of ~0.010. The only source that moves at all is LGL,
whose best case is +0.0022 EM.

## The hypothesis

`sib_adm1` compares a candidate's `admin1_name` against the *other mention
strings* in the document, so "Springfield, Ill." never fires the Illinois
signal — the sibling string is "Ill.", not "Illinois". Alias USPS codes and
AP-style abbreviations to full ADM1 names before the comparison, and the
disambiguation signal should reappear exactly where US local news (LGL) needs
it. Campaign 1 measured P(gold) rising 4.0% → 46.6% when `sib_adm1` fires on
same-name candidates, so each recovered firing looked valuable.

## Sizing (measured, not estimated)

The cue is genuinely present: 137 of 5,739 TR/LGL/GWN mentions (2.4%) are
state/province abbreviations, in 48 distinct strings (`W.Va.` 13, `Ky.` 11,
`B.C.` 10, `D.C.` 8, `Ind.` 7, `Kan.` 7, `Ill.` 6, …). The alias table is
complete: the only unmatched short/dotted mention strings in those corpora are
countries (`U.S.`, `UK`, `UAE`, `EU`), cities (`L.A.`, `Phila.`, `St. Paul`)
and two Australian states.

Recomputing `sib_adm1` on the enriched pickles with the alias-expanded sibling
set, over the held-out halves:

| | TR | LGL | GWN | total |
|---|---|---|---|---|
| entities where ≥1 candidate gains `sib_adm1` | 25 | 69 | 19 | 113 |
| entities where the **gold** gains `sib_adm1` | 25 | 33 | 13 | 71 |
| … currently **wrong** (e29 seed42) | 2 | 3 | 1 | 6 |

In the TLG-hard slice (non-country gold, gold in window, n = 1,171) that is 67
entities touched and **2 currently wrong**. Against them, 40 currently-correct
TLG-hard entities have the new firing land only on a non-gold candidate — 20×
more downside exposure than upside.

Ceilings (every touched-and-wrong entity credited as correct; frozen preds
`experiments/campaign2/preds/e29_{seed42,ens5}_w100.parquet`, window 100, so
the baseline reads 0.8832 / 0.9029 rather than the ledger's window-500 0.8730):

| variant | touched | wrong | Δ seed42 | Δ ens5 |
|---|---|---|---|---|
| **A. gold gains `sib_adm1`** (the arm) | 67 | 2 | **+0.0009** | **+0.0004** |
| B. any candidate gains `sib_adm1` | 107 | 2 | +0.0009 | +0.0004 |
| C. gold's ADM1 abbreviated anywhere in raw doc TEXT | 110 | 4 | +0.0017 | +0.0013 |
| D. fantasy cap: every wrong entity in a doc with an abbrev | 227 | 11 | +0.0073 | +0.0073 |

Variant C required recovering document text from
`raw_data/spacyed/source_{tr,lgl,gwn}.spacy` and aligning it to the pickles by
recomputing `doc_key = sha1(mean token tensor)` (0 of 1,721 unmatched). A
comma-guarded regex (`, IL`) scores *lower* than the loose one, so the guard is
not what is limiting the arm.

## Why it is this small

1. **Redundancy** — 41% of the state names an abbreviation would alias to are
   already present in the same document as a full-name mention ("Indiana" on
   first reference, "Ind." after), so `sib_adm1` already fires.
2. **The `geo` block already has it** — the abbreviation mention is itself a
   toponym and therefore an anchor. On the 71 entities whose gold would gain
   `sib_adm1`, the gold already has `anchor_same_adm1_frac > 0` in 56% of cases
   (mean 0.248) and an anchor within 150 km in 41%. Campaign 1's 4.0% → 46.6%
   figure was measured before the geometry features existed; the `sib` block's
   marginal value is bounded by what `geo` already sees.
3. **Wrong neighbourhood** — 65 of the 127 TLG-hard seed42 errors have US
   golds, but only 11 of 127 are in a document containing an abbreviation
   mention at all.

## Kept: the alias table belongs in retrieval

Of the 44 held-out TR/LGL/GWN entities whose *own* mention is a state
abbreviation, e29 scores 36.4% — and **27 of the 44 have no retrievable gold
at all**, which is **61% of every unretrievable gold in held-out TR/LGL/GWN**.
On the 17 that are answerable e29 already gets 16 right, so this is a retrieval
defect, not a ranking one. Verified against the live index: `Ind.` → *Ind,
India*; `Ky.` → *Ky Ky Pan, South Africa*; `N.M.` → a hotel in Santa Fe; `WA` →
*Wa, Burkina Faso*.

Worth **+0.0004 TLG-hard** (unretrievable golds are outside the denominator)
but real end-to-end: 27 mentions answered wrongly or blanked, each also
injecting a garbage anchor into its document's geometry. Route it to mention
normalisation before the ES query, with the Phase-1 serving fixes / gazetteer
hygiene, and measure it with `tools/end_to_end_eval.py`. The table (51 USPS
codes + 13 Canadian provinces + 62 AP forms, with case guards) is in the
sizing scripts.

## Artifacts

No checkpoints, no jsons, no edit to `mordecai3/candidate_features.py`;
`tests/test_feature_parity.py` untouched and unaffected. Cost ~0 GPU minutes.

Sizing scripts (`sizing.py`, `sizing_text.py`, `sizing_why.py`,
`sizing_selfabbrev.py`) are in the arm's worktree at
`/home/andy/projects/mordecai3/.claude/worktrees/agent-a24c16dbf871e98a0/experiments/e51_state_abbrev/`.
