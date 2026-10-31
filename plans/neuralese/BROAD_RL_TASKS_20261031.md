# Broad external RL task mix for LFM3 (2026-10-31)

Owner request: the RL cases were too narrow (20,640 cases from one procedural generator, 49 families, 1-2 instruction
texts per family). Build a broad mix from external sources with checkable answers, actively bring in new sources,
and (owner, 2026-10-31) prefer framings where Neuralese communication is natural, marking each case's framing.

Status: **in progress** (this file is updated in stages). Builder code: `scripts/rl_tasks/`. Build area:
`/home/werg/data/broad-rl-tasks-20261031/` (per-source `build/<source>/raw-cases.jsonl`, `verified/`, `assets/`).
Downloads: `/home/werg/data/external-rl-sources/<name>/` (PROVENANCE.json each; survey in `SURVEY.md` there).
Related: `LFM3_DATA_COVERAGE_AUDIT_20261031.md` (SFT coverage of the same upstream datasets; its D8/D11 rows list
BIRD, Spider, reasoning-gym and K&K as never converted for SFT — they are converted here as RL tasks, which need only
an outcome oracle).

## Case design (natlang-native)

Every item is an ordinary `natlang.program/2` curriculum case (same format as the procedural curriculum, so the
controller, `verifyCases` and custody take it unchanged):

- a typed root `.nl` function whose documents, tables, databases and services are its inputs, with a declared answer
  type (`string`, `number`, `string[]`, `Record<string, string>`, `"A" | "B" | "C" | "D"`, rows `(string | number |
  null)[][]`, ...);
- `semantics.oracle` checked by the collector's shared `checkOracle` (no per-source scoring code);
- `curriculum.reference` replayed by the collector; `curriculum.external` (never model-visible) keeps source,
  upstream id/split, split unit, shape, instruction id, **framing**, Neuralese-room note and the qualification checks;
- `split` by split unit (document / database / table / article / item), so related items stay on one side.
  Upstream evaluation splits become our held `dev` / `confirmation` halves; sources without one carve 6% of units.

Framings (`curriculum.external.framing`): `plain`, or `neuralese-helper` — the root's callable folder gets a helper
whose declared result is `Neuralese<string>` (per-document reading notes, per-question evidence, query plans,
assignment consistency notes, per-part notes of a long article) and the root instruction says how to fan those notes
in to its decision. The framing is chosen per split unit (about half), so reward/selection can be compared by framing.
Rewards stay outcome-only (correct with or without Neuralese).

## Shared oracle additions (ts-host/src/evaluation/answer-sets.ts, wired into checkOracle)

`normalized` with `normalization`:
- `result-set`: SQL execution match (multiset of rows; numbers to 9 significant digits; `ordered` optional);
- `numeric-tolerance`: relative/absolute tolerance, optional percent equivalence (FinQA);
- `answer-list`: unordered short-answer lists (WikiTableQuestions, DataBench lists, tool-call receipts);
- `qa-alias-map`: a record of free-text answers, each key token-F1 >= threshold against one of its aliases (SQuAD).

Existing kinds reused: `span` with `qa` normalization and alternates (multi-hop QA, DROP spans), `normalized` (numbers,
case/space), `exact` (MCQ letters, K&K assignments), executable `authoring` (code), service-certified receipts.
Tests: `ts-host/test/answer-sets-oracle.test.mjs`.

## Qualification per source (scripts/rl_tasks/verify.mjs)

Per case, through the collector's own execution path: (1) `verifyCases` (validation, reference replayed and admitted);
(2) negative controls — the reference's answer replaced by meaningful wrong answers (other items' gold of the same
source, flipped roles, dropped rows, wrong tool calls, a stub or a one-value-wrong implementation, and a generic
perturbation) must all be rejected; (3) false-negative probe — type-valid alternative gold forms (upstream annotator
answers/aliases, surface variants, reordered rows, explicit defaults) replayed and their acceptance recorded.
Kept = (1) passes and every negative control is rejected. Alias acceptance is measured, never a keep condition.
Lesson: alias probes must respect the declared return type (the runtime rejects a mistyped return before the
oracle); the first K&K ("Knight") and Spider (flat list) probes were type-invalid and were replaced.

## Sources and qualification (stage 2: all 22 converted sources verified; BIRD building)

Columns: families / shapes (distinct `curriculum.shape`), cases built, kept after verification, kept split
train/dev/confirmation, negative controls accepted / tried (each accepted one drops its case), alias probes rejected /
tried (false-negative probe), oracle kinds, rows with the Neuralese-helper framing, license (recorded, never a hold).

| source | fam | shapes | built | kept | train/dev/conf | wrong acc. | alias FN | oracle | Neuralese rows | license |
|---|---|---|---|---|---|---|---|---|---|---|
| 2WikiMultihopQA (new) | 1 | 4 | 1800 | 1796 | 1498/151/147 | 4/4857 | 0/3600 | span qa F1 0.8 | 915 | Apache-2.0 |
| bAbI en-10k (new) | 20 | 20 | 1720 | 1720 | 1400/167/153 | 0/4816 | 0/1720 | normalized | 0 | BSD-3-Clause |
| CLUTRR (new) | 1 | 9 | 1300 | 1300 | 1000/155/145 | 0/3900 | 0/1300 | normalized | 0 | CC-BY-NC-4.0 |
| DataBench (new) | 5 | 10 | 3054 | 3024 | 2947/77/0 | 24/6261 | - | exact, normalized, answer-list, numeric-tol | 0 | MIT |
| DROP (new) | 1 | 3 | 1800 | 1796 | 1500/145/151 | 4/5400 | 0/1882 | normalized number, span qa | 0 | CC-BY-SA-4.0 |
| FinQA (new) | 1 | 5 | 1800 | 1719 | 1433/142/144 | 81/5400 | 0/1800 | numeric-tol 5e-3 (+% equiv), exact for integers | 0 | MIT |
| GSM8K (new) | 1 | 7 | 1800 | 1800 | 1500/141/159 | 0/5400 | 0/1800 | normalized number | 0 | MIT |
| HotpotQA distractor | 1 | 11 | 1800 | 1782 | 1486/159/137 | 18/5293 | 0/3600 | span qa F1 0.8 | 888 | CC-BY-SA-4.0 |
| HybridQA (new) | 1 | 3 | 1800 | 1798 | 1499/152/147 | 1/5400 | 1/1799 | span qa F1 0.8 | 888 | CC-BY-4.0 |
| Knights & Knaves | 1 | 7 | 1500 | 1500 | 1200/139/161 | 0/4399 | 0/1500 | normalized record | 770 | CC-BY-NC-SA-4.0 |
| LogiQA 2.0 (new) | 1 | 6 | 1500 | 1500 | 1200/149/151 | 0/4500 | - | exact letter | 0 | CC-BY-NC-SA-4.0 |
| MATH (numeric answers) (new) | 7 | 5 | 1500 | 1500 | 1200/122/178 | 0/4500 | 0/1500 | normalized number | 0 | MIT |
| MBPP (new) | 1 | 1 | 895 | 895 | 354/269/272 | 0/1790 | - | executable authoring (tests) | 0 | CC-BY-4.0 |
| MuSiQue (new) | 1 | 5 | 1800 | 1751 | 1461/150/140 | 44/5400 | 0/4263 | span qa F1 0.8 + aliases | 847 | CC-BY-4.0 |
| QuALITY (new) | 1 | 2 | 1500 | 1500 | 1200/160/140 | 0/4500 | - | exact letter | 819 | CC-BY-4.0 |
| reasoning-gym | 59 | 59 | 4248 | 4248 | 3540/349/359 | 0/12384 | 0/3193 | normalized (scorer-agreement checked) | 0 | Apache-2.0 |
| Spider | 1 | 4 | 1627 | 1623 | 1158/238/227 | 4/4666 | 0/3254 | result-set (sqlite service) | 751 | CC-BY-SA-4.0 |
| SQuAD v1.1 | 1 | 3 | 846 | 796 | 643/66/87 | 50/2530 | 0/986 | qa-alias-map (several questions) | 423 | CC-BY-SA-4.0 |
| TAT-QA (new) | 3 | 9 | 1803 | 1770 | 1472/140/158 | 33/5406 | 0/1803 | numeric-tol (2dp), span qa, answer-list | 0 | MIT |
| TriviaQA rc.wikipedia (new download) | 1 | 27 | 1500 | 1500 | 1200/151/149 | 0/4498 | 0/2761 | span qa + aliases | 795 | Apache-2.0 |
| WikiTableQuestions (new) | 1 | 3 | 1799 | 1799 | 1499/147/153 | 0/5456 | 0/3598 | answer-list | 0 | CC-BY-SA-4.0 |
| xLAM 60k | 1 | 24 | 1800 | 1794 | 1689/46/59 | 0/5547 | 7/1800 | service receipts, answer-list | 0 | CC-BY-4.0 |

Reference execution: every source 100% except HybridQA (one case with a page title `$9.99`, a reserved record key),
MuSiQue (5) and Spider/xLAM (4 each); those cases are dropped.

Findings and fixes made while qualifying (each found by the negative controls or probes, then fixed in the shared
builder or oracle and re-verified):
- Long free-text golds: token F1 0.8 accepts "gold + four extra words" for golds of 15+ tokens. The negative control
  drops those cases (HotpotQA 18, MuSiQue 44, SQuAD 50); TAT-QA/HybridQA/TriviaQA keep golds of at most 8 words.
- Relative numeric tolerance on large integers: FinQA 5e-3 accepted 1,660,007 for 1,660,000 (261 cases). Integer
  golds now use exact comparison (81 left, all non-integer rounding cases, dropped); TAT-QA uses 1e-6 relative plus
  0.006 absolute (its 2-decimal rounding); DataBench uses exact for integers and 1e-6 relative otherwise.
- xLAM defaults: an explicit default written as a numeric string ("10" for an int with default "10") changed the
  receipt (4.3% of alias probes rejected). The service now compares defaults after the same coercion: 0.4% left
  (genuinely different optional arguments).
- Type-invalid probes are not oracle evidence: the runtime rejects a mistyped return before the oracle (first K&K
  "Knight" and Spider flat-list probes); probes are now type-valid.
- Spider gold execution uses a 10 s limit; two test-DB queries time out nondeterministically, so rebuilt case sets can
  differ by those items (the verified build is the one used).

False-negative evidence is strongest where upstream gives independent gold forms: SQuAD second annotator answers
(0/986 rejected), MuSiQue answer aliases (0/4263 incl. surface variants), TriviaQA aliases (0/2761), DROP validated
answers (0/1882), Spider reordered/stringified rows (0/3254), xLAM explicit defaults (7/1800). For the rest the probes
are surface variants (case, articles, punctuation, number types); a live-model false-negative sample remains to be
measured when the GPU is free (paraphrased answers of right meaning are the expected residual for span F1 0.8).

New external sources brought in by this work (downloaded with PROVENANCE.json under
`/home/werg/data/external-rl-sources/`, survey in its `SURVEY.md`): MuSiQue, 2WikiMultihopQA, SQuAD, DROP, TriviaQA
rc.wikipedia, QuALITY, TAT-QA, FinQA, WikiTableQuestions, DataBench, HybridQA, GSM8K, MATH, MBPP, bAbI, CLUTRR,
LogiQA 2.0, plus downloaded but not yet converted: MultiHiertt, ConvFinQA, MSR-SQA, WikiSQL, TableInstruct,
Qasper, StrategyQA, NESTFUL, tau2-bench, NuminaMath-1.5, DeepScaleR, ORZ-57k, WebInstruct-verified, RLVR-IF
constraints, Enigmata, Countdown, verifiable-coding (APPS/TACO-derived). Spider, BIRD, xLAM, K&K, HotpotQA and
reasoning-gym came from the existing raw archive (`/mnt/external/sdkb-archive/raw/`).

## Neuralese framing (owner, 2026-10-31)

Families with a `neuralese-helper` framing (about half of each family's split units): multidoc_qa (HotpotQA,
MuSiQue, 2Wiki: per-document reading notes), table_and_pages_qa (HybridQA: per-page notes after a table lookup),
long_page_qa (TriviaQA: per-section notes of long pages), long_article_mcq (QuALITY: per-part evidence notes),
article_multi_qa (SQuAD: per-question located evidence), sql_qa (Spider/BIRD: private query plan), roles (K&K:
consistency notes per candidate assignment). The other families are plain (single-pass computation: arithmetic,
tables, code, tool calls, short logic). `curriculum.external.framing` records it per row.
