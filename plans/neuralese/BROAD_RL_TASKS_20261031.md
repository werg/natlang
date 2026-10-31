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

## Sources (stage 1 results; counts are cases built / kept after verification)

| source | family / framing | input shape | oracle | built | kept | wrong accepted | alias FN |
|---|---|---|---|---|---|---|---|
| HotpotQA distractor | multidoc_qa, plain+Neuralese | 10 paragraphs | span F1>=0.8 | 1800 | 1782 | 18 (long golds; dropped) | 0/3600 |
| Knights & Knaves | roles, plain+Neuralese | statements | exact record | 1500 | 1500 | 0 | (probe being rerun) |
| Spider | sql_qa, plain+Neuralese | `db` service (sqlite asset) | result-set | 1627 | 1623 | 4 | (probe being rerun) |
| xLAM 60k | tool_calls | `api` service | receipts answer-list | 1800 | running | | 3% on 30-case pilot |
| others | see stage 2 | | | | | | |

(Stage 2 fills in MuSiQue, 2Wiki, SQuAD, DROP, QuALITY, bAbI, CLUTRR, LogiQA2, GSM8K, MATH, FinQA, MBPP,
WikiTableQuestions, DataBench, reasoning-gym, BIRD.)
