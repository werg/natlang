# Broad external RL task mix for LFM3 (2026-10-31)

Owner request: the RL cases were too narrow (20,640 cases from one procedural generator, 49 families, 1-2 instruction
texts per family). Build a broad mix from external sources with checkable answers, actively bring in new sources,
and (owner, 2026-10-31) prefer framings where Neuralese communication is natural, marking each case's framing.

Status: **first mixture registered** (stage 3, 2026-10-31 03:00 CET). Builder code: `scripts/rl_tasks/`. Build area:
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

## Sources and qualification (stage 3: all 23 sources verified)

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
| MuSiQue (new) | 1 | 5 | 1579 | 1542 | 1278/143/121 | 32/4737 | 0/3729 | span qa F1 0.8 + aliases | 763 | CC-BY-4.0 |
| BIRD | 1 | 8 | 1959 | 1958 | 1679/119/160 | 1/5682 | 0/3918 | result-set (sqlite service) | 1041 | CC-BY-SA-4.0 |
| QuALITY (new) | 1 | 2 | 1500 | 1500 | 1200/160/140 | 0/4500 | - | exact letter | 819 | CC-BY-4.0 |
| reasoning-gym | 59 | 59 | 4248 | 4248 | 3540/349/359 | 0/12384 | 0/3193 | normalized (scorer-agreement checked) | 0 | Apache-2.0 |
| Spider | 1 | 4 | 1627 | 1623 | 1158/238/227 | 4/4666 | 0/3254 | result-set (sqlite service) | 751 | CC-BY-SA-4.0 |
| SQuAD v1.1 | 1 | 3 | 846 | 796 | 643/66/87 | 50/2530 | 0/986 | qa-alias-map (several questions) | 423 | CC-BY-SA-4.0 |
| TAT-QA (new) | 3 | 9 | 1803 | 1770 | 1472/140/158 | 33/5406 | 0/1803 | numeric-tol (2dp), span qa, answer-list | 0 | MIT |
| TriviaQA rc.wikipedia (new download) | 1 | 27 | 1500 | 1500 | 1200/151/149 | 0/4498 | 0/2761 | span qa + aliases | 795 | Apache-2.0 |
| WikiTableQuestions (new) | 1 | 3 | 1799 | 1799 | 1499/147/153 | 0/5456 | 0/3598 | answer-list | 0 | CC-BY-SA-4.0 |
| xLAM 60k | 1 | 24 | 1800 | 1794 | 1689/46/59 | 0/5547 | 7/1800 | service receipts, answer-list | 0 | CC-BY-4.0 |

MuSiQue: about 92% of its answerable train questions share a single-hop seed question with protected held MuSiQue
worlds (`musique:seed:*`, 31,870 protected MuSiQue groups); the builder skips those (snapshot of the protected groups)
and the assembler's custody check drops any remaining relation, leaving 1,278 usable train items.
BIRD: train/dev databases of at most 40 MB (52 databases), up to 40 questions each, gold results executed at build
time (10 s limit); dev questions keep their difficulty labels. SQL end-to-end check: the gold SQL executed through the
case's own `db` service matched the build-time result on 100/100 sampled Spider/BIRD cases after two service fixes
(rows as arrays so duplicate column names survive; SQLite's double-quoted string literals accepted as in the gold run).

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

## First mixture (registered)

- Train task source: **`natlang-rl-broad-external-20261031-v2`** (`runs/natlang-rl-broad-external-20261031-v2/`,
  manifest `training/corpus-manifests/natlang-rl-broad-external-20261031-v2.json`): **32,947 train cases** from 23
  external sources plus a 1,394-row slice (whole split units, about 30 rows per family, byte-identical rows) of the
  procedural curriculum `natlang-rl-train-curriculum-lfm3-20261030-stop-evidence` for natlang-specific skills (4.2%).
  161 families, 777 family shapes, 317 distinct instruction texts (the old curriculum: 49 families), levels
  0/1/2/3 = 1,958/16,950/10,261/3,778, 6,370 rows with the Neuralese-helper framing. Per-source rows, families,
  shapes, instruction texts, levels, framings and oracle kinds: `diversity.json`. Rows are interleaved source by
  source and family by family. Per-source cap 1,500 train rows (reasoning-gym 3,520 for its 59 families).
  The first assembly (`natlang-rl-broad-external-20261031`, 33,047 rows, registered and removed the same night with
  no consumer) had 45 rows (SQuAD 43, HotpotQA 2) related to LFM3 held SFT sets through `wiki:`/question groups,
  found by the held-set group closure below; v2 excludes them, and also 55 full-table DataBench cases on six tables
  (Twitter/ID-style data) holding integers beyond 2^53: the runtime rejects binding such a table page in eval
  (`let/first/0/0: type-mismatch, expected number | ...` for 1390763244713922561), so no trajectory could read the
  table. That is a runtime limitation to fix (unsafe integers from a service), found by an end-to-end service check.
- `qualification.json` is `natlang.rl-additional-task-source-qualification/1` (task ids, row hashes, protected-world
  bindings, zero intersections), so a lineage can append the packet as an additional admitted train source through
  the existing `natlang.rl-task-source/1` manifest (`training/neuralese/natlang_neuralese/rl/task_source.py`).
- Held packets (evaluation-only, registered with `admission: evaluation-only-never-training`, so they join the
  ProtectedWorlds union): `natlang-rl-broad-external-dev-20261031` (3,527 cases, 23 sources) and
  `natlang-rl-broad-external-confirmation-20261031` (3,558 cases, 22 sources). They are upstream evaluation splits
  (HotpotQA/MuSiQue/2Wiki/SQuAD/DROP/QuALITY/TriviaQA/HybridQA validation, Spider/BIRD dev and Spider test-only DBs,
  GSM8K/MATH/MBPP test, WTQ test, TAT-QA dev+test, FinQA dev+test, LogiQA dev+test, K&K test, bAbI test, CLUTRR test)
  or a 6% unit carve-out (xLAM, DataBench, reasoning-gym).
- Custody: (1) ProtectedWorlds (registry union, now including the two held packets) plus this source's own held rows,
  by ids, groups, world bindings, world and visible fingerprints — dropped before assembly: xLAM 32 (tool sets shared
  with held items), HybridQA 30 (tables shared with held questions), reasoning-gym 20, TAT-QA 12, LogiQA 7, MBPP 4,
  Spider 2; MuSiQue protected seeds skipped at build time; after registration ProtectedWorlds relates 0 rows of this
  source and 0 of the stop-evidence curriculum. (2) The LFM3 held-set group closure of
  `/home/werg/data/lfm3-rl-dgx-20261029/scripts/protection_union.py` (joint-SFT train/held graph sets, big2 held
  tranches, packet holds, acquired-source protection): 0 conflicts for v2 (`held-set-union.json`, 32,947 rows). The original
  script also recomputes per-row world receipts, which takes over an hour on 33k large rows; the group closure was
  run with `/home/werg/data/broad-rl-tasks-20261031/union_groups_fast.py` (the script's own loading code, without
  the receipts, which (1) covers), and agreed with the original script on a 200-row sample (1 conflict each).
- Assets: 52 BIRD + 157 Spider SQLite databases and 20 DataBench tables (613 MB) in `assets/`, read by each case's
  service with a SHA-256 check; set `NATLANG_RL_TASK_ASSETS` when the source directory moves (cloud sandboxes).
  The assets are their own registry entry, `natlang-rl-broad-external-assets-20261031`, at
  `runs/natlang-rl-broad-external-20261031/assets` — the default path compiled into every case service (train and
  held). End-to-end check with that default path: gold SQL through the `db` service and paging through the `table`
  service succeeded on 69/70 sampled train and dev cases (the failure is the unsafe-integer table above).

Using it: the controller runtime must contain ts-host at 542df3c08 or later (the new `result-set`,
`numeric-tolerance`, `answer-list` and `qa-alias-map` normalizations; an older runtime rejects them as unknown
normalizations) and run Node 22.16+/24 (`node:sqlite` `setReturnArrays`). Select a broad window across sources
(rows are already interleaved); the dev packet gives per-source success rates without touching training rows.

## Blocked / next

- Live-model false-negative sample (paraphrased right answers under span F1 0.8, numeric formatting) needs actor
  rollouts; the GPU was busy (decision eval). Measure from the first RL rounds' rejected rollouts per source.
- Sources whose upstream verifier accepts many valid solutions (reasoning-gym countdown, puzzle24, n-queens, sokoban,
  rush hour, word ladder, graph colouring, integrals, ...; Enigmata; Countdown-3to4) need their verifiers ported into
  the shared scoring path (a trusted exact verifier per family); excluded until then.
- Downloaded, not converted yet: MultiHiertt, ConvFinQA (multi-turn), MSR-SQA, WikiSQL, TableInstruct, Qasper,
  NESTFUL (executable nested calls), tau2-bench (needs its environments and user simulator), APIGen-MT and ToolACE,
  KodCode / verifiable-coding (stdin/stdout or pytest tests need a Python or I/O runner in the authoring check),
  NuminaMath/DeepScaleR/ORZ/WebInstruct (decontaminate against MATH-500/AIME first), RLVR-IF constraints (map to the
  existing `constraints` oracle). Big-Math-RL-Verified needs a manual HF gate acceptance.
- Spider/BIRD gold execution has a 10 s limit; queries that time out are skipped (two Spider test-DB items vary
  between builds). BIRD databases over 40 MB are not included (about 17 of 69 train DBs).
- MBPP contributes only 350 train items (its upstream test split is held). More code needs KodCode with a Python
  test runner.

License note: bAbI rows carry `license: BSD-3-Clause` (the code license of the original repo); the downloaded en-10k
data's PROVENANCE.json records CC-BY-3.0 (Keras mirror of the official tarball). Both are recorded as facts; no row
is held for licensing. CLUTRR's card states no license (rows record CC-BY-NC-4.0 of the original release).

## v3 (in progress, 2026-10-31): runtime fix, new sources, multi-solution verifiers

Coordinator order: (1) unsafe-integer runtime bug, (2) TabFact, tool-use sets, MultiHiertt/ConvFinQA/WikiSQL/MSR-SQA,
(3) multi-answer puzzle verifiers; publish as a new source `natlang-rl-broad-external-20261031-v3` (v2 unchanged
while lineages read it).

- Runtime (66a9b1374): `coerce` accepted only safe integers for `number`, while `inferValueType` and `schemaOf` call
  every finite JS number `number`; binding DataBench rows with tweet ids (`databench-054_Joe`, `table.rows(0,500)`)
  failed with `let/first/0/0: type-mismatch`. JSON.parse has already chosen the double, so the typed boundary now keeps
  it (JS semantics; no skill text described the old rule). Re-verified: all 113 previously failing DataBench cases pass
  (113/113 reference, 0/226 wrong accepted); v3 re-verifies all DataBench cases under the fixed runtime.
- Oracle `puzzle-solution` (3ae1bf1fa, `ts-host/src/evaluation/puzzle-solutions.ts`): `expected` is the instance;
  countdown/24 expressions evaluated with exact rationals over the given numbers (each at most once, or all with
  `use_all`), n-queens placements including the fixed queens, sokoban move strings simulated (XSB grid). RL runtimes
  must be built from 3ae1bf1fa or later for these rows.
- New sources (builders c6c18fa5c): TabFact, WikiSQL (gold SQL executed over its table), MSR-SQA (WTQ table units,
  so an SQA item shares its WTQ table's split), ConvFinQA (FinQA page units/splits; history of earlier answers),
  MultiHiertt (pages sharing a long sentence with FinQA/TAT-QA dropped: 30), ToolACE (xLAM receipt framing; type-name
  mapping fixed after a 16% alias false-negative rate), NESTFUL (port of its executable arithmetic functions; 56 items
  whose gold_answer disagrees with execution dropped), tau-bench retail and airline (tools ported to a `store` service
  over a fresh database per run; answer `store.finish()` = final-state digest, tau-bench's own reward; held tau users
  never in training; tasks with required spoken outputs skipped), APIGen-MT-5k retail/airline trajectories (customer
  messages as the request; trajectories touching held tau users dropped), Countdown-3to4 and reasoning-gym puzzle24 /
  n-queens / sokoban / countdown.
- tau-bench store databases are v3 assets (`assets/stores/`), read from this source's `assets/` by default; override
  with `NATLANG_RL_TASK_ASSETS_V3`. v2 rows keep reading the v2 asset entry (`NATLANG_RL_TASK_ASSETS`).
- tau2-bench: its retail/airline tasks are revised tau-bench v1 test tasks, which stay held here; telecom needs its
  user-side tools and simulator and is not converted.
