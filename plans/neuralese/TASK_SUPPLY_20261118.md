# GRPO task supply, 2026-11-18: round 10, authored families on the held-v3 constructs (task-supply agent)

Question: which tasks give grpo-v4's construct-weighted exploration (ae1f7dacb) the constructs held-v3 tests? Held-v3 is
about 30% folder reducers, 30% Neuralese results, 18% iterateOn, 18% call depth >= 3 and 11% decide readouts. The pool has
8% folder reducers, 2.5% decide, 12% depth 3 and 13% iterate. The 2026-11-17 supply had no depth-3 programs and under 2%
iterate or Neuralese. Method: the round-9 pipeline (TASK_SUPPLY_20261117.md) with a new brief. Each program combines at
least two held constructs. The difficulty target is the round-9 one (~30% level 1). Every task is qualified on grpo-v4's
pin runtime (`pin-21dbc20c5524/ts-host`), novelty-gated, blind-solved by an independent Sonnet solver, then admitted.
Brief: `/mnt/external/natlang-rl-authoring/AUTHOR_BRIEF_ROUND10.md`, with its addendum on `decide` in references. Work
files: `/mnt/external/natlang-rl-authoring/round10/` (assignments, AUTHOR_TASK, BLIND_SOLVER, construct_count.py,
build_supply.py, final_checks.sh).

## 1. Families (Haiku 5.5 authors w205-w218, one family each)

| worker | family | admitted (train/dev/conf) | held back after blind solve |
|---|---|---|---|
| w205 | bug_report_triage_inbox | 17 (12/3/2) | 7 flagged ambiguous (two quoted error codes match two cards; "freeze" vs cue "freezes"; lower-case matching vs capitalised steps) |
| w206 | grant_proposal_screening | 25 (21/3/1) | none |
| w207 | shipment_exception_tracker | 24 (17/4/3) | none |
| w208 | systematic_review_screening | 21 (15/2/4) | 3 merge-exports (subtitle identity rule direction unstated) |
| w209 | qasper_section_dossiers (Qasper train) | 0 | whole batch rejected: blind solve 3/23 (free-text span boundaries unstated; two author answers wrong) |
| w210 | agent_run_log_audit | 28 (22/3/3), after the build | none; its train rows enter the next source (held worlds registered as `...-20261118r10b`) |
| w211 | lab_sample_custody_desk | 21 (17/2/2) | 3 fold_corrections (count wording reads two ways) |
| w212 | email_thread_filing | 24 (17/3/4) | none (one solver slip) |
| w213 | course_prerequisite_planner | 27 (17/7/3) | crs-sweep-c (a note names both the given and another term) |
| w214 | essay_rubric_grading | 20 (11/3/6) | 3 argument-essays (a period inside quotation marks decides the last sentence) |
| w215 | turn_based_game_referee | 24 (15/5/4) | none |
| w216 | expense_report_audit | 24 (14/5/5) | none |
| w217 | incident_runbook_executor | 25 (15/5/5) | none |
| w218 | version_bisect_detective | 24 (16/5/3) | vbd-p5-b (author wrong: failing component absent from the index) |

276 tasks were admitted to `natlang-rl-authoring/admitted`. The 89 held rows (47 dev, 42 confirmation) are registered as
`natlang-rl-authored-{dev,confirmation}-20261118r10`. The source holds the 187 train rows minus 2 that custody ties to a
held sibling world (w214 late-penalty a/b): **185 rows in 12 families, all new to the pool.**

## 2. Constructs and repetition (`runs/natlang-rl-task-supply-20261118/{constructs,repetition}.json`)

The shares are as the GRPO explore walk detects them (`measure_repetition.program_features`):

| construct | share | held-v3 | pool before |
|---|---:|---:|---:|
| folder_reducer (all with effect-state oracles) | 78% | 30% | 8% |
| decide call | 69% | 11% | 2.5% |
| call depth >= 3 | 66% | 18% | 12% |
| iterate | 56% | 18% | 13% |
| Neuralese result (typed 30%) | 32% | 30% | 52% |

- 173 of 185 rows carry two or more held constructs.
- Repetition: 87 distinct programs (81.5 effective) and 26 structures (top structure 22%), measured against all three
  earlier POOLS.

## 3. Checks

- **held_eval verify vs held-v3 (sha 45471060):** 0 shared ids and 0 shared split/source groups, also with the recorded
  training exposure (`held-verify/`).
- **ProtectedWorlds (registry with the r10 held packets):** 0 related after `--drop-related`.
- **LFM3 held-set group closure:** passed.
- **Exposure collector:** `ops/held-v3/exposure/collect_exposure.py` lists the new pool (the previous version is kept as
  `.pre-supply-20261118`).

## 4. Harness defect found and fixed (7f199b2fb)

Since 61c6c3614, a decision readout without a scoring driver generates its choice as JSON text on a tool-free request.
Three scripted drivers still answered it with `return_result`, which the runtime rejects as `decision-generation-invalid`:
the qualifier's structure smoke, the reference replay (`curriculum.ts referenceDriver`) and the authoring oracle
(`authoring.ts`). So every authored task whose smoke or reference called `decide` failed qualification on the grpo-v4
pins (c57e7be28 on). Round-9 tasks that had passed on the grpo-v3 pin failed the same smoke.

- **The fix.** `native/decision.ts scriptedSuccess` answers tool-free requests with the value's JSON text. Both ts-host
  drivers use it, and the smoke has its own copy so that it works against older pins.
- **Test.** `ts-host/test/scripted-decision-reply.test.mjs` replays an authored folder task whose reference calls `decide`.
- **Still open on the pinned runtime.** The reference replay runs from the pin's dist, which predates the fix. Round-10
  references therefore inline the decided values (brief addendum), and the smoke still exercises `decide`.
- **Inventory.** The other scripted drivers in `ts-host/scripts/inline-curriculum` call local JS functions named
  `decide`, not readouts.
- **Smoke reporting.** The smoke now names a missing child answer, which otherwise surfaced on a readout as
  decision-generation-invalid.

## 5. Not checked

- **Base pass rates.** No engine was free, because grpo-v4 holds the DGX. The explore walk measures them.
- **Neuralese typing.** Some authors turned typed Neuralese into strings, or dropped Neuralese altogether. They did this
  when the pin's reference replay refused plain-JSON Neuralese children ("needs a configured Neuralese write port").
  w206, w215 and w216 have no Neuralese.
- **Blind solves.** One Sonnet solver ran per batch. Tasks it flagged were held back even when they agreed. Doubts that
  an author raised but the solver did not were admitted.
- **Construct tags.** w211 P8 claims typed_neuralese on a `Neuralese<string>` result; the sampler reads programs, not
  tags.

## 6. Round 11: `natlang-rl-task-supply-20261118b` (205 rows)

- **Batches.** Haiku authors w219-w232 worked from `AUTHOR_BRIEF_ROUND11.md`, which raised the targets for typed
  Neuralese and depth 3 and folded in the round-10 lessons.
- **Admitted to `natlang-rl-authoring/admitted`:** 278 tasks in 11 families from w219-w222 and w226-w232. Three tasks
  were held back as ambiguous (w229 forfeit x2, w232 one gender-dependent relation).
- **Not in this source:** w223-w225 were still qualifying when it was built and go into the next source.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r11`.
- **Source rows.** The source holds the train rows of those batches plus w210's 22 train rows. It drops 10 rows that
  custody ties to held sibling worlds: 7 CLUTRR rows whose stories recur in held worlds, the 2 w214 late-penalty rows,
  and 1 w228 row. **Result: 205 rows in 12 families.**
- **Constructs** (explore-walk detection):
  - folder reducers 70%
  - decide 59%
  - call depth >= 3 63%
  - iterateOn 48%
  - typed Neuralese results 52%
  - 195 of 205 rows carry two or more held constructs
- **Repetition** (`repetition.json`, against all four earlier POOLS, 35,007 rows):
  - 90 distinct programs and instruction texts (82.8 effective);
  - 32 structures (23.7 effective), with the top structure at 12%;
  - 0 program or instruction texts and 0 functional programs shared with the pool.
- **Checks.** held-v3 verify found 0 shared ids and 0 shared groups, also with exposure. Custody found 0 related rows.
  The closure passed.
- **Memory.** Authoring units run at most 3 at a time beside training, through `round11/gated.py`.
