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

## 7. Round 12: `natlang-rl-task-supply-20261118c` (210 rows)

- **Batches.** Haiku authors w233-w242 worked from `AUTHOR_BRIEF_ROUND12.md` (round-11 lessons). The round-11 leftovers
  w223 and w225 were finished and admitted alongside them; their authors had gone idle after qualify, so the supervisor
  ran their novelty gates.
- **Admitted:** 312 tasks in 12 families. The Sonnet blind solve held back 7 tasks: w237 triage a-c were wrong as
  written (a "case-sensitive" cue rule against capitalised notes), and 4 were ambiguous because a cue word sat inside
  a phrase meaning the opposite (w237 team-cover-b, w240 queue-b and walk-b). w242's first blind solve ran on a stale
  packet (tasks regenerated after the packet was built) and was redone; `round11/ready.sh` now skips a batch whose
  tasks.jsonl is newer than its passing qualify report.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r12`.
- **Source rows.** Train rows of w223, w225 and w233-w242, minus 5 rows custody ties to held sibling worlds (w239 x2,
  w242 bAbI x3) and the 10 rows round 11 already left out. **Result: 210 rows in 12 families.**
- **Constructs:** folder reducers 83%, decide 62%, call depth >= 3 61%, iterateOn 50%, typed Neuralese results 59%;
  205 of 210 rows carry two or more held constructs.
- **Repetition** (against all five earlier POOLS, 35,212 rows): 91 distinct programs (84.9 effective), 35 structures
  (25.1 effective), top structure 9%, top-10 instructions 17%; 0 program texts and 0 functional programs shared with
  the pool.
- **Checks.** held-v3 verify 0 shared ids and 0 shared groups, also with exposure; custody 0 related; closure passed.
- **Next.** w224 (round-11 leftover, requalified) and round 13 (w243-w250, `AUTHOR_BRIEF_ROUND13.md` with the
  round-12 cue-word lessons) go into the next source.

## 8. Round 13: `natlang-rl-task-supply-20261118d` (130 rows)

- **Batches.** Haiku authors w244-w250 worked from `AUTHOR_BRIEF_ROUND13.md` (round-12 cue-word lessons). w224
  (round-11 leftover) was finished by a replacement author after its first author went idle.
- **Admitted:** 191 tasks in 8 families. The Sonnet blind solve held back 3 ambiguous tasks (w246 duplicate
  replacement reviewer, w247 "Outdoor seat" not an outdoor cue, w250 hyphenated whole-word match).
- **Not in this source:** w243 (GSM8K worksheet grader) was still qualifying; it goes into the next source
  (coordinator: build without waiting).
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r13`.
- **Source rows.** Train rows of w224 and w244-w250, minus 10 w249 rows custody binds to held sibling worlds and the
  15 earlier custody rows. **Result: 130 rows in 8 families.**
- **Constructs:** folder reducers 76%, decide 66%, call depth >= 3 60%, iterateOn 45%, typed Neuralese results 31%;
  124 of 130 rows carry two or more held constructs.
- **Repetition** (against all six earlier POOLS, 35,422 rows): 57 distinct programs (53.2 effective), 28 structures
  (19.9 effective), top structure 14%, top-10 instructions 24%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed.
- **Next.** Round 14 (w251-w258, `AUTHOR_BRIEF_ROUND14.md`: smoke fragment keys, per-batch construct planning,
  hyphen word splits, per-world data to avoid custody binding) and w243 go into the next source.

## 9. Round 14: `natlang-rl-task-supply-20261118e` (127 rows)

- **Batches.** Haiku authors w251-w258 (`AUTHOR_BRIEF_ROUND14.md`) and w243 from round 13. The grpo-v4 pin moved to
  `pin-a0981794b756` mid-round (the old qualify runtime was pruned); authors relaunched queued qualifies on it.
- **Admitted:** 211 tasks in 9 families. The blind solve held back 22: MuSiQue (w255) 11 of 20 for several defensible
  exact spans, GSM8K (w243) 8 of 24 for dataset wording with two readings, w254 2 cue/format near-misses. Lesson
  (`AUTHOR_BRIEF_ROUND15.md`): closed-set answers only; free-text span seeds are not worth authoring.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r14`.
- **Source rows.** Train rows of w243 and w251-w258, minus 15 w256 rows custody binds to held sibling worlds (shared
  world data) and the 25 earlier custody rows. **Result: 127 rows in 9 families.**
- **Constructs:** folder reducers 76%, decide 58%, call depth >= 3 63%, iterateOn 57%, typed Neuralese results 53%;
  124 of 127 rows carry two or more held constructs.
- **Repetition** (against all seven earlier POOLS, 35,552 rows): 61 distinct programs (56.1 effective), 25 structures
  (20.1 effective), top structure 12%, top-10 instructions 24%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed.
- **Next.** Round 15 (w259-w266, closed answers only; WikiSQL train seed with SQL-executed answers).
