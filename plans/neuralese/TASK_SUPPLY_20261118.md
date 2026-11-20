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

## 10. Round 15: `natlang-rl-task-supply-20261118f` (121 rows)

- **Batches.** Haiku authors w259-w266 (`AUTHOR_BRIEF_ROUND15.md`: closed-set answers only, after free-span dataset
  seeds lost most tasks in round 14). WikiSQL train tables seed w259, with answers computed by executing each item's SQL.
- **Admitted:** all 196 tasks in 8 families; the Sonnet blind solve agreed on every task and flagged none, the first
  clean round.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r15`.
- **Source rows.** Train rows of w259-w266 minus 16 rows (w260 x8, w265 x8) whose root `inputs` equal a held sibling
  world's (custody `inputs:` binding) and the 40 earlier custody rows. **Result: 121 rows in 8 families.**
  Round 16's brief now requires world-specific root inputs (`common/custody.py world_bindings`).
- **Constructs:** folder reducers 84%, decide 69%, call depth >= 3 62%, iterateOn 62%, typed Neuralese results 49%;
  118 of 121 rows carry two or more held constructs.
- **Repetition** (against all eight earlier POOLS, 35,679 rows): 56 distinct programs (52.3 effective), 26 structures
  (19.2 effective), top structure 12%, top-10 instructions 26%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed.

## 11. Round 16: `natlang-rl-task-supply-20261118g` (88 rows)

- **Batches.** Haiku authors w269-w274 (`AUTHOR_BRIEF_ROUND16.md`, with the custody addendum requiring world-specific
  root inputs). w267 (bAbI) and w268 were still generating or qualifying and go into the next source.
- **Admitted:** 135 tasks in 6 families. Held back 6: w270 x2 (a "not a complaint" note containing the cue word;
  one medical note naming two members), w271/w273 x1 each (ambiguous scope of a helper sentence), w274 irv-c (about
  36 model calls, near the 40-call episode budget).
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261118r16`.
- **Source rows.** Train rows of w269-w274. Custody found 0 related rows on the first build: the world-specific root
  inputs removed the `inputs:` binding that cost 15-16 rows in each of rounds 14 and 15. **Result: 88 rows in 6
  families.**
- **Constructs:** folder reducers 88%, decide 51%, call depth >= 3 65%, iterateOn 42%, typed Neuralese results 41%;
  86 of 88 rows carry two or more held constructs.
- **Repetition** (against all nine earlier POOLS, 35,800 rows): 44 distinct programs (41.0 effective), 24 structures
  (20.3 effective), top structure 10%, top-10 instructions 34%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed.

## 12. Round 17: `natlang-rl-task-supply-20261119` (103 rows)

- **Batches.** Haiku authors w275-w282 (`AUTHOR_BRIEF_ROUND17.md`) and the round-16 leftover w267 (bAbI train). The
  round-17 supervisor admitted 7 batches on 2026-11-18 and then went idle without building the source; the task-supply
  agent of 2026-11-19 picked it up (orphaned work, ~22 h).
- **Admitted:** w267 16, w275 21, w276 24, w278 24, w279 25, w281 24, w282 28 (Sonnet blind solve: all agree, none
  ambiguous). w277 (theater seats) passed novelty and its blind solve (26/26) on 2026-11-19 and goes into round 18.
  w268 (one smoke type mismatch) and w280 (three wrong answers equal to the reference) failed qualification and are
  being repaired for round 18.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261119r17` (26 + 33 rows).
- **Source rows.** Train rows of w267-w282 not already in POOLS; custody 0 related on the first build.
  **Result: 103 rows in 7 families** (museum exhibit rotation 18, procurement three-way match 17, water quality
  compliance 16, energy net metering 15, crop rotation 15, hospital bed allocation 14, bAbI counting/lists 8).
- **Constructs:** folder reducers 85%, decide 84%, call depth >= 3 73%, iterateOn 50%, typed Neuralese results 31%;
  101 of 103 rows carry two or more held constructs. Levels 31 / 44 / 28 (L1 / L2 / L3).
- **Repetition** (against all ten earlier POOLS, 35,888 rows): 52 distinct programs (48.5 effective), 25 structures
  (16.3 effective), top structure 21%, top-10 instructions 29%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed. The final checks now take
  the repetition reference set from `grpo.POOLS` (`round18/final_checks.sh`), so a new source is never left out.
- **Pass-band estimate.** Not measured on an engine (the DGX trains). grpo-v4-run2's 221 authored groups so far:
  binary pass 0.41, 75% of groups in the 0.1-0.9 band, 16% all-fail, 9% all-pass, and 15% of episodes ending on the
  40-call budget. These families follow the same design, so a similar share in band is expected.

## 13. Round 18: `natlang-rl-task-supply-20261119b` (130 rows)

- **Batches.** Haiku authors w283-w290 (`AUTHOR_BRIEF_ROUND18.md`: at most 24 model calls per world after run2 showed
  15% of authored episodes ending on the 40-call budget, a slightly easier level mix, round-17 lessons), plus the
  leftovers w277 (theater seats), w268 (dental recall, repaired smoke children) and w280 (court docket, repaired
  wrong answers that equalled the reference). w288 is seeded from TAT-QA **train** (arithmetic items recomputed in the
  generator, group `tatqa:<table uid>`). w290 (DNS zone review) was admitted after the build and goes into round 19.
- **Blind solve.** Sonnet agreed on every task of every batch, and 32 flagged tasks were held back: w280 12 ("the 5
  characters after the date" includes the separating space), w287 5 (r2/r10 file order, a cue inside a reason that
  argues against it, register.md rewrite unstated), w286 3 (an argument's source unstated), w284 3 (output contract vs
  step on a count), w288 1 (rounding order unstated), plus w277 `seat-hold-reply-03` (prefix match H-1/H-10, from the
  earlier supervisor's blind solve).
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261119r18` (38 + 38 rows).
- **Custody left out 32 rows**: all train rows of w283 (ferry deck loading) and w284 (blood bank matching). Their
  programs share one root-inputs object per world letter (`{"sailing": "N-0615-a"}` for every program's world a), so
  each train row is bound to a held row of another program. The round-16 rule said distinct inputs per sibling world;
  round 20's brief makes it distinct inputs per task across the whole batch.
- **Result: 130 rows in 8 families** (theater seats 21, package upgrades 21, kitchen tickets 19, dental recall 19,
  translation dispatch 16, TAT-QA calculations 14, archive accessions 13, court docket 7).
- **Constructs:** folder reducers 92%, decide 62%, call depth >= 3 62%, iterateOn 46%, typed Neuralese results 26%,
  skills 7%; 126 of 130 rows carry two or more held constructs. Levels 41 / 56 / 33.
- **Repetition** (against all 11 earlier POOLS, 35,991 rows): 57 distinct programs (53.5 effective), 23 structures
  (18.4 effective), top structure 11%, top-10 instructions 25%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related after the drop; closure passed.
- **Pass-band estimate.** Not measured on an engine. The same design gives 75% of run2's authored groups in band; the
  24-call cap should cut the 15% budget-exhaustion share, which moves pass rates up toward the band's middle.

## 14. Round 19: `natlang-rl-task-supply-20261119c` (105 rows)

- **Batches.** Haiku authors w291-w299 (`AUTHOR_BRIEF_ROUND19.md`: argument sources named, zero-padded file names,
  counts defined once, rounding stated, rewritten files named) and w290 from round 18. w295 (MSR SQA train seed)
  produced no batch: the author found only 11 sequences whose turns map to one explicit column/value rule that
  reproduces the gold, too few for 8 programs; it was replaced by w299 (bike repair queue, 7 programs after its
  author could not find why one program's reference replay failed).
- **Blind solve.** Sonnet agreed on every task; 27 flagged tasks were held back: w292 9 and w294 3 (unstated final
  newline of written files), w290 8 (final newline; "approved by ticket" and "Frozen" cue near-misses), w294 4 more
  (waitlist rewrite unstated, stale swap repliers), w296 2 (duty length before or after a delay), w299 1 (a repeated
  job counted once or per listing).
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261119r19` (18 + 22 rows).
- **Custody left out 19 rows** (w291 8, w294 6, w299 4, w298 1) whose root inputs equal a held sibling's. The round-20
  brief requires distinct inputs for every task of a batch, asserted in the generator.
- **Result: 105 rows in 9 families** (dunning letters 17, ski school 13, crew duty limits 13, seed bank 12, rail yard
  12, bike repair 11, DNS zone review 10, garden plots 9, orchestra library 8).
- **Constructs:** folder reducers 79%, decide 57%, call depth >= 3 54%, iterateOn 42%, typed Neuralese results 34%,
  Neuralese fan-in 29%; 95 of 105 rows carry two or more held constructs. Levels 26 / 50 / 29.
- **Repetition** (against all 12 earlier POOLS, 36,121 rows): 61 distinct programs (56.7 effective), 26 structures
  (20.3 effective), top structure 14%, top-10 instructions 26%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related after the drop; closure passed.
- **Note.** w296's author added inert `Remark:` lines and unused inputs so sibling worlds pass the near-duplicate
  gate; they are distractors the programs ignore. Round 20 forbids this padding.

## 15. Round 20: `natlang-rl-task-supply-20261119d` (114 rows)

- **Batches.** Haiku authors w300-w307 (`AUTHOR_BRIEF_ROUND20.md`: distinct root inputs for every task of a batch,
  final newline of every written file stated, rewritten files named, no character offsets, repeated list items
  defined, no inert padding). w304 is seeded from FinQA **train**: each item's `qa.program` re-executed in the
  generator, steps rewritten as numbered natural-language steps, group `finqa:<id>`.
- **Blind solve.** Sonnet agreed on every task; 27 flagged tasks were held back: w304 8 ("earlier figure" while the
  question names the later year first), w302 6 (one name for two lists; a step's scope under a branch), w303 5 (cue
  case-sensitivity unstated; a header line's newline), w306 3 and w301 2 (a file's final newline, zero-padded day).
  The newline rule cut the newline flags from 20 in round 19 to 6.
- **w301** later regenerated its batch to 29 tasks (`qualified-2`) after its first set was admitted; the admitted
  rows are the blind-solved `qualified-1` set, and `qualified-2` was not admitted.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261119r20` (31 + 25 rows).
- **Custody: 0 related on the first build.** Distinct inputs per task removed the binding that cost 32 and 19 rows in
  rounds 18 and 19.
- **Result: 114 rows in 8 families** (kennel boarding 19, snow plow routes 18, marathon results 18, lost and found
  15, exam seating 12, brewery log 12, tool library 11, FinQA program desk 9).
- **Constructs:** folder reducers 83%, decide 54%, call depth >= 3 48%, iterateOn 53%, typed Neuralese results 40%,
  Neuralese fan-in 36%; 107 of 114 rows carry two or more held constructs. Levels 32 / 55 / 27 (L0-1 / L2 / L3).
- **Repetition** (against all 13 earlier POOLS, 36,226 rows): 52 distinct programs (48.9 effective), 24 structures
  (17.1 effective), top structure 14%, top-10 instructions 26%; 0 program texts and 0 functional programs shared.
- **Checks.** held-v3 verify 0/0, also with exposure; custody 0 related; closure passed.

## 16. Round 21: `natlang-rl-task-supply-20261120` (74 rows)

- **Brief** (`AUTHOR_BRIEF_ROUND21.md`): answer variety, after the coordinator noted that bAbI count tasks mostly
  answered "one". The root return value must not be the answer of more than 25% of a batch's tasks, and sibling worlds
  must not share it. Counts must span 0-5, and every label must be some world's answer. The generator asserts this and
  the supervisor rechecks it before admission (`round21/answer_variety.py`). Rounds 18-20 already had varied answers
  (22-24 distinct answers per 24-task batch), so the rule mostly guards future dataset seeds.
- **Batches admitted:** w310 talk scheduling, w312 school bus roster, w313 housekeeping board, w314 board-game cafe,
  w315 license seat audit.
  - w309 (orchard spray schedule) was refused by the author model's content filter and replaced by w316.
  - w308, w311 and w316 were still in progress at the build. They go into round 22 so this round lands before
    grpo-v5-run1 pins.
  - The authoring gate allows 5 concurrent units now (was 3), since sft-v5 leaves ~30 GB of ledger headroom.
- **Blind solve.** Sonnet agreed on every task. 18 flagged tasks were held back:
  - w310 7: whether the next slot is tried when the first has no room;
  - w313 3: a destination value with a trailing slash;
  - w314 3: an undefined argument;
  - w312 2: an unchecked capacity, and a change to the current value.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261120r21` (15 + 17 rows).
- **Result: 74 rows in 5 families**, with 68 distinct answers (the top answer appears 4 times).
- **Constructs:** folder reducers 84%, decide 46%, depth 3 43%, iterateOn 55%, typed Neuralese 39%, skills 15%.
  64 of the 74 rows carry 2+ held constructs.
- **Repetition** (against all 14 earlier POOLS, 36,340 rows): 37 programs (34.5 effective). No program text and no
  functional program is shared with the pool.
- **Checks.** held-v3 verify 0/0, also with exposure. Custody 0 related. Closure passed.

## 17. Round 22: `natlang-rl-task-supply-20261120b` (43 rows)

- **Batches.** These are the round-21 batches still in progress when round 21 was built: w308 (parcel lockers), w311
  (car rental returns) and w316 (farmers market stalls). They were built as their own round so that they land before
  grpo-v5-run1 pins. The round-22 brief authors (w317-w324, `AUTHOR_BRIEF_ROUND22.md`) go into round 23.
- **Blind solve.** 14 tasks were held back:
  - w308 6: an "otherwise" step whose scope was unstated, a missing "otherwise nothing" clause, and a title that
    contradicts its rule;
  - w311 5: a helper that reads rates it is not given, and a waiver counted on a zero fee;
  - w316 3: real disagreements. The author files late cancellation notices under `late/`, while the program text
    moves every cancellation to `done/`.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261120r22` (5 + 10 rows).
- **Result: 43 rows in 3 families**, with 42 distinct answers.
- **Constructs:** folder reducers 81%, decide 65%, depth 3 49%, iterateOn 40%, typed Neuralese 42%.
- **Repetition:** 20 programs, none shared with the 36,414-row pool.
- **Checks.** held-v3 verify 0/0, also with exposure. Custody 0 related. Closure passed.

## 18. Round 23: `natlang-rl-task-supply-20261120c` (130 rows)

- **Batches.** Haiku authors w317-w324, briefed by `AUTHOR_BRIEF_ROUND22.md`. The brief adds the round-21 lessons:
  state the fallback when a pick fails, define every argument, never put a slash inside destination names, apply
  every stated capacity, and say whether a no-op update counts.
  - Families: ice rink sessions, community fridge log, recording studio bookings, dog walking groups, print shop
    queue, youth referee assignments, bookstore consignment, apartment maintenance tickets.
- **Blind solve.** Sonnet agreed on every task, apart from one solver slip that it fixed itself. 34 flagged tasks
  were held back. The causes were:
  - two "otherwise" branches written in sequence;
  - one name used both for a flag and for a counter;
  - opening-hour boundaries;
  - a "within" capacity boundary;
  - a variable used after its loop ends;
  - a file read order left unstated;
  - a benign note that contains a spoil cue;
  - a word-fragment cue ("injur");
  - a conditional reply treated as an acceptance;
  - a run name that is confused with an area.
- **Held worlds** are registered as `natlang-rl-authored-{dev,confirmation}-20261120r23` (19 + 16 rows) and
  `-20261120r23b` (w323's 2 + 1 rows).
- **Result: 130 rows in 8 families**, with 127 distinct answers.
- **Constructs:**
  - folder reducers 89%, decide 73%, depth 3 57%, iterateOn 69%, typed Neuralese 49%;
  - 126 of 130 rows carry 2+ held constructs;
  - levels 35 / 59 / 36.
- **Repetition** against all 16 earlier POOLS (36,457 rows): 58 programs (53.9 effective), 28 structures. No program
  text and no functional program is shared with the pool.
- **Checks.** held-v3 verify 0/0, also with exposure. Custody 0 related. Closure passed.

## 19. Fallow K3: `natlang-rl-git-history-20261120` (736 rows), repository history with git as the oracle

- **Input.** The bare repositories in `bgkit-data/repos` (12,083, full history). The first repository of each of 700
  hash-ordered owners was staged to NVMe. The shared HDD was at 96% utilisation, and git's random reads stalled on it.
  415 repositories had an object store of 0.2-25 MB. Of the 205 scanned, 120 fit the bounds: 5-800 files at the tip,
  every patch under 250 KB, no submodule, and a snapshot of at most 12 MB.
- **Model access.** No git service existed. The codebase sources serve a file snapshot (`repo`: files/read/search), so
  this source adds a `git` service in the same form: TypeScript source with a JSON snapshot bound by SHA-256.
  - It offers `info`, `log` (path, author, date, message-regex and max filters), `show` (numstat), `diff` (the patch),
    `ls`, `cat` and `grep`.
  - Each snapshot holds the newest 40 first-parent commits. A merge lists both parents, and its changes are taken
    against its first parent.
  - There is no blame and no pickaxe, so "which commit added this line" means a path-limited `log` followed by reading
    patches.
  - Assets: `natlang-rl-git-history-assets-20261120` (120 files, 231 MB) on NVMe; `NATLANG_RL_GIT_ASSETS` overrides
    the path.
- **Oracle.** Every gold value is the output of git run on the bare repository:
  - `blame --porcelain`, kept only where the line is unique in the file and exactly one window patch adds it;
  - `log --first-parent` over the window's range;
  - `show --diff-merges=first-parent --name-only`, `diff --name-only --diff-filter`, `diff-tree --numstat`, and
    `show rev:path` for the JSON values.
  - Files larger than 200 KB in any version, or with more than 4,000 changed lines in one commit, are marked `-diff`
    in a per-repository attributes file. git then reports them as binary itself.
  - `log.mailmap=false`, so author names are as the commits record them.
- **Families** (train rows; at most 7 per repository, one per kind before a kind repeats):

  | family | rows | family | rows |
  |---|---:|---|---:|
  | last_change (author or UTC date) | 152 | line_origin | 73 |
  | tree_delta (removed / added since a commit) | 114 | line_before | 65 |
  | count_commits (dir, author, author+dir, date range, subject prefix, added/deleted a file) | 86 | file_created | 52 |
  | commit_files | 82 | signature_before | 18 |
  | churn ({added, deleted} over the window) | 80 | json_value_at | 14 |

  665 distinct answers in 736 rows. Levels 1/2/3: 294/391/51. 66 patch-family rows use a `read_patch` Neuralese
  helper framing (`Neuralese<string>` per commit patch).
- **Qualification.**
  - **Reference through the service:** `git_history_solve.mjs` solves every case with the `git` service alone, as the
    model sees it, and must reach git's answer. 835/835 agreed.
  - The smoke builds before that found 24 disagreements, all of them gold-side git semantics, and each was fixed:
    - a merge's `show --name-only` printing a combined diff;
    - `log -1 -- path` walking side branches;
    - mode-only changes left out of the snapshot;
    - mailmap renaming authors.
  - **verify.mjs on grpo-v5-run1's pin runtime (f9664edfedb3):** 835 references accepted. 2,606 of 2,607 wrong controls
    were rejected; the one accepted differed from gold only in letter case, and its case was dropped. 835 alias probes
    were accepted. 834 rows kept.
  - Test: `tests/test_rl_git_history_source.py` (a synthetic repository with a merge and a `-diff` file).
- **Splits.** One unit per repository, assigned by hash: 106 train repositories (736 rows); held units dev 8 (56 rows)
  and confirmation 6 (42 rows), registered as `natlang-rl-git-history-{dev,confirmation}-20261120`.
- **Checks.**
  - held-v3 verify 0/0, also with exposure.
  - Custody 0 related, before and after the held packets were registered.
  - LFM3 group closure passed. The exposure collector lists the pool (`.pre-git-history-20261120` keeps the previous
    version).
- **Repetition** against all 17 earlier POOLS (36,587 rows): 736 distinct instruction texts, none in the pool, and all
  10 families new. Program structure is narrow: 2 structures (91% single-call service wrappers, 9% with the Neuralese
  helper). Each family has 2 phrasings, and the diversity comes from the repositories and their histories.
- **git_history.jsonl** (3,599 LLM-written free-form answers about 1,576 commits) is not converted: its answers have no
  exact oracle. Its 38 repositories are among the mirrored repositories the builder can draw on.
- **Not checked.**
  - Base pass rates: the explore walk measures them.
  - Episode length against the 40-call budget. A line-origin search reads up to one patch per commit that touched the
    file.
  - No blind solve: the gold is computed, not authored.
