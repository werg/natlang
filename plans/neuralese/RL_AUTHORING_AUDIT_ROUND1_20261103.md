# RL task authoring audit, round 1 (2026-11-03)

Auditor: "claude-opus audit 2026-11-03". This round covers the first 12 batches from six Haiku author workers
(`/mnt/external/natlang-rl-authoring/batches/w0{1..6}/b000{1,2}`), 163 tasks in all.

## How the audit was done

- **Independent solve.** Each batch had its own Opus auditor. The auditor solved every task from the program text and
  inputs alone, with `expected`, wrong answers, aliases, reference and smoke stripped out, and used Python for the
  arithmetic. Only then did it compare its answer with `expected`.
- **Other checks.** Each auditor also checked the answer contract, the construct tags against kit section 3,
  padding, functional distinctness and level labels.
- **Corrected copies.** These are in `batches/<w>/<b>/audited/tasks.jsonl` and were requalified into
  `audited/qualified[-2]/`. All of them PASS.
- **Originals and records.** The original `tasks.jsonl` files are untouched. Per-task verdicts are in the scratchpad
  verdict files and in `admitted/admissions.jsonl`.
- **Novelty.** I ran the novelty gate on each audited copy in sequence, against the broad pool plus the
  admitted pool, using `scripts/rl_tasks/authored_novelty.py` as it stood on main. Main gained commit 18104ac05
  during the audit, which adds the apostrophe fix and functional-signature novelty.
- **Admission rule.** A batch was admitted only if it passed the audit, the requalification and the novelty gate,
  and it was not held for diversity by audit judgment.

**Result: 77 tasks admitted from 6 batches** (`admitted/cases.jsonl`: 70 train, 3 dev, 4 confirmation). The 6
batches are recorded in `rotation.jsonl`. 15 tasks were rejected outright, and 5 batches holding 69 audit-clean
tasks are not admitted yet: one failed the audit-side diversity bar and four failed the functional novelty gate.

## Per-batch table

| batch | family | in | audit pass | fixed | rejected | admitted to pool | functional distinct programs | constructs genuinely covered (admitted/passing tasks) | effect_state |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| w01/b0001 | harvest_ledger_operations | 10 | 9 | 2 | 1 | **9** | 10 (9 admitted) | call_depth_3 7, decision_readout 1, iterate 1 | 3 |
| w01/b0002 | table_note_relay_answers | 10 | 10 | 5 | 0 | 0 (held) | 6 (tool: 3) | neuralese_relay 8, typed_neuralese 2 | 0 |
| w02/b0001 | tower_rota_repair | 14 | 14 | 14 | 0 | **14** | 12 | call_depth_3 14, multi_operation_codebase 14 (folder_reducer 0: fake, removed) | 4 |
| w02/b0002 | table_question_operations | 10 | 0 | 0 | 10 | 0 | 7 | none (multi_operation_codebase fake) | 0 |
| w03/b0001 | film_deposit_reconciliation | 16 | 14 | 14 | 2 | 0 (held) | 4 (tool: 5) | decision_readout 14 (call_depth_3 0: fake, removed) | 6 |
| w03/b0002 | tool_request_relay | 15 | 15 | 15 | 0 | 0 (held) | 3 (tool: 2) | neuralese_relay 13, decision_readout 3 | 0 |
| w04/b0001 | ferry_deck_boarding | 16 | 16 | 1 | 0 | 0 (held) | 8 behavioural / 1 structural | decision_readout 16, call_depth_3 16 (thin) | 16 |
| w04/b0002 | catalogue_correction_desk | 15 | 15 | 1 | 0 | **15** | 5 | iterate 15, call_depth_3 15 | 15 |
| w05/b0001 | apiary_book_ledger | 15 | 14 | 13 | 1 | **14** | 14 | iterate 10, skills 4 (10 fake tags removed), decision_readout 1, call_depth_3 1 | 8 |
| w05/b0002 | calc_plan_relay | 15 | 14 | 14 | 1 | 0 (held) | **1** | neuralese_relay 14, iterate 14 (thin) | 0 |
| w06/b0001 | sluice_shift_desk | 16 | 16 | 16 | 0 | **16** | 11 | call_depth_3 16, iterate 15 | 2 |
| w06/b0002 | fact_desk_escalation | 11 | 9 | 4 | 2 | **9** | 8 | decision_readout 9, iterate_neuralese 8, neuralese_relay 8, iterate 1 | 0 |
| **total** | | **163** | **146** | 99 | 17 | **77** | | | 54 (32 admitted) |

Admitted constructs (77 tasks):

| construct | admitted tasks |
|---|---:|
| call_depth_3 | 53 |
| iterate | 42 |
| effect_state | 32 |
| multi_operation_codebase | 14 |
| decision_readout | 11 |
| iterate_neuralese | 8 |
| neuralese_relay | 8 |
| skills | 4 |
| typed_neuralese | 0 |
| neuralese_fan_in | 0 |
| folder_reducer | 0 |

**Why the held batches are not admitted.** The novelty gate's family rule now also requires functional signatures
(at least one new signature, and new signatures at least half of the new programs):
- w01/b0002, w03/b0001, w03/b0002 and w04/b0001 FAIL it.
- w05/b0002 passes it only because its code literals differ. My audit counts it as one functional program, so I hold
  it as well.

Their audited copies are qualified and ready. They can be admitted together with a follow-up batch in the same family
that adds genuinely new operations, which would lift the family over the bar. Their individual tasks are correct.

## Wrong `expected` values found

In 163 tasks, the auditors found no arithmetic errors in a hand-computed `expected`. Haiku's arithmetic and
state-tracking on these sizes is reliable.

The `expected` failures are all semantic: the author answered from intent or from the dataset gold instead of running
the program as written.

| task | author | nature | action |
|---|---|---|---|
| w06 fact-desk-metroid-polygons | w06 | `no` copied from StrategyQA gold. The program's own router sends release-date/platform snippets to "other", so the decisive fact never reaches the case file; run literally, the program escalates. | rejected |
| w03 film-deposit-release-glen-tilt-0013 | w03 | Release depends on reading "reel 2 is the 35mm print, reel 1 unknown" as confirming an unlabelled scan. The note identifies no shelf copy, so "keep" is equally defensible. | rejected (ambiguous) |
| w03 film-deposit-release-harvest-moon-0014 | w03 | Same pattern: "reel 2 confirmed by lender letter". The author even listed "keep" among the wrong answers. | rejected (ambiguous) |
| w06 fact-desk-dioskourides-lapidary | w06 | "Artists sign their work" plus "he signed a ringstone" was treated as stating he engraved it. That is affirming the consequent. | rejected (ambiguous) |
| w05 calc-relay-b2-14 | w05 | The rule that settles 480 lives in a root preface the planner never receives. From the question alone, 211.2 is also a reasonable reading. | rejected (ambiguous) |
| w04 ferry-deck-04 | w04 | `5 - 4.2` gives `0.7999999999999998` in JS, but the exact oracle expects `0.8`. No rounding was stated. | fixed (input length set to 4.0, expected recomputed) |

Several tasks needed contract or rule fixes that leave the answer unchanged:
- **w01/b0002 0007 and 0008:** the question asked for an average or a "which" while the root returns a sum or a
  count.
- **w05/b0002 12:** the deciding rule sat in text the helper never sees.
- **w06/b0001, all tasks:** "a new fault" made worsening or recurring reports contestable. The counting rule was
  stated explicitly.
- **w04/b0002 05:** a bracketed qualifier made a row name ambiguous.
- **w02/b0001 0013:** the header contradicted the steps.

**The two deliberate escalates in w06/b0002 (manuel-tudors, florence-theocracy) are kept, despite the dataset gold
True.** In both, the program's rule removes the evidence mechanically:
- **manuel-tudors:** the three-snippet budget cuts off the Henry VIII snippet.
- **florence-theocracy:** a gaps gate applies to a file that contains no fact about Florence's rulers.

Under "answer only from the case file", `escalate` is the one correct result. I checked both myself.

## Fake or inflated construct tags (authoring reliability)

- **folder_reducer:** faked in all 14 tasks of w02/b0001. The root wrote its own inbox, and the author claimed the
  format cannot carry folders, which is false (`folders`/`expected_folders`). The auditor removed the
  directory-reducer ceremony and passed `messages` directly; the answers are unchanged.
- **multi_operation_codebase:** faked in all of w02/b0002 (one world per task). With no other construct left and the
  converter refusing empty `constructs`, the batch was rejected.
- **call_depth_3:** faked in 3 tasks of w01/b0001 and in all of w03/b0001:
  - pass-through middle helpers;
  - a one-line comparison as a sub-helper;
  - a sub-helper call inside a `readout: decision` helper. That call is unreachable: readouts are scored in one pass
    with no tools (`ts-host/src/native/agent.ts` decisionReadout).
- **skills:** decorative in 11 of 15 tasks of w05/b0001. The skill only restated the note, so the answer did not
  depend on it. Task 08's skill was unreachable, because it sat in the root folder but was read by a grandchild
  helper.
- **neuralese_relay / iterate_neuralese:** tagged where the relay never runs, in w03/b0002's gate tasks whose answer
  is "ask". Also tagged where the Neuralese file feeds no decision (w06/b0002's tally desk).
- **iterate:** used without an explicit stop condition in 9 w05/b0001 roots and in all of w05/b0002 (added in the
  audited copy). Used over a single item in w06/b0001 entry-e (tag removed).
- **Seed provenance:**
  - w03/b0002 claimed a ToolACE seed but read no item; it is recorded as invented.
  - 14 of 15 NESTFUL citations in w05/b0002 borrow only the plan format.
  - The w01/b0002, w04/b0002 and w06/b0002 citations are genuine.
- **Levels:** inflated in most batches, typically one-call filters labelled 2–3. They were relabelled in w01/b0002,
  w03/b0002 (14 of 15), w05/b0001, w05/b0002 and w06/b0002. No batch had a genuine level-3 task that was long, with
  many items, corrections or competing constraints.

## Diversity

- **Known padding cases, confirmed:**
  - **w03/b0001:** every root ended in an inert "Guard:/Tie check:/Sum check:" paragraph (removed). Underneath are 4
    operations with clause toggles, and some toggles are not exercised by their own inputs.
  - **w04/b0001:** one call graph with five toggled rule sentences. The refrigerated-first rule changes 2 of 8
    answers, the crane reserve 1 of 7, and best-fit none of the level-3 tasks. That leaves 8 behaviourally distinct
    programs, and 1 by structural signature.
  - **w05/b0002:** a per-task root preface the planner never saw was the only thing distinguishing 15 instances of one
    program (removed).
- **Thin but honest:**
  - w01/b0002: one read-filter-aggregate template.
  - w03/b0002: 3 variants of a two-call extraction.
  - w04/b0002: 5 operations × 3 worlds.
  - w06/b0001: 5 tasks are the same threshold readout, reformatted.
- **Genuinely diverse:** w01/b0001 (10 distinct programs), w02/b0001 (12 operations over a shared ledger) and
  w05/b0001 (14 programs).
- **Problem types:** 5 of the 6 admitted batches are the `running-ledger-with-corrections` problem type. The rotation
  should steer away from it.

## Tool bugs and gaps observed

1. **Qualification never executes the program**, as was known. The audit confirms the consequence: semantic and
   literal-execution errors (Metroid, the w03 releases) pass every gate.
2. **The novelty normaliser in my original reading** (`measure_repetition.QUOTED`) treated `'` as a quote opener and
   also blanked every backtick span. That erased the helper-call code in the instructions, so programs differing only
   in backticked code compared equal.
   - The apostrophe half is fixed on main in 18104ac05; I did not re-check the backtick behaviour there.
   - The new `functional_signature` is a structural proxy with errors in both directions. w05/b0002 (one program)
     gets 14 signatures because code literals differ. w04/b0001 (8 behavioural variants) gets 1, because its rule
     differences live in prose inside one helper. Audit judgment is still required.
3. **The converter requires a non-empty `constructs` list** (`scripts/rl_tasks/sources/authored.py:76`). A correct
   task whose only tag is fake cannot be admitted as construct-less. Either allow an explicit `constructs: []` (or
   `plain`) or keep rejecting such tasks; the brief and the converter currently disagree.
4. **No check that a `readout: decision` helper's body calls nothing.** Readouts run without tools, so any call
   inside one is dead, and smoke scripts the decision directly so this never surfaces. Add a convert-time error.
5. **Skills placement.** Kit section 2 says skills live at `root_folder/skills/`, but the runtime binds a folder's
   skills only to the call that owns that folder. A helper reading `read_code("skills.X")` gets nothing.
   - Fix the kit text to say "in the companion folder of the function that reads it".
   - Add a qualification check, for example a skills-ablation replay: drop the skills and the answer must change.
6. **Smoke does not check helper argument arity.** w02/b0001 passed while calling a 2-argument helper with 3
   arguments.
7. **No convert-time checks for:**
   - a `directory-reducer` root without a `folders` world;
   - a `multi_operation_codebase` tag whose world has only one root operation;
   - an `iterateOn` without a stop.
8. **`alias_answers` are only probed, never accepted** by the oracle as alternates (w04 finding). The kit says
   aliases "should be accepted". Either pass them to the oracle or change the kit wording.
9. **`memory_ledger.py run` without `--wait` exits instead of queueing** when headroom is short. My audit brief
   omitted `--wait`; the kit's command includes it, so keep it.
10. **The novelty `-o` file was not written on one FAIL run** (w03/b0002, unit `rlaudnov-w03-b0002-120709`). The log
    has the full FAIL JSON, and a rerun wrote the file. The cause is unclear.
11. **The rotation ledger credits one construct per batch.** Batches covering several constructs under-report the
    others. Its claims were also never written in this round: `rotation.jsonl` did not exist before my records.

## Recommendations for the next authoring brief

1. **Run your program literally, then write `expected`.** Simulate each helper with only the arguments it receives.
   Never copy dataset gold: if the program's routing, budget or rule removes a fact, the answer changes. Put every
   rule the answer needs in the text of the helper that decides it, never in a root preface.
2. **Each NL judgement must have one clearly right reading.** If a note does not name the item, a careful human must
   not be able to argue the other way. Prefer notes that name the box, barcode or entity explicitly. When an
   ambiguity is the lesson, make the program's stated rule settle it.
3. **Drop a tag rather than simulate it.** Don't write your own inbox to claim `folder_reducer`; use `folders` and
   `expected_folders`. Don't tag `multi_operation_codebase` unless one world runs several root operations over shared
   helper files. Remember that decision readouts are leaves, and that a skill must hold the only statement of a fact
   the answer needs.
4. **Make depth real.** The middle helper must own a different subproblem and combine its sub-helper's result with
   its own work. Exact comparisons and arithmetic stay in code.
5. **Diversity is a new computation, never a toggled sentence or an inert paragraph.** For every rule variant,
   include inputs where omitting the rule changes the answer, and say which input that is. Grow a family with new
   operations (undo, audit, carry-over, merge, repair, forecast) or new helper roles, not with more note mixes or
   readouts of the same ledger. Padding to pass the gate will be removed and counted against the batch.
6. **Contract hygiene:**
   - the question must ask for exactly what the root returns;
   - with an exact oracle, use binary-exact decimals or state the rounding;
   - wrong answers must be task-specific (no copy-pasted "colour: blue");
   - state the counting rule for recurring events;
   - add an explicit stop to every `iterateOn`.
7. **Levels:** level 3 means long inputs, many items, corrections or competing constraints. Two-call relays are
   level 0. Each family needs real level-3 tasks; none this round had them.
8. **Construct coverage:** next round, prioritise `typed_neuralese`, `neuralese_fan_in`, `folder_reducer` (real
   folders) and `skills`, which are at 0–4 admitted. Avoid `running-ledger-with-corrections` for a while.
9. **Provenance:** cite `groups` only for items you actually used (question, numbers and gold). Otherwise record
   `invented:<domain>`.

## Files

- Admitted pool: `/mnt/external/natlang-rl-authoring/admitted/cases.jsonl` (77 rows, unique ids, written atomically).
- Admission log: `/mnt/external/natlang-rl-authoring/admitted/admissions.jsonl` (6 admitted entries, plus 6
  not-admitted entries giving their reasons).
- Rotation ledger: `/mnt/external/natlang-rl-authoring/rotation.jsonl` (6 recorded batches).
- Audited copies and requalification: `/mnt/external/natlang-rl-authoring/batches/<w>/<b>/audited/` (tasks.jsonl,
  qualified*/report.json, novelty.json).
