# Task Author (`applications/task-author/`): Decomposition

Tier 0 flagship tool (WS3 T3, WS6 §2; plans/semantic-fp/03 §4): author Semantic-FP training tasks whose oracle is
qualified by replay, that no cue-word rule solves, and that never enter a held split group.

## Parts

| Part | Form | Why |
|---|---|---|
| Write the task: world, program files, prompt, oracle kind and verifier, a reference answer, ≥ 2 negative answers and the keyword-baseline answer | fn `draftTask.nl` | judgment (authoring) |
| Structure and custody: ids, construct tag, files, verifier named, answers present, split group not reserved | crisp `checkTaskStructure`; `isReserved` is a service (`held_eval.reserved` over held-v3/held-v4 reservations) | exact facts |
| Replay: reference passes the oracle, every negative and the keyword baseline fail it | crisp `replayProblems` over the `replay` service (the task's verifier or exact match) | exact; the outside world runs the oracle |
| Repair a draft from those problems | `semanticRepair` with `draftTask.nl` as the fix | exact check, judged fix |
| Does solving the task need meaning rather than wording? | fn `isSemantic.nl` (`readout: decision`), `escalate` when unsure | judgment |
| Does the oracle grade the semantic goal without leaking it or admitting shortcuts? | fn `reviewOracle.nl` (`readout: decision`), `escalate` when unsure | judgment |

A task is admitted when no exact problem is left and both judgments say yes.

## Not built yet (plan 03 §4)

- `pickSeed.nl`: choosing seeds for archetypes the pool lacks from `measure_repetition.py` output.
- `paraphraseTwin.nl` and the lexical-overlap check for metamorphic tasks.
- Writing admitted tasks through the registry path (`data_registry.py` register, `promote`, `held_eval verify`) and a
  `replay` service bound to the authored-task controls (`scripts/rl_tasks/verifier_controls.mjs`, `qualify_authored.py`)
  and `training/semantic_fp_tasks/verifiers/`.
