# Memetic improver: decomposition, part by part

Status: implemented on main 2026-11-03; its training machinery was deleted with w5 (2026-11-12) and rebuilt on the
per-call GRPO stack on 2026-11-17 ([RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md](../../plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md)
section 3): episodes are GRPO task rows (`grpo.py --memetic-tasks --memetic-share`), the supervisor's calls train as
per-call GRPO. The rows below name today's units. First design and review:
[plans/neuralese/MEMETIC_META_LEARNING_REVIEW_20261103.md](../../plans/neuralese/MEMETIC_META_LEARNING_REVIEW_20261103.md).

The memetic path is the LFM3 meta-learning technique (AGENTS.md "Meta-learning is the memetic path"). The backbone,
running this program, studies one target program P's support invocations and improves P for its future invocations.
Episodes of it are RL tasks: per-call GRPO updates on the held query-world gain of the committed change set. So
"improving a program" becomes a backbone skill that flows into general training, and every episode also exercises
`iterateOn`, typed services, evaluation and stopping, which the general mixture rarely does.

Decisions: **fn** (natural-language function), **crisp** (exact mechanism), **service** (crisp, outside world),
**data**.

## Policy (natural language, executed and trained)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Which failures matter, what caused them | fn | `improve_program.nl` steps 1-2, `refine.nl` step 1 | A judgment over traces. The model reads crisp evidence (outcome class, failed checks, value, errors, steps) and names a cause a program change could prevent on other invocations. |
| What to change, and of which kind | fn | `refine.nl` step 2 | The core improvement decision; typed `Change` values, not free text the harness guesses at. |
| Whether a change helped; what to keep | fn, bounded by measured numbers | `refine.nl` steps 3-4 | The numbers are crisp (`evaluate`); keeping a change only above `state.best` is a stated rule the model applies; the reward checks it on unseen worlds. |
| When to stop | fn inside `iterateOn`, bounded crisply | `improve_program.nl` step 3 (`withLimit({maxSteps: budget.evaluations})`), `refine.nl` step 4 (`misses >= 2`) | Structural termination: the step bound is the evaluation budget; the natural stop is "two misses or budget spent". |
| The final change set and its one-sentence reason | fn | `improve_program.nl` step 4 (`workspace.commit`) | The episode's output. |

## Mechanism (crisp)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Evidence of support attempts | service | `rl/memetic_workspace.py` `baseline`, `failures`, `attempts`, `attempt`, `source` | Exact extraction from the parent's own support rollouts, measured once per GRPO group, under the policy version that samples it (`memetic_workspace.prepare`). Never gold, never query/dev/confirmation worlds. |
| Change validation | crisp verifier | `Workspace.problems` | Exact: scope names a real function; find occurs exactly once; frontmatter parses; skill names; the natlang loader (`loadVirtualNatlang`) loads every changed `.nl`. Problems go back as the tool's answer (teach in errors). |
| Change application | crisp | `apply_changes`, `compose_guidance`, `input_skills` | Program guidance is one native state every call of the program imports; function guidance is a separate state (program part + function part) imported only by that function's calls; input-class guidance becomes a skill of the root whose description is the class, so the called model loads it only for inputs it fits (progressive disclosure). Each stored state is the native write of its exact text (span basis). Skills go to `<function folder>/skills/<name>/SKILL.md`, the program's own skills folder the runtime binds. Instruction and signature patches edit the `.nl` files. Reverts drop a change; ids make every change diff-able and reversible. |
| Evaluation on support | service | `Workspace.evaluate`, `collection_evaluator` | A nested collection (the GRPO collector's episode command) on the episode's own actor, paired seeds, full prompt contract; program guidance through the runtime's program-guidance channel (native state import). Cached by change-set digest and shared, under a lock, by the rollouts of an episode group. |
| Per-change attribution | service | `Workspace.ablate` | Leave-one-out on support. |
| Adapter fits | not offered | (deleted `rl/memetic_fit.py`) | No adapter-capable actor serves GRPO; returns with the bank (TUNING_RESTORE_20261117.md). |
| Defect reports | data | `Workspace.report_defect` | Harness/oracle defects are recorded for supervision; they earn nothing (no reward channel to game). |
| Held outcome | crisp, host only | `Workspace.score` | Committed set on the private query worlds, paired with the baselines (same worlds, same seeds, worlds and samples sized by the program's measured cost); graded quality, anchor, gain, cost, binomial noise; actual vs shuffled (another program's guidance) vs none ablation of committed guidance. Never offered to the model (`world-bridge.ts` hides `score`). |
| Answer-leak guard | crisp | `leaked_answers` | A candidate that reproduces a query answer verbatim scores zero. |
| Reward and advantage | crisp math | `memetic_reward`; `grpo.py episode_advantages` | Quality = query success minus a bounded budget cost; advantage centered on max(group mean, best trivial baseline), scaled by the group's spread. |
| Program pool, splits, admission, baselines | crisp | `rl/memetic_tasks.py`, `memetic_workspace.prepare` | Programs are exact sources from the GRPO pool; related worlds on one side; near-duplicate worlds dropped (world data, not program text); held programs (registered memetic held families) never meta-trained; admission by base pass rate at build and by the group's support measurement; baselines: no change, mechanical exemplar guidance. |
| Scheduling | crisp | `grpo.py` `--memetic-share`, `Collector._bind_memetic` | A share of new GRPO groups are memetic rows; each group gets its own spec copy (actor, runtime, sampling controls). |
| Distillation and promotion | not yet | design doc section 3 item 5 | Query runs of committed guidance as guidance-consumer data, and the held-program measurement, come next. |
| Transport | crisp | `ts-host/src/teacher/world-bridge.ts` kind `memetic` | The workspace is a world process shown as service `workspace`. |

## Not in this version

- Tool-surface edits per program (removing `delegate` from a repair program): the runtime has no per-program tool
  surface yet; guidance can only ask (failure analysis F7, F8, F11).
- Adapter fits combined with guidance in one change set (the adapted actor reading a base-written guidance state).
- A natural-language survival/merge policy across episodes (several committed sets of the same program).
