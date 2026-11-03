# Memetic improver: decomposition, part by part

Status: implemented on main 2026-11-03 (review after the fact, `plans/OWNER_REVIEW.md`). Design and review:
[plans/neuralese/MEMETIC_META_LEARNING_REVIEW_20261103.md](../../plans/neuralese/MEMETIC_META_LEARNING_REVIEW_20261103.md).

The memetic path is the LFM3 meta-learning technique (AGENTS.md "Meta-learning is the memetic path"). The backbone,
running this program, studies one target program P's support invocations and improves P for its future invocations.
Episodes of it are RL tasks: whole-graph JOLT+ updates on the held query-world gain of the committed change set. So
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
| Evidence of support attempts | service | `rl/memetic_workspace.py` `failures`, `attempts`, `attempt`, `source` | Exact extraction from recorded parent rollouts. Never gold, never query/dev/confirmation worlds. |
| Change validation | crisp verifier | `Workspace.problems` | Exact: scope names a real function; find occurs exactly once; frontmatter parses; skill names; the natlang loader (`loadVirtualNatlang`) loads every changed `.nl`. Problems go back as the tool's answer (teach in errors). |
| Change application | crisp | `apply_changes`, `compose_guidance` | Guidance changes compose into one program-guidance text scope by scope (program / function / input class); the stored state is the native write of that exact text (span basis). Skills go to `<function folder>/skills/<name>/SKILL.md`, the program's own skills folder the runtime binds. Instruction and signature patches edit the `.nl` files. Reverts drop a change; ids make every change diff-able and reversible. |
| Evaluation on support | service | `Workspace.evaluate`, `production_evaluator` | The production collector, the episode's own actor, paired seeds, full prompt contract; program guidance through the runtime's program-guidance channel (native state import). Cached by change-set digest. |
| Per-change attribution | service | `Workspace.ablate` | Leave-one-out on support. |
| Adapter fits | service | `Workspace.fit` | The LoRA-bank coefficient fit (`tune/learned/meta_bank.py`) when a bank service is configured; otherwise an explicit capability answer. |
| Defect reports | data | `Workspace.report_defect` | Harness/oracle defects are recorded for supervision; they earn nothing (no reward channel to game). |
| Held outcome | crisp, host only | `Workspace.score` | Committed set on the private query worlds; graded quality, anchor, gain, cost. Never offered to the model (`world-bridge.ts` hides `score`). |
| Answer-leak guard | crisp | `leaked_answers` | A candidate that reproduces a query answer verbatim scores zero. |
| Reward and advantage | crisp math | `memetic_reward`; `native_replay.group_episode_advantages` | Quality = query success minus a bounded budget cost; advantage centered on max(group mean, best trivial baseline), never std-normalized. |
| Program pool, splits, curriculum, baselines | crisp | `rl/memetic_stage.py` | Related worlds on one side; near-duplicates dropped; held families never meta-trained; admission by parent headroom; trivial baselines (no change; default crisp improvement). |
| Round scheduling | crisp | `rl/interleave.py` `bind_stage` → `memetic_stage.bind_round` | Every `memetic_every` RL rounds the round's task source is the memetic source for the current parent. |
| Transport | crisp | `ts-host/src/teacher/world-bridge.ts` kind `memetic` | The workspace is a world process shown as service `workspace`. |

## Not in this version

- Tool-surface edits per program (removing `delegate` from a repair program): the runtime has no per-program tool
  surface yet; guidance can only ask (failure analysis F7, F8, F11).
- In-episode adapter evaluation: needs an adapter-capable actor (the SGLang actor has no adapter path); fits are
  reported unavailable rather than faked.
- A natural-language survival/merge policy across episodes (several committed sets of the same program).
