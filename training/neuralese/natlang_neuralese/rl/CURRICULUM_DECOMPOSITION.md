# Adaptive whole-execution task curriculum

`selectTrainingTasks(tasks, groupStats, budget, curriculum) -> TaskSelection[]`
is the shared typed natural-language policy for LFM and Mellum.

1. Start with tractable exact-verifier natlang execution. Prefer tasks that can
   yield useful verified successes, then introduce tools, longer executions and
   communication as actual train-task outcomes improve.
2. Reward correct execution regardless of whether communication is used. Task
   difficulty is metadata, never an RL admission edge/node-count requirement.
3. Keep all usable admitted train tasks scheduled: reserve rotating breadth
   work alongside adaptive practice. Half of practice explores unvisited
   invocations from the least-covered declared semantic families; under policy
   `/1` new families start with their easiest admitted task, under the default
   `/2` at their median level (see step 6). Other practice retains useful mixed
   outcomes while preferring distinct families within a batch. Coverage counts
   actual scored task identities in the current source, not historical sample
   counts or physical file boundaries. Missing families remain one unknown
   bucket. Exhausted singleton families do not occupy exploration slots.
   Physical breadth and its exact cursor remain unchanged, including revisits
   of solved work after policy changes.
4. Count only scored outcomes when adapting difficulty. Repair infrastructure;
   use verified whole-trajectory teacher/RFT seeding for all-fail groups.
5. Preserve protected held separation. New static task scaffolds remain pending
   model-facing qualification until about48 live executor samples are measured in the actual intended root mode. Authored-root private-call diagnostics do not qualify whole-student-root task prompts.

The current crisp policy adapter implements this interface through existing
`production.select_tasks`, with optional external NL command implementation.
Exact packet/source bindings, row/count validation, deterministic rotation,
selection durability and outcome bookkeeping are crisp mechanisms. The existing
collector, verifier, whole-episode graph capture, shared learner and recipe
remain the only execution/training paths. Static task construction creates
program records with exact host verifiers, not invented student/teacher traces.

6. Policy `natlang.rl-task-curriculum/2` (default since 2026-11-03; the recipe
   review found 4-task x 4-sample rounds starting unvisited families at their
   easiest level, which returned 20/20 with no outcome spread):
   - new families start at the median of their task levels; once scored, a
     family practices its lowest level below the promotion rate, else the level
     after its highest promoted one (skipped easier levels count as passed);
   - consecutive newest rounds whose outcome-spread fraction is below
     `spread_target` (0.3) shift every family's practice level one step per
     round, up when the spread-free groups were solved and down when they all
     failed, bounded by `difficulty_shift_limit` (3);
   - `min_rows_per_round` 16 and `min_group` 8 are the selection budget floor.
   Continuations adopt `/2` through the recorded controller transition
   `task-curriculum-v2-spread-20261103`; a sealed selection snapshot keeps the
   policy and budget it was chosen under, and `task_curriculum_keep_v1: true`
   opts out. The controller records `task_curriculum_spread_yield` per round.

7. Construct shares are enforced here, not by dropping pool candidates (2026-11-03). Policy `/2` carries
   `class_quota` (default `{"effect_state": 0.34}`): rows tagged `curriculum.rl.mix_class` by the pool assembler (an oracle
   that verifies a world after the run: folders, services, repaired code) make up at least that share of every round.
   Rotating breadth keeps its exact cursor; the shortfall is taken from practice by the usual frontier, outcome-signal and
   family-exploration order restricted to the class, then the remaining slots fill from all other rows. A pool without
   the tag selects exactly as before; a config without `class_quota` adopts the default through the recorded transition
   `task-curriculum-class-quota-20261103`. `scripts/rl_tasks/simulate_selection.py` measures the sampled mixture, distinct
   rows, programs and families the selector actually reaches on a pool.

8. Per-row visit decay (2026-11-04). A task with a stable mixed outcome keeps the spread the selector seeks, so on the
   r7 pool (latent-outcome simulation, 1,500 rounds x 32 rows) it was re-selected without bound (one row 372 visits, 245
   rows 20+). The controller now records each scored task's actual `visits` and recent `visit_rounds` on its latest
   observation (idempotent per round; untracked history counts as one visit). Policy `/2` adds `visit_penalty` (levels of
   frontier distance per doubling of a row's visits above the pool's mean observed visits, default 1.0), `visit_cap`
   (default 2) and `visit_cap_window` (default 50 rounds): a row with `visit_cap` visits inside the window ranks behind
   every uncapped candidate (a ranking, not a drop, so a small pool still fills the 16x8 floor). Rotating breadth,
   `class_quota` and frontiers are unchanged; `visit_penalty: 0, visit_cap: 0` opts out. A /2 config without the fields
   adopts them through `task-curriculum-visit-decay-20261104`; sealed selections keep their recorded rows and policy.
