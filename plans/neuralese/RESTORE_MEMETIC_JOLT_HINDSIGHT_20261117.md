# Restoring memetic learning, JOLT+, hindsight hints and self-distillation on today's stack (2026-11-17)

Owner, 2026-11-17: "I definitely want memetic learning, JOLT+, hindsight and distillation to be brought back." They were
deleted in the w5 retirement (60d2c1209, mislabeled "SGLang actor: append-only prompts", plus 5500b44fb, c295221b3,
5642368d6, 34db2102b, 7a1cbe75f) after APPROACH_REVIEW_20261111 recommended plain per-call GRPO. This plan rebuilds the
capabilities on the per-call stack (`grpo.py`, `exp0_sft.py`, `sft_sources.py`, `lfm3_fast_train`, the SGLang actor,
held eval). It does not bring back the measured waste: JOLT+ with `beta_s = 0` and one teacher attempt per group
(advantage 0, review 2.5), privileged twins packed into every graph (2x forward tokens), the 16x whole-graph replay
overhead, and the custody/controller/transition machinery. Whole-graph Neuralese gradients come back separately as an
in-graph pass-through (NEURALESE_PASSTHROUGH_20261117.md, sibling agent); hindsight reconstruction and the JOLT+ OPSD
twin build on its interface.

Policy is natural language and mechanism is crisp (AGENTS.md): the guidance writer, the hint writer, the attempt
planner and the memetic improver are natlang programs the policy model runs and RL trains; scheduling, custody, scoring,
advantages and losses are crisp code in the training stack.

## 0. What survived the retirement

| Piece | State | Where |
|---|---|---|
| In-run guidance runtime: per call, visible facts -> `selectGuidanceFacts` (pluggable, crisp or NL) -> `writeGuidance` (policy model, returns text; host writes it as a native state) -> import (`mode: import`) or text fallback | **live code, unused since 11-12** | `ts-host/src/native/training-guidance.ts`, `runtime/node-guidance-writer.ts`, `native/agent.ts` 1425-1430, 1594-1604, 1804, 1933; run control `x_natlang_guidance` (`teacher/cli.ts` 110, 147) |
| Guidance-writer package: `writeGuidance.nl`, `selectGuidanceFacts.nl`, `planTeacherAttempts.nl`, `writeHint.nl`, `trajectoryOverview.ts`, `visibility.ts`, `crispSelection.ts` | live; `writeHint` written but **not wired** (returns `Neuralese<string>`, which the 11-03 writer root cause showed the model cannot close) | `applications/guidance-writer` |
| Privileged rendering, receipts, custody re-verification (`case_facts`, `applicable`, `PrivilegedRendering`, `guidance_plan`, `captured_block`) | live | `rl/privileged_context.py` |
| Front-side insertion of the guidance turn (text tokens or the writer's state, flagged `privileged` in the capture) | live on both Python engines | `serve/sglang_actor.py` 600-620, `serve/engine.py`, `serve/http.py` 522-532 |
| JOLT+ collection helpers: `guidance_controls`, `student_guidance`, `teacher_entries`, `plan_teacher_attempts` (crisp or NL), `group_feedback` | live code, no caller | `rl/jolt.py` 520-705 |
| JOLT+ whole-graph learner terms (`jolt_action_loss`, twins) | dead: their learner (`native_replay` objective path) has no live caller | `rl/jolt.py` 1-520, `rl/native_replay.py` |
| Scoring of `jolt_role` episodes (students-only group stats, teacher rows reported apart) | live | `rl/task_rollouts.py` 782-830 |
| Program-guidance channel (stored guidance per program or per function, imported by every call of the program) | live | `ts-host/src/guidance/program.ts`, collector `--program-guidance-file`, `run_bonsai_queue.py` 981 |
| Memetic improver program (`improve_program.nl`, `refine.nl`) | live, no world to run in | `applications/memetic-improver` |
| Memetic held programs (5 families, 111 rows) registered as a protection source | live | `training/protected-worlds.json` (`memetic-held-programs`) |
| Memetic workspace, stage (pool, splits, baselines, specs, rows), world-bridge kind `memetic`, `memetic_outcome` scoring, distillation, promotion | **deleted** | git `60d2c1209^`: `rl/memetic_*.py`; `5500b44fb^`: `world-bridge.ts`; `60d2c1209^`: `task_rollouts.memetic_outcome` |
| Hindsight objective (`rl/hindsight.py`), pilot | **deleted** | git `60d2c1209^` |
| Rollout distillation (RL successes into SFT windows) | **deleted**; replaced by `sft_sources.py` (`rl-own` source) | git `60d2c1209^` `rl/rollout_distillation.py` |
| Own-success SFT data | live: `sft_sources.py extract/replay/build`, used by sft-v3-run1 (44% of its weighted target) | `scripts/neuralese_data/sft_sources.py` |

Group outcome classes measured on grpo-v3/run1 (571 scored groups on disk): 70% with spread, **12.6% all-fail**,
17.5% all-pass. Plain GRPO learns nothing from the all-fail 12.6%, which are the tasks at the policy's frontier. A GRPO
step is ~2.7 h (46% collection, 54% learner; GRPO_STEP_THROUGHPUT_20261117.md) over 48-96 groups of 8.

## 1. JOLT+: privileged teacher attempts trained in student form

### What it does and why it should raise held task success

For a task where every student attempt failed, the same policy runs a few **teacher attempts** with privileged
guidance: `writeGuidance` (the policy, in-run) turns the task's verified facts (authored reference solution, decisive
evidence, earlier attempts' feedback) into a short guidance text that the consumer call imports. A teacher success is a
demonstration on a frontier task. The student learns from it in **student form**: the guidance turn is removed from the
teacher's prompts and its sampled actions are trained under the unprivileged prompt with the truncated importance ratio
against the actor's privileged sampling probability. This is the Guide-GRPO construction (adaptive in-context guidance
for all-fail prompts, trained off-policy with importance correction: "Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models", 2025). It turns zero-gradient groups
into groups with a positive example and trains the writer by RL on whether its guidance made the teacher succeed. The
writer is the same program that writes memetic and hindsight guidance, which is the product capability.

### Old implementation, sound parts and waste

- Sound and kept: facts and custody (`case_facts`, `ProtectedWorlds`), visibility, the declared guidance rendering and
  receipts, the in-run writer and its text-to-state write (b069f6061), teacher-attempt entries per group, the NL
  attempt plan with crisp fallback.
- Waste (review 2.5): one teacher attempt per group, normalized within the teacher group, so its advantage was 0;
  `beta_s = 0` so the student OPSD KL contributed nothing while its stop-gradient twin was still packed into every
  graph (2x forward tokens); privileged episodes landed only in the first 1-2 updates of a round.

### Minimal design on today's stack

1. **Scheduling (crisp default, NL optional).** When a student group is scored with at most `--teacher-max-successes`
   successes (default 0) and its task has facts, the collector appends `--teacher-attempts K` teacher entries
   (default when on: 4) to that group and holds the group back from the learner until they finish. Teachers run next
   (front of the queue). `planTeacherAttempts` (NL) can replace the crisp rule (`--teacher-plan nl`).
2. **Facts.** Answers are excluded by default (`--teacher-fact-kinds`, default `reference_solution, minimum_sequence,
   decisive, reference_failure, attempt_feedback, teacher_trajectory`): `expected_result`, `expected_files`,
   `expected_effects` and `child_value` would let a teacher return the answer without the work, which student form
   would then reinforce. Every pool row has a reference solution (34,822/34,822), so every all-fail group qualifies.
   Selection defaults to `crisp` (no extra model call); the writer is on (`native`: the host writes its text as state).
3. **Student form.** In the learner, a teacher episode's captured prompts drop every `privileged` segment (the guidance
   turn and its imported state) before lowering. The writer's own calls (`writeGuidance`) stay in their real context:
   they are on-policy samples whose reward is the teacher outcome.
4. **Non-zero advantage.** Students keep plain GRPO advantages over the students alone (unchanged; an all-fail group
   still gives them 0). Teachers get `A_t = (r_t - mean(r_students)) / (std(r_students ∪ r_teachers) + eps)`. In an
   all-fail group a teacher success gets a positive advantage and a teacher failure gets 0 (a guided failure in student
   form is not evidence about the student). The group counts as having spread when any advantage is non-zero.
5. **Loss.** Unchanged policy term: `-min(rho, 2) * A * logp` with `rho = exp(logp_learner(student form) -
   logp_actor(teacher form))`, the tail floor and the nucleus as for students, and the k2 KL to the start policy on the
   same tokens. Tokens the unprivileged policy finds unlikely get small `rho` and so little weight.
6. **Dense OPSD term (phase 2, after pass-through).** `beta_s * KL(pi(.|s) || sg pi(.|s, g))` on student calls of
   guided groups only, with `g` the group's teacher guidance state. It needs a forward-only twin of those student calls
   with the guidance import inserted, which goes through the learner's import path (sibling's interface). Off until
   then.

**Cost per GRPO step.** K = 4 teacher attempts on 12.6% of groups: +6.3% episodes, plus one `writeGuidance` call
(1-3 turns) per teacher episode, so about +7% collection. Teachers train like students, so about +6% learner tokens.
About +6% step wall time in total. Guided groups wait one extra episode for their teachers. The OPSD twin (phase 2)
would add a forward-only pass of ~12% of student calls, about +4% learner time.

**Dependencies.** None on pass-through for steps 1-5 (the guidance state is dropped in student form). Held eval:
unchanged (no guidance at evaluation). Task supply: every pool row carries facts.

**Entry.** `grpo.py --teacher-attempts 4` (default 0 until one live smoke on the actor shows teacher episodes with
written guidance end to end: the runtime path has not run since 11-12). It then goes into the next bundle (grpo-v5)
on by default.

## 2. Neuralese hindsight hints

### What it does and why

A verified successful trajectory tau* of a task (a teacher-generation success or the run's own earlier success of the
same task) is a much closer demonstration than the authored reference solution: it is a real run in the runtime's own
transcript shape. The hint writer `writeHint` (policy model, natlang program) reads tau* as an input value with eval
and writes a short hint. A self-teacher attempt imports the hint as its guidance. Its verified runs become student-form
targets (section 1), so the student learns from near-on-policy demonstrations on tasks it fails. With pass-through the
writer also gets a dense **reconstruction** objective: `-log pi(tau* | hint state)` through the imported state into the
weights that computed it, plus a score-function term over the writer's sampled text with a leave-one-out baseline across
K hints. That trains exactly the product capability (compress what made a run succeed into state another call uses).

### Old implementation

The 10-30 pilot (`hindsight_pilot.py`, deleted) ran `writeHint` through the native server: 6-25 min per writer episode,
hints 148-576 tokens (target 64), a small positive gain on teacher-corpus tau* (Δ +0.051 nats/token, 2.5% of the
off-policy gap), and an unstable phase-1 run at lr 2e-4. Sound: tau* as a structured value (`TrajectoryCall`), the
crisp overview, the leave-one-out writer objective, custody of tau*. Waste and defects: `Neuralese<string>` return (the
close marker is unlearned), the slow native-server writer, an import-channel KL whose scale needed a 1e-6 weight, and
a go/no-go gate the method could not reach.

### Minimal design

1. **Phase 2 first (no pass-through needed).** `writeHint` returns `string` (span basis, as `writeGuidance` since
   b069f6061). The run control `natlang.guidance-run/1` gets an optional `trajectory` (`TrajectoryCall` of the root
   call). `TrainingGuidance` runs `writeHint(consumer, execution, overview, facts)` for the root consumer when it is
   present, `writeGuidance` otherwise; the text becomes the native state exactly as before. Every engine-side check that
   names the writer (`http.py` `x_natlang_guidance_policy`, `privileged_context._request_parts`) accepts `writeHint`.
2. **tau* index.** `rl/hindsight.py trajectory_value(rollout)` builds a `TrajectoryCall` from a scored rollout's
   trajectory. `index` writes `task_id -> TrajectoryCall` from `sft_sources` episodes (teacher and `rl-own` successes,
   held closure excluded) and from the run's own archives. `grpo.py --hindsight-trajectories INDEX`: a teacher attempt
   of a task with a tau* carries it (hint teacher); others keep fact guidance.
3. **Phase 1 (after pass-through).** For a sampled set of tau* per step, K = 4 `writeHint` episodes on the actor; the
   learner builds, per hint, a reconstruction chain: tau*'s calls with the hint's producer document imported at the
   guidance place (the pass-through's `source_capture` lowering), CE on tau*'s sampled tokens, gradient into the
   producer; then the writer calls train with `A_i = R_i - mean_{j≠i} R_j`, `R_i = -NLL_i - lambda_len * L_i`. No
   import-channel noise in the first version (the hint is text-derived and short; the length cost bounds the rate).
   Δ log p(tau* | hint) and hint length are telemetry, not gates.

**Cost per step.** Phase 2: same as JOLT+ teachers (a hint teacher replaces a fact teacher); `writeHint` reads tau* in
a few eval turns, about 2-4 calls per teacher episode instead of 1-3. Phase 1 at 8 tau* per step: 32 writer episodes
(~4% collection) and 32 teacher-forced tau* chains of 5-20k tokens with backward (~10% learner).

**Dependencies.** Phase 2: none beyond section 1. Phase 1: pass-through (import from `source_capture`, gradient into
the producer). Task supply: tau* from `sft_sources` (6,483 verified trajectories over 6,206 pool tasks in
sft-verified-20261117-v2) and own archives.

**Entry.** `--hindsight-trajectories INDEX` with `--teacher-attempts`; phase 1 `--hindsight-reconstruction K` once
pass-through lands.

## 3. Memetic meta-learning

### What it does and why

The policy runs `improve_program` on one program P: it studies P's support invocations (outcomes, failed checks, step
traces), proposes typed changes (guidance by scope, skills, instruction and signature patches, tool withholding),
measures them on support worlds, and commits a change set. The reward is the committed set's success on **query**
worlds it never sees, anchored on the best trivial improvement (no change, or mechanical exemplar guidance). This is the
product's meta-learning loop (soft skills learned from several invocations). Held task success gains indirectly:
improver episodes are long multi-call natlang work with measured feedback, and their committed guidance is consumed
by the same model.

### Old implementation

Sound (kept): the typed action space and checks (`memetic_workspace.py`: exact patches, loader check, placeholder and
verbatim-answer refusals, revert, ablate, evaluation cache), program identity `family@sha12`, world components and
near-duplicate dedupe, support/query split, the anchored reward `quality = query_success - bounded cost`, the anchor
`max(none, mechanical default)`, unscored runs as no measurement, per-rollout workspace directories, the registered held
families. Waste: the stage machinery around it (recipe kind, controller hook, per-parent receipts and admission
memos, distillation admission queues, promotion gates), the bank `fit` path (no adapter actor), and evaluation sizes
that made one episode take hours on the GB10.

### Minimal design

1. **Rows.** `rl/memetic_tasks.py build`: programs from the GRPO pool (held-v3/v2 closure excluded, memetic held
   families excluded), `family@digest` with at least `support + query` independent worlds (671 programs with 8 or more
   rows), one episode row per program (`family: memetic_improve`, root `improve_program.nl`, world
   `{kind: memetic, task: spec}`). Held programs come from the registered memetic-held packet only and get
   development/confirmation slices for a later held measurement.
2. **Workspace.** `rl/memetic_workspace.py` (restored lean): evidence, typed changes, evaluation by a nested
   collection on the episode's actor (`task_rollouts.prepare` + `run_bonsai_queue.py` + `task_rollouts.score`, as the
   GRPO collector runs episodes), guidance delivered through the program-guidance channel (text written as a native
   state on the actor), `score` on query worlds. Preparation (support baseline, query baselines none/default) is
   computed once per group by its first rollout under a lock, so every sample of a group is anchored on the same
   measurement under the same policy version.
3. **Transport and scoring.** World-bridge kind `memetic` (spawns `memetic_workspace serve`; the program sees
   `workspace`), `task_rollouts` reads `natlang.memetic-outcome/1` from the world score as a graded reward (binary
   success = quality above the anchor).
4. **GRPO.** `--memetic-tasks PATH --memetic-share F`: a fraction F of new groups are memetic rows (group 4). Each such
   group gets its own spec copy in its group directory. The supervisor's calls train as ordinary per-call GRPO.
5. **Later.** Query runs of committed guidance as guidance-consumer SFT data (with pass-through: the guidance state's
   producer trains through the consumer), held-program measurement (development gain vs the start policy), and
   deployment bindings of committed guidance.

**Cost.** Lean sizes: support 4 worlds x 1 sample, query 6 worlds x 2 samples, 2 evaluations per episode. Per group of
4: preparation 4 + 12 + 12 = 28 rollouts, per episode 2 x 4 + 12 = 20, so 108 inner rollouts, about 13 normal groups.
At one memetic group per ~50 ordinary groups (F = 0.02) that is about +25% collection. F = 0.01 gives ~+12%. Episode wall
time is 1-2 h (fits under `--max-staleness 2`).

**Dependencies.** None on pass-through for the RL loop. Inner rollouts share the training actor.

**Entry.** `--memetic-tasks PATH --memetic-share 0.01`, off by default until one real episode completes on the
pinned stack; then in a bundle.

## 4. Self-distillation of verified own successes

### What exists

`sft_sources.py` extracts every `task_success` episode from GRPO archives (`rl-own`), replays it under the current
runtime and oracle, and builds capped, deduplicated, held-excluded per-call rows for `exp0_sft.py`. sft-v3-run1 trains
on them (44% of its weighted target). The old `rollout_distillation.py` (expert iteration into SFT windows with
per-family shares) is replaced by this.

### What is missing

1. **Interleaving into GRPO.** A self-imitation term inside the GRPO step: CE on recent verified successes from
   groups with low success (at most 2/8), from a replay buffer of the run's own last W policy versions, lowered with
   exact actor ids (no replay needed within one pin), weight `--sd-weight` (rare successes already get a positive GRPO
   advantage; the CE term keeps them after the group leaves the pass band). Truncated ratio against the stale actor.
2. **On-policy recency.** `sft_sources` is a batch build between experiments. The in-loop buffer is current by
   construction; between experiments `extract` keeps covering the newest archives.
3. **Teacher-attempt successes.** `sft_sources extract` and `lower_rollout_turns` take students only. Teacher
   successes (section 1) should enter in student form, with the privileged segments dropped as in the learner.
4. **Neuralese whole-graph rows** (`sft_sources neuralese-graph`, sibling) once pass-through lands.

**Cost.** At most 16 replayed episodes per step: about +5% learner.

**Dependencies.** Items 1 and 3 touch the learner's batch assembly and loss, which the pass-through work is changing
now. They are implemented after it lands. Item 3's student-form transform is the same function as section 1's.

**Entry.** `--sd-weight 0.1 --sd-window 4` (off until implemented), in a bundle.

## 5. Inventory of components touched

| Component | JOLT+ | Hindsight | Memetic | Self-distillation |
|---|---|---|---|---|
| `scripts/neuralese_data/grpo.py` Collector (`_score`, `_record_outcome`, `_new_group`) | teacher phase, students-only pass rates | tau* in teacher controls | per-group spec, share | - |
| `grpo.py` `load_group`, step assembly | student form, guided advantages | - | - | replay buffer, CE term (later) |
| `rl/jolt.py` collection helpers | fact-kind filter, `guided_advantages`, `student_form` | `trajectory` in `guidance_controls` | - | `student_form` reused |
| `rl/privileged_context.py` | - | writer function `writeHint` accepted | - | - |
| `serve/http.py` (`x_natlang_guidance_policy`) | - | `writeHint` accepted | - | - |
| `serve/sglang_actor.py`, `serve/engine.py` | none: guidance insertion is shared through `guidance_plan` | none (same) | none | none |
| Managed llama-server, in-page wllama | not applicable: training guidance is only requested by training collectors with `--neuralese-server` (`teacher/cli.ts` 110-111); the runtime refuses it elsewhere | same | program-guidance channel already on every engine | none |
| `ts-host/src/native/training-guidance.ts`, `runtime/node-guidance-writer.ts` | - | `writeHint` mode, run `trajectory` | - | - |
| `applications/guidance-writer` | (writeGuidance unchanged) | `writeHint.nl` returns string | - | - |
| `applications/memetic-improver` | - | - | unchanged program | - |
| `rl/memetic_workspace.py`, `rl/memetic_tasks.py` (new) | - | - | restored lean | - |
| `ts-host/src/teacher/world-bridge.ts`, `collector.ts` | - | - | kind `memetic` | - |
| `rl/task_rollouts.py` | (scoring of `jolt_role` already live) | - | `memetic_outcome` | - |
| `rl/hindsight.py` (new) | - | `trajectory_value`, index | - | - |
| `lower_rollout_turns.py`, `lfm3_fast_train.py`, `exp0_sft.py` | none (student form is applied to the row before lowering) | phase 1 via pass-through | none | after pass-through |
| `sft_sources.py` | teacher successes in student form (later) | tau* source | - | teacher role (later) |
| `rl/held_eval.py`, `behaviour_probe.py` | none: no guidance at evaluation; the probe selects plain calls | none | held-program eval (later) | none |

## 6. How they enter a run

All four are flags on the pinned GRPO loop, bundled by expected merit (engineering, not attribution). Recommended
next bundle (grpo-v5), after each has run once end to end on the pin: `--teacher-attempts 4`
`--hindsight-trajectories INDEX` (`--teacher-fact-kinds` default), `--memetic-tasks PATH --memetic-share 0.01`; with
pass-through landed, `--hindsight-reconstruction 4` and `--sd-weight 0.1`. Expected step cost of the bundle: about
+20% wall time (JOLT+/hindsight ~6%, memetic ~12%, SD ~2-3%). Paired held eval against the pin's base judges the bundle;
a regression is bisected by branch states with the flags off.

## 7. Status

Implemented in this round (sections 1-3, pass-through independent parts): see DECISIONS.md, entry "2026-11-17:
JOLT+, hindsight writer and memetic rows rebuilt on the per-call stack", and FALLOW_REGISTER rows P3, P5, P6.

2026-11-18: hint check, top-level telemetry, memetic support 4 with measured rates and the re-smoke launcher
(`ops/restore-smoke/resmoke.sh`); DECISIONS entry "2026-11-18 00:30, restore fixes for grpo-v4's resume".
