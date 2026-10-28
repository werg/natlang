# Shared guidance writer: privileged context, Neuralese guidance and whole-graph teacher scoring

One natlang writer family serves every user that turns privileged information into guidance for a consumer call
(owner, 2026-10-28): the JOLT+ self-teacher (oracle facts and earlier-attempt feedback), the memetic learner
(development failures as attempt feedback) and the learned optimizer (`fit_hindsight`; its instruction/skill-state
writes, with the mixture choice as a finite decide head). The policy model writes the guidance itself as a
Neuralese producer span, a consumer imports it with consumer markers, and whole-graph RL trains the writer on the
consumer's outcome. Inputs differ per user; the writer, the producer/consumer mechanism, the RL treatment and the
custody are shared. Sources: `TRAINING_INSIGHTS_20261028.md` §7, `LFM3_RECIPE_20261028.md` "RL default: JOLT+".

## Flow of one JOLT+ episode

1. The controller (`rl/jolt.py` `teacher_entries`, `student_guidance`) gives each attempt run-level controls
   `x_natlang_guidance` (`natlang.guidance-run/1`): role (teacher or student), the world's custody receipt and all
   of its facts (teacher attempts add the group's earlier-attempt feedback).
2. The natlang runtime that executes the episode runs, per consumer invocation on its first model turn:
   visibility (crisp), `selectGuidanceFacts` (NL, pluggable) and `writeGuidance` (NL, `Neuralese<string>`,
   template readout), in-process as child calls recorded in the same episode graph. Their model turns carry
   `x_natlang_guidance_policy`.
3. Every model turn of the consumer sends `x_natlang_privileged_context` (`natlang.guidance-request/1`). The
   engine (`privileged_context.guidance_plan`) inserts the declared guidance turn after the leading system turn:
   the header, then the writer's state between consume markers (`import`, teacher). For a student it only records
   the request (`write-only`, the OPSD twin's guidance). If the writer failed, a teacher gets the selected facts as
   text (`text`, with the reason recorded). The receipt goes into the capture.
4. The learner (`rl/jolt.py` `privileged_source`, `add_privileged_twin`) re-verifies the captured guidance and adds
   each episode's whole-graph twin to the packed forward:
   - a teacher attempt's twin is `student_form`, with the guidance turns and their imports removed;
   - a student's twin is `teacher_form`, which imports its write-only state at the declared place.

## Parts

| Part | Form | Where | Why |
| --- | --- | --- | --- |
| Protected world custody (`ProtectedWorlds`) | crisp verifier | `privileged_context.py` (moves to the shared `common/custody.py` when it lands) | Exact identity check against every registered `evaluation-only-never-training` packet; fails closed. Only `split == 'train'` worlds teach. Custody runs before any fact exists. |
| Fact extraction (`case_facts`) | crisp helper | `privileged_context.py` | Mechanical reading of the authored reference, oracle data, decisive evidence, verified same-world trajectories and earlier-attempt feedback. |
| Visibility (`applicable`) | crisp mechanism (exact data rule) | `applications/guidance-writer/visibility.ts` (runtime); `privileged_context.applicable` re-verifies at the engine and the learner | A child sees only its own matched reference value. The root (and a learned-optimizer `binding`) sees world-level facts. Feedback and `fit_hindsight` may be bound to one invocation. This keeps a policy from routing another call's answer. The runtime enforces it and the Python side verifies it. |
| Fact choice and order (`selectGuidanceFacts`) | policy: own NL function, crisp alternative behind `pluggable` (default `nl`) | `applications/guidance-writer/selectGuidanceFacts.nl` | Which facts help this call is a judgment. It also chooses which earlier-attempt feedback to include, from the crisp candidate pool (`group_feedback`: the last 3 scored earlier attempts of the group). Its turns are graph nodes, trained by the teacher outcome. Failure or an invalid answer falls back to crisp, recorded. |
| Guidance writing (`writeGuidance`) | policy: NL function executed by the policy model, `Neuralese<string>` producer span | `applications/guidance-writer/writeGuidance.nl` | The guidance itself is the model's own Neuralese, not a fixed template. It is trained by whole-graph RL through the producer/consumer import. |
| Consumer import (`PrivilegedRendering`, `guidance_plan`) | crisp, hash-bound declaration | `privileged_context.py`, `serve/engine.py` | One ChatML turn per backbone template (header, then consume markers around the state), placed after the leading system turn, else after BOS. Exact positions are needed so the learner can rebase the graph. |
| Text fallback | crisp mechanism, recorded | `guidance_plan` mode `text` | Used only when the writer fails: whole facts are packed in selection order into `budget_tokens`, and the reason is recorded in the receipt and the learner telemetry. |
| Receipt and verification (`captured_block`, `verify_captured_block`, `writer_saw`) | crisp durability and verifier | `privileged_context.py` | Binds mode, rendering hash, selection identity, writer identity, custody, fact ids and hashes, insertion index, token hash and state id. The learner re-verifies custody, every fact, visibility against the graph's role, the rendering, and that the in-graph writer's own prompt carried the facts. |
| Graph edit (`teacher_form`, `student_form`, `writer_span`) | crisp mechanism | `privileged_scoring.py` | Rebases token IDs, producer spans and import boundaries/spans of the actual compiled `NativeReplay`. Inserts or removes a guidance turn together with its import of the in-graph writer. Refuses edits that cross spans or share an import boundary. |
| Teacher scoring (`teacher_scores`, `teacher_role_logps`) | crisp mechanism | `privileged_scoring.py` | Stop-gradient one-pass forward of the edited whole graph, aligned 1:1 with student actions, and teacher-role log-probs with gradients. |
| Loss treatment of policy turns (`policy_turns`) | crisp mechanism | `rl/jolt.py` | Teacher attempts give the selection and writer turns the teacher outcome term (score-centered), with no KL twin. Student episodes give them no loss, since they do not affect the student's outcome; they stay in the forward because the OPSD twin imports the writer state by name. |
| Leakage telemetry (`reproduces_reference`) | crisp telemetry, not a gate | `privileged_context.py`, `summarize_jolt` | Counts hints that copy a reference line or answer verbatim. |
| Teacher attempts per group, and skipping a group | policy: NL function `planTeacherAttempts`, crisp alternative (default count for every group with facts) | `applications/guidance-writer/planTeacherAttempts.nl`, `rl/jolt.py` `plan_teacher_attempts` | Which groups benefit from a privileged attempt (all-correct groups, hopeless worlds, informative failures) is a judgment. The count bound, group identity and budget are checked crisply. It runs in the Python controller through `natlang call` against the actor endpoint, a process boundary because the controller is not a natlang runtime. A failure falls back to crisp, recorded. |
| Curriculum task selection | policy: `applications/curriculum-advisor/selectTrainingTasks.nl`, crisp default | `rl/production.py` `select_tasks` | Unchanged here. Its NL form exists, and its default belongs to the curriculum work. |
| Losses (beta_S, beta_T, reverse KL, score-centered estimator, group centering) | crisp math | `rl/jolt.py`, `rl/policy_math.py` | Exact policy math. |

## Memetic and learned-optimizer users

- Memetic learning authors stored guidance with the same `writeGuidance` from development failures (attempt
  feedback facts of the development worlds). It trains the writer with JOLT+ rounds whose consumer is the
  program's root call. A stored artifact is accepted on untouched confirmation worlds (shared acceptance gate). The
  external teacher-ferry author path is removed.
- The learned optimizer renders its instruction/skill-state writes as `writeGuidance` calls (consumer role
  `binding`; a forced variant teacher-forces the binding text with declared extra slots for contextual
  re-encoding), and reads the mixture choice as a finite decide head over its own learned rows. It keeps its
  cotangent-replay outer step.

## Open

Semantic-codec graphs (`semantic_dependencies`) are refused until the codec executor rebases its source spans.
Model-facing texts (header, `selectGuidanceFacts`, `writeGuidance`, `planTeacherAttempts`) are measured live on
the LFM3 instruct base (`plans/MODEL_FACING_CHANGES.md`).
