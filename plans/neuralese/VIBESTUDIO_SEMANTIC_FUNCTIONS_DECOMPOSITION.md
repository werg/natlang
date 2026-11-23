# Vibe Studio semantic functions

Application functions own judgment and return application values. Callable folders retain specialist stages and installed skill mounts. Tasks choose a meaningful entry point, inputs and independently observable outcome. Runtime call records carry execution history.

## Mechanism inventory

| Component | Existing mechanism | Required change |
| --- | --- | --- |
| Base skills/open-user-goal | Universal runUserGoal and UserGoalReport union with domain wrappers | Expose existing domain and composite functions directly, with native signatures and hierarchical specialist scope. Remove the universal router from collection entry points. |
| System testing workers/system-test-runner/authored-teacher.ts | Hardcoded root program and root name | Load the task's declared application entry and retain its actual typed result and complete call tree. |
| System testing workers/system-test-runner/index.ts | Per-world service binding and verifier invocation | Bind application entries without prescribing child routes. Pass the actual application result into outcome verification. |
| System testing and domain managed verifiers | Final envelope plus exact call counts, route checks and independent state checks | Express requested outcomes and scope protection separately from execution conventions. Answers remain required when the user asks for them. |
| Host scripts/export-live-workflow-teacher.mjs | Catalog pins runUserGoal and admission repeats route shape checks | Bind declared task entry, typed result and independent evidence, with no required generic report or fixed child sequence. |
| Natlang scripts/neuralese_data/live_teacher_sources.py | Admission hardcodes runUserGoal path and goal-only input | Bind actual declared entry, input hashes, trace identity and application result. Preserve older captures under their recorded program identity. |
| Native disk verifier `ts-host/scripts/observed-workspace-evidence.mjs`, called before rendering by `ts-host/scripts/materialize-native-teacher.mjs:135` | Independent JS admission also hardcoded passed-only, runUserGoal and one goal input, and trusted projected approval flags | Re-read SHA-bound manifests, calls, native argument JSON, partial evidence and original failures. Recompute each decision's eligibility and optional failed root-return exclusion before native materialization. Historical router captures retain their original entry. |
| Native TS projection `ts-host/src/teacher/observed-workspace.ts`, dispatched by `ts-host/src/teacher/native-materializer.ts:2014` | Passed-only outcome and completed-invocation restriction excluded independently proved partial decisions; empty-target evidence was rejected | Admit exactly scope-proved generations in later-failed invocations, preserve unapproved failed children, enforce explicit return exclusions, and retain valid evidence-only rows with zero learner targets. This path consumes observed trajectory IR directly; it never replays a workspace program or requires Program IR. |
| Natlang scripts/neuralese_data/vibestudio_campaign.py | Collection, receipt, deferred native trajectory export | Preserve data, recheck rejected captures with recorded verification provenance, expose meaningful corpus coverage. |
| Runtime engines and learner | Consume regular typed calls and native trajectory IR | No language or engine protocol change. spec/SPEC.md describes typed return values. live_teacher_sources.py converts captured model exchanges into native graph and decisions. Training consumes those decisions without a Vibe Studio report contract. |

## Decomposition

Each existing domain function and composite becomes a public callable with its existing typed return. Specialist lambdas and TypeScript remain colocated below it. Explicit services enforce outside-world scope. The task descriptor is exact plumbing that binds the entry and initial inputs, never instructions encoding a teacher trajectory. The outcome verifier reads state and retained source evidence. Source identity, held separation and immutable receipts remain exact plumbing.

Migration proceeds across the entire task registry. Existing task identities and source-group splits stay intact. Old captures retain original source identities. Reassessment produces an additional attestation and exposes successful subtrees only where evidence supports them. Diversity measurements count observed functions, depth, branching and domain compositions rather than imposed routes.


## Implemented migration and evidence

All 2,327 catalog tasks retain their original identities and goals and resolve to
application entries. The declared entry registry is 116 kinds across 27 compiler
roots; the complete catalog uses 114 entry programs. Native typed results replace
the model-visible universal report. The verifier's remaining historical envelope
is an internal adapter only. Existing traces retain their actual original entry.

Shared provider mechanisms permit authorized rereads, extra discovery and
concurrent identical requests. Outcome and scope checks remain independently
attested; none requires the model to follow a particular child-call tree.
Optional incorrect root returns are retained as observations and removed from
supervision. Partial recovery admits only explicitly evidence-bound original
ordinals, including independently valid actions in later-failed invocations.

Runtime authority review: ambient .d.ts inclusion only changes compiler type
roots. ProgramLoader still narrows explicit service bindings; NativeSession still
filters services per actual frame scope. No engine, serving or learner protocol
change is needed because all consume ordinary typed calls and trajectory IR.


The independent JS materializer is a separate data-admission implementation,
not a runtime engine. Its disk verifier authenticates originals before the pure
TS renderer projects learner targets. Cross-language tests generate native IR
through the Python loader and run the actual JS CLI for typed multiple inputs,
partial scopes, failed invocations, optional incorrect returns, evidence-only
captures and forged approvals. An archived Luna capture exercises the same path
offline when its manifest is supplied. Ordinary serving engines consume the
resulting model calls; they do not interpret the Vibe Studio task descriptor.

Other direct calls to the pure materializer keep their existing source dialect:
`ts-host/scripts/rewrite-student-trajectories.mjs:68`,
`project-student-trajectories.mjs:71` and `rewrite-student-chunks.mjs:34`
require an exact `task.program_ir` and pinned source closure before rendering;
observed workspace trajectories have no such Program IR and cannot enter those
paths. Static reference and recurrence producers construct their own Program IR
rows (`inline-curriculum/replay-demonstrations.mjs:38` and
`verify-recurrence-pool.mjs:22`). These do not consume workspace entry descriptors
or partial live receipts. The actual live Python training and reasoning-fill
paths use `sft_sources.candidate_turns` and `filled_reasoning.Renderer.chains`,
which re-read original captures and project the same decision masks.

Cohort export retains the historical `verified` row count for existing readers,
while separately recording `completed_goals`, `partial_trajectories`,
`supervised_decisions` and `evidence_only` in its marker and registry metadata.
A recovered partial remains an unfinished goal even when its checked decisions
are usable training targets.

Additive completed-outcome reassessment uses the existing offline recovery command (`scripts/neuralese_data/recheck_live_trajectories.py`), alongside partial decision recovery. A later independently checked whole outcome preserves the original failed packet/result and exact model replies; it adds an immutable checker verdict, source bytes, and native witnesses. Both Python admission (`verify_completed_reassessment` in `live_teacher_sources.py`) and the independent JavaScript materializer (`completedReassessment` in `observed-workspace-evidence.mjs`) rehash original capture, verdict, checker sources and witnesses, and bind the exact original goal, application entry and actual result. Native IR marks completed outcomes explicitly and keeps partials distinct. Existing passed captures without reassessment retain their direct native verification path.

The archived wave27 local presentation is a concrete whole-outcome recovery: corrected `checkPresentation` accepts its requested values and grounded optional title; the original pinned native preview implementation returned `visible` only after DOM observation passed at the owned attempt/build and the source revision remained current. The original native receipt's MDX digests, current source/revision and root application value match the corrected rendering. `reassess_local_presentation.mjs` runs this check without a teacher, and recovery produces one completed task with 25 original supervised decisions. Its original negative receipt remains retained. Tests run Python admission through disk JavaScript verification, TypeScript materialization and the native CLI, including altered witness/value/completion negatives. This is reassessment of original native execution evidence, not a rerun of its program or fabrication of a new trajectory.
