# Portable evaluation fixtures in canonical teacher IR (2026-11-17)

The semantic program remains the original typed natural-language functions and their real callable folders. A fixture
is an outside-world service and oracle; neither its source nor selected-case labels enter model-owned files. Teacher
collection must execute a `natlang.program/2` row, rather than build a second trajectory format around a direct suite run.

## Shared mechanism inventory

- `teacher/program.ts` describes source/input IR; `teacher/collector.ts` validates, executes, records real model
  exchanges and writes canonical program rows. Extend this dispatch, preserving `nativeJobRunner`/`programRow` output.
- `evaluation/suite.ts`, `runner.ts`, `worker.ts`, `worker-entry.ts` already compile/execute isolated fixtures and their
  original scorers. Reuse them. `FixtureContext.programRoot` supplies original-shaped host asset paths, while the
  compiled entry comes only from the isolated model project.
- `teacher/runtime-capabilities.ts` executes probes, so advertise the new contract only after a real fixture/root/scorer
  execution. `training/neuralese/natlang_neuralese/common/task_roots.py`, `scripts/neuralese_data/teacher_gen.py`,
  `training/neuralese/natlang_neuralese/rl/task_rollouts.py` and `scripts/neuralese_data/sft_sources.py` consume the
  canonical task and rollout shapes; preserve actual fixture inputs and use shared replay/admission (parallel strand).
- The managed llama-server, browser wllama, native Python and SGLang transports consume unchanged ModelTurnRequest/
  ModelTurn contracts through ModelDriver (`runtime/driver.ts`). This adds no serving field/cache/sampling feature:
  selection/materialization and scoring live outside every engine. Hosted teacher demonstrations deliberately expose
  generation without the optional scoring-only `decide` capability; finite returns generate JSON on the scored
  student's exact short decision prompt, as detailed below.
- Ordinary CLI builds and nonfixture program kinds keep their existing frontend. Live module-valued call stores and
  native guidance/state are refused by this worker frontend: structured-clone workers cannot transport those live
  objects. No fabricated native sampling receipts or decision scores are substituted.

## Decomposition

1. Original `.nl` root/helpers: unchanged semantic functions, named explicitly in `semantics.files`.
2. Authored TypeScript entry helpers: retained exact source when the real application entry is a typed branch or
   wrapper; `root_role: semantic-anchor` does not claim the anchor executed.
3. Original fixture/world services: exact host source/assets in `evaluation_fixture.host_files`, with only the selected
   complete case. They remain outside the model file tree. Existing score/observe functions remain the oracle.
4. Portable task materialization/import binding: crisp filesystem/compiler plumbing. No generated NL forwarding root.
5. Input binding verifier: crisp equality over declared JSON inputs at the actual invoked export path, before dispatch;
   service argument names designate host-constructed live handles. This is an exact verifier, not semantic policy.
6. Trace/result adapter: crisp serialization into existing ProgramRun, preserving actual invocation IDs, source
   definition ledger, original outcome, score/gates, observations and constrained-oracle strength. A typed zero-model
   branch has no fabricated model trace; training admission rejects zero-target episodes separately.
7. Independent semantic review, replay and split custody: existing admission outside execution, never a new hidden
   evaluator judge request. Canonical replay must re-execute the same task IR and compare original scorer evidence.

The descriptor schema is `natlang.evaluation-fixture/1`, on outer kind `evaluation_fixture` so collectors without this
frontend reject explicitly. Its paths use workspace-relative host source/assets, model-relative `program_entry` and
`entry.module`, an actual `entry.export_path`, and positional `input_arguments`/`service_inputs`. The executed fixture
must call that binding with the same JSON inputs; unchanged source traces must bind every model invocation.

Targeted checks cover original host assets, original scorer, actual argument mismatch before model execution,
provenance, source boundaries, finite generation without scoring, explicit settings and typed zero-model branches.
The full repository gate runs on an immutable combined commit after these changes and shared compiler fixes settle.

## Decision reply-surface correction

Inventory: `native/agent.ts:decisionReadout` constructs the scored messages; `native/decision.ts` lists finite values;
`runtime/kernel.ts` propagates scored distributions only from phase `scored`; `calls/recorder.ts` counts actual model
request end events; `native/model-recording.ts` requires generation receipts to link to a generation graph node without
`readout: decision`. `contracts.ts`, runtime model options, capture workers and native teacher trajectory recording
all consume the existing ModelTurn generation contract. Training `model-call-kinds.json` now records the generated
choice surface; canonical `task_rollouts`/`sft_sources` retain its actual plain JSON reply. No engine-specific wire
changes: managed llama-server, wllama, Python native and SGLang already generate tool-free chat requests.

The prior no-scorer fallback taught tool calls and a long tool prompt, while the scored student executes a short JSON
choice prompt. The default now generates one JSON choice on the exact encoded scored messages. Exact finite-set
validation, generation recording and budget reservations are crisp verifiers/plumbing. There are no synthetic
probabilities or decision receipts. A real generation graph node is marked as generation; separate decision evidence
marks phase `generated`. `decisionGeneration: tool-loop` is an explicit diagnostic opt-out. Existing multi-model
handoff/fallback configuration still owns an ordinary loop, preserving its established handoff semantics; fixture
teacher collection uses no fallback chain. Native scoring implementations and their probability semantics do not change.
