# Builtin optimization of natlang programs

Research and architecture proposal, 2026-09-29. This is a source-based assessment, not an optimizer benchmark or implemented API. Local references describe the working tree inspected on this date. External implementation findings refer to the snapshots listed below; confirm published-package exports before adopting them.

**Decision update:** The chosen direction is now entirely TypeScript, using a focused vendored adaptation of Ax GEPA. The [complete implementation plan](../plans/ADAPTATION_SYSTEM_IMPLEMENTATION.md) supersedes this report's initial Python-backend recommendation and staged scope. The comparisons below remain research background.

## Recommendation

Build a small, natlang-owned adaptation and evaluation layer. Use standalone **GEPA** as the initial reference optimizer through a Python-to-Node bridge, and spike **Ax GEPA** as the TypeScript-only alternative. Keep both behind the same interface. Natlang should own execution, component identity, prompt composition, fixtures, scoring, artifact compatibility, and deployment. The optimizer should propose candidates and allocate evaluation budget.

Start with **one authored lambda's instruction body**, holding its signature, captures, callable folder, runtime protocol, model configuration, and evaluation environment fixed. Execute the actual lambda, including its nested calls and tools. Next optimize several authored lambdas against a whole-program metric. Only then introduce shared per-program system guidance, demonstrations, and structural program search.

This makes optimization builtin without requiring programs to be rewritten into DSPy, Ax, or another execution framework. There is no need to implement a textual-autodiff engine first.

## What can be used off the shelf

| Candidate | Relevant capability | Fit for natlang | Decision |
|---|---|---|---|
| Standalone GEPA, Python | Named text components, custom execution/evaluation adapter, trace-based reflection, candidate selection | Closest boundary to an existing interpreter; natlang remains authoritative | First reference backend |
| Ax GEPA, TypeScript | Custom evaluation adapter, component discovery/application, reflective optimization and optimization artifacts | Strongest TypeScript candidate; some Ax program scaffolding remains | Spike alongside GEPA; likely option for a Node-only distribution |
| Classical DSPy, Python | BootstrapFewShot, MIPROv2, GEPA and other optimization methods | Excellent algorithm baseline; ordinary optimizers expect DSPy predictors/signatures | Use selectively, not as the natlang runtime |
| ruvnet/dspy.ts | Bootstrap, instruction/demo search, GEPA-style search | Its optimizer paths construct its own prediction modules; significant adapter work | Do not adopt as the foundation |
| AdalFlow, Python | Trainable parameters, textual gradients, few-shot optimization, training orchestration | Useful credit-assignment ideas; custom graph/backward integration needed | Later experimental backend or design reference |
| TextGrad, Python | Text variables, textual feedback propagation, gradient-based text updates | Useful proposal mechanism; does not supply natlang's evaluation/deployment architecture | Later proposer experiment |

**Standalone GEPA.** Its adapter accepts a candidate map of component names to text and returns outputs, per-example scores, and optional trajectories. `make_reflective_dataset` turns those trajectories into feedback for selected components; proposal logic can also be customized. This lets a Node worker execute ordinary natlang programs while Python handles search. Per-example program failures should produce scores and diagnostics; infrastructure failures need a distinct failure path. [GEPA adapter contract](https://gepa-ai.github.io/gepa/api/core/GEPAAdapter/), [source](https://github.com/gepa-ai/gepa/blob/d771eb21b5dd3228bc3f567293d2ccfc423fc900/src/gepa/core/adapter.py).

GEPA also has a higher-level `optimize_anything` API. It is worth considering for a single text component, but the explicit adapter is a clearer starting point for natlang's component-specific traces and eventual multi-component program optimization. This is an architectural preference, not a measured performance difference. [API source](https://github.com/gepa-ai/gepa/blob/d771eb21b5dd3228bc3f567293d2ccfc423fc900/src/gepa/optimize_anything.py).

**Ax.** Current source exposes `AxGEPAAdapter.evaluate`, `make_reflective_dataset`, and optional `propose_new_texts`. `AxGEPA.compile` accepts `gepaAdapter` in its options but still takes an `AxProgrammable`. A natlang facade can enumerate components and route execution to our evaluator; wrapping each lambda in `AxGen` is unnecessary. Validate the actual released types/exports in a small integration spike: some options in the inspected implementation are accessed through casts. [Adapter](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepaAdapter.ts), [compile implementation](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepa.ts).

Ax's component metadata includes constraints, dependencies, preserved strings, length limits, and validators. That maps well to frozen natlang contracts. Its framework also requires signature, usage, and other program methods, so this is a facade rather than a completely dependency-free optimizer call. [Component mapping](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepaComponents.ts), [program interfaces](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/types.ts).

One important integration behavior: the inspected evaluator catches adapter failures and falls back to direct program evaluation. The facade must route that path through the same natlang evaluator, or we should adapt the backend to fail closed. Never let an adapter failure silently switch the system being measured. Candidate state must also be isolated across concurrent evaluations. [Evaluation source](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepaEvaluation.ts).

**DSPy.** BootstrapFewShot is a useful low-cost baseline; MIPROv2 searches instructions and demonstrations jointly and uses Optuna's TPE sampler. DSPy's GEPA integration discovers named predictors and extracts their signature instructions. A generic Python wrapper around a natlang call does not automatically expose its internal lambdas as tunable predictors. We could implement that facade, but standalone GEPA avoids it for the first use case. Later, MIPROv2 is especially relevant if instruction-only optimization stalls and verified demonstrations help. [Optimizer overview](https://github.com/stanfordnlp/dspy/blob/9c900c7de0a3cc3114c23fe8202ebe48e2206ce1/docs/docs/learn/optimization/optimizers.md), [MIPROv2 source](https://github.com/stanfordnlp/dspy/blob/9c900c7de0a3cc3114c23fe8202ebe48e2206ce1/dspy/teleprompt/mipro_optimizer_v2.py), [GEPA integration](https://github.com/stanfordnlp/dspy/blob/9c900c7de0a3cc3114c23fe8202ebe48e2206ce1/dspy/teleprompt/gepa/gepa.py).

**dspy.ts.** Do not infer parity from optimizer names. The inspected MIPROv2 samples instruction/demo combinations with a seeded RNG, rejects a raw Pipeline without a signature, and constructs an `OptimizedModule`. Its GEPA path also expects a signature-bearing module. The README explicitly lists a Bayesian surrogate as remaining work. It may be useful for simple declarative JS prediction tasks, but adapting it to natlang would mean replacing substantial execution assumptions without an obvious advantage over Ax. [MIPROv2 implementation](https://github.com/ruvnet/dspy.ts/blob/535d0e41901fb291a8af8b4131fd46af456829d4/src/optimize/miprov2.ts), [GEPA implementation](https://github.com/ruvnet/dspy.ts/blob/535d0e41901fb291a8af8b4131fd46af456829d4/src/optimize/gepa.ts), [documented limitations](https://github.com/ruvnet/dspy.ts/blob/535d0e41901fb291a8af8b4131fd46af456829d4/README.md).

**AdalFlow.** Its main optimization architecture uses `Parameter`, `Generator`, `AdalComponent`, and `Trainer`; textual gradient optimization is relevant when downstream failures need to improve upstream instructions. Natlang would need adapters that represent executed dependencies and translate semantic feedback through tools and ordinary TypeScript. Start by borrowing structured feedback and proposal/acceptance concepts. [Project overview](https://github.com/SylphAI-Inc/AdalFlow/blob/810de99d86191b3aa0c939aa6d6d1a21977555aa/README.md), [textual optimizer](https://github.com/SylphAI-Inc/AdalFlow/blob/810de99d86191b3aa0c939aa6d6d1a21977555aa/adalflow/adalflow/optim/text_grad/tgd_optimizer.py).

The newer `adalflow.optim.optimize_anything` should not be mistaken for that full machinery: the inspected implementation mutates strings by appending hints, prepending objectives, swapping lines, or normalizing spaces, plus line crossover. It does not call a reflection LLM in its mutation function. Its benchmark describes itself as a scaffold. This observation applies to that API, not the whole AdalFlow library. [Implementation](https://github.com/SylphAI-Inc/AdalFlow/blob/810de99d86191b3aa0c939aa6d6d1a21977555aa/adalflow/adalflow/optim/optimize_anything.py), [benchmark description](https://github.com/SylphAI-Inc/AdalFlow/blob/810de99d86191b3aa0c939aa6d6d1a21977555aa/benchmarks/optimize_anything/README.md).

**TextGrad.** Its textual gradient descent supports constraints and gradient history, with a momentum variant. A lambda's instructions could be a trainable variable and its execution a custom operation. But a scalar final score alone is not a backward rule for a multi-tool natlang execution: we still need meaningful feedback and dependency attribution. Try its update strategy behind the same evaluator later; there is little reason to make its variable graph the core runtime representation. Textual gradients are model-generated critiques, not mathematical derivatives or correctness guarantees. [Project](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/README.md), [optimizer](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/textgrad/optimizer/optimizer.py).

The inspected repositories report MIT licensing for DSPy, GEPA, dspy.ts, AdalFlow, and TextGrad, and Apache-2.0 for Ax. Preserve the relevant license/notice files if adapting source; audit the selected package's dependency tree separately.

## Existing architecture and exact integration points

| Existing implementation | What it provides | Proposed change |
|---|---|---|
| `runtime/loader.ts`: `NatlangRecord`, `parseNatlang` | Named source ID, revision, typed frontmatter, instruction body, callable tree | Produce a component descriptor; overlay bodies without changing the frontmatter |
| `compiler/inline.ts`: `InlineLambdaPlan` | Source span, revision-sensitive definition ID, template segments, typed parameters/returns/captures | Emit an adaptation manifest and optional durable site identity |
| `runtime/lowered.ts`: `inline`, `planDefinition` | Turns plans into definitions; interpolates template values | Apply validated template adaptation before interpolation; preserve live capture accessors |
| `runtime/kernel.ts`: `invokeDefinition`, `definitionNode` | Shared entry for named/inline calls, child frames, outcomes and traces | Resolve and record the effective adaptation before node construction |
| `runtime/runtime.ts`: runtime/task options | Task propagation, services, traces, model; runtime-wide extra `systemPrompt` | Add immutable task-level adaptation/program context |
| `native/agent.ts`, `native/prompt.ts` | Protocol prompt, depth/tool/folder-specific guidance | Compose explicit immutable prompt layers |
| `runtime/virtual-project.ts` | In-memory source loading and compilation | Candidate evaluation without modifying working files |
| `teacher/oracle.ts`, `native/scenario.ts` | Semantic/structural/file oracles and trace contracts | Extract reusable evaluator interfaces into an evaluation package |
| `native/trace.ts`, `native/effects.ts` | Trace reconstruction and recorded service effects | Add adaptation provenance and explicit fixture capture |

Paths above are under `ts-host/src/`. The authoring guide in `skills/natlang-authoring/SKILL.md` supplies useful proposal constraints, but is not itself an evaluation objective.

Several details matter:

* `definitionNode` already accepts `options.instructions`; the kernel is a natural resolution point. A kernel-only string override is insufficient for interpolated templates and coherent `read_code` behavior.
* Named IDs are source-path-based. Inline IDs hash revision/path/span and are explicitly stable only within a source revision. Do not persist cross-revision adaptations against those IDs alone.
* Captures are inferred from exact identifier mentions in instruction literals and interpolation expressions. Rewriting inline source can add/remove a capture even when its prose sounds equivalent.
* `NatlangTask.systemPrompt()` appends runtime guidance to `TOOLS_PROMPT`. This already supports extra system text, but not a versioned program-specific component or per-task override. The callback can be consulted again during a call; snapshot candidate guidance for optimization runs.
* `read_code` returns `record.text`, and `edit_code` reparses and replaces callable records. An overlay must make the effective source visible and define how subsequent runtime edits interact with it.
* Service effects are executed, not rolled back by the trace recorder; effect previews are truncated at 400 characters. Existing traces are not complete fixture snapshots.
* Callable-folder modules execute once per source revision. A fresh task alone is not a guarantee of fresh module state; use fixture factories and fresh module instances or worker processes.

## The natlang-owned adaptation model

Treat optimization output as a **versioned build artifact**. Retain authored source as the readable baseline. Explicitly exporting an optimized source patch can be a later action; candidate search should not rewrite the checkout.

Separate four identities:

1. Logical component key: for example `triage/classify:instructions`.
2. Source definition ID/revision: the exact compiler/loader definition being executed.
3. Contract fingerprint: signature, captures and mutability, template slots, callable/service scope, relevant helper/dependency revisions, and runtime/compiler protocol.
4. Candidate/artifact digest: the actual adaptations selected for a run.

For named functions, derive logical identity from package/program identity plus normalized source path. For authored inline functions, support an explicit compiler-recognized site label when durable identity is needed. An AST-derived fallback can be tied to the exact build. A matching label does not authorize reuse after an incompatible contract change.

Store the baseline instruction hash, effective text/template, component key, contract fingerprint, source/build fingerprint, target model and inference configuration, evaluator/dataset versions, split hashes, search seed/budget, parent candidate, results, and runtime compatibility in the artifact. Include quantization/adapter identity for local models where relevant. Reject stale or incompatible artifacts explicitly. An optional fallback to authored source must be visible in the trace.

The first component kind should be `lambda.instructions`. Later add `program.guidance`, then verified demonstrations. Keep signatures, captures, tools, service authority, helper source, and runtime protocol out of the initial search space. Prompt-only changes cannot guarantee semantic equivalence; fixed contracts and independent tests are still required.

For inline templates, optimize the static template, never one rendered invocation containing user data. Preserve slot count, order, and expressions initially. Either restrict v1 to interpolation-free templates or use structured template segments with exact slot validation. Preserve the original capture set for runtime overlays. When exporting source changes, recompile and compare captures, mutability, signatures, and helper scope before acceptance. A later explicit capture declaration could decouple lexical scope from prose, but is not required for the named-lambda MVP.

Runtime-generated `nl`/Python/delegate calls are a different category: their text and identities depend on execution. Initially let them run and contribute to the parent's score and trajectory; do not persist independent adaptations for them. If repeated generated children prove important, promote them to authored helpers or design an explicit dynamic-family identity later.

## Prompt composition and whole-program scope

Use separate layers in a documented order:

```
runtime protocol + capability/depth-specific runtime guidance
application-owned system guidance
selected program adaptation guidance
lambda instruction template rendered with invocation values
invocation inputs, scope opening, tool observations
```

These are conceptual layers: the first three are system guidance; lambda instructions remain in the invocation opening. Application/runtime authority stays enforced in code; text ordering is not a security boundary. Avoid letting an optimizer replace the interpreter's tool protocol or type checks.

Add `programId`/`adaptation` to task options, with an optional runtime default. Freeze the resolved bundle at task creation and inherit it through existing frames, including `iterateOn` and descendants. Define program boundaries explicitly: one runtime may host several apps, and imported library lambdas should not acquire unrelated program guidance accidentally. An artifact should declare its target closure; do not select instructions by basename.

Prefer an immutable per-task effective program view shared by invocation, callable loading, `read_code`, and tracing. If `edit_code` is allowed, use a task-local copy-on-write revision and record it as a runtime patch layered over the candidate. Never reapply the original overlay over a subsequent edit silently. For prompt-only evaluations, begin with fixtures that do not require code editing, or expose a clear evaluation policy that disables it; later evaluate under the production editing policy explicitly.

Program guidance should be a separately selected component with a token budget, not a global mutable callback. Evaluate its effect across all affected lambdas. Optimizing a shared prompt for one easy function can degrade every other function in the program.

## Evaluation boundary and optimizer interface

Proposed interfaces, not existing APIs:

```ts
type Candidate = Readonly<Record<string, string>>;

interface OptimizationTarget {
  components(): readonly ComponentDescriptor[];
  validate(candidate: Candidate): ValidationResult;
  evaluate(candidate: Candidate, cases: readonly CaseRef[],
    options: { captureTraces: boolean; seed: number }): Promise<EvaluationBatch>;
  feedback(batch: EvaluationBatch, componentKeys: readonly string[]): ReflectiveDataset;
}
```

Component descriptors supply editable text and frozen contract context. Evaluation records include per-case score, acceptance gates, semantic diagnostics, typed output/outcome, trace reference, component/call attribution, tool/model usage, and latency. Keep rich structured records internally; translate to each optimizer's wire types in its adapter.

For standalone GEPA, use a long-lived Node worker with framed JSON messages over stdio. Python sends candidate maps and case IDs; the worker constructs fixtures, runs the exact natlang implementation, and returns structured evaluations. Stdout is protocol-only, diagnostics go to stderr. Candidate evaluation must load no live JS closures across the bridge: fixture factories remain in Node. Large traces can be referenced by content digest. Python is an optional optimization dependency, never an inference dependency.

For Ax, implement a small program facade plus its GEPA adapter against the same target. Ensure both adapter and fallback `forward` use the same natlang semantics. Convert Ax's best component map into the natlang artifact; do not make Ax serialization the deployment format. If its release/API coupling proves awkward, extract or port the search implementation into a development-only package rather than importing Ax into the runtime.

An evaluation case should construct fresh arguments, captures, services, folder state, and module state. Pure JSON inputs can be stored directly; handles, live objects, mutable captures, and external services require fixture factories or explicit snapshot/restore implementations. Evaluate folder reducers on fresh virtual copies. A prompt change can choose different tools, so service replay needs defined behavior for unseen calls; fuzzy replay of old model outputs cannot measure a new prompt's quality.

Reuse the current oracles, but move generic evaluation types out of teacher-data generation. Exact/structural/file checks should lead where possible; semantic judges should have fixed independent rubrics and include an uncertain/review outcome. Format/type success is a gate, not proof of task success. Expected blocked/failed cases should score according to the task contract, not automatically as zero. Distinguish candidate failures from outages/configuration errors.

Maintain separate training, selection-validation, and locked test splits. Group related cases by document, source task, or scenario family to avoid leakage. The optimizer may see training feedback; validation used repeatedly for selection is not an unbiased final estimate. Keep test labels out of proposer context. Run baseline and finalist on matched cases, repeat stochastic evaluations, and report uncertainty plus per-slice regressions.

Track rollout count, all model calls including reflection/judges, input/output tokens, elapsed time, and monetary cost when available. Start with semantic quality and hard correctness gates; use cost as a tie-breaker or constrained objective. GEPA's per-example Pareto selection and a quality-versus-cost deployment frontier are different concepts. A short prompt can cause more tool turns and cost more overall.

Cache only under the full effective program, model/config, fixture, evaluator, and seed identity. A cache accelerates repeated identical evaluations; it does not substitute for independent stochastic replicates. Bound concurrency and support cancellation/checkpointing outside the production runtime.

## Per-lambda first, then the whole program

There are two useful evaluation modes for one component. Isolated cases give precise feedback when the lambda has a local contract. Whole-program cases vary just that lambda while measuring the application's outcome. Use both when possible: an isolated improvement can still harm downstream behavior. Whole-program optimization does not require changing program structure; a fixed TypeScript program with several trainable instruction components already qualifies.

Start component selection from actual invocation traces. Group repeated calls by authored definition; aggregate their feedback rather than treating each invocation as a separate parameter. Parent/child call IDs show nesting, but do not establish complete data provenance through host TypeScript. For the first multi-component optimizer, let end-to-end evaluation decide acceptance and use traces to suggest where to edit. Do not claim automatic causal credit assignment. Later add explicit data dependencies or controlled ablations before introducing textual backpropagation.

Verified demonstrations are a separate adaptation kind. A successful program trace does not prove every intermediate answer was correct. Admit demonstrations only under suitable local checks, render them through natlang's own invocation format, enforce context budgets, and re-evaluate them on the actual small/local interpreter. Teacher-generated exemplars and student-model deployment must not be conflated.

Changing helper decomposition, TypeScript orchestration, parameter types, or service access is structural program optimization. It needs source patches, recompilation, broader regression tests, and artifact migration. Preserve an interface that can eventually support this, but do not couple it to v1 prompt replacement.

## Implementation sequence and decision gates

**1. A minimal live experiment with existing machinery.** Select a named `.nl` function such as `examples/triage/classify.nl` or `is_urgent.nl`. Load its complete source context through `loadVirtualNatlang`, replace only its body in memory, and execute fixture cases through the normal runtime with the intended deployment model. Compare authored baseline, a simple failure-reflection hill-climber, and standalone GEPA under matched budgets. No production runtime refactor is required to establish whether optimization helps.

**2. Native adaptation infrastructure.** Add component descriptors, contract fingerprints, immutable effective program views, trace provenance, and artifact loading. Start named functions, then authored inline sites with slot/capture checks. Add task-level program context now even if program guidance is not yet an optimization target. Expose tooling such as `natlang optimize`, `natlang eval`, and `natlang inspect --adaptation` only after their underlying APIs exist.

**3. Choose the search backend empirically.** Spike Ax against the same fixtures. Compare held-out quality, total cost, repeatability, adapter complexity, dependency footprint, resume/cancellation behavior, and failure semantics. Prefer Ax if its published API works cleanly and Node-only installation matters. Keep standalone GEPA if fidelity and low adapter burden win. Write a native search engine only if measured integration/distribution needs justify maintaining candidate selection, sampling, merges, and checkpoints ourselves.

**4. Multi-lambda program optimization.** Use a classification case, a compositional helper case such as moderation, and a stateful/folder case with independent final-state/effect checks. Add coordinate/component updates and whole-program acceptance, then program-guidance ablations: lambda-only, guidance-only, and joint. Validate that shared guidance helps the program rather than moving failure between functions.

**5. Further adaptation.** Experiment with demonstrated traces, AdalFlow/TextGrad-style proposal feedback, model-specific profiles, and distillation into the existing training pipeline. Keep optimizer-training data distinct from locked tests and record the exact prompt artifact in exported training trajectories.

Required infrastructure checks should cover: no-artifact behavior, named and inline resolution, unchanged capture/slot contracts, `read_code` coherence, nested inheritance, simultaneous tasks with different candidates, stale artifact rejection, cancellation, failed calls, and runtime patch provenance. Include a save/load parity check and a Node/browser artifact-consumption check. Semantic gains require live-model evaluations; scripted drivers validate wiring only.

The initial deliverable should be an evaluated artifact and a repeatable report, not an unsupported claim that one optimizer is universally best. The largest work natlang must own is the reliable adaptation/evaluation boundary; once that exists, search algorithms are replaceable.

## Source snapshots inspected

| Repository | Commit |
|---|---|
| ax-llm/ax | `b780a14a3cb94d5ac572db04038399aef655c76c` |
| ruvnet/dspy.ts | `535d0e41901fb291a8af8b4131fd46af456829d4` |
| SylphAI-Inc/AdalFlow | `810de99d86191b3aa0c939aa6d6d1a21977555aa` |
| zou-group/textgrad | `75e912e210864b61999781778cdf756d4468120f` |
| gepa-ai/gepa | `d771eb21b5dd3228bc3f567293d2ccfc423fc900` |
| stanfordnlp/dspy | `9c900c7de0a3cc3114c23fe8202ebe48e2206ce1` |

No dependencies were installed and no paid model evaluations were run for this research. No inference implementation was changed.
