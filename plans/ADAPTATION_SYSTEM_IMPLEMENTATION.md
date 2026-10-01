# Complete TypeScript adaptation system

Status: implementation plan, 2026-09-29. No implementation is implied by the APIs below.

This plan supersedes the Python-backend recommendation in [the research report](../docs/PROGRAM_OPTIMIZATION_RESEARCH.md). Deliver one integrated release: a natlang-owned adaptation system using a focused vendored adaptation of Ax GEPA. Implementation proceeds in dependency order, but stopping after the single-function prototype does not satisfy this plan.

## 1. Release contract

A developer can define an evaluation suite for an existing natlang program, select its trainable components, optimize them through the real runtime, interrupt/resume the search, inspect evidence and instruction changes, and deploy or roll back a portable artifact. Everything added for adaptation is TypeScript. There is no Python optimizer, Ax runtime, Ax model client, or network dependency at artifact-load time.

The release includes:

- Named `.nl` instruction optimization, preserving frontmatter and callable scope.
- Authored inline `nl` instruction optimization, including interpolation and live captures, in application TypeScript and callable-folder TypeScript.
- Single-component, selected multi-component, and whole-program instruction optimization against local or end-to-end metrics.
- Per-program system guidance as a selectable component, including guidance-only and joint searches.
- A native TypeScript GEPA engine adapted from Ax, plus a small reflection hill-climber for comparison and debugging.
- Evaluation fixtures, independent scoring, component feedback, budgets, cancellation, caching, checkpoints, and reports.
- Versioned artifacts, strict compatibility validation, explicit activation, source-patch export, and rollback.
- Node SDK/CLI optimization; Node and browser artifact consumption.
- Packaged examples, documentation, automated integration tests, and a recorded live-model evaluation.

Not required for this release: optimizing model weights, executing the optimizer in a browser, automatic online learning, structural rewriting of TypeScript or helper topology, automatic capture/signature changes, independently persisted runtime-generated children, or demonstration search. These require separate semantics and are not necessary for a complete instruction/guidance adaptation lifecycle. The artifact format reserves no unimplemented executable features; unsupported component kinds fail validation.

## 2. Decisions fixed by this plan

1. **Own the public abstractions.** No `AxProgrammable`, `AxGen`, Ax signature, or Ax artifact type crosses a natlang API boundary.
2. **Vendor a bounded source subset.** Adapt its search implementation, not its provider/runtime/framework infrastructure.
3. **Preserve authored source.** Search operates on immutable candidate overlays and private fixture state. Source export produces a reviewable patch; it never silently overwrites the program.
4. **One effective program view.** Execution, callable discovery, `read_code`, source inspection, and traces use the same selected definitions.
5. **Resolve at task scope.** No process-global current candidate or mutable instruction callback. Nested calls inherit the task's selection.
6. **Separate editable text from contracts.** Types, capture accessors/mutability, slot expressions, capabilities, and helper topology remain fixed.
7. **Evaluate actual natlang execution.** Never fall back to an Ax prediction path or score replayed model outputs as fresh inference.
8. **Keep deployment lightweight.** Runtime modules import artifact validation/resolution only; optimizer/evaluation modules remain separate entry points.
9. **Explicit model identity.** Production artifact binding checks the executor identity/configuration it was evaluated with. Custom drivers supply a caller-declared identity; identities are compatibility metadata, not proof of provider immutability.
10. **No promised improvement.** Baseline is always a candidate. A completed search may correctly retain it and produce a no-improvement report.

## 3. Code and package layout

Keep implementation in the existing `ts-host` build to reuse its compiler/runtime without packaging a second runtime instance. Expose optimization through a lazy Node subpath, not the root barrel. This is module-level separation in this release; `@natlang/node` contains the code on disk, but normal inference and browser bundles do not import it. A separate npm distribution can be introduced later without changing these interfaces.

```text
ts-host/src/adaptation/       # platform-neutral runtime support
  types.ts                   # descriptors, templates, artifacts, binding types
  schema.ts                  # strict parsing, diagnostics, schema versions
  identity.ts                # canonical hashes, component/build/model fingerprints
  compatibility.ts           # artifact/program/model contract checks
  program-view.ts            # effective definitions and task-local source revisions
  prompts.ts                 # pure prompt-layer composition
  index.ts
ts-host/src/evaluation/       # Node orchestration, neutral data contracts
  types.ts, suite.ts, runner.ts, worker.ts, worker-entry.ts
  fixtures.ts, metrics.ts, feedback.ts, usage.ts, cache.ts
  oracles.ts                 # reusable checks extracted from teacher tooling
  report.ts, index.ts
ts-host/src/optimization/
  types.ts, optimize.ts, budget.ts, proposer.ts
  checkpoint.ts, run-store.ts, promotion.ts, export-patch.ts
  strategies/gepa.ts, strategies/reflection.ts
  vendor/ax-gepa/            # adapted engine, selection, dependencies, Pareto helpers
  index.ts
ts-host/src/cli/adaptation.ts
spec/adaptation-artifact.schema.json
spec/adaptation-run.schema.json
vendor/ax-gepa/              # pristine reference subset, provenance, license, notices
examples/adaptation/         # pure, compositional, inline, and stateful suites
docs/ADAPTATION.md
```

Add `@natlang/node/adaptation`, `/evaluation`, and `/optimize` exports and corresponding development-host exports. Browser root exports only portable descriptor/artifact binding APIs. CLI dynamically imports its optimization command implementation. Update npm staging to ship the adapted code's license/notices, keep evaluation free of `teacher/` runtime imports, and exclude optimizer/evaluation declarations from browser publication unless referenced portable types are moved into `adaptation/types.ts`.

## 4. Vendoring and engine extraction

Pin Ax commit `b780a14a3cb94d5ac572db04038399aef655c76c`, the snapshot inspected during research. Fetch raw files at that commit and record hashes; do not silently substitute current main. The starting dependency inventory is:

| Upstream file | Treatment |
|---|---|
| `gepa.ts` | Extract search state, candidate evolution, acceptance, merges, and selection loop |
| `gepaSelection.ts` | Adapt component selection; inject serializable RNG |
| `gepaDependencies.ts` | Adapt update grouping using native component keys |
| `paretoUtils.ts` | Retain relevant selection/frontier utilities with attribution |
| `gepaEvaluation.ts` | Replace runtime-specific paths with one mandatory native evaluator |
| `gepaReflection.ts` | Retain useful proposal logic/constraints; replace Ax generation with `ModelDriver` |
| `gepaComponents.ts`, `gepaAdapter.ts` | Reference for native contracts; remove framework discovery/application |
| `gepaBootstrap.ts` | Exclude demonstration bootstrap from this release |
| Relevant upstream tests | Port algorithm behavior cases and their attribution |

Walk imports before extraction and list every copied helper, including anything taken from `optimizer.ts`. Remove dependencies on `AxBaseOptimizer`, `AxOptimizedProgramImpl`, `ax()`, providers, signatures, telemetry, and direct prediction. Do not retain unused Ax compatibility methods or `any` casts to emulate its program object.

Maintain `vendor/ax-gepa/UPSTREAM.json` with repository, commit, source paths and SHA-256 hashes, copied test origins, license files, and adaptation mapping. Keep pristine files outside the TS compilation root and adapted files inside it. Add `CHANGES.md` explaining deliberate behavior differences. Ship Apache-2.0 license text and applicable notices in npm artifacts; annotate modified source files.

Behavior changes required immediately: explicit per-case errors, no execution fallback, immutable candidates, complete budget accounting, injected seeded randomness, stable tie-breaking, abort propagation, serializable checkpoints, and rejection of malformed evaluator/proposal output. Retain upstream selection/merge behavior where compatible and record where natlang differs. Describe the result as an Ax-derived GEPA implementation, not proven equivalent to Python GEPA.

Build deterministic scripted tests against the pinned original algorithm's relevant pure operations before changing behavior. Compare selected parents/components, frontier membership, merge decisions, and accepted candidates under fixed scores. Integration-specific differences get explicit tests rather than a misleading blanket parity claim.

Reference sources: [Ax engine](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepa.ts), [evaluation](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepaEvaluation.ts), [component contracts](https://github.com/ax-llm/ax/blob/b780a14a3cb94d5ac572db04038399aef655c76c/src/ax/dsp/optimizers/gepaComponents.ts).

## 5. Component contracts and identity

Use these public concepts (exact names may change together before release; semantics may not):

```ts
type ComponentKind = 'lambda.instructions' | 'program.guidance';
type ComponentKey = string;

type InstructionTemplate = {
  segments: readonly string[]; // N+1 static strings around N original slots
  slotIds: readonly string[]; // expressions/values are not optimizer-owned
};

type ComponentValue =
  | { kind: 'lambda.instructions'; template: InstructionTemplate }
  | { kind: 'program.guidance'; text: string };

type Candidate = Readonly<Record<ComponentKey, ComponentValue>>;

interface ComponentDescriptor {
  key: ComponentKey;
  kind: ComponentKind;
  origin: 'named' | 'authored-inline' | 'program';
  definitionId?: string;
  source?: { path: string; start: number; end: number };
  baseline: ComponentValue;
  baselineHash: string;
  contractHash: string;
  contract: FrozenComponentContract;
  constraints: { maxChars: number; requiredBindings: readonly string[] };
}
```

The engine sees an encoded `Record<ComponentKey, string>` through a codec: plain text for guidance and slot-free bodies, strict JSON template segments for interpolated instructions. Decoding checks the exact known key set, kind, segment count, slot identities, size limits, and full compile-time contracts. No arbitrary slot-marker parsing and no interpolation of optimizer-generated JavaScript.

Logical keys are qualified by program/package ownership and normalized path. Named example: `triage::classify.nl::instructions`. Inline example: `triage::main.ts::site:urgency::instructions`. Program guidance: `triage::program.guidance`. Encodings must escape delimiters or use canonical tuple serialization internally. Do not match by basename, display name, or line number alone.

Support an optional `/* @natlangSite urgency */` annotation immediately attached to an `nl` tagged expression. Define its exact comment attachment grammar in the spec; reject duplicate/ambiguous labels. Unannotated sites get exact-build keys based on original source identity/span. Labels make reports and explicit rebinding stable; they do not bypass compatibility checks. Keep optimization identity separate from recursion guards and existing `iterateOn` site identity.

`FrozenComponentContract` contains ordered parameter/return types, type aliases, subtype, open-parameter status, capture names/types/mutability, ordered original slot expression hashes, callable/helper identities, service declaration/scope hashes, and compiler/runtime protocol versions. Track both a contract hash and complete behavior/build fingerprint: unchanged types do not make changed helper behavior compatible. Conservative full source/dependency fingerprinting is the default; performance-oriented dependency minimization is not required.

Extend `DefinitionManifest` in `compiler/project.ts` with versioned optimization descriptors. Inventory named helpers recursively and authored inline sites inside callable-folder modules, not only ordinary application source files. Update `compileModule`, lowering, and virtual-project compilation to return/register the same metadata. Record origin explicitly so a compiler invocation on model-generated eval code cannot accidentally produce a persistently trainable authored site.

Anonymous runtime-generated `nl`, `delegate`, Python children, and unregistered `defineNatlang` calls remain executable but nonselectable. They receive trace lineage and eligible program guidance. Optimizing their authored parent can improve them. Inspection must clearly report why they are not persistent components rather than silently ignoring selection requests.

## 6. Artifact, binding, and compatibility

`natlang.adaptation/v1` is data-only JSON. It contains:

- Artifact content digest and schema version; canonical serialization excludes the digest field itself.
- Program identity and source/dependency build fingerprint.
- Selected component descriptors' baseline/contract hashes and exact replacement values.
- The program-guidance target scope, including explicit imported-library inclusion.
- Runtime/compiler/protocol compatibility and executor model/configuration identity.
- Inference policy used during evaluation, including code-edit behavior, budgets, and external model settings that affect results.
- Provenance: strategy/vendor revision, run ID, suite/evaluator/split hashes, seed and budget summary, selection evidence reference, promotion status.

Keep full datasets, model conversations, API keys, endpoint credentials, and large traces out of deployable artifacts. Compact evidence can state score/cost summaries and digests. An artifact is not trusted executable code and cannot add tools, services, filesystem paths, or model configuration overrides.

Provide `parseAdaptation`, `validateAdaptation`, and `bindAdaptation(artifact, program, executorIdentity)`. Binding returns an immutable validated object used by runtime/tasks. Validate all entries, even currently unexecuted ones; reject unknown components, duplicate keys, nonfinite scores, wrong kinds, stale source, capture/slot changes, and incompatible executor identities. Use a parser that detects duplicate JSON object keys rather than silently taking the last value.

Strict compatibility is the default. Reusing an artifact with another source revision or executor requires an explicit `revalidate` workflow: map stable component keys, check contracts, rerun the selection/regression suite, and issue a new artifact with its own provenance. Do not offer a generic production `force` flag. Unannotated changed inline sites require a human-authored mapping or regeneration; never fuzzy-match them automatically.

Provide explicit authored-baseline selection via `adaptation: null`; `undefined` inherits the runtime default. A task-specific binding replaces, rather than merges with, the runtime default. Artifacts are immutable; rollback selects the prior artifact or baseline for new tasks. Active tasks retain their original binding.

Avoid hash cycles: the source/build fingerprint excludes adaptation files, search outputs, and target activation pointers. The encompassing package archive digest still covers deployed artifacts. Version these hash domains and include dependency resolutions/lockfiles where they affect executable behavior.

## 7. Runtime and compiler integration

Modify `runtime/runtime.ts`, `context.ts`, `kernel.ts`, `callable.ts`, `lowered.ts`, `loader.ts`, `modules.ts`, and `hooks.ts` as one coherent change.

Add a `ProgramDescriptor`/program handle and immutable adaptation binding to runtime defaults and `TaskOptions`. Add an optional executor identity beside `ModelConfig`; the launcher supplies it from the resolved model profile, and SDK users with custom drivers supply it explicitly when binding adaptations. Separate search/reflection and judge identities from the executor identity.

At task creation, establish a `ProgramView` over baseline records and the selected artifact. It resolves definitions by original identity, materializes effective named source text, renders effective inline templates, and gives callable trees consistent records. Existing imported callable objects must resolve through the active task at invocation, rather than retaining a stale adapted body from their construction time. Cover direct calls, child properties, `Folder.apply`, `iterateOn`, and bound callbacks.

For inline lambdas, retain original template values and capture accessors. Apply the replacement static segments before interpolation, using existing interpolation/coercion behavior. Never optimize one rendered invocation or reevaluate interpolation expressions; preserve normal JS evaluation timing. An inline callable created before task selection must resolve the active adaptation when invoked. Calls with live/mutable captures must retain the existing accessors and conflict rules.

Validate adapted inline text by projecting it into the original AST and rerunning analysis. Require identical captures, mutability, slots, parameter/return types, and callable policy. Text that unintentionally mentions a new lexical binding is invalid; text that drops the sole mention of an existing capture is invalid. Keep original site identities through an explicit source-span/site mapping when projecting source, rather than recomputing deployment identity from the longer/shorter replacement text.

`read_code` for `.nl` returns effective source. For modules containing adapted inline sites, synthesize effective TypeScript source from AST-aware template replacements and original expressions. Compile the projected module with origin mappings and contract validation; do not concatenate unescaped template text. Share this projection with source export and inspection so they cannot disagree. Browser consumption must support already compiled plans and the callable-folder compilation path; it must not require an on-disk checkout.

Introduce task/program-view-aware callable-folder module instance caching. Cache compiled code by effective source/compiler/dependency digest, but isolate mutable module instances for evaluation fixtures and task-local code revisions. Preserve legacy sharing for no-adaptation production tasks unless an explicit isolation option is requested; document this policy. Top-level application ESM state requires separate evaluation isolation and is not fixed by changing this module cache alone.

Define code-edit semantics now. Base artifact view is immutable. `edit_code` produces a task-local revision on top of it, updates descendants consistently, invalidates affected compilation/module entries, and records original/effective/patched hashes. Existing active calls use their captured revision; calls starting after a committed edit resolve the new revision. Concurrent edits use expected-revision checks and return a conflict rather than losing an update. Contract-changing edits use existing compiler/runtime checks; they cease to inherit a stale component override and are explicitly traced as runtime revisions. They never change the persisted artifact or another task.

Evaluation exposes `codeEdits: 'allow' | 'deny'`, with the production default preserved. Denial must affect actual tool authorization and tool descriptions, not just a prompt sentence. Include the policy in compatibility/evaluation fingerprints. Tests cover both policies; live acceptance must use the intended deployment policy.

Without an adaptation, existing programs retain their current behavior. Bump compiled-output protocol when new required metadata is emitted; accept older compiled modules only through an explicit no-adaptation compatibility path or fail with a rebuild instruction.

## 8. Program system guidance

Refactor `native/prompt.ts` and `native/agent.ts` into explicit composition without changing baseline text unnecessarily:

1. Fixed interpreter protocol, including depth-specific variant.
2. Application-owned system guidance, preserving its current position.
3. Capability-dependent runtime guidance for functions/folders.
4. Selected program-guidance text, if the current call belongs to its declared scope.
5. Existing invocation opening with the effective lambda instructions and scope.

System layers 1–4 remain system text; the invocation opening remains in its existing message role. Preserve the existing baseline ordering, separators, and text when layer 4 is empty. Apply depth-limit rewriting only to the runtime protocol layer. Refactoring prompt composition must not introduce an unmeasured new baseline.

Snapshot adaptation guidance for a task. Resolve legacy dynamic application `systemPrompt` behavior unchanged for unadapted calls; adapted tasks snapshot it at creation and include its digest in their binding context. Reject undeclared prompt drift during evaluation. Compute effective system text once per invocation and reuse it through compaction/retries.

Guidance applies to the program's authored calls and their generated descendants. Imported libraries are excluded unless explicitly included in the program's guidance scope. Dynamic children inherit the nearest owning program; beginning a different program/task establishes a new boundary. A shared runtime can host simultaneous programs with different guidance. Runtime authority and depth limits remain enforced by code.

Register `program.guidance` with an empty or configured baseline, explicit semantic objective, and size limit. Support `lambda-only`, `guidance-only`, and `joint` target selection. Whole-program tests must exercise every affected major component before guidance promotion.

## 9. Evaluation SDK and fixture lifecycle

Add a typed `defineEvaluationSuite()` authoring API. A suite is a trusted local TypeScript module compiled with the existing project toolchain and loaded by Node. It declares:

- Program descriptor and entry fixture module/export.
- Stable case IDs, grouping/slice labels, dataset version/content digest, and explicit train/validation/test membership.
- Fixture factory and teardown, input construction, entry invocation, observable result capture, and independent scoring.
- Local component metrics where available, required regression gates, executor/reflection/judge profiles, and run budgets.
- Selected component keys or validated selectors; selection resolves to a fixed recorded inventory before search.

Use a worker protocol of JSON case IDs, candidate values, model/profile references, seeds, and result/trace references. Do not serialize live closures or handles. Workers import the suite's factory and create the live values themselves. Default to a fresh worker per case replicate so Node module globals, captures, services, and folder state cannot contaminate another candidate. Share model servers and immutable compilation caches outside workers. An opt-in reusable worker requires a declared/tested reset contract.

Proxy executor model requests from workers to a coordinator-owned driver using request IDs and cancellation messages. This supports programmatic custom drivers without serializing functions, centralizes concurrency/usage accounting, and avoids starting a model server for every case. Reflection and judge requests use the same accounting gateway with separate role/model identities. Fixtures remain local to the worker; the coordinator returns model turns, not application handles.

Fixture lifecycle is `create -> execute -> capture -> score -> dispose`, with teardown in `finally` even on cancellation. Capture includes typed return/outcome, folder final state/diff, ordered service effects where order matters, and application-specific observations. Expected blocked/failed cases have their own accepted results. `NatlangCallError` is a task result for scoring, not necessarily a worker infrastructure error.

Register all in-flight child invocations with the evaluation task. Before final observation and worker disposal, await their completion or cancel and drain them under the task budget. An unawaited child must not disappear from usage accounting or mutate fixture state after scoring. This tracking must include generated children and service calls; force-terminated work is recorded as incomplete, not successful.

Construct virtual folders and fake/snapshot-backed services afresh. Recorded service traces are diagnostics, not a complete replay dataset: current effect previews truncate data. Add a fixture observation store when full payloads are required. Exact replay must define unseen calls as a fixture failure; alternative actions generally require a simulator or real read-only fixture service. External mutable effects require an explicitly supplied resettable evaluation backend; ordinary optimization examples use isolated fixtures.

Keep expected outputs and evaluator code outside the model-visible callable/source tree. The worker may know gold answers; the executor must not receive them through arguments, manifests, services, or source tools. Reflection sees training feedback only. Use explicit feedback projections rather than forwarding arbitrary worker objects.

Extract reusable oracle implementations from `teacher/oracle.ts` into `evaluation/oracles.ts` and leave backward-compatible teacher re-exports. Preserve existing oracle tests and behavior. Keep generation-specific dataset rules in teacher modules. Reuse trace contracts in `native/scenario.ts` through an adapter. Avoid a production package importing files excluded by npm staging.

An `EvaluationResult` includes status (`scored`, `candidate-invalid`, `infrastructure-error`, `cancelled`), quality in [0,1], hard gates, metric vector, textual feedback, outcome, component coverage, usage, latency, and trace references. Reject nonfinite values and output-length mismatches. Treat unavailable/uncertain judge decisions explicitly; never turn them into silent passes. Infrastructure failures follow bounded retries and then suspend/fail the run without assigning the candidate a semantic zero.

## 10. Feedback, search, and credit assignment

All strategies use one target interface:

```ts
interface OptimizationTarget {
  components(): readonly ComponentDescriptor[];
  validate(candidate: Candidate): Promise<ValidationResult>;
  evaluate(candidate: Candidate, cases: readonly CaseRef[],
    context: EvaluationContext): Promise<EvaluationBatch>;
  feedback(batch: EvaluationBatch,
    keys: readonly ComponentKey[]): Promise<ReflectiveDataset>;
}
```

Evaluation contexts carry run/evaluation IDs, replicate seed, trace-capture choice, abort signal, and budget reservation. The evaluator is the only execution path for both strategies. Candidate maps are complete over the selected component set; missing/unselected components resolve to the fixed baseline, not to the last candidate evaluated.

The proposer uses the existing `ModelDriver`/model profile infrastructure directly. Its prompt includes component purpose, immutable signature/scope, original objective, current candidate, and a bounded selection of successful/failing training examples with compiler/tool/semantic feedback. Require structured replacement values, validate them, and permit a small configured repair budget. Repairs, failed proposals, and reflection requests consume model budgets. No executor tool privileges are inherited by the proposer.

Build feedback from `InvocationTrace` trees and component keys. Group repeated calls to the same definition; keep call IDs so contradictory examples remain distinguishable. Separate local oracle feedback from downstream program feedback. Include observed inputs through explicit bounded projectors, outputs/outcomes, relevant tool failures, and context needed to preserve the parent criterion. Do not infer that parent/child nesting proves data dependency or causal blame.

Support isolated lambda suites and full-program evaluation with only selected components mutable. Inline sites generally require a harness that exercises their surrounding code; do not invent serialized closures for standalone replay. Dynamic call topology is observed afresh for every candidate.

GEPA search requirements: baseline scoring, seeded component/parent selection, reflective mutation, paired minibatch acceptance, validation-based candidate tracking, per-example frontier selection, compatible-component merges, bounded population/history, and incumbent preservation. Resolve static update dependencies independently from observed call nesting. If a component has no coverage, report it and seek covering training cases before spending mutation budget; report never-covered selections as a coverage gap.

Keep final selection policy separate from search mechanics. Required validation gates must pass; choose mean quality first and use declared cost/latency policy for ties or configured constraints. Preserve per-example frontier diversity during search. A quality/cost frontier can be reported separately. Invalid candidates never replace the incumbent. Include the simple hill-climber using the same proposer, evaluator, and budgets as a baseline, with no alternate execution semantics.

## 11. Splits, budgets, caching, and resumption

Reject duplicate case IDs and group leakage across train/validation/test. Record split manifests and hash the evaluator code/configuration. Training examples feed reflection; validation is used for selection; test data is withheld from both. Freeze the selected candidate before final test evaluation. A test failure is reported, not used to choose a different candidate against the same test set. If developers iterate after inspecting it, record a new development cycle and reserve a fresh holdout for unbiased claims.

Separate search PRNG seeds, case/fixture seeds, and model-request seeds. Derive seeds by stable case/component/replicate IDs, not scheduling order. Record unsupported provider seeding and stochastic limitations. Sort results by scheduled evaluation IDs before updating deterministic search state; changing worker completion order must not change algorithm decisions in scripted runs.

Budget dimensions: case rollouts, candidate proposals, all model requests, input/output tokens, elapsed time, and optional monetary cost. Count nested natlang calls, repairs, judges, and reflections. Reserve capacity before scheduling; stop launching new work when exhausted, drain/cancel as configured, and return the valid incumbent. Record reserved and actual use plus any unavoidable in-flight overrun. A hard dollar cap requires known pricing and bounded requests; unknown usage/pricing must remain unknown rather than zero. Separate finalist/holdout evaluation reservations from search allowance.

Exact cache keys include candidate/program/protocol/model/prompt/suite/fixture/evaluator digests, seed and replicate identity, trace-capture completeness, and execution policy. Do not reuse incomplete records. Independent stochastic replicates get distinct IDs even if the provider does not honor seeds. Changed criteria invalidate scoring caches; cached immutable rollout observations can only be rescored through an explicitly versioned path.

Persist under `.natlang/adaptation/runs/<run-id>/`: immutable run manifest, event log, candidate store, evaluation ledger, traces/blob references, checkpoints, selected artifact, and JSON/Markdown reports. Writes are atomic; use a run lock and content-addressed blobs. Checkpoint the engine version, RNG state, candidate ancestry/population/frontiers, component-selector state, case schedule, scores, pending evaluations, budget ledger, and cache references.

Resume validates all fingerprints and restores a completed decision boundary. Completed evaluations are idempotent by ID. Interrupted evaluations are marked unknown and rerun only from fresh fixtures; possible already-billed model calls stay visible in usage accounting. Checkpoint after accepted/rejected proposal decisions and orderly interrupts. A second interrupt may terminate workers; recovery must still preserve the last committed checkpoint. Resume compatibility is exact for engine/state schema version; unsupported versions fail with an actionable message.

## 12. SDK, CLI, deployment, and source export

Public lifecycle: `defineEvaluationSuite`, `inspectComponents`, `evaluate`, `optimize`, `resumeOptimization`, `loadAdaptation` (Node file helper), `bindAdaptation` (portable), and `exportAdaptationPatch`. Programmatic optimization accepts drivers; CLI resolves named profiles using current model configuration without storing credentials in artifacts.

Implement a suite contract that supports this authoring shape (all shown methods/options are proposed):

```ts
export default defineEvaluationSuite({
  id: 'triage-v1',
  program: { root: '.', id: 'triage', entry: 'main.ts' },
  components: ['triage::classify.nl::instructions'],
  cases: './evaluation/cases.jsonl', // id, group, split, input, expected
  models: { executor: 'local', reflection: 'teacher' },
  budget: { maxRollouts: 300, maxProposals: 20, maxModelCalls: 1500 },
  fixture: {
    async create(testCase, context) {
      const app = await context.loadFreshProgram();
      return {
        async execute() {
          return context.runtime.run(() => app.triage(testCase.input));
        },
        async observe(result) { return { result }; },
        async dispose() { /* close fixture resources */ },
      };
    },
  },
  score(testCase, observation) {
    const ok = labelsMatch(observation.result, testCase.expected);
    return { quality: ok ? 1 : 0, gates: { correct: ok },
      feedback: ok ? 'Correct.' : explainMismatch(observation, testCase) };
  },
});
```

The case loader passes gold answers only to fixture/scoring code, not automatically into the executor. `context.loadFreshProgram()` loads the compiled application in that case's worker; the runtime carries the selected binding. Support outcome observations when `execute` throws, and make `observe`/`dispose` optional with documented defaults. Helpers such as `labelsMatch` are ordinary evaluator code outside the model-visible program tree. Budget numbers here illustrate configuration, not a recommended universal allowance.

CLI surface to implement and document:

```sh
natlang adapt inspect ./project --json
natlang eval ./adaptation.config.ts --split validation --baseline
natlang optimize ./adaptation.config.ts --strategy gepa --out .natlang/adaptation/runs/run1
natlang optimize resume .natlang/adaptation/runs/run1
natlang eval ./adaptation.config.ts --artifact ./adaptations/best.json --split test
natlang adapt inspect ./adaptations/best.json --project ./project
natlang adapt revalidate ./adaptations/best.json --suite ./adaptation.config.ts --out ./adaptations/revalidated.json
natlang adapt export-source ./adaptations/best.json --project ./project --out ./adaptations/best.patch
natlang run ./project --adaptation ./adaptations/best.json
natlang run ./project --no-adaptation
```

Also support `--adaptation`/`--no-adaptation` for relevant `call`/`apply` paths; reject unsupported use on unregistered ad hoc `ask` components with clear diagnostics. `--json` keeps machine-readable stdout; progress goes to stderr. Dry-run/preflight validates inventories, splits, model availability, fixtures, and budgets without paid inference. Define stable exit codes: success, invalid input/compatibility, evaluation regression, infrastructure failure, and interrupted run. A completed no-improvement optimization is successful.

SDK example, with both objects already validated:

```ts
const binding = bindAdaptation(artifact, program, executorIdentity);
const runtime = createNatlangRuntime({ model, program, adaptation: binding });
const result = await runtime.run(() => application(input));
// A second task can explicitly run the authored baseline:
const baseline = await runtime.run(() => application(input), { adaptation: null });
```

Extend `NatlangTarget` and both manifest parsers/schema with optional `adaptation: <relative artifact path>`. CLI overrides target defaults; explicit `--no-adaptation` selects baseline. Validate that packaged references exist and are covered by `include`. Pass binding/program metadata through `TargetContext` and initialize `context.runtime` consistently. Apps creating their own runtimes must receive/use the supplied binding explicitly; do not imply CLI activation can affect unrelated runtimes invisibly. Update bundled application construction paths and document this embedding contract.

Program guidance belongs to the artifact or an explicit authored guidance file, not user-level model configuration. Browser apps load artifact JSON through their own asset pipeline and call the same portable binder with embedded build metadata. No Node filesystem/compiler-server API is required merely to consume an artifact.

Source export uses AST-aware projection for inline sites and frontmatter-preserving body replacement for `.nl`. Escape backticks and literal interpolation sequences correctly, preserve original expression ASTs, then recompile and assert frozen contracts. Emit program guidance to a named guidance file plus explicit integration instructions when no authored guidance hook exists; never pretend it is representable as a lambda-only patch. Emit a manifest of original/new hashes. Export is for review; applying it changes source identity and requires revalidation/rebuilding of adaptation artifacts. Never apply both an exported body and its stale overlay silently.

## 13. Observability and reports

Extend invocation traces with optional adaptation provenance: program/component identity, original definition/build revision, artifact/candidate digest, effective instruction hash, system-guidance hash, runtime-patch revision, executor identity, and evaluation/case/replicate IDs. Record selection once per invocation, not only at task end. Keep old trace readers compatible with absent fields; version any changed required trace semantics.

Record actual model/tool usage for every branch, including failures and dynamic children. Reflection/judge events have distinct roles. Reports show baseline and selected quality, per-case paired changes, slices, outcome/gate counts, coverage, uncertainty from replicates, total execution/search costs, prompt lengths, model identities, search history, and compatibility status. Summaries must distinguish validation selection results from locked-test results.

Store original and effective instructions so users can inspect why behavior changed. Reports distinguish no improvement, regression, insufficient coverage, missing judge decisions, budget exhaustion, cancellation, and infrastructure failure. Generating a report must not depend on asking another model.

## 14. Work breakdown and acceptance gates

These are ordered work packages for one delivery, not separately optional releases.

| Work package | Main files | Completion gate |
|---|---|---|
| A. Contracts and vendor baseline | `adaptation/types/schema/identity`, `spec/*`, `vendor/ax-gepa` | Schemas/examples parse; snapshot hashes/licenses recorded; dependency closure documented |
| B. Compiler inventory | `compiler/inline/project/lower`, `runtime/loader/modules` | Complete authored inventory, labeled/unlabeled identities, slot/capture contracts, origin mapping |
| C. Effective runtime view | `runtime/*`, `adaptation/program-view`, `native/runtime` | All invocation/read/edit paths agree; simultaneous bindings and runtime patches are isolated |
| D. Program guidance | `native/prompt/agent`, task/frame options | Scope/inheritance, depth variants, compaction consistency, no-adaptation regression checks |
| E. Evaluation infrastructure | `evaluation/*`, oracle extraction | Fresh workers/fixtures, local/end-to-end metrics, split isolation, usage and failure semantics |
| F. Native strategies | `optimization/vendor`, strategies/proposer | Scripted algorithm tests, reflection baseline, no Ax framework dependencies |
| G. Durable runs | budget/cache/checkpoint/run-store | Interruption/resume equality with scripted models, ledger correctness, atomic artifact/report writes |
| H. Public workflows | SDK exports, CLI, package manifests/staging | Optimize/evaluate/inspect/revalidate/deploy/rollback/source-export work from an installed package |
| I. Browser and examples | browser entry/build, examples/docs | Same artifact consumed in Node/browser; no optimizer bundled into inference browser assets |
| J. Live acceptance | versioned suite fixtures and result reports | Real deployment-model runs completed; evidence published, failures not replaced by scripted claims |

Before moving past C, resolve stale callable references and effective source projection; these are architecture work, not polish. Before H, do not expose a public CLI that only optimizes an in-memory toy path. Before release, complete all gates even if live optimization finds no quality gain.

## 15. Required verification matrix

| Area | Required meaningful cases |
|---|---|
| Baseline behavior | Existing named/inline/iterate/folder programs with no artifact; older compiled-output handling |
| Inventory/identity | Same names in different packages; nested helpers; inline sites in modules; duplicate labels; moved sites; helper implementation changes |
| Template/captures | Multiple slots, backticks, literal `${...}`, interpolation evaluated once, changed/new/dropped captures, mutable conflicts, created-before-task callables |
| Effective program | `read_code` equals executed source, compiler projection retains origin, child-property and folder calls, existing callable references after edit |
| Isolation | Two artifacts concurrently on one runtime, nested calls, browser await propagation, module globals, fixture teardown, cross-task edit separation |
| Guidance | Lambda-only/guidance-only/joint, imported-library scope, generated children, depth-limit guidance, compaction and callback snapshot behavior |
| Evaluator | Correct/incorrect typed answers, expected blocked outcome, service effects, folder final state, judge uncertainty, malformed metrics, infrastructure outage |
| Search | Baseline retained, paired acceptance, frontier diversity, merge conflicts, invalid proposals, uncovered components, deterministic ordering |
| Budget/resume | Nested request accounting, overrun reporting, crash during evaluation/write, repeat resume, locks, changed fingerprint rejection |
| Artifacts | Round trip, unknown/duplicate keys, tampering, stale source/model, baseline rollback, revalidation creates new provenance |
| Distribution | Installed CLI and SDK, packaged artifact path, declarations resolve, browser loading, vendor notices shipped, no optimizer in browser graph |
| Export | Named/inline patches compile, captures unchanged, guidance export explicit, stale overlay rejected after source changes |

Use the existing `node:test` suite pattern. Add focused `adaptation-*.test.mjs`, `evaluation-*.test.mjs`, and `optimization-*.test.mjs` tests plus CLI/package/browser integration coverage. Run targeted tests while developing, then the repository's full `npm test` (build, application build, browser types, tests, conformance), staged-package checks, and the browser pilot with artifact coverage before release. Keep all existing teacher/oracle tests passing after extraction.

Live acceptance uses three checked-in suites: (1) triage classification with ambiguous/blocked cases; (2) a compositional moderation-style program with local and final metrics; (3) a small folder/stateful workflow with independent state/effect assertions, plus authored inline sites exercised by at least one suite. Use genuine held-out cases grouped by source family, the intended small/local executor, and an explicitly configured reflection model. Do not turn benchmark values into fixed accuracy promises before measurement.

Run baseline, reflection baseline, and GEPA under matched search allowances; report actual spend separately. For guidance, compare lambda-only, guidance-only, and joint on the compositional suite. Repeat baseline/finalist evaluation sufficiently to expose instability; record sample counts and uncertainty. Save/load the selected artifact and repeat representative calls through packaged Node and browser consumers. A lack of improvement is acceptable evidence; silently dropping failing cases or selecting on the locked test set is not.

## 16. Documentation and final definition of done

Update `spec/SPEC.md`, `NATIVE_PACKAGES.md`, `ts-host/README.md`, npm package READMEs, model/profile documentation where identities are added, and the authoring/integration guides. Explain how to define local versus whole-program objectives, preserve capture names, label inline sites, write resettable fixtures, interpret blocked results, deploy artifacts, and handle stale bindings. Link the research report as background and this plan as the implementation contract.

The work is complete only when an ordinary application author can, using documented installed-package APIs:

1. Discover all eligible authored components and understand exclusions.
2. Supply a typed evaluation suite with independent scoring and fresh fixtures.
3. Optimize one lambda, multiple lambdas, or program guidance using the real interpreter.
4. Stop and resume without losing the incumbent, provenance, or budget ledger.
5. Inspect a useful report and reject a result without touching source.
6. Load a validated artifact in Node or a browser and observe it in traces and source inspection.
7. Run concurrent tasks with different artifacts safely and roll back explicitly.
8. Receive clear failures for stale contracts, unsupported models, malformed artifacts, and infrastructure errors.
9. Export a compilable source patch and revalidate after adoption.

All ten work-package gates, the verification matrix, licensing/distribution checks, and recorded live evaluation are part of that completion criterion. No backend selection decision, Python bridge, or Ax framework migration remains open.
