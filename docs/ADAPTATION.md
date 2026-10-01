# Adaptation

Natlang has a TypeScript adaptation path for discovering instruction components, evaluating candidate instructions with a real program, running a bounded search, and binding a validated JSON artifact to a runtime. The optimizer is Ax-derived GEPA plus a reflection hill-climber. It is a focused native implementation; no Python GEPA runtime or Ax provider is loaded by inference.

The implementation follows [the adaptation implementation plan](../plans/ADAPTATION_SYSTEM_IMPLEMENTATION.md). It includes authored `.nl` and inline inventory, strict artifacts, task-scoped execution/source overlays, program guidance, fresh-worker evaluation, both strategies, persistent runs, revalidation, CLI activation and rollback, and source-patch export. Staged-package tests exercise the SDK and CLI, and the Chromium pilot consumes the same Node-generated artifact while running concurrent adapted and baseline tasks. The [three example suites](../examples/adaptation/README.md) include recorded local-model failures as well as scored results; they make no accuracy or improvement promise. The [implementation evidence](ADAPTATION_IMPLEMENTATION_EVIDENCE.md) maps the completed plan gates to verification and live findings.

## Portable artifact use

Import portable data contracts and validation from `@natlang/node/adaptation`:

```ts
import { bindAdaptation, loadAdaptation } from '@natlang/node/adaptation';
import { createNatlangRuntime } from '@natlang/node';

const artifact = loadAdaptation('./best.json');
const binding = bindAdaptation(artifact, programDescriptor, executorIdentity);
const runtime = createNatlangRuntime({ model, program: programDescriptor,
  executorIdentity, adaptation: binding });
const answer = await runtime.run(() => app(input));
const baselineAnswer = await runtime.run(() => app(input), { adaptation: null });
```

An artifact contains replacement values and compatibility evidence. It does not contain executable code, provider credentials, or instructions for changing the executor configuration. Binding checks its program and executor identity. A task can choose a separate validated binding or explicitly request the authored baseline with `adaptation: null`.

Configure the runtime with the inference settings and task limits used by the suite. Artifacts check these settings; they do not apply them for you. `executorIdentityForChoice()` produces a sanitized identity for a resolved model profile. Custom drivers declare an identity themselves. Reflection and judge identities are recorded separately. Profile changes, helper edits, authored guidance changes, and dependency lockfile changes require revalidation.

Use `inspectComponents(programDescriptor)` to inspect eligible components. Named `.nl` instructions and authored inline `nl` sites have stable program-qualified identities; optional `@natlangSite` labels make inline identities easier to refer to. Program guidance is selectable as a component. Captures, interpolation slots, types, and helper behavior are compatibility contracts rather than editable instruction text.

Attach a label as `/* @natlangSite urgency */ nl<string>\`...\``. One block annotation belongs to the immediately following tag; malformed or repeated labels in a file fail compilation. Keys escape delimiters, so obtain exact keys from inspection. Generated `nl`, `delegate`, Python children, and unregistered `defineNatlang` functions execute but are not independently persistent components. Generated children inherit their owning program's guidance; imported programs require explicit inclusion in `importedGuidancePrograms`.

## Evaluation and optimization

A suite is an ordinary TypeScript module. Case data may be inline or read from a JSONL path relative to the suite module. The fixture creates a fresh program instance for each case, and the scorer returns an independent quality score and optional gates:

```ts
import { defineEvaluationSuite } from '@natlang/node/evaluation';

export default defineEvaluationSuite({
  id: 'triage-v1',
  program: { root: '.', id: 'triage', entry: 'main.ts' },
  components: ['triage::classify.nl::instructions'],
  cases: './cases.jsonl',
  executorIdentity: { id: 'local-model-v1', configuration: { revision: 1 } },
  budget: { maxRollouts: 100, maxProposals: 12, maxModelCalls: 500 },
  fixture: {
    async create(testCase, context) {
      const app = await context.loadFreshProgram();
      return { async execute() { return context.runtime.run(() => app.main(testCase.input)); } };
    },
  },
  score(testCase, observation) {
    const correct = observation.result === testCase.expected;
    return { quality: correct ? 1 : 0, gates: { correct } };
  },
});
```

Prepare a suite with `prepareEvaluationSuite(suite, moduleURL)` or load one from disk with `loadEvaluationSuite(path)`. `evaluate(prepared, options)` runs selected cases using a driver. `optimize(prepared, options)` runs GEPA by default; `strategy: 'reflection'` selects the simple hill-climber. `resumeOptimization(prepared, runDirectory, options)` resumes a compatible persisted run. Separate training and validation cases are required; test cases remain locked until the candidate is selected. The optimizer writes checkpoints, evaluation records, and reports under `.natlang/adaptation/runs/` by default.

Scorers receive observations and execution outcomes. Keep expected answers in fixture/scorer code and do not pass them into the model-visible program unless that is the evaluation objective. Fixture teardown is optional. Search records usage where the driver reports it; unknown provider cost or token counts remain unknown.

Use `observe()` to project live handles into finite JSON. The default observation has `{ result, outcome }`, omitting an undefined result. A blocked call is a scored execution outcome, not an automatic infrastructure failure. `context.observations.put(key, json)` captures full service/effect payloads independently of truncated interpreter previews. `context.judge(request)` uses a separately supplied `judge` driver and the same accounting gateway; unavailable or uncertain judge decisions fail their gate. Feedback projectors must also return finite JSON and receive training results only.

Infrastructure errors expose the failed case, retry attempt and incurred ledger through `EvaluationInfrastructureError`. You may also pass an explicit `UsageGateway` into `evaluate()` to persist usage after success, cancellation or failure. A failed model request counts as a request and leaves its billed tokens/cost unknown.

`components` accepts exact keys, `lambda-only`, `guidance-only`, or `joint`. Local metric vectors and the final quality objective can coexist. Changed whole-program guidance cannot become the incumbent without validation coverage of the program's authored lambdas. Never-covered selections appear in the report. Optional `selection` declares quality-tie policy (`baseline`, `modelCalls`, `latency`, or `cost`) and cost/latency/request constraints; unknown cost cannot satisfy a cost constraint.

Budgets count rollouts, proposals, and every actual executor/reflection/judge request, including failed requests and repairs. Token limits stop new work after exhaustion and report unavoidable overrun. Configure `requestBounds` to reserve token capacity before requests; the driver/provider must guarantee those input/output bounds. A hard monetary cap additionally requires `pricing` in dollars per million input/output tokens, applicable to every configured role. Output requests are bounded to the declared maximum; unknown billed usage stops further capped work. Rollouts and at least one model request per test case are protected for the holdout. SDK `holdoutReservation` can reserve larger request/token/cost allowances before search.

Runs use a process lock, atomic durable files, content-addressed evidence, exact cache identities, and integrity-checked checkpoints. Resume restores the last complete decision boundary; requests outstanding at a crash retain unknown incurred usage. Cancellation or incomplete holdout writes an artifact marked `incumbent` and a report without marking promotion complete. A frozen final selection writes `selected.json` after its required locked test completes; failed test gates prevent promotion and do not choose another candidate. JSON and Markdown reports distinguish validation, locked test, coverage gaps, failures, actual usage, and original/selected instructions.

`maxPopulation` bounds search candidates (default 16), and `maxHistory` bounds checkpoint/report history (default 1000); older events remain in the durable event log. Custom model drivers receive an optional second `AbortSignal` argument and should forward it to their transport. The managed local/provider sessions and CLI forward that signal. Cancellation still records unknown incurred usage when a provider cannot report the interrupted request.

Fresh evaluation workers isolate top-level ESM globals. Callable-folder module instances are also task-local with an adaptation, explicit `isolateModules`, or a runtime revision. Compiled module code is shared through a bounded source/contract cache. Unadapted production tasks retain legacy module sharing. Allowed `edit_code` changes only a task's program view; expected revisions prevent lost updates, and trace/tool evidence records source hashes. Edits are compiled before commitment. Retained inline callables resolve revised static text while preserving their original slots and live capture cells; a changed contract requires recreating the callable. Deny code edits for fixed-instruction evaluation. Declare service signatures/scopes through `buildProject({ services })` and `suite.services`; adapted tasks reject undeclared capability additions or changed declarations. An additive application `systemPrompt` is snapshotted for adapted tasks and requires its evaluated `systemPromptHash`; use program guidance for trainable text.

## Review, revalidation, and source export

Use `loadAdaptation(path)` to read an artifact. `revalidateAdaptation(old, preparedSuite, driver, mapping?)` explicitly checks an old artifact against a newly prepared suite and emits a newly attributed artifact only after validation gates pass. Component mappings are caller supplied; identity is not guessed from names or line positions.

`exportAdaptationPatch(artifact, program, executorIdentity, projectRoot)` returns a reviewable source patch and original/updated hashes. Inspect and test that patch before adopting it. Changing source changes the program fingerprint, so the old overlay must be revalidated after source adoption.

Revalidation executes paired authored-baseline and candidate validation through one budget ledger, checks contracts, selection constraints and regression gates, and emits new provenance. Changed program guidance also requires validation coverage of every authored lambda. Explicit mappings must be one-to-one. Source export escapes static template text, preserves original interpolation expressions and frontmatter, recompiles projections, and checks contracts. Guidance export includes a separate text file and explicit build integration instructions, including when guidance is cleared to an empty string.

## CLI

```sh
natlang adapt inspect ./project --json
natlang optimize ./project/suite.ts --dry-run
natlang eval ./project/suite.ts --baseline --split validation --json
natlang optimize ./project/suite.ts --strategy gepa --out ./runs/search
natlang optimize resume ./runs/search --suite ./project/suite.ts
natlang adapt revalidate ./old.json --suite ./project/suite.ts --out ./new.json
natlang adapt export-source ./new.json --project ./project --out ./review.patch
natlang run ./project --adaptation ./new.json
natlang run ./project --no-adaptation
```

Preflight checks compilation, inventory, split/budget/profile configuration and every fixture's creation/disposal without executing paid inference. Remote provider reachability and credentials still depend on that provider when execution starts. Progress goes to stderr and command results to JSON stdout. Exit codes are 0 success (including no improvement), 2 invalid input/compatibility, 3 regression, 4 infrastructure failure, and 130 interruption. `call` and `apply` support explicit activation/rollback; unregistered ad hoc `ask` rejects adaptation flags.

## Package targets

A package target may declare `adaptation: "artifacts/best.json"`. The artifact must be included in the package archive. The target context also has optional `program`, `adaptation`, and `executorIdentity` fields for hosts that validate and supply those values. Merely declaring the path does not configure an unrelated runtime constructed by application code; the application must pass the binding into the runtime it uses.

Bundled terminal applications and the triage target use `context.runtime`, so launcher selection reaches their calls. SDK applications creating another runtime must pass the context program, binding and executor identity explicitly. Browser apps import portable binding APIs from `@natlang/browser`, load artifact JSON through their asset pipeline, and supply embedded build metadata and a matching configured driver. Browser inference contains no evaluation workers or optimizer.
