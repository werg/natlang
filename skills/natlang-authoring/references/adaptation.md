# Adaptation and evaluation

Use the installed `@natlang/node` declarations and CLI help to confirm the APIs below.
Adaptation discovers authored components, evaluates actual natlang execution, and
searches immutable instruction overlays with Ax-derived GEPA or a reflection
hill-climber. Search runs in Node; compatible artifacts can be consumed in Node or
a browser. In a checkout, `docs/ADAPTATION.md` has additional API and deployment detail,
and `examples/adaptation/` contains complete suites and recorded live evidence.

## Choose the editable boundary

Named `.nl` instructions and authored inline `nl` static template segments are
eligible. Program guidance is a separate selectable system layer. Types, frontmatter,
interpolation expressions, capture accessors and mutability, helpers, and capabilities
remain fixed. Improve those structures through normal source edits when needed.
Preserve exact capture names in replacement instructions. Inline interpolation is
evaluated at its original creation point; live captures remain live at invocation.

Inspect component keys with `natlang adapt inspect ./project --json` or
`inspectComponents(programDescriptor)` from `@natlang/node/adaptation`; do not guess
keys from filenames. Label an authored inline site immediately before its tag:

```ts
const decide = /* @natlangSite urgency */ nl<string>`Classify message ${message}.`;
```

Labels must be valid and unique within the file. Runtime-generated children,
`delegate`, and unregistered ad hoc callables are not independently persistent
components. Generated children inherit owning-program guidance; imported programs
need explicit guidance inclusion. Component selection accepts exact keys,
`lambda-only`, `guidance-only`, or `joint`.

## Define independent evidence

Use typed cases grouped by source family into training, validation and locked test
splits. Related variants stay in one split. Keep expected answers and oracle state
outside callable folders and model inputs. Define gates for required behavior and a
quality objective for selection; local helper metrics can coexist with final quality.

```ts
import { defineEvaluationSuite } from '@natlang/node/evaluation';

export default defineEvaluationSuite({
  id: 'triage-v1',
  program: { root: '.', id: 'triage', entry: 'main.ts' },
  components: 'lambda-only',
  cases: './cases.jsonl',
  executorIdentity: { id: 'my-executor', configuration: { revision: 1 } },
  budget: { maxRollouts: 100, maxProposals: 12, maxModelCalls: 500 },
  fixture: {
    async create(testCase, context) {
      const app = await context.loadFreshProgram();
      return {
        async execute() {
          return context.runtime.run(() => app.main(testCase.input));
        },
      };
    },
  },
  score(testCase, observation) {
    const correct = observation.result === testCase.expected;
    return { quality: correct ? 1 : 0, gates: { correct } };
  },
});
```

Each case has a fresh worker and fixture. Use `context.runtime` so candidate selection
reaches actual calls, and create private folders, services and mutable state per case.
Use fixture teardown for resources. Supply `observe()` for state/effect assertions and
project handles into finite JSON; default observations contain `result` and `outcome`,
omitting an undefined result. `context.observations.put(key, json)` retains full effect
payloads; `context.judge()` uses a separately configured, accounted judge. Treat
uncertain judgments as failed gates.

A blocked result can be an expected semantic outcome. Provider outages, worker
failures and malformed scoring are infrastructure errors; do not score them as
incorrect answers. `EvaluationInfrastructureError` exposes the case, attempt and
incurred ledger. An explicit `UsageGateway` passed to `evaluate()` retains usage on
success, failure or cancellation.

If helper behavior is part of the objective, exercise and observe those helpers;
correct final answers alone do not prove local behavior. Changed guidance requires
validation coverage of all authored lambdas before becoming the incumbent. Deny
runtime code edits for fixed-instruction evaluation, or declare the intended deployment
edit policy explicitly and keep it consistent. Record executor/reflection identities,
sampling, seeds, runtime limits and service declarations.

## Search and review

```sh
natlang optimize ./project/suite.ts --dry-run
natlang eval ./project/suite.ts --baseline --split validation --json
natlang optimize ./project/suite.ts --strategy reflection --out ./runs/reflection
natlang optimize ./project/suite.ts --strategy gepa --out ./runs/gepa
natlang optimize resume ./runs/gepa --suite ./project/suite.ts
```

Preflight compiles, inspects configuration and creates/disposes fixtures without
paid inference. SDK entry points are `loadEvaluationSuite`, `prepareEvaluationSuite`
and `evaluate` from `@natlang/node/evaluation`, and `optimize` and
`resumeOptimization` from `@natlang/node/optimize`.

Choose allowances appropriate to the task and authorized model usage. Compare
strategies under matched allowances and report actual spend separately. Request
budgets count executor, reflection and judge calls, including repairs and failures.
Unknown billed tokens or pricing remain unknown. Token/elapsed limits can overrun
on work already running; hard cost caps require guaranteed request bounds and pricing
for every role. Forward the driver's optional `AbortSignal` to its transport.

Inspect JSON/Markdown reports for quality, gates, coverage gaps, local metrics,
instruction changes, usage, failures and holdout status. Resume requires compatible
fingerprints and restores the last committed decision boundary. Preserve ledgers from
interrupted requests. A retained baseline is valid; do not promise improvement.
Repeat fresh baseline/finalist evaluations to report sample counts and uncertainty.
Validation may select a candidate; the locked test cannot trigger reselection.
`selected.json` is written after the required holdout completes and passes gates;
an interrupted `incumbent` artifact is not a completed promotion.

## Adopt or roll back explicitly

```ts
import { loadAdaptation, bindAdaptation } from '@natlang/node/adaptation';
import { createNatlangRuntime } from '@natlang/node';

const artifact = loadAdaptation('./runs/gepa/selected.json');
const adaptation = bindAdaptation(artifact, programDescriptor, executorIdentity);
const runtime = createNatlangRuntime({ model, program: programDescriptor,
  executorIdentity, adaptation });
const selected = await runtime.run(() => app(input));
const baseline = await runtime.run(() => app(input), { adaptation: null });
```

Supply the compiled descriptor and matching model/settings/limits; artifacts check
compatibility and do not configure a model for you. Custom drivers declare an identity;
managed profiles can use `executorIdentityForChoice()` from `@natlang/node/model`.
Concurrent tasks may choose separate bindings. Effective source inspection and traces
follow the same task selection.

CLI activation uses `natlang run ./project --adaptation ./selected.json`; rollback
uses `--no-adaptation`. `call` and `apply` support these flags too. For a browser,
import portable binding APIs from `@natlang/browser`, load artifact JSON through the
asset pipeline, and supply matching embedded build metadata and driver identity.
Use the existing application's runtime; creating an unrelated runtime loses selection.

After source, helper, contract, model or relevant configuration changes, let strict
binding reject stale artifacts. Use `natlang adapt revalidate ./old.json --suite
./project/suite.ts --out ./new.json` for explicit paired baseline/candidate validation
and new provenance. Component mappings must be explicit and one-to-one.
`natlang adapt export-source ./new.json --project ./project --out ./review.patch`
produces a reviewable, compiled patch without overwriting source. Review and test it
before adoption, then revalidate: adopting source changes its fingerprint. Guidance
export includes separate text and explicit build integration instructions.
