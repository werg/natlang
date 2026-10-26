# Independently designed NatLang applications

The 100 programs in [`codebases/design_briefs`](../../../codebases/design_briefs/)
implement different application briefs. Each has its own native application API,
domain representations, reasoning decomposition, control flow, and operational
recovery. The [application index](../../../codebases/design_briefs/INDEX.md)
links each program's public operations, source, design decisions, and evaluation
cases. `application.json` declares the actual native signatures; it is not a
universal semantic request or response contract.

## Design requirement: independent approaches

The initial release imposed a common five-stage wrapper and repeated
preparation, decision, construction and audit helpers. That structure overrode
the differences between the applications. The current release removes that
requirement and replaces those implementations with domain-specific reasoning.
Renaming a shared sequence, adding unused helpers, or attaching a review summary
after an action does not provide a different application architecture.

Examples of the resulting distinctions include conditional commercial package
search, receipt and liability reconciliation, device event journals, causal
investigation of negative experiments, misconception testing with hidden learner
state, temporal identity migration, whole-family dataset withdrawal, and narrative
knowledge tracking. Their intermediate values mean different things. Their branch
conditions decide which child procedures execute and which results can be used.
A call-tree report describes potential source calls, never observed execution.

Semantic interpretation, precedence, causal analysis, alternatives, search
strategy, review and recovery decisions belong in `.nl` functions. Crisp code is
restricted to exact mechanisms such as integer arithmetic, interval checking,
source hashing, type validation, storage and transport. There is no required
number of functions, common workflow shape, or minimum communication-edge count.
The original brief's five algorithms and five operational requirements are
coverage obligations, not a prescribed five-step execution sequence.

## Running an application

Every program is directly callable through the existing native runtime. The
optional [application service](../../../applications/design-briefs-host/README.md)
adds an authenticated HTTP interface and SQLite persistence without putting its
transport implementation into model-visible program sources:

```sh
npm ci
npm run build:node
node applications/design-briefs-host/main.mjs describe 020
node applications/design-briefs-host/main.mjs serve /path/to/deployment.json
```

`describe` reports that program's named operations, inputs, types and outputs.
Supply its own native inputs when creating or invoking an application case. The
service executes real native source with a configured model. It does not load
fixture answers or replace domain reasoning with a crisp implementation.

Applications own any carried state. A declared `stateParameter`, optional nested
`stateInputPath`, and `stateResultPath` bind a native checkpoint to the next
invocation. The service provides tenant isolation, per-operation role checks,
immutable invocation receipts, optimistic versions, durable idempotency, audit history and
transactional outbox storage. Only a configured application-specific adapter can
serialize domain intentions for delivery. Actual receiver receipts determine
outcomes; ambiguous sends require evidenced reconciliation before resending.
The host no longer requires the same proposal, second-person approval and commit
sequence for every program. Domain approvals, reviews and interruptions belong
to each application's own Natlang operations and saved domain state.
See the service documentation for configuration, routes and the receiver contract.

These are runnable source applications with a tested custody and transport
implementation. Production qualification still needs execution evidence for the
chosen model, installed domain integrations and operational environment. A
compiler result or scripted runtime test cannot establish that qualification.

## Original contracts and independent cases

`source-manifest.json` retains the original 100 brief hashes and supplied source
provenance. Each `brief.json` preserves its algorithms, acceptance gates and
operational details. `domain-contracts.json` retains all 600 original domain
records and 3,032 fields, including enumerations and exact Decimal, Ratio,
Quantity and Money representations. `operational-contracts.json` retains all
500 concrete operational requirements.

Each program's `application.json.coverage` identifies actual source procedures
responsible for these requirements. The verifier checks their existence and
identity; semantic adequacy still requires source review and real scenarios.
There is no inference that mentioning a requirement proves successful execution.

Independent `cases.json` files remain outside model-visible source. They contain
three exact qualification tasks and five operational workflows per application:

| Task | Evaluation | Use |
| --- | --- | --- |
| Supported, contradicted and insufficient-evidence domain gates | Exact `pass`, `fail` or `needs-review` answer | Finite task checks and later RL preparation |
| Original operational workflows | Independent semantic rubric and view, durable-state and effect assertions | Full-program execution and independently reviewed trajectories |

Migration to application-specific APIs preserves original expected answers,
oracles, observations and source records. The release's fixture migration audit
compares them with the original PR commit. Input containers and root paths may
change; independent evaluation truth is not rewritten to fit a new implementation.
A judged result stays unresolved without the independent judge, even when it
parses or sounds plausible.

## Freeze and verify

```sh
node ts-host/test/design-brief-applications.test.mjs
node ts-host/scripts/design-briefs/check.mjs --report runs/design-briefs-check.json
node ts-host/scripts/design-briefs/build.mjs --out data/teacher/design-briefs-v3
```

The output directory must be new. The compiler, service and freezer read exactly
`application.json.source_files`: native `.nl`, non-generated `.ts`, and the root
`natlang.json` package boundary. Retired sources, generated declarations, fixture
answers and deployment code cannot silently enter a frozen program. Paths are
confined and symlinks rejected. Native `uses` imports retain their package scope.

Checks cover the real compiler and loader, public signatures, positional call
arity, helper scope, checkpoint compatibility, original record contracts,
requirement ownership, typed inputs, exact expected values, source hashes,
fixture identity and split closure. Source references form a potential call graph
for review. They do not prove that a model called a child or used its result.

The freezer creates six `natlang.program/2` JSONL packets: `train`, `dev` and
`test`, each separated into exact `gates` and judged `workflows`. It also writes
an inventory and SHA-256 manifest. Whole applications are held out: in each
five-application domain, the first three are TRAIN, the fourth development and
the fifth locked evaluation (60 / 20 / 20 applications). Every later variant
keeps its original application source group. This prevents application-family
split leakage; it does not prove that conceptual motifs never overlap.

## Collection preparation and admission

```sh
NATLANG_PYTHON=python3 node ts-host/scripts/design-briefs/prepare.mjs \
  --packet data/teacher/design-briefs-v3 \
  --out runs/design-briefs-prepared-v3
```

Preparation verifies packet hashes and invokes the existing
`natlang_neuralese.rl.task_rollouts prepare` with `--root-mode student`. It writes
four distinct seeded requests per exact TRAIN gate and two teacher job
descriptions. It makes no model requests and does not resume paused services.
Resolve the teacher and judge endpoint/model bindings to approved deployed
identities before any separately authorized collection. The repository's model
selection and generation policy remain authoritative.

The owner resumed six Luna generation workers on 2026-10-26 and authorized Sol
or Luna for judging. For independent model judging, bind the actual Sol model
separately from the Luna teacher:

```sh
NATLANG_PYTHON=python3 node ts-host/scripts/design-briefs/prepare.mjs \
  --packet data/teacher/design-briefs-v3 --out runs/design-briefs-configured-v3 \
  --teacher-model-id gpt-6-luna --teacher-provider openai-codex \
  --judge-model-id gpt-6-sol --judge-provider openai-codex
```

The reviewed six-slot dispatcher now accepts these native projects without
inventing scripted reference trajectories. Its immutable plan binds
`judge_model: { model_id: "gpt-6-sol", provider: "openai-codex" }`, which reaches
each case's queue and the collector. Judged cases fail preparation without a
judge binding; the judge shares the whole-case model-request budget. Complete
source-bound accepted trajectories still pass the normal verifier and admission
path before either student trains on them.

Real collection must retain complete native journals, actual sampled/forced
producer turns, consumer imports and occurrence bindings. Correct plain-text
execution remains rewardable. Do not fabricate call edges, force communication
traffic or substitute detached state for a missing in-episode producer. Failed
and unresolved executions retain their actual classification. Existing whole-graph
materialization and admission run only after actual execution and review.

The v2 source migration and source review are complete. Deterministic validation
results and review rationale are recorded in [readiness.json](release-v2/readiness.json),
[REVIEW.md](release-v2/REVIEW.md), and the linked compiler, fixture, architecture,
type-pruning, and validation artifacts. The release remains **source-only and
unqualified for live learned-runtime or production deployment**, with
`training_admission: false`. No teacher trajectories, student rollouts, optimizer
updates or learner exposure are claimed. Model-facing measurement debt is recorded
in [`plans/MODEL_FACING_CHANGES.md`](../../../plans/MODEL_FACING_CHANGES.md).

## Release records

[`release-v3`](release-v3/) binds the reviewed source, compiler evidence and
TRAIN/development/test packet. Its [review](release-v3/REVIEW.md) records effect
and spending fixes, collection/judge integration, RL eligibility and validation
limits. Use this source identity for new collection.

[`release-v2`](release-v2/) records the revised inventory, compiler evidence,
fixture migration, architecture review, type pruning, packet hashes and
validation limits. The [readiness record](release-v2/readiness.json) is the
summary; [`REVIEW.md`](release-v2/REVIEW.md) explains the source review and its
limits. Each changed source release gets a new immutable identity and corpus
snapshot.

[`release-v1`](release-v1/) is historical evidence for the original source at
commit `5648b99b73d3b94af382f9e480dd2fe46915e66e`. Its source hashes, old fixture
paths and old registry patch describe that commit. They must not be used to
validate or register the revised source tree. Existing generated trajectories,
if any are collected elsewhere later, retain the exact source identity under
which they were produced.
