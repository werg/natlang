# Native design-brief program corpus

This corpus implements the semantic programs in all 100 supplied application
briefs. The canonical source is in [`codebases/design_briefs`](../../../codebases/design_briefs/).
Each application has its own native root, typed domain records, five domain
stages, nested preparation/decision/construction/audit procedures, and the
cross-cutting evidence, reconciliation, change, and review procedures.

The corpus is **source and evaluation inputs**, prepared for the repository's
existing self-hosted teacher collector and whole-graph RL runner. It contains no
generated teacher replies, sampled student actions, fabricated native-state
receipts, or training-exposure claims. Production connectors, durable services,
web front ends, and operational qualification remain deployment work; the native
programs return typed proposals for those host boundaries.

## Canonical inputs and provenance

`source-manifest.json` binds all 100 original brief paths and SHA-256 digests,
the supplied archive, the expansion-plan attachment, the shared specifications,
and the repository revision used for implementation. Per-application
`brief.json` files retain the domain algorithms, acceptance gates, and five
operational details. Each source package also explains its decomposition.
The [application index](../../../codebases/design_briefs/INDEX.md) maps every
brief to its entry point, cases, domain, and split.

`domain-contracts.json` retains all six original domain record definitions per
brief. The build checks their field names, required fields, array shapes,
enumerations, and exact Decimal, Ratio, Quantity, and Money representations
against the actual native type environment. Each application declares its
reference envelope in `types.ts`; training adapters use either string reference
IDs or the full source binding. Production reference resolution, authorization,
and persistence are host responsibilities.

`operational-contracts.json` retains the five concrete use cases in section 9 of
every original brief. Each workflow binds one of those cases and supplies its
own observable world, independent review rubric, and assertions for the user
view, proposed durable state, and proposed effects. The freezer checks all 500
use-case mappings and rejects copied requirement text in observed evidence.

Each directory here contains `cases.json`. Expected outcomes and judging
rubrics stay outside the model-visible project. The freezer includes only the
native source files and the case's declared input parameters. It uses the real
native loader/type checker, rather than inferring validity from Markdown or
filenames.

There are two kinds of task:

| Task | Scoring | Role |
| --- | --- | --- |
| Concrete domain gate and its contradicted/insufficient-evidence variants | Exact finite `pass`, `fail`, or `needs-review` result | Tractable task execution and RL preparation |
| Complete operational workflow | Independent domain-specific semantic rubric with the original observations and acceptance gates | Full-program teacher generation, review, and later whole-episode learning |

A compiler pass establishes source validity. A typed fixture establishes its
input/result contract. Neither establishes a model's task success. A workflow
without an independent judge remains unresolved in the existing collector; it
does not become a positive example because its output parses.

## Split policy

Splits are assigned to **whole applications** before any teacher generation.
Within each five-application domain, the first three are TRAIN, the fourth is
development, and the fifth is locked evaluation: 60 / 20 / 20 applications.
All counterfactuals, repairs, paraphrases, and workflow scenarios of an
application retain that application source group. Development and test packets
are emitted separately and never enter the prepared RL queue.

This is a conservative application-family holdout. Shared platform patterns are
intentional; the split is not a claim that every conceptual motif is novel in
the held applications. Further world expansion must retain the original
application group and preserve the same split.

## Build and verify

From the repository root, install its locked Node dependencies and build the
native host if needed:

```sh
npm ci
npm --workspace @natlang/typescript-host run build:node
node ts-host/scripts/design-briefs/check.test.mjs
node ts-host/scripts/design-briefs/check.mjs --report runs/design-briefs-check.json
node ts-host/scripts/design-briefs/build.mjs --out data/teacher/design-briefs-v1
```

The output directory must be new. The build emits six `natlang.program/2`
JSONL packets (`train`, `dev`, `test`, each divided into `gates` and
`workflows`), an application inventory, and a SHA-256 manifest. It rejects
missing applications/stages, duplicate IDs, source-hash mismatches, invalid
native signatures, mismatched fixture inputs, invalid exact outputs, incomplete
counterfactual coverage, erased domain contracts, out-of-scope helper calls,
missing decomposition records, and split leakage. Named helper checks inspect
the runtime's resolved callable scope, including nested audit/qualification
children; merely mentioning a sibling function in prose does not make it callable.

The compiler command checks each application in a separate process to bound
compiler-cache memory. Generated `.d.nl.ts` files are excluded from canonical
source and from the frozen projects.

## Prepare collection and RL

```sh
NATLANG_PYTHON=python3 node ts-host/scripts/design-briefs/prepare.mjs \
  --packet data/teacher/design-briefs-v1 \
  --out runs/design-briefs-prepared-v1
```

This verifies the packet hashes, invokes the existing
`natlang_neuralese.rl.task_rollouts prepare` implementation with
`--root-mode student`, and creates four separately seeded rollout requests per
exact TRAIN task. It also writes two teacher job descriptions. It makes no
model requests and does not resume paused training or generation services.

The teacher job descriptions bind the train packet, output namespace, collection
role, root seed, and existing collector entrypoint. Resolve `TEACHER_MODEL_ID`
and `TEACHER_SERVER` to the actual approved self-hosted teacher. The standing
repository selection is Qwen3.6-35B-A3B; use its exact deployed checkpoint,
tokenizer, quantization, runtime, and actor identities, not a guessed public
model-name string. Workflow collection also needs `JUDGE_MODEL_ID` and
`JUDGE_SERVER` for independent review. The default one-worker launch description
is a preparation setting, not a measured capacity recommendation.

For example, after the generation service is resumed through its normal
operational process and those environment variables are bound:

```sh
node ts-host/scripts/teacher-collector.mjs \
  data/teacher/design-briefs-v1/train.gates.ir.jsonl \
  runs/design-briefs-generation/gates/jobs \
  runs/design-briefs-generation/gates/results.jsonl \
  --model-id "$TEACHER_MODEL_ID" --server "$TEACHER_SERVER" \
  --root-seed 20261025 --all --workers 1 --collection-role teacher

node ts-host/scripts/teacher-collector.mjs \
  data/teacher/design-briefs-v1/train.workflows.ir.jsonl \
  runs/design-briefs-generation/workflows/jobs \
  runs/design-briefs-generation/workflows/results.jsonl \
  --model-id "$TEACHER_MODEL_ID" --server "$TEACHER_SERVER" \
  --root-seed 20261025 --all --workers 1 --collection-role teacher \
  --judge-model-id "$JUDGE_MODEL_ID" --judge-server "$JUDGE_SERVER"
```

Collection uses the existing native journals, actor provenance, result checks,
and execution graph capture. Preserve complete trajectories and actual
producer/consumer occurrences. Correct plain-text execution remains rewardable;
do not force communication traffic, count authored nodes as observed edges, or
replace a missing in-episode producer with a detached input.

After real executions, use the existing materializer and native-graph admission
pipeline. Failed and unresolved outputs keep their actual classification.
Never promote source packaging, scripted checks, or a teacher's self-assessment
to trajectory admission. Live model-facing qualification is tracked in
[`plans/MODEL_FACING_CHANGES.md`](../../../plans/MODEL_FACING_CHANGES.md).

## Updating the corpus

Edit the native source and its independent cases together. Retain the original
brief hash, create a new release identity for changed frozen packets, re-run the
compiler and fixture validation, and bind the new source manifest in the corpus
registry. Preserve old generated trajectories with their original identities.
Both LFM and Mellum consume the same task contracts through the shared runner;
no per-model corpus fork is needed.

## Release v1 readiness (2026-10-26)

The frozen `design-briefs-100-native-20261025-v1` source packet covers 100
applications across 20 domains, with 500 domain stages, 4,100 native functions,
600 original domain records and 3,032 fields. Its 800 tasks split by whole
application family into 300 exact gate tasks and 500 workflows requiring
independent judgment: train has 180 gates and 300 workflows, while development
and test each have 60 gates and 100 workflows. The packet inventory and exact
file manifest are recorded in [`release-v1/inventory.json`](release-v1/inventory.json)
and [`release-v1/packet-manifest.json`](release-v1/packet-manifest.json).

All 100 compiler checks passed with source hashes bound, and source/fixture
metadata was verified. The 10 release unit tests passed. The current-base quick
check passed at `2d14506aa343e3a7b8993a0f6b645fae8dec0243`. A separate full-suite
run at earlier base `1b6b4102cf338f6eae6a449e93f17b8e91f48b2d` was red; see
[`release-v1/readiness.json`](release-v1/readiness.json) for its exact counts,
reported failure classes, evidence paths, script/input hashes, and the required
full-suite gate before merge.

Preparation produced 180 student-root TRAIN gate tasks, 720 rollout requests
(four seeds each), and two teacher-job descriptors: one for the 180 exact TRAIN
gates and one for the 300 judged TRAIN workflows. No models were executed. Training admission remains false; live
qualification, production-host readback, teacher and student outputs, optimizer
steps, and whole-graph live trajectories remain absent. This is source-only
readiness, not evidence of training or production capability.

The immutable local snapshot is committed under
[`training/corpus-manifests`](../../corpus-manifests/design-briefs-100-native-20261025-v1.json).
Central registration is pending: the GitHub connector could not complete the
47 MB registry upload, and this workspace has no authenticated Git CLI push.
The exact entry is preserved in [`release-v1/registry-entry.json`](release-v1/registry-entry.json),
with the minimal change in [`release-v1/registry.patch`](release-v1/registry.patch).
Apply that patch to `training/neuralese_corpora.json` on the PR branch, preserve
any newer registry entries, run the quick check, and commit the registry change
before merging. Packet building, source checks, and collection preparation work
independently of this registration step; normal corpus synchronization and
production admission still require it and the later live evidence.
