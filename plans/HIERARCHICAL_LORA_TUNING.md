# Hierarchical LoRA tuning for natlang programs

Status: complete-scope implementation assignment, 2026-10-26.

## 0. Mandate

Implement this entire plan in one continuous effort. There are no phases,
pilot stopping points, approval gates, go/no-go reviews or deferred work items.
Every section below is required scope: data generation, the joint trainer,
placement, the family level, distillation back into the base, runtime binding,
serving on the reference server, vLLM and the llama.cpp fork, the `natlang
tune` build step, evaluation and documentation.

Work proceeds in dependency order, but no item waits for a measurement to
justify it. Measurements inform tuning choices (ranks, penalties, thresholds,
teacher routing); they never suspend implementation. When something fails,
fix it, rewrite it, recollect the data or rerun the job, and keep progressing
other available work in parallel. Report empirical results honestly, including
where an arm does not beat its baseline, and then improve that arm.

Both students are first-class targets from the start: LFM2.5-350M and Mellum.

## 1. Outcome

Every natlang program gets a build step that specialises the student to that
program. Per-lambda training data comes from teacher runs inside the real
runtime. The result is a tree of small LoRAs that mirrors the program's context
tree. A call runs the base plus the sum of the adapters on its lexical path, so
shared behaviour is trained once and only what is genuinely lambda-specific is
stored per lambda.

The deliverable is a working system:

1. **Data.** Inputs captured by running whole programs, teacher trajectories,
   a verification ladder, on-policy corrections, all stored per lambda.
2. **Training.** One joint run per program trains every level of the tree;
   depth-weighted penalties push shared behaviour up the tree; a placement pass
   keeps only the adapters that pay; a cross-program family level and periodic
   distillation into the base complete the hierarchy.
3. **Runtime and serving.** Adapters are data items in contexts; each call's
   stack follows its lexical path; each stack is exported as one standard LoRA
   and served with batched multi-LoRA on vLLM and the llama.cpp fork.
4. **Lifecycle.** Adapters are build products keyed on source, interface, base
   and data; `natlang tune` builds them incrementally; datasets are the durable
   asset.

The success measure is held-out program-level task success at matched
inference cost.

## 2. Starting point in this repository

| Existing piece | Where | Use in this plan |
| --- | --- | --- |
| Contexts as content-addressed folder values; each call steps down the context graph | `spec/SPEC.md` (Contexts) | The tree adapters attach to; lexical scoping rule |
| Callable folders, `natlang.d/` binding | `spec/SPEC.md` | Folder-level nodes |
| `nnt1` native adapter coefficients, frozen SVD bases, native graph gradients, Adam/SGD continuation | `ts-host/src/neuralese/learning.ts`, native trainer | Training parameterisation |
| `withNativeAdapters` (AsyncLocalStorage scope, nested scopes add up) | `ts-host/src/neuralese/learning.ts` | Replaced by path-resolved stacks for tuned programs; kept as the explicit experiment API |
| Invocation tracing | runtime | Call capture for inner-lambda inputs |
| Source/world connected-component split selector | `natlang_neuralese.rl.memetic_batch` | Splits that keep related cases together |
| Task oracles, `classify_case` / `reward_of` | collector and RL code | Top rung of the verification ladder |
| Self-hosted Qwen3.6-35B, Luna, Bunny teachers | `TEACHER_SETUP.md` | Trajectory generation |
| Per-request LoRA in llama.cpp (no mixed-adapter batching) | `vendor/` fork | Serving, extended with grouping and batched kernels |
| Native-state import refusal for adapted KV | native serving path | Boundary rule for lambdas that pass native states |

## 3. The adapter hierarchy

Adapters attach to nodes of the context tree that the spec already defines.
A call's effective weights are the base plus the sum of the adapters on its
lexical path, plus the family adapters its lambda belongs to.

| Level | Attaches to | Trained on | Initial rank |
| --- | --- | --- | --- |
| Family | A library entry shared across programs (classify, extract, summarise, agentic glue, and every family clustering discovers) | Lambdas of that family in all programs | 16 |
| Tree | A program root, and every subtree root marked by the author or discovered by placement | Every lambda in the tree | 8 |
| Folder | A callable folder or `natlang.d/` node | Lambdas bound to that folder and below | 4 |
| Lambda | A named `.nl` function or an inline `nl` call site | That lambda only | 2–4 |

Initial ranks are tuned automatically by the rank sweep in §5.6.

```text
Base student (frozen)
 + Family adapters (cross-program, mixed by learned scales)
 + Program A root ............................ tree adapter
     natlang.d/ ............................. no adapter after placement
         summarise.nl ....................... lambda adapter
     triage/ ................................ folder adapter
         label-ticket.nl .................... lambda adapter   (family: classify)
         inline nl call ..................... lambda adapter

Call to label-ticket.nl runs on:
  base + classify + Program A + triage/ + label-ticket.nl   → exported as one LoRA
```

**Composition.** Adapters are residual. LoRA deltas add, so a stack is one LoRA
whose rank is the sum of its parts, with a per-adapter scale. Scale 0 removes
an adapter, which gives ablation and rollback without retraining.

**Lexical scoping.** The spec binds each function to its own context, and every
call steps down the context graph. A callee runs under its own path's stack and
never inherits its caller's stack. Explicit `withNativeAdapters` in user code
still appends to the resolved stack, for experiments.

**Shared contexts.** Contexts form a DAG. A folder reachable from two parents
takes its stack from its containing folders in the source tree, so the stack is
deterministic and cacheable however the folder is reached.

**Multi-family membership.** A lambda may belong to several families. Each
membership carries a learned mixing scale, trained jointly with the program's
adapters (§5.4). Membership comes from frontmatter tags and from clustering
(§5.5); both sources are implemented.

**Empty nodes.** Any node may have no adapter. Placement decides which nodes
keep one; an empty node contributes nothing to the stacks below it.

## 4. Data generation

Inner lambdas get their inputs by running whole programs, never by inventing
arguments from signatures alone.

1. **Program inputs.** Real invocation traces wherever they exist. In addition,
   a large model synthesises program-level inputs from the root signature, its
   types and real examples. Every input is tagged real or synthetic.
2. **Call capture.** Run each program on those inputs with the teacher as
   executor. Every inner call's arguments, context ID, source hash and result
   become a lambda-level record.
3. **Teacher trajectories.** For each captured call, sample k teacher
   trajectories with every model turn kept, so the student trains on the turn
   sequence the runtime actually renders.
4. **Teacher routing.** Each family is calibrated on a sample: the self-hosted
   Qwen3.6-35B and the paid teachers (Luna, Bunny) run the same verified
   cases, and each family routes to the teacher with the highest verified
   success per unit cost. Routing is recomputed whenever a teacher changes.
5. **Verification ladder**, strongest available label first:
    - the program's task oracle, propagated to inner calls whose results led
      to a verified program result;
    - typed return checks and refinement types;
    - agreement among the k teacher samples, with the agreement rate stored as
      label confidence;
    - a judge, for free-text results, stored as the weakest label source.
6. **On-policy corrections.** After every training round, the student runs
   with its current stacks. At each student-visited turn boundary the teacher
   continues from the student's state, and that turn becomes a target.
   Turn-level targets work across the Qwen, LFM and Mellum tokenizers.
   Correction rounds repeat until held success stops improving.
7. **Splits.** The existing connected-component selector assigns development,
   confirmation and held sets per program. Held inputs are never shown to the
   input synthesiser or the teachers' prompts.

Each record carries: lambda ID, context ID, source hash, interface hash, input
provenance, label source and confidence, teacher identity, split, and the
student stack it was collected under (for on-policy rows). Records stay valid
across base-model changes; they are the durable asset.

**Lambda identity.** Named functions use their path. Inline `nl` call sites get
a stable ID from the compiler's plan for the call: enclosing function, ordinal
among its `nl` calls, declared parameters and return type. The instruction-text
hash is kept separately for staleness (§8).

## 5. Training

### 5.1 Joint objective

One joint run trains all of a program's adapters at once. Each example
activates exactly the adapters on its lambda's path plus its family adapters,
so the tree adapter sees every example, a folder adapter sees its subtree, and
a lambda adapter sees only its own calls.

Turn-level cross-entropy on verified teacher turns, weighted by label
confidence, plus an L2 penalty on each adapter's delta that grows with depth:

```latex
L = \sum_{i} w_i \, \mathrm{CE}_i\Big(W_0 + \sum_{f \in F(i)} \alpha_f B_f A_f + \sum_{n \in \mathrm{path}(i)} s_n B_n A_n\Big) + \sum_{n} \lambda_{d(n)} \lVert B_n A_n \rVert_F^2
```

d(n) is a node's depth, λ increases with depth, F(i) is the lambda's family
set and α its learned mixing scales. A behaviour that several lambdas share is
cheaper to learn once in a shallow adapter than many times in deep ones, so
shared structure migrates up the tree. Per-lambda sampling is balanced, so a
high-volume lambda cannot dominate the shared levels.

### 5.2 Placement

After the joint run, each node is ablated (scale 0) on the development split,
deepest first. A node keeps its adapter when removing it costs more than the
placement threshold in program-level success per kilobyte, measured together
with serving throughput (§7). Removed nodes are folded away and the remaining
adapters are re-fitted briefly. Authors can pin a node to force or forbid an
adapter.

The threshold has a global default and a per-program override in the program
manifest. The default is fitted across all tuned programs to maximise held
success per unit of serving cost.

**Subtree discovery.** Placement also proposes new tree roots: when a folder's
adapter carries most of its subtree's gain and its siblings' adapters do not
overlap with it, the folder is promoted to a tree root with its own rank-8
adapter. Authors can mark tree roots explicitly as well.

### 5.3 Model placement and parameterisation

Adapters cover all layers of each student. On LFM2.5 that is all 16 layers:
attention projections, the convolution blocks' input and output projections,
and the MLPs. On Mellum it is attention, shared and expert MLP matrices.

Training uses the native `nnt1` coefficient path, restricted to top-r singular
subspaces, so every adapter converts exactly to a standard LoRA. The trainer
also implements free LoRA factors; the rank sweep (§5.6) selects per level
which parameterisation each student uses.

### 5.4 Family mixing

Family mixing scales α are leaves in the same joint run. They are initialised
from tag membership (1 for tagged families, 0 otherwise) and from clustering
similarity, and trained with an L1 penalty, so a lambda keeps only the
families that help it.

### 5.5 Family training across programs

1. Collect lambda adapters and datasets across all tuned programs.
2. Cluster lambda adapters by delta similarity (cosine over stacked factors)
   and merge with frontmatter tags into family assignments.
3. Train each family adapter on the union of its members' data.
4. Re-fit each member program's tree with the family adapters frozen beneath
   it and the mixing scales trainable.

This runs on a schedule and after every batch of newly tuned programs.

### 5.6 Rank and penalty sweep

A standing sweep over ranks per level, depth penalties and parameterisations
runs on a rotating sample of programs. It includes the capacity measurement:
flat per-lambda LoRA and full per-lambda fine-tuning at 8, 32, 128 and 512
verified examples, against untuned LFM, untuned Mellum and the teacher. Its
results set the defaults used by `natlang tune`, and they decide per program
which student is the default executor. Where full fine-tuning also stays far
below the teacher, `natlang tune` reports the lambda as a decomposition
candidate, with its failure categories, to the program-improver workflow.

### 5.7 Distillation into the base

At every base-model training round, outputs of the best adapter stacks across
all tuned programs join the base SFT mix, together with the verified datasets.
A better base shrinks every adapter, and programs that were never tuned improve
too. After a base update all adapters are rebuilt (§8).

## 6. Runtime

**Binding.** Adapters are data items in contexts, like `.nz` files. A folder's
adapter lives in that folder; a lambda's adapter sits next to its `.nl` file or,
for inline sites, in the enclosing folder under the call-site ID. A
program-level manifest, `natlang.adapters.json`, lists every adapter with its
node, level, scale, family memberships and mixing scales, base hash and build
receipt. Adding or replacing an adapter produces a new context revision, so
promotion and rollback are rebinding.

**Stack resolution.** For each call the runtime computes the ordered stack:
families with their scales, then tree, folders and lambda. The stack ID is a
hash of the ordered adapter IDs and scales. The resolved stack replaces any
inherited one; an explicit `withNativeAdapters` scope appends on top of it.
Every trace records the stack ID a call ran with.

**Degradation.** A missing, stale or wrong-base adapter is skipped and logged,
and the call runs on the rest of its stack. A call never fails because an
adapter is unavailable.

**Native states across boundaries.** A native state written under one stack and
read under another is re-encoded from its text at the boundary, as the
serving path already requires for adapted KV. Lambdas that pass native states
to each other can be pinned to share one stack; placement honours that pin.

## 7. Serving

**Export.** Each distinct stack is exported as one LoRA: factors concatenate
in rank, with scales folded into B. Stacks are precomputed per lambda at build
time, so servers load only standard LoRAs.

| Server | Work |
| --- | --- |
| Native reference server | Training and gradient sessions; per-call stacks through the existing adapter scope |
| vLLM | Load exported stacks; route requests by stack ID; batched multi-LoRA for rollouts and production |
| llama.cpp fork | Group concurrent requests by stack ID; add batched multi-LoRA kernels (SGMV-style) as an upstreamable patch; GGUF LoRA export for every stack |
| Browser (wllama) | Load exported stacks through the fork's LoRA path; store them in OPFS beside the model |

**Throughput accounting.** Every serving benchmark records batch
fragmentation by stack ID. The placement threshold (§5.2) is fitted against
these measurements.

## 8. Lifecycle and `natlang tune`

Adapters are build products. `natlang tune <program>` produces them from
datasets, and any change to their inputs marks them stale.

**Build key.** Each adapter's receipt records the base checkpoint hash, the
lambda's instruction-text hash, its interface hash (signature, captures,
callable context items), the dataset receipt, and the hashes of the adapters
above it in the stack.

| Change | Effect |
| --- | --- |
| Instruction text of one lambda | That adapter is stale; the call runs on its parent stack while an incremental build retrains it warm-started from the stale one |
| Signature, captures or callable items | That lambda's adapter and dataset are dropped; data is regenerated and the adapter rebuilt |
| A folder adds or removes lambdas | Folder adapter stays valid; a joint re-fit of that subtree is scheduled |
| A family or tree adapter is retrained | Every adapter below it is re-fitted with the new one frozen |
| Base checkpoint | Every adapter is rebuilt from stored datasets |

**Incremental builds.** A changed lambda retrains only its own adapter, with
everything above it frozen. A full joint re-fit runs on a schedule and after a
configured number of incremental builds.

**Commands.**

- `natlang tune <program>`: full pipeline (capture, teachers, verification,
  joint training, placement, export, manifest).
- `natlang tune <program> --incremental`: rebuild stale adapters only.
- `natlang tune <program> --watch`: incremental rebuilds in the background
  while an author edits.
- `natlang tune --families`: cross-program family clustering and training.
- `natlang tune --report <program>`: held results, per-level ablation, adapter
  bytes and throughput.

**Provenance.** Each receipt names its teachers, dataset split receipts,
placement decision and evaluation results, so a regression bisects to one
adapter.

## 9. Evaluation

Evaluation runs continuously alongside implementation and on every build. It
reports; it never blocks work.

**Arms**, all on the same held sets at matched inference cost:

| Arm | What it shows |
| --- | --- |
| Untuned student (LFM, Mellum) | Baseline |
| One adapter per program | Value of per-lambda structure |
| Flat per-lambda adapters | Value of sharing |
| Hierarchy before placement | The joint objective |
| Hierarchy after placement | Success kept per byte and per unit throughput |
| Hierarchy with families | Cross-program sharing |
| Base after distillation, untuned | Transfer to programs that are never tuned |
| Teacher | Ceiling |

**Per-level ablation.** Leave-one-level-out (scale 0) on held sets, per
program, in every report.

**Reported per program:** held success on real and synthetic inputs
separately, per-level contributions, adapter bytes, serving throughput and
batch fragmentation, teacher tokens spent, build time.

**Regression checks.** Programs without adapters stay byte-identical in
behaviour. Programs with adapters are re-evaluated after every base update and
every family retrain.

## 10. Decisions on former open questions

| Question | Decision |
| --- | --- |
| Can a callee inherit its caller's adapters? | No. Stacks follow the lexical path; explicit `withNativeAdapters` appends for experiments |
| Can a lambda belong to several families? | Yes, with learned mixing scales under an L1 penalty (§5.4) |
| Placement threshold per program or global? | Both: a fitted global default plus a per-program override (§5.2) |
| Who marks tree roots? | Both: authors mark them, and placement discovers and promotes them (§5.2) |
| Which teacher serves which family? | Calibrated routing by verified success per cost, recomputed on teacher changes (§4) |
| Which student? | Both LFM2.5-350M and Mellum are implemented; the sweep picks each program's default executor (§5.6) |

## 11. Work list

All items are required and are implemented in this effort.

**Data**

- [ ] Real-input import from invocation traces, with provenance tags
- [ ] Program-level input synthesiser seeded from real examples and types
- [ ] Call capture of every inner call into lambda-level records
- [ ] Teacher trajectory runner with k samples and full turn retention
- [ ] Teacher calibration and per-family routing
- [ ] Verification ladder with oracle propagation and label confidence
- [ ] On-policy correction rounds at student-visited turn boundaries
- [ ] Split assignment through the connected-component selector
- [ ] Stable inline call-site IDs in the compiler plan

**Training**

- [ ] Path-routed joint trainer on `nnt1` with depth-weighted penalties and balanced sampling
- [ ] Free LoRA factor parameterisation beside top-r subspaces
- [ ] Family mixing scales as trainable leaves with L1 penalty
- [ ] Placement pass: deepest-first ablation, fold, re-fit, author pins, subtree discovery
- [ ] Cross-program clustering and family adapter training
- [ ] Rank, penalty and capacity sweep with automatic default updates
- [ ] Stack outputs and datasets added to the base SFT mix at each base round
- [ ] LFM2.5-350M and Mellum support throughout

**Runtime**

- [ ] Adapters as context items and the `natlang.adapters.json` manifest
- [ ] Path-resolved stack resolution replacing inherited scopes
- [ ] Stack IDs in every trace
- [ ] Graceful degradation for missing, stale and wrong-base adapters
- [ ] Native-state re-encoding at stack boundaries and shared-stack pins

**Serving**

- [ ] Stack export to standard LoRA (safetensors and GGUF)
- [ ] vLLM multi-LoRA routing by stack ID
- [ ] llama.cpp request grouping by stack ID
- [ ] llama.cpp batched multi-LoRA kernels, written for upstreaming
- [ ] wllama stack loading and OPFS storage
- [ ] Throughput and fragmentation benchmarks feeding placement

**Lifecycle and tooling**

- [ ] Build receipts and staleness rules from §8
- [ ] `natlang tune` with `--incremental`, `--watch`, `--families` and `--report`
- [ ] Scheduled full re-fits and post-base-update rebuilds

**Evaluation, tests and documentation**

- [ ] Continuous evaluation arms and reports from §9
- [ ] Conformance cases: stack resolution, lexical scoping, DAG-shared folders, degradation, export parity against the reference server
- [ ] Spec text for adapter context items and scoping in `spec/SPEC.md`
- [ ] Author guide for `natlang tune`, pins and manifests
- [ ] API migration entry under `training/api-migrations/` for the scoping change

## 12. Risks and responses

| Risk | Response |
| --- | --- |
| Synthetic inputs differ from real ones | Call capture from real program runs; real and synthetic results reported separately; synthesiser seeded from real examples |
| Weak verifiers distil teacher mistakes | Verification ladder with confidence weights; oracle-backed rows weighted highest; on-policy rounds correct drift |
| Too few examples for a lambda | The hierarchy pools data upward; placement removes lambda adapters that do not pay |
| Capacity limits on some lambdas | The sweep detects them; `natlang tune` routes the program to Mellum or reports the lambda for decomposition |
| Many adapters fragment serving batches | Throughput-fitted placement; grouping and batched kernels |
| Frequent edits keep adapters stale | Parent-stack fallback; fast incremental and watch builds |
| Native states change meaning across stacks | Re-encoding at boundaries; shared-stack pins |

## 13. Definition of done

- Every program in the production task packets can be tuned with one
  `natlang tune` command on both students.
- Tuned programs run with path-resolved stacks on the reference server, vLLM,
  the llama.cpp fork and wllama, with export parity checked.
- Families exist and are trained across programs; distillation into the base
  has run at least once and all adapters have been rebuilt on the new base.
- Evaluation reports for every tuned program include all arms, per-level
  ablations, throughput and costs, with results stated as measured.
- Spec, guides, conformance cases and the API migration entry are merged.
