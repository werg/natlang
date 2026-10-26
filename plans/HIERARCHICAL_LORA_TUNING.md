# Hierarchical LoRA tuning for natlang programs

Status: complete-scope implementation assignment; reviewed 2026-10-26 against
repository commit `6a128e5e79fa94cf17bf4b098c567cf8a433ab74`.
Unchecked work below is required work, not a claim of implemented capability.

## 0. Mandate

Implement this entire plan in one continuous effort. There are no pilot stopping
points, approval gates, go/no-go reviews or deferred work items. Data generation,
the joint trainer, placement, family learning, distillation into the base,
runtime binding, all serving targets, `natlang tune`, evaluation and documentation
are required. Both LFM2.5-350M and Mellum are first-class targets throughout.

Work proceeds in dependency order. Measurements inform ranks, penalties,
placement, teacher routing and implementation repairs; a disappointing result
does not cancel an arm or suspend independent work. Report measured results
honestly, investigate failures against exact references, and improve the system.
Correctness checks still apply: a corrupt artifact, leaked evaluation row or
wrong-base state must not be relabelled as usable to keep a pipeline moving.

### Natlang-native decomposition

Follow [the repository's owner rules](../AGENTS.md): policy is natural language;
exact mathematics, verification, durability and external I/O are crisp. Before
implementation, write `DECOMPOSITION.md` in the owning tuning application and
map each unit to an existing mechanism or a justified new one.

| Unit | Natural-language policy | Crisp mechanism |
| --- | --- | --- |
| Data | Propose root inputs; route authorized teachers; diagnose disagreements | Trace capture, replay, split enforcement, typed/task verifiers |
| Training | Propose families, rank changes and decomposition candidates | Shared native trainer, autodiff, optimizers, deterministic tensor mappings |
| Placement | Propose removals, scope roots and shared-stack pins | Execute ablations, calculate costs, verify constraints |
| Build/runtime | Explain failures and propose recovery | Dependency hashes, lexical binding, cache identities, atomic publication |

Extend the existing specializer, call store, pluggable policies, shared trainer,
recipe inheritance and hashing. Weight adapters complement native positional
KV/hybrid-state communication; they do not replace it. Preserve complete
trajectories and actual producer/consumer graph dependencies in training.

## 1. Outcome

Every natlang program gets a build step that specializes a student to that
program. Data comes from whole-program execution. Jointly trained residual
adapters mirror a declared lexical scope tree over the program's bound
definitions; a call uses its own binding's path and family adapters.

The deliverable comprises:

1. **Data:** real and synthetic root invocations, captured inner calls, full
   teacher trajectories, explicit verification evidence and on-policy repairs.
2. **Training:** joint program adaptation, measured placement, cross-program
   families and periodic distillation into each student's base.
3. **Runtime/serving:** immutable adapter bindings, exact effective-model
   identities, exported stacks and batched serving on the reference server,
   vLLM, the llama.cpp fork and wllama.
4. **Lifecycle:** incremental builds, dependency-aware invalidation, atomic
   activation/rollback and reusable source datasets with exact provenance.

The primary outcome is held-out program-level task success under a declared
inference-cost envelope. Report success, latency, throughput and memory together;
do not substitute token loss or adapter size for task success.

### 1.1 Author workflow: set up, tune, call normally

Seamless inference-engine integration is a primary product requirement. An
author sets up an ordinary natlang program, trains it through one command, and
uses the same program and call sites immediately. The intended CLI workflow is:

```sh
natlang setup                 # one-time model/runtime configuration
natlang tune ./my-program     # collect/reuse data, train, validate and activate
natlang run ./my-program      # ordinary execution automatically uses the result
```

`natlang tune` is a planned command, not an existing CLI capability at the
reviewed revision. Its setup preflight must resolve the selected student,
training capability, authorized teacher and serving backend from the existing
profile/settings. Reuse those settings across programs. An author supplies the
normal typed program plus available root examples, tests or traces; the build
discovers its callable/site inventory and generates the adapter metadata.

The default path requires no handwritten adapter manifest, adapter IDs, manual
export/conversion, server-loading command, `withNativeAdapters` wrapper or
alternate inference API. Normal `natlang run`, `natlang call` and program-aware
SDK calls resolve the active compatible binding automatically. An ordinary
typed function call stays an ordinary typed function call. Package distribution
may pin a release, and experts may override it, without adding steps to local use.

A successful tune means the chosen backend has loaded and accepted the release
and a normal engine call has exercised it with a safe recorded test input. It
does not mean merely that training produced files. Report the active program,
student/backend, measured validation outcome and time to readiness. If activation
fails, retain the previous compatible release and report the failure explicitly.

**Fast first use.** Fit the requested program for its selected executor using a
declared default recipe, existing compatible families and reusable admitted
data. Warm runs reuse source/tokenization/basis/export caches and incremental
state where valid. Cold runs collect the missing data through the same command.
An absent family is a valid empty hierarchy level (§3).

Per-program readiness does not wait for a global rank/capacity sweep, a new
cross-program family generation, a base-distillation round, training the other
student, or exporting to every other backend. Those remain required system
capabilities and separately scheduled/explicit maintenance work; a build waits
only for its selected release's real dependencies and correctness checks. Both
students and every serving target must support this same author workflow.

Measure time from invoking `tune` to the first successful tuned call through
the normal engine, separating queue/setup, data, training, validation, export
and loading time. Report cold and warm cases, data volume and hardware. Optimize
this measure alongside quality; do not promise seconds or minutes before
measurement. Once ready, inference has no teacher or trainer in its request path.

## 2. Verified starting point

These are integration points at the reviewed revision, not claims that the
hierarchical pipeline already exists.

| Existing piece | Source | Required extension |
| --- | --- | --- |
| Content-addressed context DAG; companion-folder binding and `.in(context)` | [Contexts](../spec/SPEC.md#contexts) | A lexical adapter-scope overlay with explicit binding identity |
| `nnt1` frozen-SVD bases, coefficient cores and functional weight substitution | [native_adapters.py](../training/neuralese/natlang_neuralese/model/native_adapters.py), [basis math](../training/neuralese/natlang_neuralese/model/native_adapter_math.py) | Multi-node joint updates, shared basis inventory, free-factor option and export |
| `withNativeAdapters`, recorded objectives and optimizer continuation | [learning.ts](../ts-host/src/neuralese/learning.ts) | Separate resolved lexical bindings from explicit dynamic experiment scopes |
| Runtime contexts and callable bindings | [contexts.ts](../ts-host/src/runtime/contexts.ts), [callable.ts](../ts-host/src/runtime/callable.ts) | Immutable per-invocation stack selection and trace fields |
| Program discovery, normal run/call launch and runtime model driver | [CLI launch](../ts-host/src/cli/main.ts), [runtime](../ts-host/src/runtime/runtime.ts) | Automatic active weight-binding discovery and per-request stack dispatch on the existing inference path |
| Source/world connected-component selector | [memetic_batch.py](../training/neuralese/natlang_neuralese/rl/memetic_batch.py) | Extend train-only development/confirmation selection; preserve external protected sets |
| Explicit `same-base-adapter-state` imports and consumption receipts | [native_adapters.py](../training/neuralese/natlang_neuralese/model/native_adapters.py), [native_grad.py](../training/neuralese/natlang_neuralese/serve/native_grad.py) | Retain cross-stack producer/consumer identity and gradients; distinguish cache reuse |
| Teacher configuration and current operating instructions | [TEACHER_SETUP.md](../TEACHER_SETUP.md), [AGENTS.md](../AGENTS.md) | Calibrated routing among currently authorized actors |
| Shared training recipes and line parity | [TRAINING_RECIPE.md](neuralese/TRAINING_RECIPE.md) | One shared hierarchical recipe with declared backbone differences |

The existing split selector rejects non-train inputs and makes development and
confirmation groups; it does not already implement the complete split protocol
below. The native adapter path stores FP32 bases/cores, binds the exact base
identity, and applies `U C V^T`; it is not yet a generic free-factor LoRA trainer.

## 3. Hierarchy and binding semantics

| Level | Scope | Training population | Initial rank per target matrix |
| --- | --- | --- | --- |
| Family | Versioned library family, separately for each student/base | Members across training programs | 16 |
| Tree | Program root and author-marked or discovered subtree roots | Lambdas below that root | 8 |
| Folder | Callable folder or `natlang.d/` scope | Lambdas in that scope and below | 4 |
| Lambda | Named `.nl` definition or inline `nl` site | That bound lambda | 2–4 |

Ranks are starting settings, selected on development data (§5.6). Every level
may be absent. Author pins can require/forbid a node or require a set of calls to
share an effective stack. Conflicting pins produce an actionable build error.

### 3.1 Lexical scopes over a DAG

The context DAG is not itself a tree with a unique parent. Compile an explicit
adapter-scope tree over **bound definitions**, with a manifest entry for each
binding: owning program/library, definition or site ID, semantic context digest,
and ordered scope ancestry. Resolve this entry from the callee value, never by
walking the dynamic caller chain or choosing the first path through a DAG.

Two aliases of the same bound callable retain one binding and stack. Two scoped
instances of content-identical code/context may have different binding IDs and
stacks. An imported callable retains its defining binding; a program-specific
instance requires an explicit binding in that program's manifest. `.in(context)`
creates a new binding revision and rechecks compatibility. A compatible context
without a tuned binding uses the declared fallback; it must not inherit the
caller's program adapter by accident.

For example, a bound `triage/label-ticket.nl` can use
`classify + program-root + triage-folder + label-ticket`. Calling it from
another folder does not add that caller's adapters. An explicit
`withNativeAdapters` experiment scope appends separately (§6).

### 3.2 Composition and identity

For each target matrix `m`, the effective weight is
`W0,m + sum_f a[l,f] D[f,m] + sum_n s[n] D[n,m]`, where `n` ranges over the
binding's lexical path. Membership and scales are static per binding/version,
not token-dependent routing. `s[n]` is a fixed deployment/ablation scale (normally
1 or 0); only the family gates specified in §5.4 are learned.

Addition is exact algebraically. The sum's rank is **at most** the sum of factor
ranks, and can be much smaller with shared bases (§5.3). Floating-point kernel
and quantization parity require measurement; algebraic equality does not imply
bit-identical outputs across backends.

Family adapters are model-specific artifacts. LFM and Mellum share family labels,
data and policy, not tensor coordinates, factors or native states.

### 3.3 Soft instructions and soft skills (owner, 2026-10-26)

The tunable state of a lambda is not only its weight residuals. Its **instructions** and the **skill documents
bound to it** (its companion folder's skills, as disclosed to that invocation) also become directly tuned native
state: positional KV and hybrid (LFM convolution/linear) state spans in the existing `nzs1` native-state format.

| Element | Initialization | Tuned parameters | Where it is consumed |
| --- | --- | --- | --- |
| Soft instructions | The lambda's instruction text written by the exact executing reader (base plus the binding's lexical stack at initialization) | Every KV/hybrid tensor of the state span, directly | In place of the instruction text in the call's rendered prompt, at its exact position |
| Soft skill | The skill document's text written the same way, one state per bound skill file | Same | Wherever that skill's text would enter the invocation (listing or read), at its position |

Rules:

1. A soft state is a parameter, not a cache: after initialization it is optimized directly with the shared
   native-state optimizer (`native_state_derivation.optimize_state`), keeping its immutable parent lineage. It is
   never silently recomputed from text.
2. It retains its source text, tokenizer and initializing actor so it can be **re-initialized** from text when the
   executing base/checkpoint or tokenizer changes; tuned values do not transfer across bases (§5.7 rebuild).
3. It is part of the binding's release: the manifest binds instruction/skill states per binding (bound to the
   exact base), their text digests and initialization receipts. Changing the instruction text or skill file makes
   that state stale exactly like the binding (§8).
4. Training updates soft states jointly with the binding's weight residuals in the same logical update: gradients
   flow through the actual consumption positions in each trajectory. A state shared by several calls of one
   binding receives every occurrence's gradient once per occurrence.
5. Serving requires a backend that imports native state (native reference server, the llama.cpp fork's Neuralese
   server). A release declares whether its bindings require soft state. On a backend without native-state import,
   a soft-state binding is not served as tuned: it runs the declared fallback (text instructions with the base, or
   a separately validated text-instruction reduction) and is reported degraded. `natlang tune` for such a backend
   trains the text-instruction form instead.
6. Soft states are model-facing changes: the measurement rules for model-facing changes apply to the instruction
   text used for initialization, and task success is measured with the tuned states, not inferred from loss.

## 4. Data generation and evidence

### 4.1 Split before expansion

Partition root invocations by connected source/world components before input
synthesis, trajectory sampling, inner-call extraction or student correction.
Every derived row inherits its root's partition and lineage. Extend the existing
selector rather than independently splitting each lambda's rows.

| Partition | Permitted use |
| --- | --- |
| Training | Gradient updates, input synthesis seeds, family discovery and distillation |
| Development | Teacher calibration, stopping, ranks/penalties, gates, placement and student/default selection |
| Confirmation | Evaluate a selected candidate; record selection history if its feedback guides further work |
| Protected held/test | Report a predeclared candidate and ablations; never select configurations or generate training repairs |

Confirmation used repeatedly for decisions is adaptive validation, not an
untouched test. Keep protected held data outside all tuning, family and base-SFT
inputs. If held failure details are used for repairs, retire that set as a clean
final estimate and use new protected components for the next such estimate;
retain the original protection/provenance records.

Enforce component separation globally across programs, families and distillation,
including duplicate tasks, source variants and synthetic descendants. For
unseen-program transfer, exclude the entire evaluation program and related
source/world components from family training, clustering and base distillation.
An isolated evaluator may query a teacher on held inputs for the teacher arm;
those prompts, answers and traces never enter the training/generation pipeline.

### 4.2 Whole-program collection and teacher routing

1. Import real root invocation traces with source/world and provenance tags.
   Synthesize additional root inputs from training-only signatures, examples
   and context; do not invent inner arguments from signatures alone.
2. Run the real runtime with a teacher executor. Capture every inner invocation,
   bound definition/context, arguments, captures, tool interactions and outcome.
   Replay side effects against a captured world or sandbox so multiple samples
   do not repeat real writes or see different external state unnoticed.
3. Sample `k` complete trajectories for captured calls, preserving all model
   turns, rendered prompts, tool results, return/stop decisions and graph edges.
   Record `k`, decoding settings and failed attempts, including useful repair
   or preference examples; do not label failure traces as positive targets.
4. Calibrate Qwen3.6-35B, Luna and Bunny integrations on common development cases,
   but only execute actors authorized by the current owner instructions and
   available deployment. The 2026-10-26 instructions resume Luna generation;
   they do not implicitly resume paused CoreWeave services or authorize every
   other external provider. Preserve each actor's exact original identity.
5. Choose a teacher on the measured quality/cost frontier with a declared minimum
   task-quality requirement. Include retries, verification, input/output tokens
   and elapsed compute in cost. Recalibrate on actor/runtime changes; a cheap
   weak teacher must not win solely through a success-per-token ratio.

For the current operating configuration, the owner calls for six supervised
parallel Luna generation workers, separate from engineering helpers. Integrate
with that supervisor-managed pool; a program build must not launch a duplicate
pool. Worker concurrency remains operational configuration, not a fixed property
of the learning algorithm.

### 4.3 Verification evidence, not automatic credit assignment

Store separate evidence fields instead of collapsing all checks into one
"verified" flag.

| Evidence | What it establishes | Training treatment |
| --- | --- | --- |
| Direct task oracle/local semantic contract | Correctness under that oracle's declared coverage | Highest-confidence positive target when it passes |
| Successful enclosing program | Association with a successful trajectory | Weak inner-call attribution unless a local check or controlled replay verifies the call |
| Type/refinement checks | The specific structural/property constraints tested | Necessary validity evidence; not general task correctness |
| Teacher agreement | Consensus among sampled answers | Weak evidence; record dependence and disagreement |
| Free-text judge | That judge's assessment | Calibrate on development oracle cases; retain judge/version/rubric |

A correct final answer does not prove every intermediate result was correct;
agreement can repeat a shared mistake. Direct contradictory evidence overrides
consensus. Confidence weights and abstention are calibrated on development data,
with separate counts/results for oracle-backed and weakly labelled rows.

### 4.4 On-policy correction and durable records

After each training round, run the current student on training roots. At each
visited turn boundary, ask an authorized teacher to repair or continue the
serialized student-visible prefix in the same captured world. Retain the whole
trajectory and distinguish the student's mistaken prefix from the corrected
target. Reverify the continuation and downstream result. Mix admitted repairs
with earlier usable data and select further correction rounds on development
performance, never held performance.

Cross-tokenizer transfer is through text/tool semantics, rendered and tokenized
for each student. It is not token-index alignment or transfer of the teacher's
KV. Preserve native communication graphs with model-specific producer/consumer
state and actual occurrence bindings; do not detach a producer when training
its downstream consumer.

Each record includes root invocation/component IDs, binding and site IDs,
source/interface/semantic-context hashes, captures/world/service receipts,
full trajectory and graph, real/synthetic ancestry, split receipt, verifier
evidence, teacher checkpoint/tokenizer/runtime, decoding settings and collection
policy. On-policy rows also record exact student checkpoint, requested/effective
stack, prefix and correction round. Record actual admitted and trained exposure
separately from capture or registration.

Raw text/world evidence is durable across base changes. Rendered tokens, native
states and correction distributions have compatibility/provenance requirements;
revalidate/rerender them rather than declaring every derived artifact universal.
Interface changes invalidate admission, not the existence of useful source data.

**Site identity.** Named definitions use a package-relative definition ID.
Inline sites use compiler-persisted IDs, with an optional author ID and structural
matching on edits. Instruction text is a separate digest. An ordinal alone is
not stable when another `nl` call is inserted. Ambiguous matches receive new IDs
or an explicit migration map; never silently attach another site's adapter.

## 5. Training

### 5.1 Joint objective and updates

One logical program training run updates all its tree/folder/lambda adapters
and family gates jointly, with published family tensors frozen. Every example
activates its own bound path. A complete native graph retains the actor at each
invocation and gradients through actual producer/consumer dependencies.

For trajectory `i`, let `z[i,t]` be student-tokenized tokens, `h[i,t]` the rendered
prefix/graph state, and `b[i,t]` the declared role/mask weight. Model turns and
return/stop decisions have primary weight; prompt/input/tool text has the lower
auxiliary weight required by the shared recipe. Padding, unavailable targets
and rejected positive targets have zero weight. Normalize by positive token
weight so long trajectories do not silently dominate; reject rows with no
positive target weight. Assign losses to actual recorded turn/graph occurrences
so a producer used by several consumers is not accidentally counted repeatedly.

```text
ell_i = - sum_t b[i,t] log p_W(z[i,t] | h[i,t]) / sum_t b[i,t]
L = mean_lambda mean_{i in lambda}(w_i ell_i)
    + sum_n lambda_depth(n) sum_m q_m ||D[n,m]||_F^2
    + tau mean_lambda sum_{f in F(lambda)} a[lambda,f]
```

Here `w_i` is the calibrated evidence weight and `q_m` is a fixed, recorded
matrix-size normalization (initially inverse element count). Family gates are
nonnegative (§5.4). Balance lambdas in the training sampler; separately evaluate
both macro program success and the observed production input mix. Keep trajectory
sampling and correction-round weights explicit to avoid counting `k` correlated
teacher samples as `k` independent invocations.

Depth-weighted L2 encourages economical sharing; it does **not** identify a
unique decomposition or prove behavior migrates upward. Parent and child deltas
can substitute for one another. Establish sharing through refitted ablations,
placement and transfer results, not a penalty interpretation alone.

Use the existing native FP32-master/optimizer and continuation mechanisms.
Accumulate all active-node/gate gradients against one immutable parameter
snapshot before applying the logical update; do not update a parent midway
through a graph. Check zero-adapter equivalence, active/inactive gradient routing,
actual parameter changes and resume equivalence. Persist optimizer/RNG,
sampling/exposure cursors and dependency identities for resumable jobs.

### 5.1a SFT and RL objectives (owner, 2026-10-26)

Program tuning supports both supervised fine-tuning and reinforcement learning over the same tunable state
(weight residuals, family gates, soft instruction/skill states) and alternates them in one recipe.

| Objective | Data | Update |
| --- | --- | --- |
| SFT | Verified teacher trajectories and admitted repairs (§4) | §5.1 trajectory cross-entropy |
| RL | The student's own whole-program rollouts on training roots, `k` samples per root, each call under its own binding stack | Group-relative advantages from task rewards; clipped token policy loss over every sampled model turn of the actual graph |

RL rules:

1. Rewards come from task outcomes: the root's direct oracle and declared local semantic checks, never an exact
   historical message as a reward when other continuations are correct. A root with no oracle is not an RL root.
   A failed child is not excused by a successful parent; a correct blocked/declined result counts when the oracle
   says so.
2. The policy update covers the entire executed graph: every model turn of every inner invocation, each with its
   recorded actor (its binding's effective stack and soft states), producer/consumer native-state dependencies and
   exact sampled tokens (native recorded turns), with gradients through the real dependencies. Reuse the existing
   recorded-policy machinery (`rl/native_replay.py` `prepare_objective`, `rl/policy_math.py`,
   `rl/rollout_math.py`, `serve/native_grad.py`); do not detach in-episode producers.
3. Rollouts record the exact behavior policy (release, stack identities, decoding settings) so off-policy
   corrections are exact; a rollout from another release is not on-policy.
4. Only training roots are rolled out. Development/confirmation/protected roots never produce rewards for updates.
5. The schedule alternates SFT and RL rounds on development-selected settings (round sizes, `k`, temperature,
   clipping, KL telemetry). Base-relative KL is telemetry, never a stop.
6. RL requires a backend that records native turns (the native reference server); llama.cpp backends serve the
   result but do not produce training rollouts.

### 5.2 Placement and subtree discovery

Use development data and a fixed serving workload. Record allowable task-success
loss, latency/memory envelope, cost weights and author pins in the manifest.
Prefer the lowest measured serving/storage cost among candidates within that
quality envelope; also report the full success/cost frontier.

Start with deepest-first ablations, then test interacting node/level groups where
single-node results are ambiguous. This order is a search heuristic, not an
additive attribution theorem. Refit remaining adapters after proposed removals
and remeasure the complete candidate against the unpruned stack using paired
root invocations. Include uncertainty; an underpowered ablation is not evidence
that a node is useless. Removing a node zeros it; folding its delta into another
node is a separate rank/mapping change that must be verified.

Fit global defaults across training/development programs only, with per-program
overrides. Serving cost includes exported-stack duplication, active rank,
resident memory, load/eviction and batching costs (§7), not just source adapter
kilobytes. A shared source adapter can still be duplicated in every export.

Propose a subtree root when residual errors or measured ablations suggest useful
local sharing. Promote it, allocate its declared rank, refit and compare the
result under the same workload. Factor overlap alone is not a behavioral test.
Honor explicit tree roots and shared-stack pins throughout.

### 5.3 Target matrices, parameterization and exact export

Enumerate targets from each pinned model and record names, shapes, expert axes,
orientation, dtype and backend mapping. LFM2.5-350M has 16 configured layers;
cover attention, convolution-block **input/output projections**, and MLP
matrices. Mellum covers attention plus shared and expert MLP matrices. A
convolution kernel/state tensor is not an ordinary linear projection. Preserve
the native target inventory's protected embeddings/output/interface/control
parameters unless a separately declared recipe explicitly changes that scope.

Implement both parameterizations through the shared trainer:

| Parameterization | Delta for one matrix | Export |
| --- | --- | --- |
| Native frozen subspace | `U_r C V_r^T`, with FP32 `r x r` core | `B = U_r C`, `A = V_r^T` |
| Free LoRA factors | `(alpha_lora/r) B A` | Fold the declared scaling into an exported factor |

For native subspaces, compute/store one canonical maximum-rank basis per base
matrix and use exact leading slices. If all stack members use nested slices of
that same basis, their sum is `U_R (sum_j s_j pad_R(C_j)) V_R^T`, with rank at
most `R = max_j r_j`, not the sum of ranks. Verify shared basis IDs/slices;
independent SVD calls are not a sufficient nesting contract. Store/report basis
bytes separately from the small coefficient cores.

For free or unrelated bases, concatenate `B_export = [s_1 B_1 ... s_k B_k]`
and `A_export = [A_1; ...; A_k]`, including per-adapter LoRA scaling exactly once.
Use unit export scaling or an equivalent documented config. Mixed target
inventories require explicit zero contributions and per-module ranks. Apply the
same rule per expert with a verified packed/unpacked and sharding map.

Free factors start with one factor zero and the other nonzero, so the initial
delta is zero without killing both gradients. The rank sweep compares capacity,
parameter count and cost of both forms: a frozen top-singular subspace is a
restriction on direction, not merely a cheaper equivalent of free LoRA.

### 5.4 Family gates

Use per-binding gates `a[lambda,f]`, not one global scale per family. Initialize
tagged memberships at 1 and cluster candidates from calibrated similarity;
project gates into `[0, a_max]` (initial `a_max = 1`) and apply the L1 term in
§5.1. Zero is an exact removable membership. Dev-selected gate bounds and
thresholds are recorded; absent families are not silently eligible for updates.

Published family tensors and their normalization/amplitude convention are frozen
while program adapters and gates train. During a family update, memberships and
their training weights are fixed; publish a new immutable family version before
refitting program gates. Never jointly leave an unconstrained family amplitude
and its gates free: shrinking a gate while inflating its factors would defeat
the intended sparsity penalty. Record a canonical weight-space norm/scale for
each published family; raw-factor normalization is not invariant.

Concretely, use `||D_f||_q = sqrt(sum_m q_m ||D[f,m]||_F^2)` and fix its nonzero
amplitude `rho_f` from a training-only seed fit. Hold `rho_f` fixed during ensuing
family/gate alternation, normalizing updated family directions to that amplitude;
changing it is an explicit development-selected recipe change. A near-zero
family is absent until a useful nonzero fit exists. This convention works for
both coefficient cores and free factors without choosing a raw-factor gauge.

### 5.5 Cross-program family learning

1. Collect training-only member data and same-base adapter deltas, with context,
   label quality and exposure receipts. Start with author tags when learned
   families are not yet available; run discovery again as data accumulates.
2. Cluster using normalized **effective deltas** or behavioral signatures on
   training/development probes. Cosine over stacked raw factors is invalid:
   `BA = (BQ)(Q^-1 A)` permits different factors for the same delta. Low-rank
   Gram products can calculate delta similarities without dense expansion.
3. Merge tag and discovered proposals with explicit membership/version records.
   Train each family on its members' data against the frozen base, with private
   program/folder/lambda residuals disabled and membership weights fixed. This
   lets a new family learn behavior already present in private adapters instead
   of adding a second copy on top of a fully fitted stack. Use balanced
   program/lambda sampling and the shared trainer; record any other fixed family
   contributions in the training actor.
4. Publish a versioned family, then refit each affected program's adapters and
   gates with the new family frozen. Compare members and non-members for
   negative transfer; preserve the previous coherent release for rollback.

Run after newly tuned program batches and on a configured schedule. Never
cluster incompatible bases or compare LFM and Mellum tensor factors directly.
Keep held-program transfer data outside family discovery as well as gradients.

### 5.6 Rank, penalty and capacity sweep

Sweep ranks, target coverage, depth penalties, gate settings and parameterization
on a rotating training/development sample. Include flat per-lambda LoRA and full
per-lambda fine-tuning at 8, 32, 128 and 512 verified root-derived examples, plus
the full-data point where available. Deduplicate related trajectories; report
independent root/component counts, not only turn counts.

Compare untuned LFM, untuned Mellum and the teacher reference. Record actual
trainable parameters, tokens/exposures, updates, tuning trials, seeds and compute;
use declared comparable training budgets or label differences. Match inference
constraints and measure actual cost, since tokens differ across tokenizers.
Use repeat runs for close/noisy decisions. Select defaults/student on development
data and evaluate the selected candidate on confirmation/protected sets.

If full fine-tuning also trails the teacher, first check data, rendering, gradients,
optimizer state and inference parity. Report remaining lambda failure categories
to the program-improver as decomposition candidates; a weak run alone does not
establish an intrinsic capacity limit.

### 5.7 Distillation into each base

At each base-training round, add admitted, verified training-split outputs from
selected stacks and the durable datasets to that student's shared production
SFT recipe. Reverify stack-generated targets, deduplicate and balance sources;
preserve existing native graph and full-trajectory obligations. Do not admit
development, confirmation or protected evaluation examples through this loop.

Measure untuned-base task success on unseen programs, retention on existing
tasks, and adapter size/cost after rebuilding. Base improvement need not shrink
every adapter. Rebuild all weight adapters against the new exact base and
requalify native-state compatibility; never carry old tensor coordinates forward
merely because architecture and tokenizer names are unchanged.

## 6. Runtime contracts

### Integration into the normal inference engine

Extend the existing program descriptor/compiler inventory, CLI launch,
`createNatlangRuntime` and model-driver path. Resolve the program's active
weight binding during normal program loading/task creation; select each
callee's stack at the model-driver/server request boundary. All model turns,
decision readouts and native-state operations for that call use its pinned
effective actor. Applications do not pass stack IDs or select a special runner.

Use one resolver and binding contract across CLI, SDK and browser entrypoints.
The loader supplies program identity/root metadata automatically when available;
an embedded host registers its program once at runtime construction, not on
every call. Extend `natlang call`'s cwd/project resolution to obtain that identity
without an explicit adaptation flag. Generated `natlang.adapters.json` and the Natlang-managed active
release record drive discovery. A deployment/package can carry an immutable
release reference instead of relying on a mutable local active pointer.

The existing `adaptation` artifact carries instruction/guidance changes, not
LoRA weights. Reuse its program inventory and compatibility principles without
reinterpreting that schema or reinstating obsolete memetic adapters. Combine
compatible instruction, native-guidance and weight bindings through the normal
runtime with one explicit effective-source/executor identity. An explicit model
profile override must be checked against the trained binding; a fallback cannot
be reported as tuned execution (§6, Effective identity).

Activate only after the selected driver/backend acknowledges the exact loaded
artifact and passes a normal-call readiness probe against a replayable test world
that does not repeat real side effects. Publish the active reference
atomically; existing tasks finish on their pinned release and subsequent tasks
resolve the new one. Watch mode uses the same path. At a new task boundary the
default program loader constructs a fresh immutable program/context/callable
binding snapshot from the active release. Existing callable handles retain
their captured contexts/releases, and explicit pins stay pinned; never replace
the context behind a callable already held by application code. Surface
`untuned`, `tuning`, `ready`, `stale` and `degraded` status through normal CLI/API
diagnostics, with actionable reasons rather than exposing internal tensor names.

### Binding, publication and scopes

Adapters are immutable data items in contexts. Folder adapters reside in their
folders; named-lambda adapters beside their definitions; inline adapters under
their site IDs. `natlang.adapters.json` is the versioned binding manifest with
scope ancestry, family versions/gates, source/data/build receipts, target maps,
export hashes and fallback bindings. Generated records are implementation data,
not new model-visible prompt content by default.

Prepare and verify artifacts before atomically publishing the complete manifest
and context binding. Pin one release/base for a top-level invocation, including
its inner calls; in-flight work completes on that release. Rollback selects the
previous coherent release rather than mutating an adapter under an existing ID.

Maintain separate lexical and explicit experiment scopes. At a call boundary:

1. Resolve the callee's binding and its family/tree/folder/lambda stack.
2. Replace the caller's lexical stack with that stack.
3. Append the explicit `withNativeAdapters` scopes in deterministic nesting order.
4. Validate compatibility and snapshot the effective stack for all turns of that
   call. Async siblings and post-`await` continuations cannot mutate each other.

The browser gets the same semantics through its runtime scope mechanism; do
not assume Node's AsyncLocalStorage is available there. Publish the scoping
change and recovery advice in the spec, authoring/integration skills and API
migration entry when implementation changes those semantics.

### Effective identity, degradation and caches

Canonical stack identity covers exact base/checkpoint, tokenizer/embedding
identity, target-map/parameterization version, ordered artifact hashes and finite
canonical scales. Extend the existing native effective-identity contract. An
execution identity additionally binds backend, quantization/layout and rendering
semantics where these affect outputs or state compatibility.

Record **requested and effective** stack IDs, resolved release and skipped
artifacts/reasons. An unavailable, stale or incompatible adapter triggers the
manifest's declared fallback. Dropping a parent also drops children that require
that parent version unless that reduced stack has been explicitly validated.
Do not assemble an arbitrary mixture of individually loadable residuals.

Ordinary availability fallback can reach the compatible base executor and is
marked degraded; it is never reported as success of the requested tuned stack.
Evaluation/export-validation uses strict resolution so missing artifacts cannot
produce misleading benchmark results. Invalid state imports or absent compatible
execution paths remain explicit errors, not silently altered semantics.

Call-result, prefix/KV and hybrid-state caches include effective actor identity,
relevant source/context, rendered tokens, position/layout, explicit imported-state
receipts and decoding/world inputs as applicable. Never reuse a cached result or
implicit KV entry merely because the source lambda or requested stack matches.
Test cold/warm and concurrent reuse, base changes, eviction and rollback.

### Native communication versus implicit cache reuse

Preserve the existing distinction:

| Boundary | Rule |
| --- | --- |
| Ordinary prefix/KV/hybrid cache reuse | Require compatible effective actor and layout/positions; a stack change invalidates that reuse |
| Explicit native communication across adapters | Use the existing declared `same-base-adapter-state` mode, exact immutable base, source/consumer bindings, state ID, positions, layout and consumption receipt |
| Different base/model/tokenizer | Use the declared text reconstruction or learned semantic adaptation path with its own provenance/qualification; raw state is not interchangeable |

Explicit same-base communication may have different producer and consumer
adapter stacks. Preserve producer and every consumer occurrence in captured
graphs and backpropagate through their real dependencies. Shared-stack pins
remain an optimization, not a requirement that erases valid explicit imports.

Text reconstruction is a separately measured path using retained text/context;
it is not guaranteed to reproduce the original KV or LFM convolution state.
Do not silently replace a native communication edge with text or claim that a
LoRA-capable server already implements native-state import/export and gradients.

## 7. Serving and export

Export every distinct static effective stack using §5.3 as safetensors/standard
LoRA and GGUF. Record the exact base, module/expert mapping, scaling, actual
exported rank, exporter version and artifact digest. Deduplicate identical
effective stacks. Explicit experimental stacks are exported/cached on demand
under the same identity contract rather than misusing a precomputed lambda ID.

| Server | Required work |
| --- | --- |
| Native reference | Joint gradient sessions and per-call actor bindings; whole-graph/state reference |
| vLLM | Route exported stacks by unique IDs; configure ranks/residency; batch multiple LoRAs; implement required target/native-state gaps |
| llama.cpp fork | GGUF mapping; request grouping; batched multi-LoRA kernels with verified sequence isolation, written for upstreaming |
| wllama | Fork-compatible exports and binding scopes; OPFS artifact storage; measured loading, eviction and browser memory behavior |

Maintain a tested capability matrix for **each student x backend x target class
x dtype/quantization x state mode**. Upstream architecture-level LoRA support
does not prove coverage of this repository's native weights, all LFM projections,
Mellum expert layouts or graph/state operations. Missing coverage is explicit
implementation work; never silently omit tensors, truncate ranks or claim a
backend complete using the reference server in its place. Browser measurements
state the hardware/model size actually exercised.

Verify dense-delta algebra first, then reference/exported forward logits and
greedy/tool decisions, then cached decoding and full programs. Set and record
dtype/backend tolerances before comparison, including the native path's rounded
weight arithmetic versus an unmerged LoRA kernel. Test each target type,
multi-adapter scales, variable ranks and mixed-stack batches. An SVD compression
of an export is approximate and needs a new artifact/measurement; exact
concatenation or common-basis summation does not.

**Cost accounting:** report unique source cores/factors and basis bytes,
materialized export bytes, CPU/GPU/OPFS residency, active ranks, load/eviction
time, adapter and prefix-cache hits, prefill/decode throughput, p50/p95 program
latency and fragmentation by stack ID. Benchmark observed request mixtures and
concurrency with both warm and cold caches. Include text/state conversion and
fallback frequency. Placement uses these measurements, not an assumption that
rank or source-file bytes alone predict serving cost.

## 8. Lifecycle and `natlang tune`

### Dependency graph and build keys

Adapters are build products of immutable source/data/configuration snapshots.
Keep separate keys for dataset admission/rendering, trained artifacts, exports
and deployed bindings so an exporter change need not regenerate teacher data.

A training receipt covers base checkpoint/architecture/tokenizer/embedding,
semantic source and context dependencies, binding/site-ID schema, target map,
dataset/split/verifier receipts, family versions/gates, parent artifacts,
parameterization/basis, optimizer/seed/recipe and training-policy versions.
Joint-run outputs share one input snapshot and run receipt; exports then bind
the exact final stack plus exporter/backend/quantization configuration.

Avoid a content-hash cycle: semantic source/context digests exclude generated
adapter payloads and manifests. Hash source/data/configuration first, then
trained artifacts, then stack exports, then the published context/release.
Do not hash a generated adapter back into its own training input or make
jointly trained parent/child output hashes depend recursively on each other.

Track reverse dependencies: shared adapters depend on their contributing lambda
data/semantics; residual descendants depend on the shared versions used in fit.

| Change | Invalidation/rebuild |
| --- | --- |
| Lambda instruction text | Revalidate its rows; mark its artifact and shared ancestors trained on those semantics dirty, then follow residual dependents |
| Signature, captures, used data/tool contract or callable context | Invalidate affected row admission/bindings; retain raw evidence; regenerate/reverify/rerender as needed |
| Add/remove/move a lambda or change scope ancestry | Recompute bindings and the affected shared-training population; invalidate affected ancestors and residual dependents |
| Retrain family/tree/folder or change a family gate | Rebuild dependent stacks; refit residuals whose trained parent composition changed |
| Append admitted data | Update data receipts and schedule affected fitting; existing published release remains immutable |
| Change base/checkpoint/tokenizer | Revalidate/render data and rebuild all affected adapters; requalify state transport |
| Change export format/backend/quantization | Re-export and repeat backend parity; retrain only when the target/math contract changes |

Distinguish binary compatibility, semantic freshness and measured quality. A
folder adapter trained on removed/changed descendants is not automatically
fresh. An edit may invalidate siblings through a shared ancestor; incremental
cost is the dependency closure, not necessarily one lambda.

### Incremental execution and commands

Incremental builds may first fit a leaf against explicitly pinned frozen parents
as a fast **candidate**, then complete the affected shared refit. The candidate
records which older semantics those parents saw and is not advertised as a
fully refreshed joint build. Warm-start compatible weights; preserve optimizer
state only when its exact continuation contract remains valid, otherwise record
the new optimizer history. Re-evaluate the actual fallback and candidate stacks.

Jobs are resumable and idempotent by input receipt. Watch mode coalesces edits,
isolates in-progress builds, and compares the source/release revision before
publication so an older job cannot overwrite a newer result. Full joint refits
run on a configured schedule and after a configured number of incremental builds.

- `natlang tune <program>`: discover the program; capture/reuse data; verify, fit,
  place and export for its selected executor/backend; load, validate through the
  normal inference path and atomically activate. Success means ready to call.
- `natlang tune <program> --incremental`: rebuild the stale dependency closure.
- `natlang tune <program> --watch`: resumable background incremental builds.
- `natlang tune --families`: versioned cross-program discovery/training/refits.
- `natlang tune --report <program>`: exact candidate/release results, ablations,
  provenance, costs, time to first tuned call, degradation and incomplete work.

Progress reports identify the current stage and a resumable job; rerunning the
same command resumes/reuses valid work. The ordinary run/call path remains
available during tuning with its current compatible release or declared base
fallback. Training is an explicit build operation; a normal inference request
does not unexpectedly start a teacher job or block on retraining.

Reports distinguish collected, admitted, trained, exported and actually served
artifacts. Receipts identify all affected dependencies so a regression can be
bisected without pretending one adapter necessarily has sole causal credit.

## 9. Evaluation

Evaluation runs alongside implementation and on builds. It informs repairs and
publication choices, without cancelling required work. Protect data according
to §4.1 and report a versioned candidate, workload and measurement protocol.

The foreground per-program build validates its selected candidate, compatible
backend and readiness through the ordinary engine. Full capacity sweeps,
cross-backend comparisons and fleet-wide evaluation run through the wider work
schedule; they do not add unrelated prerequisites to first use (§1.1). Include
time to first tuned call and the number of manual author actions in acceptance
reports. Exercise the same CLI/SDK entrypoints authors actually use.

| Arm | Question |
| --- | --- |
| Untuned LFM and Mellum | What is each exact base's task performance? |
| One program adapter | Does per-lambda structure help at comparable capacity/cost? |
| Flat per-lambda adapters | Does sharing help relative to independent specialization? |
| Hierarchy before placement | What does joint adaptation achieve? |
| Hierarchy after placement | What quality/cost tradeoff does removal/refitting achieve? |
| Hierarchy with families | What is the within-program gain and unseen-program transfer? |
| Distilled base without program adapters | Does transfer persist for programs excluded from distillation? |
| Teacher reference | What does the selected teacher achieve under the stated tool/input constraints? |

The teacher is a reference, not a guaranteed ceiling. Report both native cost
and comparable cost envelopes rather than claiming exact cost matching where
there is no comparable operating point. Fix tool access, root cases, decoding
policy, retry rules and concurrency per comparison; report total program cost,
including inner calls and errors. State training-budget differences separately.

Use paired root cases and confidence intervals clustered by independent
source/world component; turns and correlated teacher samples are not independent
trials. Report counts, seeds/repeats and unresolved uncertainty. Leave-one-level-
out results are conditional effects in a particular stack, not contributions
that can be summed. Select ablations on development data and predeclare those
reported on held sets.

**Per program:** task success and failure categories; real/synthetic and
oracle-backed/weak-evidence strata; macro and traffic-weighted results; all
source/export/resident bytes; latency/throughput/cache/fragmentation; teacher
tokens and cost; training/build time and compute; actual exposure; requested
versus effective stacks and fallback rates. Quantify family negative transfer
and the fraction of calls whose low data counts prevent a reliable estimate.

**Regression contract:** with the same base, backend, inputs, seed and runtime,
disabling all adapters preserves the untuned execution path and its numerical
tolerance. A base update is expected to change outputs and requires task
regression/transfer evaluation; do not promise byte-identical behavior across
base revisions. Re-evaluate affected programs after shared/family updates and
verify that unrelated bindings retain their own identities and behavior.

## 10. Decisions and their limits

| Question | Decision |
| --- | --- |
| Caller adapter inheritance? | No lexical inheritance; explicit experiment scopes append separately |
| Shared DAG node's parent? | Binding manifest declares scope ancestry; aliases keep the callee's binding |
| Multi-family membership? | Yes: static per-binding, bounded learned gates over frozen family versions |
| Placement policy? | Development-selected global defaults and per-program cost/quality overrides |
| Tree-root discovery? | Author declarations plus measured, refitted placement proposals |
| Which teacher? | Authorized actors calibrated on development quality and complete cost |
| Which student? | Implement both; select the program executor on development measurements |
| Exact exported rank? | Common nested bases: at most maximum rank; general factors: at most rank sum |
| Native state across stacks? | Explicit same-base communication with receipts; no implicit cache reuse |
| Missing/stale artifact? | Declared dependency-consistent fallback, visible effective identity; strict evaluation |

## 11. Required work list

All items remain in this effort. Implement in dependency order while advancing
independent work; a checked item needs its artifact and actual evidence.

**Data**

- [ ] Real-root import and training-only root synthesis with source/world lineage
- [ ] Global split enforcement through expansion, family learning and distillation
- [ ] Complete inner-call/trajectory capture and replayable service worlds
- [ ] Authorized teacher integrations, calibration and routing with full costs
- [ ] Integrate the current six-worker Luna generation instruction with its
  supervisor, separately from engineering helpers and without duplicate pools
- [ ] Direct/weak evidence schema, confidence calibration and disagreement handling
- [ ] Student-prefix corrections with verified continuations and graph provenance
- [x] Stable binding/inline-site IDs and explicit edit/rename migration (`ts-host/src/tune/sites.ts`)

**Training**

- [ ] Path-routed joint trainer, balanced trajectory objective and depth penalties
- [ ] SFT and RL objectives over the same tunable state, alternating in one recipe (§5.1a): whole-graph
  on-policy rollouts with exact recorded actors, oracle rewards, group advantages and clipped policy updates
- [ ] Shared nested `nnt1` bases/cores and free-factor parameterization
- [ ] Immutable-snapshot gradient updates, actual-update checks and exact continuation
- [ ] Per-binding family gates with fixed family scale/version contracts
- [ ] Placement, interactions, refits, pins and subtree discovery using measured costs
- [ ] Invariant family clustering, balanced training and dependent program refits
- [ ] Rank/penalty/capacity sweeps and development-only defaults/student selection
- [ ] Verified stack/data distillation into each base followed by adapter rebuilds
- [ ] LFM and Mellum coverage with shared recipes and declared architecture differences

- [ ] Soft instructions and soft skills (§3.3): text-initialized native states per binding and bound skill,
  manifest binding, joint state/weight updates through actual consumption positions, re-initialization on base
  change, runtime substitution in prompts and skill disclosure, and backend capability fallback

**Runtime and serving**

- [ ] Manifest-bound lexical scopes over context DAGs, `.in()` and library imports
- [ ] One setup/tune/ordinary-call workflow with automatic active-binding
  discovery, backend loading and activation across CLI, SDK and browser
- [ ] Explicit experiment scopes and async/browser isolation
- [ ] Atomic release pinning, requested/effective identities and coherent fallback
- [ ] Call/prefix/KV/hybrid cache compatibility and rollback behavior
- [ ] Existing explicit same-base native imports, complete occurrence receipts/gradients
- [ ] Safetensors/GGUF export with verified scale/rank/module/expert mappings
- [ ] vLLM multi-LoRA routing, residency and required native-state integration
- [ ] llama.cpp grouping and batched multi-LoRA kernels, written for upstreaming
- [ ] wllama/OPFS loading, binding scopes and hardware-specific measurements
- [ ] Full capability matrix and algebra/numerical/cached/program parity evidence
- [ ] Cost/fragmentation/cache measurements feeding placement

**Lifecycle, evaluation and documentation**

- [ ] Acyclic build keys, reverse dependencies and invalidation matrix
- [ ] All five CLI modes, resumable jobs, watch races and scheduled full refits
- [ ] First-use defaults, trace/data/cache reuse and progress/readiness reporting;
  global maintenance does not block a program's selected release
- [ ] Every evaluation arm, paired uncertainty and held-program transfer reports
- [ ] Conformance: DAG aliases/rebinding, nested/parallel scopes, exact exports,
  wrong-base rejection, explicit cross-stack states/gradients, degraded cache
  identities, ancestor invalidation, interrupted publication and rollback
- [ ] End-to-end author-flow cases: setup/tune/run and tune/call without manual
  manifests or flags; unchanged SDK calls; readiness/activation failure;
  new-task release refresh with old tasks/pins preserved; cold/warm timings
- [ ] Spec, author guide, authoring/integration skill updates and API migration
- [ ] Live measurement of actual model-facing changes under repository rules

## 12. Risks and responses

| Risk | Response |
| --- | --- |
| Synthetic distribution shift or correlated duplicates | Real-root capture, component lineage and stratified program results |
| False inner-call credit or correlated teacher errors | Separate evidence, direct checks and controlled replay where useful |
| Adaptive evaluation leaks into training | Global split receipts; development decisions; protected final estimates |
| Sharing exists only in the regularizer's interpretation | Refitted ablations, flat/program baselines and transfer measurements |
| Frozen subspaces restrict capacity | Free-factor/full-tuning controls with matched exposure and cost disclosure |
| Factor/gate ambiguity or invalid clustering | Frozen family scale/version, bounded gates and effective-delta similarity |
| Export rank, tensor coverage or rounding differs | Explicit target inventory and backend-specific parity before claiming support |
| Source bytes look small but exports fragment serving | Measure duplication, residency, ranks, cache/load costs and request mixtures |
| Frequent edits invalidate shared ancestors | Dependency-closure builds, explicit provisional candidates and coherent fallback |
| Native state or cached results cross actor identities | Exact effective identity; explicit communication mode and occurrence receipts |
| Distillation loses specialized behavior | Verified source mix, per-task retention/transfer measurement and rebuilt adapters |

## 13. Definition of done

- Every production-packet program can be tuned from its permitted training data
  with one `natlang tune` command on both students; protected rows stay excluded.
- From a configured installation and ordinary program/examples, the author can
  tune once and execute through the existing run/call/SDK path without handling
  adapter files, IDs, exports or server-loading steps. Successful completion
  means the selected tuned release has been exercised and is ready in that path.
- Cold and warm time to first tuned call is measured on declared hardware;
  required global family/sweep/distillation work is independent of program
  readiness, and runtime requests do not unexpectedly initiate training.
- Bound tuned programs execute on the reference server, vLLM, the llama.cpp fork
  and wllama with declared target/state coverage, measured parity and actual
  hardware evidence. Required backend gaps remain visible until implemented.
- Families have been learned across programs for each student, and each base
  has completed at least one distillation/rebuild cycle with measured outcomes.
- Reports include all arms, real/synthetic success, evidence strata, uncertainty,
  ablations, unseen-program transfer, source/export/resident cost and throughput.
  Report negative results honestly; implementation completion is not proof of lift.
- Incremental edits, shared-parent changes, missing artifacts, concurrent calls,
  interrupted builds and rollback satisfy the conformance contracts.
- Spec, guides, skills, API migration and implementation decomposition are merged;
  actual model-facing changes have their required live measurements.

## 14. External integration references

Consult the pinned implementation as well as these upstream contracts. Versions
and exact capabilities used by a build belong in its receipt.

- [LFM2.5-350M model configuration](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/config.json): layer and architecture inventory.
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/): architecture-level support, not proof of every target/state mode.
- [vLLM LoRA serving](https://docs.vllm.ai/en/latest/features/lora/): request IDs, loading and rank/residency configuration.
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora): factor/scaling, target-module and expert-parameter integration.
