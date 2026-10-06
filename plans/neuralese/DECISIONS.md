# Neuralese programme: decision log

Every decision of the programme, with date and reason. Newest entries go at the
end. The [README](README.md) decisions table is the summary; this log is the record.

| # | Date | Decision | Reason |
| --- | --- | --- | --- |
| 1 | 2026-10-03 | bgkit and Schnitzeljagd stay as they are; they are sources of data and lessons only. No checkpoints, encoders or machinery are reused. | The owner wants a clean implementation of the single sketch-writer design. |
| 2 | 2026-10-03 | Use all data from all teachers; restrictive licences (including CC-BY-NC) are acceptable, recorded per record. | Maximise data; licences stay traceable. |
| 3 | 2026-10-03 | The `werg/neuralese-` repository is obsolete; the specification lives in this repository. | The source documents' footnotes point to it, but it was design-only and superseded. |
| 4 | 2026-10-03 | All code lives in the natlang repository. | One home for spec, runtime, training and servers. |
| 5 | 2026-10-03 | Start from the best Natlang crisp student (LFM2.5-350M with LoRA), selected by execution evaluation; merge its LoRA into a frozen base with S3 changes as separate deltas. Final model size is decided later. | Natlang competence is the point; lower held-out loss has not yet meant task wins; deltas-off gives the self-distillation teacher. |
| 6 | 2026-10-03 | The whole DGX Spark is available; no throughput-costly memory savers such as gradient checkpointing. | Training speed over memory frugality. |
| 7 | 2026-10-03 | Neuralese is a first-class target of llama.cpp (and wllama) and vLLM, in forks designed for upstreaming; no Python-only serving and no text fallback. Browser storage is OPFS. | Make Neuralese an attractive target for others. |
| 8 | 2026-10-03 | No separate KL teacher model; the teacher is the same model given the full source. | Simplicity; tokenizer compatibility. |
| 9 | 2026-10-03 | `Neuralese<T, D = DefaultDialect>` is a real type with full infrastructure. There is no `Crisp<T>`; ordinary types keep their exact meaning. `Handle<T>`, `Structured<S>`, `NeuraleseFunction<F>` are not needed. | Neuralese extends the semantics; records of soft fields and applicable soft functions cover the helper types. |
| 10 | 2026-10-03 | The combinators are library functions, not syntax: `map`, `zip`, `ap`, `combine`, `empty`, explicit `split`/`splitList`, `read`, `convert`, `gloss`. Each is a system natural-language function with a trainable soft body. | A sensible functional algebra around the type; trainable operators. |
| 11 | 2026-10-03 | The laws hold approximately and are trained to some degree; the compiler rewrites programs with them, a rule enabled automatically once whole-program comparison on the current model shows no harm, every rewrite traced. | Cheaper programs; the owner wanted law-based rewrites included. |
| 12 | 2026-10-03 | Soft functions are applicable: `Neuralese<F>` is callable. | Owner decision. |
| 13 | 2026-10-03 | Purpose is not part of the type and not an argument of any combinator; it is encoded at the write site. Re-encoding for a purpose through a separate call is an anti-pattern. | The best place to encode purpose is where a value or function is written. |
| 14 | 2026-10-03 | One literal form: the markers `<|neuralese|>`…`<|/neuralese|>` wrap only the vectors; type, signature and captures are plain TypeScript before it. The same form is used for writing (eval code, tool arguments) and listing. Function literals are `nl.with({ … })` with a soft body. | It ties the write and read ports to the language; the annotation conditions the writer. |
| 15 | 2026-10-03 | Two renderings only: reference form (`$neuralese` with a unique ID) everywhere outside the model, model form at the token level. Glosses are diagnostics. | Simplicity. |
| 16 | 2026-10-03 | Training data is rewritten to eager typing and explicit captures by the compiler. | Types come before literals; the model learns to name captures; function-literal samples. |
| 17 | 2026-10-03 | No budget in the language; the stop head decides length; the runtime has a hard maximum. | Writers produce content until done; limits are a runtime matter. |
| 18 | 2026-10-03 | One general `.nz` file format (safetensors with a typed header of named exports); soft functions and skills are exports. | Neuralese-bearing data needs a typed file format of its own. |
| 19 | 2026-10-03 | Learning uses immutable values: `grad`, `valueAndGrad`, `stopGradient`, nested `grad` (first-order by default), `logLikelihood` and law objectives, optimisers, `save`, with steps run by `iterateOn`. No mutable parameters. | `grad` is elegant over arguments; gradient descent and learned updaters share one shape. |
| 20 | 2026-10-03 | A function is a curried function of its immutable, content-addressed context; rebinding with `in`; skills are context items; directory reducers compute new contexts; self-improvement maps contexts to contexts. Existing spec and data are migrated. | Makes skills (currying) and `grad` over embedded values explicit. |
| 21 | 2026-10-03 | New executable nodes (`.nl` and TypeScript functions) come only from files; existing nodes may be edited within their interfaces; data may be added and edited freely; unions of file contexts are allowed. | Keeps runnable code compiler-checked while leaving skills, data and learned values open. |
| 22 | 2026-10-03 | No explicit fuel. Termination: acyclic contexts, function-typed captures by value, no recursive function types, the existing nesting limit on runtime-written functions; `iterateOn` with a TypeScript predicate needs a hard bound, a natural-language predicate runs under a dedicated anti-divergence prompt; `checkProgress('off')` needs a bound. | Explicit fuel is the lazy option; semantic, approximate Turing incompleteness is acceptable; crisp predicates were a backdoor. |
| 23 | 2026-10-03 | Explicit captures of Neuralese function literals are snapshots by default, `live(x)` opts in; text `nl` keeps implicit live captures and `let` write-back. | Stored functions need snapshots; immutability is a design tool, not a goal. |
| 24 | 2026-10-03 | Dialects are lightweight version tags; when the model changes, bump the version; no holding of dialects or conformance-preservation objectives; adapters only where a different-width model must share a dialect. | Dialect stability matters only once there is a large user base. |
| 25 | 2026-10-03 | Literals stay inside the eval code string of the native Pythonic tool call, control tokens inside the quoted string. | One uniform form. |
| 26 | 2026-10-03 | Skills are standard agent-harness skills (`SKILL.md` + files) extended by a `natlang:` frontmatter block; `.nz` skills hold the same fields in metadata; progressive-disclosure loading plus automatic scope injection; skills are bound in contexts, with a global mutable pool at the top level that affects programs only on rebinding. A skill may become purely soft. | Compatibility with existing skills; optimisation needs fixed contexts. |
| 27 | 2026-10-03 | Crisp skill authoring is a prerequisite, integrated into self-improvement with its own corpus; it is the basis for our meta-learning. | Soft skills are initialised from crisp authoring. |
| 28 | 2026-10-03 | All execution environments are built (SQLite, test runners, world simulators, SWE containers). | Checkable tasks for S1, S5 and S7. |
| 29 | 2026-10-03 | Paid teachers (Luna, Bunny) are used alongside local Qwen3.6, without a planning ceiling. | Data volume and quality. |
| 30 | 2026-10-03 | No fixed pass/fail thresholds; stages evaluate continuously against rubrics and review checkpoints decide next steps. Mechanical tolerances are fine. | Evaluate and react rather than pre-commit to numbers. |
| 31 | 2026-10-03 | Models and `.nz` libraries are published on Hugging Face; specifications and conformance suites on GitHub. Release licensing is not part of the plan. | Owner decision. |
| 32 | 2026-10-03 | Control tokens use LFM2.5's `<|reserved_7|>` and `<|reserved_8|>`. | Unused reserved IDs avoid resizing the vocabulary. |
| 33 | 2026-10-03 | S0 declares the surface in `spec/neuralese.d.ts` as three ambient modules: `natlang:neuralese` (type and combinators), `natlang:context` (contexts and rebinding) and `natlang:learning`. Model-written literals are replaced before type checking by `__neuralese.value(id)` or `__neuralese.body(id)`. | Lets TypeScript enforce opacity, applicability, nesting and dialect mismatch; the natlang compiler adds the rest. |
| 34 | 2026-10-03 | Neuralese writing is stochastic, gated by a Neuralese temperature: the content projection predicts mean and per-dimension scale, payload `μ + τ·σ⊙ε`, deterministic at `τ = 0`. Training uses it with an optional KL prior; `logLikelihood` includes payload log-densities; `.nz` blocks may be Gaussian distributions. | VAE-style robustness of representations, and a tractable likelihood so encodings can be trained directly, including through RL and sampling-based estimators. |
| 35 | 2026-10-03 | The port uses transformers 5.18 (multi-token continuation after cached conv state) or an equally efficient own forward, in a dedicated natlang venv. | Avoid the inefficiency of the 5.5 workaround. |
| 36 | 2026-10-03 | The llama.cpp fork is a separate checkout (`/home/werg/llama.cpp-neuralese`, branch `neuralese`, based on upstream `1537a0a8b`), pinned by `training/neuralese/llama-cpp-fork.json`, not vendored or a submodule. The core patch is one general API, `llama_set_layer_range` in the staging header `llama-ext.h` (implemented for LFM2), reusing upstream's `llama_set_embeddings_layer_inp`. Neuralese heads ship as a sidecar projector GGUF loaded by a `tools/neuralese` library, like mtmd's mmproj; the model GGUF stays a stock LFM2 GGUF with control rows merged. The write procedure runs the sketch and completion passes on two scratch sequences sharing the prefix (`kv_unified`, `n_seq_max >= 3`), so the main sequence needs no snapshot/restore. | Keeps the natlang repo small; a minimal, generally useful core patch (layer ranges also serve layer-skip self-speculation) for upstreaming; no model-loader changes. |
| 37 | 2026-10-04 | Programs may ship small weight adapters (`Adapter<Base, Kind>` exports in `.nz`) as context artifacts scoped to a program, function or skill; `grad` and learned updaters may produce them. Every learning regime (supervised, RL, conditioned distillation) is available both as direct training and through a learned improver distilled from recorded direct-training trajectories; a reward-blind `improve` operator is a programme goal. Plan: [LEARNING_CONTINUUM.md](LEARNING_CONTINUUM.md). | Owner direction: weights should be adaptable and shippable with programs, and the improvement process itself should be meta-learned end to end. |
| 38 | 2026-10-04 | The design choices of LEARNING_CONTINUUM.md §14 are defaults, not rules: RL for tiny adapters, adapters on layers after the cutoff, LoRA-convertible parameterisations, adapters as Neuralese codes decoded by a trained projection per base model, deltas for learned updates, the memetic optimiser, two reward-blind visibility classes. Runs may override them and record why. | Owner: keep these as starting points that evidence can revise. |
| 39 | 2026-10-04 | Models never write deltas. Learned updaters write ordinary Neuralese blocks; a trained delta projection `D` per artifact kind (part of the base model's port machinery, like the adapter projection `P`) turns a written block into `Delta<A>`. The `#delta` dialect is host bookkeeping for computed deltas (`diff`, `compose`, scaled ablations) only. | Owner: asking the model to emit deltas directly is a stretch; one semantic space for everything the model writes keeps updates readable and the writer's training unchanged. |
| 40 | 2026-10-04 | Runtime-owned prompts are soft artifacts. The central system prompt and every extension (depth-limit variant, function tools, directory reducers, program guidance, decision readout, natural-language predicate, compaction notice, handover frame, automatic note, turn notices) has a stable piece ID and a `Neuralese<SystemPrompt>` form initialised from its text. The forms are trained with the base model (S3, S5: parameters next to the LoRA), ship with the checkpoint as a `.nz` bank in the model's dialect, are used by the runtime under a Neuralese driver (crisp text otherwise and as the initialisation), and are artifacts of the self-improvement system (gradient, memetic and learned-updater operators may adapt them, recorded as `system-prompt` artifacts). | Owner: of course the system prompt and its additions should be text-initialised, freely trainable and adaptable. |
| 41 | 2026-10-04 | The training-data converter turns everything that can be Neuralese into Neuralese and inventories the rest. Always converted: runtime-owned prompt pieces (decision 40) and compaction handover notes (the `compact_history` note becomes a model write; the pinned note reads that block; the crisp note is the teacher's view and the write's source). Converted by curriculum step (S5 §6): call instructions (soft bodies), skill bodies, `nl` literals, values with only model consumers. Every remaining site is counted with the reason it stays exact (exact by nature, host consumers, no consumer trace). | Owner: make sure the converter turns everything that can be Neuralese into Neuralese, including compaction handover notes. |
| 42 | 2026-10-04 | What becomes Neuralese is decided by use. Convert a value when it is reused (prompt pieces, program guidance, skills, instructions serving several calls: one shared soft parameter), when one agent produces it and another consumes it (handover notes, child-call results), or when it is large and a digest saves context (cut-off listing values). A value read once by the call that produced it stays text. A share of single-use stored instructions converts anyway so that position has Neuralese samples. Text becomes Neuralese without a summarising call: `encode` reads it in one forward pass through the port (supplied-input write, one vector per token); only digests and model-produced values go through the writer. Amends 41. | Owner: don't pay an extra pass to convert single-use values; the ideal cases are heavy reuse and agent-to-agent handoff; large items get separate summaries to save listing context. |
| 43 | 2026-10-04 | A dedicated `digest` operator writes large values compactly for a call's opening listing. When an argument's listing would be cut off, the runtime writes its digest at a write site conditioned on the receiving call's instructions (system: the `digest` prompt piece; user: instructions, then the full value; reply: `const digest: Neuralese<Digest> = ` and the write) and lists the block with a note that the variable holds the whole value for exact reading. The stop head decides the digest's length; it trains through the listing's consumers with a length cost (S3 phase E). A value longer than one write site's window (the model's context less the site's own text) is split into token chunks: each chunk is digested at a part site and a combine site reads the part digests and writes the final one, so no value is cut. Every write in the plan is the differentiable write procedure, so the listing's consumers train the writer. Runtime: `neuralese.digest` (`serverDigester`); server: `/v1/neuralese/digest` (the plan, `digest.py`) and `/v1/neuralese/write`; training: converted digest sites and `train.trajectories --digest written`. | Owner: encode larger items separately as summaries to save context in the initial listing. |
| 44 | 2026-10-04 | Neuralese operators answer by template readout, not free decoding. A call that declares `readout: template` (every standard-library combinator does) keeps the agentic trajectory format: its opening is rendered as for any call, and its first reply is forced to `return_result(status="success", value=…)`, cut from the model's own chat template at the value. A Neuralese result is written as a block at the value and the call is closed (the stop head decides the length); any other result is decoded from the value on. The value is checked as any return; a rejected value falls back to ordinary turns. Servers take it as the `neuralese_template` request field (reference and fork), and the trajectory trainer forces its write sites with the same cut (`chat.call_reply`), so what runs is what trains. | Owner: operators and combiners should not run as autoregressive decoders; generate a sensible trajectory template and read out the resulting encoding or return value at the end, keeping the natlang agentic trajectory format. |

### 2026-10-05 — foundation before compression

Pause compressed recurrence after failed embedding controls. Require an explicit raw, token-aligned identity reference and held causal next-token feedback qualification before compression. Distillation completion is not qualification. Same-position hidden-state inversion is experimental; no production protocol/dialect change is approved by these diagnostics. Preserve the paused optimizer state and all diagnostic failures. See HANDOVER.md for evidence and outstanding unified runtime work.

### 2026-10-05 — exact raw foundation before shallow optimization

The shared default first qualifies the actual full-depth output head and native
normalization as an exact next-token embedding reference. A zero-initialized
state correction already meets that target; optimize only where a shallow
variant needs it. Required source/context gates and actual serving/replay handoff
are shared code, not per-machine conventions. Raw transport changes positions and
scale relative to legacy RMS/marker checkpoints: preserve their diagnostics,
re-encode text sources, and prohibit silent soft-parameter reuse/native export.
Autonomous stopping and semantic compression remain separate learned qualities.

### 2026-10-05 — Typed writer warm-up and explicit curriculum continuation

The shared raw recurrence recipe now declares producer text supervision (weight 1)
under its actual soft ancestor context, and an uncompressed one-vector-per-token
warm-up. Previously the reader objective could replace the producer's own gold
body with its opaque output, leaving no direct source-body supervision. Evaluation
never uses those labels to generate a block. Curriculum changes require named
`--curriculum-change` declarations while retaining full optimizer/RNG state.

Structured `Neuralese<unknown>` writes must start at the model's native unquoted
value boundary. The host quotes only its opaque wire placeholder for tool parsing;
the model cache remains native typed syntax. Serving, training, and execution
probes share this boundary, with string writes preserving their quoted boundary.

Paired evaluation completeness counts eligible writer/reader pairs, not every
held-out turn. A lower own-context CE than shuffled-context CE can select a
candidate checkpoint; it does not qualify semantic correctness or stopping.
Nonfinite gradients must fail before an optimizer update.

The frozen local v2 run stopped after its finite full-state step-400 checkpoint:
staged replay saw all-NaN output. The checkpoint's floating tensors were checked
and are finite. Preserve that failure; the numerical cause remains under review.

### 2026-10-05 — Production typed boundary gate and local allocator recovery

Runtime qualification now independently compares an actual structured template
write against ordinary causal generation after the native unquoted value prefix,
and checks restoration of the opaque host argument. The CPU proof passed all
controls and is registered as `local-raw-typed-runtime-proof-20261005-v4`; this
qualifies foundation transport, not trained recurrence semantics or stopping.

The producer-supervised v3 warm-up passed staged replay but hit backward OOM on
the nine-producer step406. Preserve its failure and step400 recovery state. The
v4 retry retains identical frozen code and optimizer/RNG, removes temporary
anomaly tracing, enables expandable CUDA allocations, and increases only the
machine resource envelope from7 to7.35GiB. Resource allowances are excluded from
training identity; objective, weights, data, and learning rates remain pinned.

### 2026-10-05 — Stable finite-gradient clipping

Accumulate recurrence gradient norms in FP64 before clipping. BF16/FP32 squared
norms can overflow even when every individual adjoint is finite, spuriously
rejecting a valid step or scaling it to zero. Tests exercise finite3e30/4e30
adjoints and preserve their normalized direction; NaN gradients still reject
before any optimizer update. This numerical safeguard does not establish the
cause of the old v2 replay failure. The current frozen v4 run keeps its existing
code until an explicit full-state handoff.

### 2026-10-05 — Profile recurrence dispatch and narrow checkpoint cache inputs

A40-second live py-spy profile of Pop v4 sampled the main trainer thread:
explicit full GC in staged add accounted for7.8% and graph-object release for2.5%.
The process saturated one CPU core; a separate12-second GPU sample ranged0–31%
(mean20.6%). This is a phase sample, not a whole-run utilization estimate. Training
is batch1; sequential full-depth writer steps, per-layer checkpointing and staged
replay emit many small GPU operations. Large chains reach6.62GiB allocated, so
increasing batch indiscriminately risks OOM.

Each LFM layer checkpoint formerly passed every layer's cached tensors despite
using only its own state. Pass only the current layer's tensor fields and return
only its updated tensor fields; reconstruct the cache outside checkpointing.
No opaque cache or unrelated tensor graph is captured in the closure. Cached
recurrence output, input gradients, weight gradients and cache-container release
pass the CPU regression. GPU replay and performance still need verification in
an explicit full-state code continuation. Do not claim a throughput gain yet.

### 2026-10-05: typed read boundaries and measured batching equivalence

Unknown-valued tool arguments transport their type alongside the opaque block.
Preserve argument order and remove only the synthetic quote around a direct
unknown argument; an unknown block embedded in a code string keeps that string's
quotes. Serving and gradient rendering share the resolver.

Equal-length batched GEMMs are numerically equivalent, not necessarily bitwise
equivalent. Content-addressed IDs remain hashes of actual payloads. Tests compare
payloads and each reply's real references instead of demanding identical hashes
from distinct matrix execution shapes. Old frozen runtime reproduces the former
false assertion; numerical test tolerance is 1e-5.

GPU bfloat16 step450 known-source controls reproduce CPU results (crisp4/4,
encoded0/4), so the encoded-source regression is a failed quality gate rather
than a CPU-only precision artifact. Instruction-parameter intervention flags do
not affect source-control arms, which retain crisp instructions.

### 2026-10-05: split local supervision tapes without truncating recurrence

Staged producer execution releases the writer graph before creating its
independent gold-text auxiliary graph. Reverse replay accumulates both into the
same fixed parameters and child leaves before replaying children or updating
parameters. Preserve one combined local term per supervised producer and the
original chain/batch normalization. Joint execution retains its original
combined graph. Router observations sum the two independent tape costs because
they still coexist in joint mode; using the smaller staged peak there would
incorrectly reduce predicted joint memory. Branching-DAG input/parameter
gradients agree with joint execution at multiple scales, with/without stop loss.
GPU memory/throughput validation pending; do not claim true tensor batching.

Known-source encoding diagnosis: direct embedded and transparent controls4/4;
trained encoder0/4; diagnostic zero content residual restores4/4. The latter is
an explicitly modified-weight intervention, not checkpoint qualification.
`--content-projection identity` is diagnostic only and never rewrites checkpoints.
The content residual is implicated; decide shared raw-stage anchoring/freezing
policy separately, preserving later semantic/compression optimization capability.

### 2026-10-06: offload hook graph ownership

Offload retained-small-tensor paths must return detached views, preserving data
and version rather than storing original tensors with their grad_fn. Original
tensors through a Python saved-tensor hook can retain C++ autograd cycles, keeping
staged local graphs alive. Live CPU budgeting alone does not fix that. v9 failed
753 before the old759 case; its frozen runtime and failure log are preserved.
New GPU regression asserts intermediate weakrefs disappear after each staged
primal/replay, and exact gradient/storage-reuse tests remain12/12 passing.

### 2026-10-06: share differentiable attention prefixes

Autograd attention caches retain an immutable differentiable prefix and a growing
suffix. Checkpoints save these separate tensors; repeated prefix entries share
storage rather than retaining a full context concatenation per generated vector.
Attention receives the same ordered materialized K/V, and gradients still reach
the prefix and all suffix positions. No detachment/truncation/shortened context.
Inference's copy-on-write KVBuffer path is unchanged; batch-row selection and
checkpoint reconstruction preserve prefix fields.5/5 CPU tests cover tensor
identity, prefix/suffix gradients, cached transformer input/weight gradients,
fast/reference primal equivalence and snapshot branching. GPU lineage fit pending.

### 2026-10-06: GPU utilization and recurrence memory follow-up

v11 (frozen24c2fc3f) passed a thirteen-producer chain at step816 with
zero replay error, 1.58GiB allocated peak, no CPU offload and 96.862s.
Earlier v10 comparable, different cases took about111s with6.34GiB GPU
and2GiB live CPU offload; this is not a controlled speed comparison.
The job remains CPU-dispatch/sequential-autoregression bound; --batch is
serial accumulation, not tensor batching. Do not claim full utilization.

The shared estimator now versions cache geometry, drops stale memory
calibration/routes when layout changes, and counts one shared prefix plus
growing suffixes. It preserves weights, optimizer and RNG. Only the LFM
checkpoint implementation declares this geometry. Full-depth frozen exact
zero feedback correction skips its MLP with weight-version/trainability
guards; load or modification invalidates the shortcut. Targeted CPU tests
cover primal/input-gradient equality and invalidation. Runtime fit remains
separate from semantic qualification: prior encoded/written semantic probes
failed; training completion or replay agreement does not grant admission.

Audit: 48/670 producer gold payloads exceed the current128-vector bound
(24 train,24 heldout; max145). This is silent payload clipping, not a prompt
context cap. Shared recipe declares512; explicit capacity extension and
continuation/qualification work remains required. No clipping fix claimed.

Closed v7-v10 recovery artifacts are registered and SHA-verified on DGX;
local v7/v9 checkpoint750 copies were then evicted with availability receipts.

### 2026-10-06: probe-local deterministic producer reuse

Paired written/shuffled evaluation reuses producer payloads across readers
within one no-grad probe, keyed by producer+depth. Weights/leaves are fixed
for the scope, and the memo is discarded on return. Existing admission for
sharing requires an acyclic DAG, no random max-writes and no sampled stop
policy. Training graph/penalty accumulation remains per-reader unchanged.
Report cache hits/entries; probe-local cache cannot survive optimizer updates.
5/5 producer tests cover shared gradients/depth, stochastic opt-out, and
exact deterministic multi-reader results with fresh values after weight change.
This reduces duplicate evaluation autoregression; it is not tensor batching.

### 2026-10-06: opt-in attention-only layer checkpoints

The all-layer policy remains the default. --checkpoint-attention-only requires
--checkpoint-layers and retains convolution-layer activations while retaining
attention checkpointing/shared-prefix storage. This is a resource trade-off,
not a shortened graph or changed loss. Mixed geometry accounts explicitly for
uncheckpointed layers and resets incompatible calibration; execution policy
is recorded in full checkpoints and accepted by the shared recipe handler.
Six CPU tests pass, including real350M cached-prefix recurrence primals, input
gradients and convolution-weight gradients against fully uncheckpointed and
fully checkpointed execution. Actual GPU memory/speed qualification pending.

First50-minute v13 sweep: steps898–1048,129 joint updates averaging4.728s,
22 staged averaging106.923s (79% of measured update seconds),errors0/replay0.
Joint maxpeak3.28GiB,staged1.60GiB; CPUoffload0. Probe1024 reuse8hits/12entries,
writtenCE.42292 vs shuffled.42348 (6/12better); gap nearlyzero, not semantic
qualification. Fetch/merged DGX127249f1 context512 Maple recipes.

### 2026-10-06: scope selective checkpoints to staged writers

Global attention-only v14 fits thirteen-producer1070 (82.788s,4.12GiB,
replay0,CPUoffload0), but larger retained tapes also force small previously
joint-shaped cases to stage. Different source cases are not a controlled speed
comparison. New --staged-checkpoint-attention-only selects the lighter
checkpoint policy only inside staged writer gradient replays when live baseline
plus mixed-layer geometry fits the graph budget. Readers, gold auxiliary
losses and joint attempts keep all-layer checkpoints; oversized contexts keep
all-layer checkpoints too. Global and staged-only options are mutually
exclusive; both opt-in and available through declared recipe parameters.

Writer policy restores on return/exception. Attention recomputation reconstructs
its own exact layer/cache inputs, independent of the later policy. Selective
staged-writer tape observations cannot contaminate all-checkpointed joint
admission calibration. Full weights/optimizer/RNG continue unchanged; execution
policy is checkpointed.17 CPU memory/state tests pass including live-budget
admission,32k fallback on Pop and larger-envelope admission, and unchanged
layout dictionary. Earlier real350M selective primal/gradient tests6/6 passed.
GPU staged-only fit/performance still pending.

### 2026-10-06: eliminate silent supervised payload clipping

Supervised source length now uses exact ceil(gold-token-count / explicitly
declared tokens_per_vector) and raises when capacity is insufficient, before
training updates for indexed handover producers. No implicit min(capacity).
New --max-write-vectors is a named continuation curriculum change; shared
raw-recurrence recipe explicitly declares512, aligning with its runtime stage.
This is a payload bound, not the65536-token prompt context bound.48/670
current producer sources exceed128 (24train/24heldout,max145).

Constant-position stop heads always read row0. Their write capacity can change
without replacing any parameter, resizing any tensor, consuming RNG or changing
optimizer moments; unused position storage remains unchanged. Position-dependent
heads reject bounds beyond learned rows. Deployment reconstructs stored position
tensor shapes and separately restores explicit capacity; trajectory evaluation
restores the continuation's capacity too. Existing checkpoint bytes remain
immutable. Expanded runtime/control behaviour and trained channels still need
exact-weight semantic/transport qualification; no certificate inherited from
this resource/curriculum change.

v15 staged-only selective policy actual1072:17nodes,34 grad-enabled writer
calls (primal+replay),replay0,CPUoffload0,154.524s INCLUDING a cold joint-budget
miss. It restores small joint cases to~6s/2.11GiB and3producer joint~16s/3.26GiB.
Planner still omits explicit gold-writer auxiliary geometry, relying on adaptive
observations; improving cold admission for that known objective is outstanding.

### 2026-10-06: preserve execution decisions across replay

v15 failed1093 in staged return replay (75.8% payload elements differ,
maxabs.003662109375). Strict guard stopped before optimizer update; original
frozen runtime/log preserved. Last full state1075; completed unsaved1076–1092
updates cannot be recovered from scalar logs. GPU exit was discovered during
active sweep rather than a running50-minute watcher; keep an exit watcher
active while editing too. Do not claim the scoped policy is fully qualified.

Live-budget selection was repeated for primal and backward replay despite
different live allocations. Each producer now owns a ReplayResourceChoice,
choosing once and replaying the same checkpoint path without remeasurement.
Both true and false selections are pinned. This is a suspected cause fix;
actual failing-case recovery remains pending and must not loosen replay tolerance.

Pre-optimizer accumulation now emergency-checkpoints completed updates and
rewinds incomplete cursor/samplers/baseline for RuntimeError/AssertionError/
ValueError as well as OOM. It still never certifies a partially executed
optimizer step (the optimizer runs outside this catch). Preserve failure.json
and raise; recovery is explicit. Capacity continuation512 separately removes
known gold clipping; changing that curriculum means subsequent successes are
not an isolated unchanged-input reproduction of v15.

### 2026-10-07: protect raw content identity during recurrence warm-up

Pop v16 cleanly paused after the 1408 periodic probe (written CE 0.393961,
shuffled 0.411216, only 6/12 better). A fresh CUDA/BF16 conditional-return
probe of that paused checkpoint passed crisp 4/4 and learned encoded-source 0/4;
the diagnostic raw-content identity intervention passed encoded-source 4/4.
These are fixed teacher-prefix final-value probes, not autonomous task scores.
Outputs: `runs/neuralese-raw-semantic{,-identity}-probe-20261007-v16`.

The shared raw recurrence recipe now explicitly uses `content_transport=raw-identity`.
This bypasses the learned residual at deterministic transport, retaining every
parameter and optimizer moment rather than resetting them or changing optimizer
groups. Writer/adapter/stop learning and differentiable sketch transport remain.
Learned residual transport remains an explicit separate mode for future qualified
compression stages. Mode is checkpoint metadata, restored by serving and evaluation;
changing it requires a named continuation curriculum decision. No foundation or
semantic admission certificate is inherited from the diagnostic intervention.
Sampling with nonzero payload temperature remains a separate noisy intervention.

Training log `step` remains its historical zero-based iteration index; new explicit
`iteration_index` and `completed_updates` disambiguate it from checkpoint/evaluation
`step` (completed updates). Length/context metrics now report the current update,
including true maximum and capacity, instead of stale cumulative last-eight values.
Periodic evaluations already persist in `eval.jsonl`; the earlier monitoring concern
about stdout-only persistence was incorrect.

### October 6 UTC: native value length, not compact source serialization

Auditing the 670 rich-cohort producer values against the actual LFM template
found all 670 supervised source-text lengths too short, by up to 18 tokens.
The compact JSON source omits spaces present in the native tool value. Counting
source tokens therefore truncates generated values and trains incorrect stopping.
The largest native body is 163 tokens, within the explicit 512 capacity.
Audit: `runs/neuralese-native-writer-length-audit-20261006-v1`.

The shared recipe and default new runs now use `writer_length_policy=native-value`.
`serve.chat.write_value_text` obtains the exact gold span using the same template
and typed/string boundary as `write_reply`; it validates prefix/suffix instead of
inventing a serialization. Trainer preflight, writer sizing, selective checkpoint
admission, and graph geometry use the same cached native source span. Source-text
length remains explicit for reproduction of old experiments; moving a checkpoint
requires a named curriculum change, preserving optimizer/RNG and fixed source data.
Digest preview sizing is unchanged. This changes training lengths, not runtime's
learned free stopping, which still needs separate qualification. All 670 helper
spans and token counts match an independent native-target rendering audit.

A stale rendering test assumed tool arguments were a content-part list. The current
shared contract is a JSON argument string containing typed block parts within the
argument value; the test was updated to assert that existing contract. No production
rendering behavior was changed to satisfy the stale assertion.
