# Neuralese in Natlang: programme and stage plan

Proposal, 2026-10-03. This is the top-level plan for adding Neuralese to Natlang and training a Natlang/Neuralese model with its directly trained artifacts. Each stage below gets its own detailed plan in this directory; this file fixes scope, decisions, dependencies and gates.

Design inputs, kept verbatim in [sources/](sources/):

- [Port mechanics and training](sources/port-mechanics-and-training.md): the read port, the single sketch-based write procedure, chat-template integration, port training.
- [Semantics and training](sources/semantics-and-training.md): language semantics, trainable blocks, skills, meta-learning, RL.

These inputs are source material, not the specification. Where they conflict with the decisions below, the decisions win. Their footnote links to `werg/neuralese-` point to an obsolete repository; the specification now lives here.

## 1. Goal

1. A general Neuralese capability in a causal LM: reading and writing soft tokens through one port contract, with learned stopping.
2. Neuralese as an extension of Natlang semantics, implemented end to end: language, native runtime, model servers (including llama.cpp), training graph.
3. A Natlang/Neuralese-focused model trained with this machinery, plus its directly trained Neuralese artifacts (core operators, skills, data blocks), tagged with the dialect version they were trained for.

## 2. Decisions

| Area | Decision |
| --- | --- |
| Predecessors | bgkit (`~/bgkit`) and Schnitzeljagd (`~/sdkb`) stay as they are. They are sources of data and lessons only. No checkpoints, encoders, slot pricing, compaction spine, ratio sampling or KB retrieval machinery are reused. |
| Data | Use all available data from all teachers. Restrictive licences (including CC-BY-NC) are acceptable; licence is still recorded per record. Model- and harness-specific conventions are stripped; content and supervision are kept. |
| Code | Everything lives in this repository. |
| Starting point | The best Natlang crisp student (currently the LFM2.5-350M LoRA line). The final model size is decided later. |
| Backbone facts | LFM2.5-350M: 16 layers, width 1024, vocab 65,536, full attention at layers 2, 5, 8, 10, 12, 14, short convolutions elsewhere. (`TRAINING.md` §0 says 14 layers; that is wrong for 350M.) |
| Compute | The whole DGX Spark is available. Train without throughput-costly memory savers such as gradient checkpointing. |
| Serving | Neuralese is a first-class target of real inference servers: llama.cpp (which also gives the browser via wllama) and vLLM, parity-tested against the PyTorch reference. The llama.cpp and vLLM changes live in forks designed for upstreaming: general APIs, small patches, upstream conventions. The browser stores blocks and `.nz` files in OPFS. |
| Distillation teacher | No separate teacher model. The KL target is the same model given the full source. |
| Gradient plumbing | The simplest efficient system: the runtime records an execution graph; the trainer replays it on GPU, holding discrete choices and effect results fixed. In-program learning uses the same replay inside the model server. |
| Feature scope | The full feature set is specified now. Implementation follows real dependencies, not feature releases. |
| Assets | Stored assets carry a **dialect version tag**. Several models may speak one dialect, for example a small browser model and a larger server model. |
| Skill authoring | Crisp skill authoring is a prerequisite, built into the self-improver with its own generated corpus. It is the basis for our meta-learning. |
| Skills | Skills are standard agent-harness skills (`SKILL.md` with `name`/`description` frontmatter plus code and reference files), extended by an optional `natlang:` frontmatter block. `.nz` skills keep the same fields in their safetensors metadata; a skill may become purely soft, with its crisp ancestor kept in provenance. Loading is standard progressive disclosure plus automatic injection of metadata-declared scope data. Skills are part of the context bound to a function: implicit per-program selection by default, and a global mutable pool at the top level that affects programs only when they are rebound. |
| Evaluation | No fixed pass/fail thresholds. Every stage evaluates continuously against a rubric, and reviews decide what to improve next. Mechanical tolerances (parity, cache agreement) are fine. |
| Semantics | **`Neuralese<T>` is a real type** in the Natlang type system, with full attendant infrastructure: type checking, compiler and runtime enforcement, a tensor store, dialect tagging, typed readout, closures, context-held trainable values and training-graph support. Ordinary types keep their existing exact meaning, so there is no `Crisp<T>`; Neuralese extends the semantics rather than re-annotating existing programs. |
| Algebra | `Neuralese` comes with library combinators, not syntax: functor `map`, applicative `zip`/`ap`, monoidal `combine`, Kleisli composition through ordinary natural-language functions, and `read` as the only, typed, fallible way out. Each combinator is a system natural-language function with a trainable soft body; a model may keep its own tuned bodies as values in its context, with the library's as the default. The algebraic laws (identity, fusion, associativity, read/map commutation) hold approximately and are trained to some degree as consistency objectives. Splitting a structured soft value into soft parts is explicit. The compiler uses the laws for rewrites (fusion and similar). A rule is enabled automatically once whole-program comparison on the current model shows it does no harm, re-measured when the model changes; every applied rewrite is traced. |
| Soft functions | Neuralese functions are applicable: a `Neuralese<F>` with a function type `F` can be called. |
| Dialect in types | `Neuralese<T, D = DefaultDialect>`: the dialect is a type parameter with a default, so ordinary code writes `Neuralese<T>` and dialect conversion stays an explicit, typed function. |
| Learning | No mutable parameters. Trainable blocks are immutable Neuralese values in a program's context (`.nz` files). Learning is `grad` with respect to a soft argument plus step functions run with `iterateOn`; gradient descent and learned updaters have the same shape. Promotion binds a new context revision. |
| Captures | Explicit captures of Neuralese functions are snapshots by default, so soft functions can be stored; `live(x)` opts into live `let` captures. Text `nl` keeps implicit live captures and `let` write-back. Function-typed bindings are always captured by value. Immutability is a design tool where it helps, not a goal. |
| Executable nodes | New executable nodes (`.nl` and TypeScript functions) come only from files. Existing nodes may be edited within their interfaces; data (including skill files and Neuralese values) may be added and edited freely. |
| Environments | All execution environments are built (SQLite, test runners, world simulators, SWE containers). |
| Files | One general safetensors-based file format for Neuralese-bearing data: a module of typed named exports (natlang type syntax), with soft functions as one kind of export. |
| Length | No budget in the language. The writer stops when its stop head says so; a hard runtime maximum bounds runaway generation. |
| Purpose | Purpose is not part of the type. It is encoded where a value or function is written: by the producing function's instructions, declared result and causal context. Combinators take no purpose argument, and re-encoding a value for a purpose through a separate call is treated as an anti-pattern rather than a core operation. |
| Literal form | Neuralese has a literal form, and the markers wrap only the vectors: `<|neuralese|>⟦z1…zL⟧<|/neuralese|>`. Everything else (type annotation, function signature, explicit captures) is plain TypeScript around the literal, so it precedes the block in the token stream and conditions the writer and stop head. A function literal is the `nl` construct with a soft body and an explicit capture object. The same form appears in model-written eval code (the write port runs) and in listings of Neuralese-typed values and arguments (the read port splices the vectors in). |
| Rendering | Two forms only. Outside the model (source, JSON, traces, logs, UI) a Neuralese value is an ordinary data value that points to tensor-store content by unique ID. At the model-token level the runtime substitutes marker-wrapped vectors for the reference when rendering, and stores emitted blocks and substitutes references when parsing. Human-readable glosses are optional diagnostics, not a third form. |
| Typing and captures in data | Training data is rewritten so that model-written natlang is eagerly typed: the compiler inserts inferred annotations, so every literal has its type in front of it. The compiler also makes the implicit captures and arguments of inline `nl` calls explicit at data-generation time, which teaches the model to name captures and gives function-literal samples. |
| Dialects | Dialect stability is not a concern until there is a large user base. A dialect is a lightweight version tag on stored values: when the model changes, the version is bumped. Mismatched versions are rejected and the value is regenerated or converted. Adapters exist only where a model of a different width needs them. |
| Tool-call literals | Literals stay inside the eval code string of the native Pythonic tool call, with the control tokens inside the quoted string. |
| Teachers | New data generation uses the paid teachers (Luna, Bunny) as well as the local Qwen3.6 on the DGX, including crisp revisions as warm-start targets for S6 and teacher rollouts for hard RL families early on. |
| Publication | Models and `.nz` libraries on Hugging Face; specifications and conformance suites on GitHub. |

### Open points

No plan has open points. Context as a curried argument is adopted (S0 §§7–8).

## 3. Stages

Gates are review checkpoints, not fixed thresholds: each stage evaluates continuously against a rubric, and the review decides whether to hand results on or what to improve next.

The stages form a dependency graph, not a sequence of releases. Several run in parallel from the start.

```text
S0 Spec & foundations ──┬──────────────────────────────┐
                        │                              │
S1 Data migration ──────┼──────────┐                   │
   (continuous)         │          │                   │
S2 Crisp skill authoring┼───────┐  │                   │
   (continuous)         ▼       │  ▼                   ▼
                  S3 Port model & training ──G1──► S5 Program-level training ──G2──┐
                        ▲                          ▲                               │
S4 Runtime & servers ───┴──────────────────────────┘                               ▼
                                                   S6 Soft skills & meta-learning ─G3─► S7 Task RL ──► S8 Target model & release
                                                    ▲
                                       S2 ──────────┘
```

### S0. Specification and foundations

**Goal.** One coherent specification before code depends on it.

- The Neuralese semantics extension, in full: the `Neuralese<T>` type and its checking rules, the combinator library and its laws, typed readout, write-site purpose, soft instructions, closures and captures, contexts, `grad` and the learning library, effect and authority rules. Written into `spec/` and `TYPES.md` with conformance cases, not as a side document.
- The model-port contract: block syntax inside the native chat template, control tokens, cache and readback rules.
- Dialect version tags and the compatibility check.
- The execution-graph record: what the runtime records, what replay recomputes, how definitions and context revisions are identified.
- Repository layout: `spec/`, `ts-host/src/neuralese/`, a Python package for model, trainer and server, `vendor/` patches for llama.cpp and vLLM.
- A decision log in this directory.

**Exit.** The spec is reviewed and its conformance cases exist (failing). No open contradictions with the decisions table.

### S1. Data inventory and migration (continuous)

**Goal.** Every eligible dataset from bgkit, Schnitzeljagd, Natlang and the external drive has a disposition, and accepted data is available in two products.

- **Port records.** A model-neutral record for read and write training: source or causal prefix, purpose, consumer context, target, outcome labels, lineage, licence, split group. Sources include the bgkit task stores (compaction, tool digests, purpose-conditioned QA, repository QA and localisation, web browsing, memory, SWE traces, replay), Schnitzeljagd probe corpora and teacher episodes, and raw text and repositories.
- **Natlang executable tasks.** `natlang.program/2` records produced by adapters where typed decomposition and checked results add value, then run through the existing collector, materializer and renderer.
- Splits closed over source groups across all imported corpora; cross-corpus deduplication; protected held-out sets; a per-dataset ledger of accepted, rejected and excluded rows with reasons.
- Pre-tokenized data with no text (bgkit1) is decoded only where it is recoverable, otherwise excluded with a reason.

**Exit for G1 use.** Port records for the main task families pass rendering, causal-mask and leakage checks. Natlang task adapters keep producing data for S5–S7 after that.

### S2. Crisp skill authoring (continuous)

**Goal.** The text self-improver creates, revises, binds and reuses skills, judged on separate later tasks.

- Extend `ts-host/src/improvement/` and the optimizer curriculum with skills as first-class assets: procedures, structured knowledge, typed helpers, applicability, dependencies and tests.
- A support/query episode format: authoring from support, evaluation on separate query tasks, comparison with the unmodified program.
- A large teacher-generated, execution-checked authoring corpus covering creation, revision, binding, repair and reuse across task families, with failures kept as labelled feedback.
- Crisp student training on that corpus, with held-out authoring and transfer evaluation.

**Hand-off.** Reviewed continuously against the S2 rubric; when authoring helps and transfers on held-out families, it initialises soft skills and becomes the basis for the meta-learning in S6.

### S3. Port model and port training

**Goal.** The crisp student reads and writes Neuralese through the single procedure.

- PyTorch reference implementation on LFM2.5: read-port splicing, shallow sketch recurrence with feedback projection, stop head, blockwise completion, content projection, cache restore and readback (attention KV and convolution state).
- An evaluation harness from day one: correct, shuffled and zeroed payloads at matched length; full-text reference; premature stopping, excess length and truncation; norm, rank and collapse monitors; latency per phase; cached-versus-recomputed agreement.
- A curriculum following the port document: bootstrap with known-text embeddings and span labels; shallow distillation from the frozen model; transition to generated sketches; producer/consumer training with self-distillation; purpose pairing through write-site context; explicit stop-policy learning; gradual unfreezing with ordinary-text replay.
- A cutoff sweep over depths just after attention layers (k = 3, 6, 9 for 350M).
- The first dialect version tag, `nd:natlang@1`, assigned to the trained model's space.

**Checkpoint G1 (port viable).** Reviewed against the S3 rubric: on held-out consumers, correct payloads beat shuffled ones. Stopping is reliable with self-generated sketches. Post-block text quality and ordinary-text capability hold. The cached and reference paths agree.

### S4. Runtime and inference servers

**Goal.** Natlang programs execute with soft values on real servers.

- Native runtime: the `Neuralese<T>` type, literal parsing and rendering, typed `read`, soft instructions and closures, contexts and `.nz` loading, dialect version tags, the extended model-turn protocol (block references in and out), execution-graph recording.
- Python model server: tensor store, block writing, readback, replay-based `grad` sessions.
- llama.cpp: embedding splicing for reads, a shallow-loop and blockwise-completion graph for LFM2, a block-writing API, GGUF metadata for the new heads; the wllama build for the browser, with OPFS storage.
- vLLM: prompt-embedding reads and the write procedure, for high-throughput rollouts.
- Parity: each server against the PyTorch reference.

S4 starts in parallel with S3, using an untrained port (payloads from text embeddings) so that runtime work does not wait for G1.

**Exit.** Spec conformance cases pass on the Python server and on llama.cpp, with trained S3 weights.

### S5. Program-level training

**Goal.** Soft values pay off inside complete Natlang programs.

- Soft conversion of executable tasks: trace consumers, decide which values may stay soft, rewrite instruction sites to soft bodies, withhold replaced sources from consumers.
- A replay trainer over recorded graphs: credit through chains of calls, multiple consumers, closures, discrete choices fixed.
- Core operators and combinators trained as system functions with soft bodies; law-consistency objectives.
- Increasing substitution by site and type; crisp/soft agreement checks; fresh rollouts on unseen instances.

**Checkpoint G2.** Soft programs match or beat crisp programs on held-out complete-program correctness at matched inference budget, with exact values preserved and type and authority checks passing.

### S6. Soft skills and meta-learning

**Goal.** Programs improve themselves through Neuralese values in their contexts, and the improver itself is learned.

- Direct tuning of context-held blocks (instructions, skills, data, operators) with `grad` and `iterateOn` steps.
- Soft skills initialised from S2's crisp authored skills, then trained through multiple downstream uses.
- The gradient-tuning baseline, then a learned updater warm-started from S2's authoring episodes and trained on separate query performance; structured gradient digests as inputs.
- Self-revising updaters within bounded episodes.

**Checkpoint G3.** On held-out task families, learned updates beat the gradient-tuning baseline and crisp authoring at matched compute.

### S7. Task-level reinforcement learning

**Goal.** Policy improvement on fresh, complete-program rollouts across the validated task mix.

Rewards come from independently checked results, state changes and effects. Policy learning covers sampled decisions (decomposition, helper and skill selection, skill authoring, stopping); differentiable objectives cover blocks and operators. Supervised data and ordinary-language replay stay in the mix. vLLM rollouts on the DGX.

**Review.** Pre-RL and post-RL programs compared on held-out tasks at matched inference budget.

### S8. Target model and release

**Goal.** The Natlang/Neuralese model and its artifacts, published.

- Choose the target backbone and size from S3–S7 evidence; it takes the current dialect version or bumps it.
- Run the full pipeline at target scale.
- Publish the model, the artifact library (core operators, skills, data blocks), GGUF builds and conformance results.

## 4. Cross-cutting work

- **Evaluation.** One suite grows from S3 onward: correct/shuffled/zeroed comparisons, stopping, exactness, composition, latency and compute, skill transfer, ordinary capability.
- **Ordinary capability.** Replay and capability checks in every training stage.
- **DGX scheduling.** Training, teacher generation and rollouts share one machine; each detailed plan states its DGX schedule.
- **Run identity.** Neuralese runs are separate from crisp runs. Crisp training may continue and be rebased onto.
- **Migrations.** Every runtime or API change gets an entry under `training/api-migrations/`. Data built under different runtime versions is never mixed.

## 5. Detailed plans

| File | Covers |
| --- | --- |
| [S0_SPEC.md](S0_SPEC.md) | Semantics extension, contexts, termination, learning, files, port contract, execution graph |
| [S1_DATA.md](S1_DATA.md) | Inventory, dispositions, port records, Natlang adapters, splits |
| [S2_SKILL_AUTHORING.md](S2_SKILL_AUTHORING.md) | Skill assets, episodes, corpus generation, student training |
| [S3_PORT.md](S3_PORT.md) | Reference implementation, harness, curriculum, cutoff sweep, G1 |
| [S4_RUNTIME_SERVERS.md](S4_RUNTIME_SERVERS.md) | Runtime, protocol, Python server, llama.cpp, vLLM, parity |
| [S5_PROGRAM_TRAINING.md](S5_PROGRAM_TRAINING.md) | Soft conversion, replay trainer, operators, G2 |
| [S6_META_LEARNING.md](S6_META_LEARNING.md) | Block tuning, soft skills, updaters, G3 |
| [S7_RL.md](S7_RL.md) | Environments, rewards, algorithms, rollout infrastructure |
| [S8_TARGET.md](S8_TARGET.md) | Model selection, scale-up, release |
