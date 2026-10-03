# Neuralese in Natlang: Types, Semantics, and Training

## Executive overview

Natlang combines typed TypeScript programs with model-executed natural-language functions. This proposed extension adds **`Neuralese<T>`**, a purpose-conditioned soft representation of a value, and **soft closures**, callable functions whose instructions are themselves neuralese. Persistent Neuralese blocks are independently trainable parameters for data, skills and core semantic operators. Exact computation, program structure and authority remain explicit in the host language.[^natlang]

The representation and execution contract is defined in [Neuralese: Port Mechanics and Training](neuralese-port-mechanics-and-training-updated.md). Every newly generated soft output uses its single procedure: shallow autoregressive sketch generation, upper-stack completion, then consumption of the completed payload. The language adds no alternative block-generation mode.

The training objective is useful, composable program behaviour. First train the crisp Natlang self-improver to author reusable skills from checked trajectories; use that ability to bootstrap soft skills. Downstream losses train intermediate values, instructions and skills. Once basic execution and authoring are established, substantial task-level reinforcement learning trains execution and improvement against checked outcomes. Meta-learning evaluates updates on separate subsequent cases.

## 1. Type and value contract

`Neuralese<T>` is an opaque reference to an ordered sequence of soft tokens representing a **view of `T` for a declared purpose**. It is not a subtype of `T`, a text encoding, or proof that every fact about `T` can be recovered. The runtime retains the tensor, actual length, type and purpose descriptors, representation version and producer or parameter reference.[^types]

| Form | Meaning and use |
| --- | --- |
| `Neuralese<T>` | Semantic data consumed through the read port. Compiler and runtime checks reject implicit field access, arithmetic and use as a branch condition. |
| `Handle<T>` | An exact reference to host-held content. Resolution preserves the selected content's bytes; selecting the correct handle remains a separate decision. |
| `NeuraleseFunction<F>` | A callable contract with a soft instruction body, explicit captures, skill dependencies, revision and permitted effects. |
| `Crisp<T>` | A return annotation requiring a checked, ordinary host value rather than a soft result. |
| `Structured<S>` | A return annotation applying the result rules separately to each field. |

**Readout is explicit.** A soft value becomes a branch condition or ordinary result only through a typed readout. Validation checks the declared structure and admissible values; it does not establish factual correctness. Invalid readouts use the normal call failure or repair path, not an unchecked cast.

**Exactness stays outside the latent representation.** Identifiers, paths, quotations, code and numbers used in exact arithmetic remain crisp values or handles. A handle may have a companion soft view, but that view is not responsible for byte recovery. Host code performs exact calculations and transformations. A data handle does not confer service or filesystem authority.

A value may have different views for different consumers, with an exact backing handle where available. Behaviour is assessed by purpose fidelity and by substituting or composing views in complete programs. No lossless round-trip or exact composition identity is assumed.

### Persistent trainable blocks

A Neuralese block can be an **independently stored parameter**: an ordered matrix `Z ∈ ℝ^(k × d)` of `k` embeddings of width `d` in the reader's input space. It has its own identity, revision, shape and representation version, separate from model weights. Initialise it typically from text, using token embeddings or a soft encoding, then optimise its coordinates directly through downstream losses. It need not be regenerated from that text or its original producer.

The runtime stores and reloads these blocks, with optimiser state for resumable training. Each block can be selected for optimisation or frozen independently, including while the backbone remains frozen. Length is fixed within a revision; resizing is an explicit revision change. Stored blocks can represent data or instructions; callability requires the typed closure contract, not merely a tensor.

## 2. Operations, calls and closures

### Core operations

| Operation | Contract |
| --- | --- |
| `soft(value, purpose, { tokens })` | Write a purpose-conditioned view. `tokens` is a maximum payload budget; the stopping head determines actual length. |
| `crisp(view)` | Produce and validate an ordinary value of the view's declared type. |
| `nnl` | Define a natural-language function with soft instructions. Parameter names, argument positions and types remain explicit. |
| `merge(views, purpose)` | Compute a new soft result from several views through an ordinary function invocation. |
| `handle(value)` / `resolve(handle)` | Introduce or resolve an exact reference under the host's access rules. |
| `neuralese.function(parts)` | Construct a callable closure from soft instructions and explicit bindings. |
| `ask(view, question)` / `gloss(view)` | Produce a question-specific textual readout or a diagnostic rendering. Neither becomes the canonical value. |

Calls remain asynchronous. A function can receive ordinary values, soft views of the corresponding types, or exact handles. A missing purpose defaults to the consumer's instruction, declared result contract and context already available at the write site; otherwise it is a general overview. Generated soft instructions can supply the purpose. Future observations and expected answers are never purpose inputs.

**Return representation follows the declared kind.** `Crisp<R>` returns an ordinary `R`; an existing neuralese value, handle or closure retains its identity and kind; `Structured<S>` applies these rules field by field. Other semantic results become `Neuralese<R>`. A returned closure remains callable. A neuralese description of a function is data, not an executable function.

### Trainable core operators

The semantic behaviour of `soft` (ingesting crisp data), `crisp` (producing typed crisp data) and `merge` is implemented by **stored Neuralese functions with independently trainable instruction blocks**. Each operator may bind its own soft skills. Optimise operators and their selected dependencies through consumer or typed-readout losses, separately from task lambdas and backbone weights.

Tensor registration, port execution, exact-reference resolution and type validation remain runtime primitives. Learned operators use these primitives; they do not replace their guarantees or introduce another generation mode. Direct block registration allows text initialisation without invoking the operator being initialised.

### Soft closures

A closure contains its identity and revision, typed signature, soft instruction body, captures, skill bindings and effect allowance. Captures are explicit because an opaque instruction body cannot be scanned for variable names. Each capture is a **snapshot** taken at construction, a **live** binding resolved at invocation, or a **pinned** revision. The invocation records the actual bindings used.

Binding leading arguments produces a specialised closure with a residual signature. Revising instructions or skills produces a new revision; running calls retain the revision they started with. Specialised computations are keyed by the closure revision and relevant bindings, with a generic execution fallback.

Constructing or passing a closure performs no external action. Invocation can use only capabilities supplied or delegated by the host, within Natlang's callable scope, iteration and recursion rules. Soft instructions and learned updates cannot enlarge that authority.[^types]

## 3. Representation during inference

**Host representation.** TypeScript carries opaque value and closure references, not arrays of floating-point coordinates. A model-side tensor store resolves soft references. The runtime checks representation compatibility, validates exact references and retains the metadata needed to invoke a closure. Incompatible payloads require explicit regeneration or conversion rather than silent reuse. Stored parameter blocks are loaded at a pinned revision and read directly as completed payloads, without sketch generation.

**Model representation.** The runtime renders the native chat template and inserts each instruction or argument payload between `<|neuralese|>` and `<|/neuralese|>`, with a typed header identifying its role and actual length. Each vector occupies one model position. Exact references and structural control data remain discrete, outside the continuous payload positions. The sketch generator retains its continue/stop contract. Block closure ends the block, not the assistant message or Natlang function.

**Execution.** The compiler supplies the call signature and allowed bindings. The model consumes supplied values, invokes permitted helpers or host services, and generates the declared result. New soft outputs use the companion document's single writer. Consumers receive final payloads and declared context, never private sketch states or producer caches. Same-invocation continuation follows the specified completed-payload readback before ordinary decoding resumes.

The host validates crisp results, reference selections and effect requests. A function finishes through Natlang's normal checked completion mechanism, not merely because a neuralese block stopped. Ordinary inference retains no automatic-differentiation graph; explicitly authorised learning operations may request gradient computation without changing the port's execution semantics.

## 4. Representation during training

Training uses a **typed execution graph** linking calls, parameter blocks, arguments, purposes, captures, soft writes, readouts, decisions, effects and updates. Nodes identify both the function definition and its particular invocation, with relevant model versions, budgets and stochastic choices. This records which computation produced each value and which consumers used it.[^graph]

Live soft outputs retain their producer connections, including across model-server calls. Consumer losses therefore train the writer, generated instructions, captured views and upstream computations; multiple consumers accumulate credit at shared producers.

**Persistent blocks have explicit parameter ownership.** For direct tuning, `learn.parameters` selects them as optimisation leaves: gradients from all uses accumulate at the block, not its text or generating initialiser. Storing a generated value alone does not promote it to a parameter. Conversely, a revision produced by a differentiable updater retains its update edge during meta-learning. Replay distinguishes direct parameter tuning from differentiation through generation or updates.

**Continuous and discrete credit are distinct.** Teacher-forced scores of correct outputs train the continuous path even when serving returns a sampled crisp value. Cross-entropy and, where compatible teacher distributions exist, token-level distillation supervise results and actions. Sampling, reference selection, tool choice and block stopping require explicit supervision or outcome-based learning; gradients through payloads do not differentiate these decisions. A deterministic host branch is not an additional sampled action.

During replay, recompute differentiable quantities while holding recorded discrete choices fixed. Restore the closure, operator and parameter-block revisions used at each point, and supply recorded effect results rather than repeating external actions. Fresh rollouts evaluate current decisions in the task environment; after a changed action, an old observation is not automatically a valid response.

Port adaptation, generated-sketch training and ordinary-text retention follow the companion document. Language-level losses add program correctness, purpose fidelity, composition and resource use.

## 5. Generating training data from existing trajectories

Use a template-neutral executable task record as the durable source: typed function definitions, inputs, available services, initial state, expected results and permitted effects. Chat messages and tokenized training examples are rebuildable products, not the canonical task.[^trajectories]

1. **Normalise and validate.** Recover the task and action/observation sequence; retain call identities, source snapshots and completion or failure labels. Check accepted demonstrations against results, effects and resulting files or state. Keep repair episodes and rejected actions with their labels. Unexecuted traces remain imitation data, not verified program successes.
2. **Identify safe substitutions.** Trace value consumers. Values used only by model-executed functions may become soft; values used by host arithmetic, branching or exact operations remain crisp. Mixed-use values retain their exact form alongside a soft view. Large inputs can acquire views without removing access to their backing data.
3. **Construct purposes and soft instructions.** Derive each purpose from its actual consumer and causally available context. Rewrite selected natural-language sites to use soft instructions and soft inputs or returns, preserving argument contracts, capabilities and host control flow. Capture dependencies explicitly.
4. **Train against preserved behaviour.** Keep the original checked actions and results as targets. Withhold replaced source material from the consumer where necessary so it must use the soft channel. Train through connected calls, progressively increase substitution by site and type, and use self-generated sketches wherever the program generates new blocks.
5. **Validate complete programs.** Compare original and rewritten executions on unseen instances, not only next-token likelihood. Check result correctness, effect arguments, exact-reference resolution, failure handling and budgets. Build new rollouts to expose errors that fixed teacher trajectories cannot reveal.

This produces paired crisp and soft executions, local read/write examples, multi-call composition examples and failure-and-repair cases from the same underlying tasks. Outcome labels and future answers remain supervision, never earlier model inputs.

### Skill-authoring data

Build a substantial, dedicated **crisp skill-authoring corpus** before relying on soft skill generation. Group suitable trajectories into support experience and separate later-use tasks. A teacher authors or revises a reusable skill from support alone, specifies its applicability and dependencies, supplies tests, and integrates it into a task lambda. Record the authoring actions, resulting assets, checks and later executions, including comparisons with the unmodified program. Keep successful examples for imitation and unsuccessful revisions as labelled feedback.

Cover creation, revision, binding and reuse across task families, including repair of missing, irrelevant or incorrect skills. Scale execution-checked teacher generation until held-out authoring, integration and transfer are reliable; track coverage and learning curves rather than raw row count alone. These examples establish the authoring capability used to initialise soft skills, not merely a fixed library of skills to consume.

## 6. Selective conversion of bgkit and Schnitzeljagd tasks

**Natlangify tasks where typed decomposition and execution checks add value**, rather than wrapping every training row in a nominal function. Reuse the bgkit inventory and migration workstream in the companion document, with an additional adapter from suitable records to executable Natlang tasks.

| Source | Suitable task shapes | Natlang conversion |
| --- | --- | --- |
| bgkit | Purpose-conditioned context use, large tool-output processing, browse-and-answer episodes, repository tasks and trajectory compaction. | Explicit source-access helpers produce views; subsequent functions reason over them and return checked answers or actions. Preserve exact paths, source revisions and effect results. |
| Schnitzeljagd | Schema-grounded tool use, conversational decisions, bounded coding workflows and support/next-task episodes. | Represent tools as typed host services, observations as inputs, and decisions as checked returns. Use separate support and query instances for learning reusable procedures. |

The source projects document relevant context-compression workflows and trajectory adapters; availability and execution validity must be established during import.[^imports] Missing environment state requires a fixture, a new validated teacher execution, or exclusion from executable-task training—not invented observations.

Pure continuation and reconstruction examples can remain in port training or ordinary-text replay. Keep conversions only when they preserve task meaning and provide a checkable result, state transition or effect. Split by repository, document, task instance or tool-schema group before generating windows or adaptation episodes, keeping all attempts at one task together; deduplicate across both projects and track conversion coverage and exclusions. A query task's later solution must never appear among its support examples.

## 7. Skill authoring, soft skills and self-improvement

### Crisp skill authoring

Extend the **basic text-based Natlang self-improver** to create and revise reusable skills, not only edit the current task's instructions. Skills may contain textual procedures, structured task knowledge or typed helper functions, with explicit applicability, dependencies and tests. The improver must bind them into task lambdas, check their use and evaluate reuse on separate cases. This capability is trained and evaluated in the crisp system before depending on Neuralese.

### Soft skills

A **soft skill** is a reusable Neuralese asset: task knowledge, a rubric, an extraction strategy, a diagnostic method or a verification routine. A Natlang lambda selects or retrieves a small explicit set per task, binding versioned data blocks for consumption or stored closures for invocation. Callable skills retain their typed contracts and captures. Core operators can bind skills in the same way.

Use the trained crisp authoring capability as the **bootstrap for soft skill creation**: initialise semantic instruction and data blocks from authored text, retaining signatures, exact code, bindings, tests and capabilities in host form. Then train soft authoring and revision through multiple downstream applications, including direct optimisation of persistent blocks. Evaluate transfer rather than reproduction of a particular answer. Skill selection and revision remain explicit program operations and grant no additional capabilities.

### Self-improvement and meta-learning

Self-improvement produces a new parameter-block or closure revision, including instructions, skills and operator implementations. The learning interface exposes backend-owned operations such as `learn.score`, `learn.parameters`, `learn.grad`, `learn.digest` and `learn.update`; the host receives opaque loss, parameter and gradient references rather than raw tensors.[^learning]

A bounded adaptation episode has two parts. **Support cases** provide experience and feedback used to revise the program. **Separate query cases** measure the revised program and supply the outer objective that trains the updater. Adaptation to support examples is not itself evidence of transferable improvement.

Once crisp authoring and soft initialisation are established, use **gradient-based tuning of selected instruction, skill and operator blocks** as the soft-adaptation baseline. Warm-start a learned updater from evaluated authoring and revision episodes, then train it on subsequent task performance at matched compute and evaluation budgets. Its inputs can include the current instruction, skill and operator blocks, a trajectory digest, feedback and a structured gradient digest. The latter retains parameter identity, role, shape, direction, scale and optimiser state; raw gradient magnitude is not a semantic diagnosis.

Differentiate through continuous update outputs where supported, and state whether gradient-digest inputs are treated as observed features or differentiated further. Discrete choices—such as which diagnostic, example, helper or candidate revision to try—use supervised or outcome-based objectives. Experience summaries must preserve attempted actions, observations, failed hypotheses and unresolved questions.

The same framework can train generated learning procedures that select exercises, objectives and update steps. An updater's own soft instructions can also be revised, using finite, explicitly bounded episodes under the same scope and capability rules. Ordinary model-weight training remains distinct from applying an authorised local Neuralese parameter update.

## 8. Reinforcement learning on Natlang tasks

After supervised training establishes basic typed execution and skill authoring, make **substantial task-level reinforcement learning** a core training phase. Use fresh, complete-program rollouts in executable task environments to improve decomposition, helper and skill selection, skill creation and revision, error recovery and block stopping. Expand task difficulty and rollout length as performance permits.

Use independently checked results, state changes and permitted effects as the primary reward. Judge authored skills and updates by their contribution on separate query tasks; constrain token, tool and adaptation costs without rewarding shortcuts that fail the task. Type checks, capability boundaries and execution limits remain host-enforced, not merely reward penalties.

Combine policy learning for sampled decisions with differentiable objectives for continuous blocks and operators. Retain supervised examples and ordinary-language replay during RL, and compare against the supervised baseline on held-out tasks. Task success, skill transfer and resource use—not training reward alone—determine progress.

## 9. Delivery and evaluation

First extend the crisp self-improver with skill authoring, build the dedicated corpus and establish held-out authoring and reuse. In parallel, implement typed values, persistent parameter storage and optimisation, closure invocation, graph replay and task converters. Use validated crisp skills to initialise soft assets; establish crisp/soft agreement before increasing substitution. Introduce learned updaters with separate support/query evaluation and the gradient-tuning baseline. Once basic execution and authoring are reliable, scale task-level RL across the validated Natlang task mix, including suitable imported tasks.

Evaluate complete-program correctness, type and capability compliance, exact-value preservation, stopping reliability, composition across calls, latency and total compute. Compare correct, shuffled and zeroed payloads and skills at matched lengths. Evaluate independently tuned operators against their text-initialised baselines. Measure skill-authoring coverage, cross-task reuse and adaptation on held-out task families; retain ordinary-language capability checks. Compare pre-RL and post-RL programs at matched inference budgets. Promote a revised skill or updater only on measured downstream benefit, not on diagnostic renderings or training-loss reduction alone.

[^natlang]: [Natlang language specification](https://github.com/werg/natlang/blob/main/spec/SPEC.md).
[^types]: [Proposed neuralese type and callable surface](https://github.com/werg/neuralese-/blob/main/spec/neuralese.d.ts). The contracts in this document specify the extension; they do not assert that its runtime is already implemented.
[^graph]: [Typed execution-graph specification](https://github.com/werg/neuralese-/blob/main/spec/graph.d.ts).
[^trajectories]: [Executable task records and reproducible training builds](https://github.com/werg/natlang/blob/main/PROGRAM_IR_PIPELINE.md).
[^imports]: bgkit, [training workflows](https://github.com/werg/bgkit/blob/main/docs/02_training_plan.md) and [context-compression task families](https://github.com/werg/bgkit/blob/main/plans/capability_packaging_2026_08_20.md); Schnitzeljagd, [teacher datasets and episode construction](https://github.com/werg/schnitzel/blob/main/docs/datasets.md).
[^learning]: [Neuralese learning operations and objectives](https://github.com/werg/neuralese-/blob/main/docs/design.md#learning-to-improve).
