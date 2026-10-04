# S5: Program-level training

Draft, 2026-10-03. Detailed plan for stage S5 of the [Neuralese programme](README.md). It builds on the specification in [S0](S0_SPEC.md), the data products of [S1](S1_DATA.md), the authoring corpus of [S2](S2_SKILL_AUTHORING.md), the port model and trainer of [S3](S3_PORT.md), and the runtime and servers of [S4](S4_RUNTIME_SERVERS.md). The relevant source sections are [semantics and training](sources/semantics-and-training.md) §§4–6.

S3 makes the model read and write Neuralese through one port. S5 makes Neuralese pay off inside complete natlang programs: soft values passed between calls, soft function bodies, the combinator library, and the laws that the compiler relies on. Its review point is Checkpoint G2.

## 1. Starting point

- **Model.** The model released at Checkpoint G1: the crisp base with the S3 deltas, speaking `nd:natlang@1`. S5 continues the same lineage and run discipline (S3 §4.1). The self-distillation teacher is still the same model with the source in full view instead of a block; there is no separate teacher model.
- **Runtime and servers.** The S4 runtime with Neuralese types, literals, contexts, `.nz` files, the graph-record trace and `natlang:learning`; the Python reference server with replay sessions. llama.cpp and vLLM reach parity during S5 and are used for evaluation and rollouts once they pass.
- **Data.** From S1: the migrated, eagerly typed natlang corpus with explicit captures; executable tasks with value-consumer traces; the environments (SQLite, test runners, world simulators, SWE containers) as host services. From S2: crisp skills and authoring episodes whose data skills are the first candidates for soft forms.
- **Compute.** The whole DGX, without gradient checkpointing.

## 2. Soft conversion of executable tasks

S5 needs paired crisp and soft executions of the same tasks. They are produced from `natlang.program/2` records by a compiler-driven conversion, not by hand.

### 2.1 Consumer tracing

For each accepted trajectory, the S4 graph record already links every value to its producer and consumers. The conversion classifies each value by its consumers:

| Consumers | Treatment |
| --- | --- |
| Only model-executed functions (text or soft) | May become `Neuralese<T>`. |
| Host code that observes it: field access, arithmetic, branching, effects, exact comparison, serialisation | Stays exact. |
| Both | Keeps its exact form; model-executed consumers receive a soft value alongside. |
| Exact by nature: identifiers, paths, quotations, code, numbers used in host arithmetic | Stays exact, whatever its consumers. |

Large inputs that only feed model consumers are prime candidates: tool outputs, documents, repository context, earlier conversation, intermediate analyses.

### 2.2 Rewriting sites

The compiler rewrites a converted program at three kinds of site, preserving signatures, effects, services and host control flow:

- **Value sites.** A producing call whose result only feeds model consumers is retyped to return `Neuralese<T>`. Its instructions (the write site) carry the purpose. Consumers are retyped to accept `Neuralese<T>`.
- **Function sites.** An inline or named text function is turned into a soft function literal: `nl.with({ … })` with explicit captures from S1's explicit-capture pass, its body initialised from its instruction text (token embeddings) and then written or tuned. Function-typed captures stay by value (S0 §6).
- **Skill sites.** Data skills from S2 gain `.nz` forms beside their markdown; `scope` bindings may inject Neuralese values.

Every rewritten program recompiles under the S0 rules, including eager typing (S0 §11.4).

Two site kinds are runtime-owned and always converted (decisions 40, 41):

- **Prompt sites.** The system message and every runtime-written frame (central interpreter prompt and its depth-limit variant, function-tool and directory-reducer extensions, program guidance, decision and predicate prompts, compaction notice, handover frame, automatic note, turn notices) are split into registered pieces; each piece becomes a reference to its trainable `Neuralese<SystemPrompt>` form, initialised from the piece's text. Text that matches no registered piece is a versioned piece of its own (`<kind>@<sha12>`), so historic runtime versions keep their wording as initialisation; training may tie versions of a piece.
- **Handover sites.** A `compact_history` note is a model write (`Neuralese<HandoverNote>`); the pinned note message reads that block. The write's source and the teacher's view are the crisp note (and, for self-distillation, the elided history it summarises). Consumers of the note are the later turns of the same trajectory, so the producing and consuming records are linked by the write's name.

The converter reports, per record and in total, every candidate site with its treatment: converted, converted at a later curriculum step, or kept exact with the reason (exact by nature, host consumers, no consumer trace yet).

### 2.3 Withholding sources

Where a soft value replaces source material, the consumer's rendering omits the source, so the consumer must use the channel. The exact source stays available to host code that observes it and to the self-distillation teacher, which sees the full source in place of the block.

### 2.4 Preserved targets

The original checked results, effects and file states remain the targets. Unexecuted traces stay imitation data and are labelled as such. Failure-and-repair episodes keep their labels. Future observations and expected answers are never inputs to earlier writers.

## 3. The replay trainer

### 3.1 Relation to S3 and S4

There is one differentiable execution engine. S3's trainer (`training/neuralese/`) provides the port forward and backward passes, batched producers and consumers, and the self-distillation teacher pass. S4's replay sessions use the same code to serve `grad` requests at inference time. S5 adds the graph layer on top: it replays whole recorded programs rather than single producer–consumer pairs.

### 3.2 What a training step does

1. Load a batch of recorded graphs: S1/S2 trajectories converted in §2, and fresh rollouts from the current model (§3.4).
2. Restore the definitions and context revisions recorded at each node.
3. Supply recorded effect results; never repeat effects.
4. Hold discrete choices fixed: sampled text tokens, tool choices, stop positions, readout choices, host branches.
5. Recompute every differentiable path: written blocks (through the sketch recurrence, completion and content projection), reads, soft bodies, combinator calls.
6. Compute losses at the program's checked outputs and at intermediate consumers (§3.3).
7. Backpropagate. Gradients accumulate at every producer from all of its consumers, across call chains, closures and multiple consumers of the same value.

Calls within a graph are batched across graphs where their structure allows; long chains are processed in topological order with the whole graph held in memory.

### 3.3 Losses

- **Cross-entropy** on the checked outputs and on the recorded actions of each model turn, teacher-forced.
- **Self-distillation.** Token-level KL from the same model given the full source (the crisp rendering of the same call) to the model given the soft value.
- **Log-likelihood objectives** for discrete choices that were good: recorded successful trajectories train decisions such as block opening, helper choice and stopping through `logLikelihood`; outcome-weighted variants carry over to S7.
- **Law consistency** (§5).
- **Ordinary replay.** Ordinary text and the crisp natlang corpus with KL to the crisp base, as in S3 §5.2.

Continuous and discrete credit stay separate: gradients through payloads never stand in for training a sampled decision.

### 3.4 Fresh rollouts

Teacher-forced replay cannot reveal errors in states the recorded trajectories never visited. A share of every batch comes from fresh executions of the current model on training tasks in the real environments. Their graphs are recorded and replayed like any other. After a changed discrete choice, an old observation is not reused; the rollout produces its own.

## 4. Core operators and combinators

### 4.1 What is trained

The combinators of S0 §4 (`map`, `zip`, `ap`, `combine`, `split`, `splitList`, `read`, `convert`, `gloss`) and the composition operator `compose` (S0 §4.3) are system natural-language functions whose bodies are soft values in the standard library's `.nz` file. These library bodies are the default; a model or program may keep its own tuned bodies as values in its context. They are trained as values: `valueAndGrad` with respect to their bodies, independently of task lambdas and of backbone weights.

### 4.2 Initialisation

Each operator body is initialised from a text description of the operator (token embeddings through direct registration, not through the operator itself), so a working text version exists before any soft training. The text versions run first as crisp natural-language functions to collect trajectories.

### 4.3 Objectives per operator

| Operator | Trained by |
| --- | --- |
| `read` | Cross-entropy of the exact value against the known source, through typed validation; self-distillation against reading the exact value. |
| `map` | Downstream consumer losses on the mapped value; read/map commutation (§5). |
| `zip`, `split`, `splitList` | Consumers of the packed and split values; split/zip law. |
| `combine` | Consumers of combined views of the same value; associativity and identity laws. |
| `ap` | Consumers of the result of applying soft functions to soft arguments. |
| `convert` | Consumers in the target dialect; trained only when a second dialect exists. |
| `compose` | Map fusion: the composed function must agree with sequential application. |
| `gloss` | Human-readable diagnostic quality; never a training target for other operators. |

Operators bind their own skills where useful (S0 §4.1), and those skills train with them.

## 5. Laws

### 5.1 Consistency objectives

Each law of S0 §4.2 becomes a loss measured through `read` and through downstream consumers, never as vector equality:

| Law | Loss |
| --- | --- |
| Map identity | Consumer loss and `read` agreement between `map(v, x => x)` and `v`. |
| Map fusion | Agreement between `map(map(v, g), f)` and `map(v, compose(f, g))`. |
| Read/map commutation | Agreement between `read(map(v, f))` and `f(read(v))`. |
| Combine associativity | Agreement between the two groupings, by consumers and `read`. |
| Combine identity | Agreement between `combine(v, empty())` and `v`. |
| Split/zip | Agreement between `split(zip(a, b))` and `[a, b]`. |

Laws are trained "to some degree": their weight is small relative to task losses, and it is tuned from evaluation so that consistency improves without harming task results.

### 5.2 Law measurements and rewrite enabling

S5 produces the law-agreement measurements that enable the compiler rewrites (S0 §4.3):

- A fixed law suite of `.nz` values and functions per law, part of the S5 evaluation set.
- For each law, agreement measured on held-out cases through `read` and consumers, and the effect of the rewrite on whole-program results: the same programs run with and without the rewrite.
- A rule is enabled automatically for the current model and dialect version once the whole-program comparison shows it does no harm. The comparison reruns whenever the model or dialect version changes; every applied rewrite is traced.

## 6. Curriculum

S5 increases how much of a program is soft, step by step, on the same tasks:

Runtime-owned prompts and handover notes (decisions 40, 41) are soft from the first step: they belong to the runtime, not to the program, and are trained in every mix.

1. **Single values.** One converted value per program, with the source withheld from its consumer. Mostly large inputs and tool outputs.
2. **Several values and chains.** Multiple soft values; values that pass through several calls; Neuralese-to-Neuralese computation across functions.
3. **Soft function bodies.** Functions with soft bodies initialised from their text, tuned by `grad` on their call sites' losses, and then written by the model.
4. **Combinators.** Programs that use the combinator library, with operators trained alongside.
5. **Model-written soft code.** The model writes literals and function literals in eval code (§7) inside complete tasks.
6. **Skills.** Data skills with soft forms, bound through contexts.

Substitution also increases by type: plain text, then records and arrays, then function types. Each step is evaluated against the crisp version before the next becomes the default mix. The mix keeps earlier steps and the crisp corpus throughout.

## 7. Model-written literals and function literals

The model must learn to write Neuralese in code, not only consume it:

- **Value literals.** From converted trajectories, eval code in which the model declares `const x: Neuralese<T> = <|neuralese|>…<|/neuralese|>;` and passes `x` on. The type annotation comes first because the corpus is eagerly typed.
- **Function literals.** S1's explicit-capture pass yields every inline `nl` site as a candidate. Each becomes a pair: the text form `nl.with({ … })\`instructions\`` and the soft form with the same captures and a soft body. The model learns to write the capture object explicitly and then the body.
- **Returning soft values** through `return_result` and staged results.
- **Native form.** All of this is rendered inside LFM's Pythonic tool calls, with the literal inside the eval code string (S4 §11).

Losses on the text around the literal are ordinary cross-entropy. The literal's content is trained through its consumers. Opening and stopping are trained by log-likelihood on successful trajectories and by the stop-policy objective from S3.

## 8. Crisp/soft agreement

Every converted task is evaluated in both forms on unseen instances:

- **Result agreement.** Do the soft and crisp programs reach the same checked results and effects?
- **Exactness.** Are exact values (identifiers, paths, numbers, code) preserved where they matter, and is no exact value accidentally routed through a soft channel?
- **Channel use.** For each soft site, correct versus shuffled versus zeroed values at matched length, as in S3, but measured on whole-program results.
- **Cost.** Tokens, model calls, latency and stored entries for the two forms.

Disagreements are read and categorised. They decide which sites, types and families return to the crisp form, need more data, or need operator training.

## 9. Data products for S6 and S7

- **Graph records** of soft and crisp executions with outcomes, for S6's gradient tuning and learned updaters and for S7's policy learning.
- **Trained operators** in the standard library's `.nz` file, with their law measurements.
- **Soft forms of data skills** beside S2's crisp skills, as the starting point for S6's soft skills.
- **Converted task families** with environments, ready as RL environments for S7.
- **Function-literal pairs** for authoring soft functions in S6.

## 10. Checkpoint G2

G2 is a review point, not a fixed threshold. Evaluation runs continuously on held-out task instances and held-out families, and each review decides whether soft programs are ready to hand to S6 and S7 or what to improve first.

| Dimension | Questions the review answers |
| --- | --- |
| Program correctness | Do soft programs match or beat crisp programs on held-out complete-program results at matched inference budget, and in which families? |
| Channel use in programs | Do correct soft values beat shuffled and zeroed ones on whole-program results, not only on local losses? |
| Exactness | Are exact values preserved, and do type and capability checks always pass? |
| Composition | Do results hold across multi-call chains, closures and combinator use? |
| Laws | How well does each law hold, and which rewrites can be enabled for `nd:natlang@1`? |
| Operators | Do trained operators beat their text-initialised versions? |
| Model-written soft code | Does the model write well-typed literals and function literals with correct captures in fresh tasks? |
| Cost | How do tokens, calls, latency and compute compare between soft and crisp forms? |
| Ordinary capability | Has crisp natlang execution or ordinary text regressed? |
| Failure reading | A sample of failed soft programs, read and categorised, tracked over time. |

## 11. Risks

| Risk | Response |
| --- | --- |
| Soft values that do not carry content (consumers learn a format prompt, as Schnitzeljagd saw with SQL) | Shuffled and zeroed comparisons on whole programs from the start; prefer tasks where the withheld source is necessary. |
| Errors compounding along chains | Increase chain length gradually; self-distillation at each consumer, not only at the end. |
| Exactness leaks into soft values | Consumer tracing and the exact-by-nature rule; exactness checks in every evaluation. |
| Laws that cannot be trained without hurting tasks | Laws are weighted lightly; rewrites stay disabled for laws that do not hold. |
| Replay drift from fixed discrete choices | Fresh rollouts in every batch. |
| Ordinary capability regression | Replay with KL to the crisp base; ordinary capability in every review. |
| Graph replay too slow for long programs | Batch by node type across graphs; profile early; keep long-chain episodes a minority until throughput is known. |

## 12. Work items

1. Consumer tracing and classification over S4 graph records (§2.1).
2. Compiler conversion pass for value, function and skill sites, with recompilation checks (§2.2–2.3).
3. Converted-task builds from S1 executable tasks and S2 episodes, with both forms executed and validated (§2.4).
4. Graph-level replay trainer on top of S3's trainer and S4's replay sessions (§3.1–3.3).
5. Fresh-rollout pipeline in the real environments, recorded as graphs (§3.4).
6. Text versions of the operators and `compose`; trajectory collection (§4.2).
7. Operator training as soft values (§4.1, §4.3).
8. Law losses and the law suite; per-dialect law measurements and the rewrite comparison (§5).
9. Curriculum mixes for each substitution step (§6).
10. Literal and function-literal training data and losses (§7).
11. Crisp/soft agreement evaluation (§8).
12. G2 review harness and failure-reading workflow (§10).
13. Publication of data products to S6 and S7 (§9).

## 13. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Rewrite enabling:** automatic once a whole-program comparison on the current model shows no harm; traced (§5.2).
2. **Operator bodies:** the standard library's bodies are the default; a model or program may keep its own tuned bodies as values in its context, promoted to the pool when they help broadly.
3. **Backbone:** S5 continues releasing shared backbone layers (as in S3 phase F). When the model's space changes, the dialect version is bumped and stored values are regenerated or converted; dialect stability is not a concern yet.
