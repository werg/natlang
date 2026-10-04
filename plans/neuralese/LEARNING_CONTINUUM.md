# The learning continuum: search, gradients, adapters and learned improvers

Draft, 2026-10-04. A cross-cutting plan that revises [S6](S6_META_LEARNING.md) and extends [S5](S5_PROGRAM_TRAINING.md) and [S7](S7_RL.md). It puts every way a natlang program can improve into one framework, so that improvement methods can be combined, compared at matched compute and distilled into one another. The design choices listed in §14 were accepted by the owner on 2026-10-04 as **defaults**, not rules: each is where work starts and what a run uses unless it says otherwise. Any of them may be overridden per experiment, artifact or program when evidence or a task calls for it, and the override is recorded with the run. Decision 37 (adapters as program artifacts) is the owner's direction of 2026-10-04.

## 1. Thesis

Every improvement method we have or want is the same act: an **operator** reads a program's current **state** and some **evidence**, and proposes a new state. The methods differ along three axes:

1. **What changes:** crisp text (instructions, skills), soft values (Neuralese instructions, soft skills, operator bodies), or weights (tiny adapters).
2. **Which signal the operator may see:** labels and targets (supervised), rewards from checks (reinforcement), a privileged teacher's behaviour (conditioned distillation), or none at all (reward-blind learning from experience).
3. **How the operator computes the proposal:** discrete search (DSPy/GEPA-style reflective mutation), zeroth-order estimates (evolution strategies, score-function RL), first-order gradients, second-order gradients, or **amortised** prediction by a trained natlang function (a learned updater or an adapter writer), which replaces an optimisation loop with one call.

The programme makes all three axes first-class and orthogonal. Any operator can act on any artifact kind under any signal regime that the artifact admits. Every application of an operator is recorded in one format (§8). Those records are what the amortised operators are distilled from. This gives the path the owner asked for: direct training first, record the trajectories, then distil the improvement process into a meta-learner.

The machine is end-to-end meta-optimisable because the operators are themselves natlang programs bound to contexts. The search policy, the learned updater, its faceted meta-skills and the adapter writer are artifacts like any other, so the same operators improve them (§12).

## 2. What exists today

| Piece | Where | Role in this plan |
| --- | --- | --- |
| Immutable learning surface: `grad`, `valueAndGrad`, `stopGradient`, optimisers, `iterateOn` steps | `natlang:learning` (ts-host/src/neuralese/learning.ts), S0 §9 | The first-order operator for soft artifacts. |
| Objectives: `crossEntropy`, `decision` (Brier/RPS/log loss on the decision readout), `selfDistill`, `logLikelihood`, `klPrior`, law objectives | same; server terms in serve/grad.py | Supervised, conditioned-distillation and RL signals. |
| Gradient replay with discrete choices fixed, per-term accumulation, `decision_backward` | Python reference server | The engine for every gradient operator. |
| Crisp authoring with GEPA-style search, sealed query evaluation, ablations | S2, `run_skill_authoring_queue.py`, `improveProgram` | The discrete-search operator and its protocol. |
| Episodes with support/query/transfer, the gate, headroom screens | S2 §4, §5.1b, §5.1c | The common evaluation substrate. |
| Soft-skill arms none / text-init / generic / tuned / specific | `soft-skill-decision.mjs` | The first matched comparison of direct soft training. |
| Decision readout | spec/SPEC.md | Smooth, calibrated objectives on typed decisions; the cleanest loss landscape we have. |
| LoRA injection and deltas-off teacher | training/neuralese/natlang_neuralese/train/adapters.py | The starting point for weight artifacts. |
| Decider and Clef labels, 58,500 decision cases | decision-data-20261004 | Teacher signals for distillation into readouts, soft skills and adapters. |

## 3. Artifacts

An artifact is an immutable, content-addressed value in a program's context (decision 20). This plan adds one artifact kind and one wrapper.

| Artifact | Type | File | Differentiable | Searchable |
| --- | --- | --- | --- | --- |
| Instructions | `.nl` body text | `.nl` | No | Yes (rewrite) |
| Crisp skill | `SKILL.md` + files | folder | No | Yes |
| Soft instructions | `Neuralese<F>` soft body | `.nz` | Yes | Yes (resample, write) |
| Soft skill / scope value | `Neuralese<SkillBody>`, `Neuralese<T>` | `.nz` | Yes | Yes |
| Operator body | soft bodies of `map`, `read`, … | `.nz` | Yes | Yes |
| **System prompt piece** (decision 40) | `Neuralese<SystemPrompt>` per piece ID (interpreter, extensions, decision, predicate, compaction frames, program guidance) | `.nz` bank shipped with the checkpoint | Yes | Yes (rewrite the text, re-embed, refine, merge) |
| **Adapter** (new, §6) | `Adapter<Base, Kind>` | `.nz` export | Yes | Yes (ES, merge) |
| **Delta** (new, §5) | `Delta<A>` over any of the above | `.nz` / patch | Where `A` is | Yes |

Crisp and soft forms are connected by five **bridge operators**, which make discrete search and gradient descent interchangeable rather than parallel:

- **encode** (crisp → soft): the text read in one forward pass through the port, its token embeddings supplied at the
  block positions as in S3 phase A, so the output port gives one payload vector per token
  (`/v1/neuralese/encode`). No summarising call and no compression; the default initialisation of soft artifacts
  (system-prompt pieces, skills, instructions) since 2026-10-04;
- **embed** (crisp → soft): raw token-embedding initialisation (no forward pass), used by the earlier soft-skill runs;
- **write** (crisp → soft): the writer reads the text and writes a block through the port, so length and content are learned;
- **verbalize** (soft → crisp): `read<string>` of a soft artifact under instructions that ask for a usable skill text. Its output is a *candidate*, judged by evaluation like any crisp proposal. Glosses stay diagnostics (decision 15); a verbalized skill is a new crisp artifact with provenance;
- **distil-to-weights** (soft or crisp → adapter): context distillation into an adapter (§6.4), so knowledge carried in a prompt can move into weights when that is cheaper to ship or run.

## 4. Signal regimes

Every regime is available to every differentiable artifact through `natlang:learning`. Every regime is also available to search, which uses the same quantities as fitness.

### 4.1 Supervised

Loss on support cases with known targets: `crossEntropy` of the expected result, `decision` against gold or teacher distributions, `logLikelihood` of accepted trajectories. This is what the soft-skill runs use today.

### 4.2 Reinforcement

Rewards from checks (S7 §1), through `logLikelihood` of sampled trajectories weighted by advantage. The payload temperature `τ > 0` gives written blocks a density, so blocks and adapters can be trained by score-function estimates where no differentiable path exists. Importance-weighted off-policy reuse with ratio clipping follows the stop-head exploration already in the S3 trainer.

TinyLoRA's controlled finding is relevant: with very few parameters, RL is far more parameter-efficient than supervised fine-tuning. A 13-parameter adapter trained with GRPO recovered most of the gain on maths benchmarks. **Default (§14):** the regime for the tiniest adapters (§6.2) is RL, and the default for soft skills stays supervised and distillation, with RL as the comparison arm.

### 4.3 Conditioned distillation

The teacher is the same model on the same task, **conditioned on knowledge the student will not see**. The student is the program with the artifact being trained and without the knowledge. The loss is the KL from the teacher's behaviour to the student's, or the cross-entropy against the teacher's decision readout.

`selfDistill` already takes `teacher_messages` (the full-source teacher of decision 8). This plan generalises it into `objectives.conditionedDistill(student, teacher)`, where `teacher` is any run of the same program with extra context. The teacher can be conditioned on any of the following:

| Teacher conditioning | Student lacks | Trains |
| --- | --- | --- |
| The crisp skill text | the skill (or has only its soft form) | soft skill initialisation and repair |
| A reference document, schema, API manual | the document | knowledge artifacts: soft scope values or adapters (Doc-to-LoRA's setting) |
| Gold answers or worked solutions (privileged information) | the answer | instruction blocks that elicit the behaviour, not the answer |
| Hindsight: the outcome of this attempt | the outcome | the reward-blind improver (§10) |
| A stronger model's trajectory (Qwen, paid teachers) | the trajectory | anything, at a cross-model KL where tokenizers match, else on decision readouts and checked outputs |

For decision readouts the conditioned loss is exact and cheap: the teacher's distribution over options is a fixed target (the hold-to-generic term in `soft-skill-decision.mjs` already does this). The gate gains a *privilege check* for conditioned distillation data. No fragment of the teacher's privileged context may appear in the student's view (`answerFragments` from S2 already implements the fragment test).

## 5. The residual update operator

**Default (§14).** Updates produce **deltas**, not replacements: `Delta<A>` is a value with `apply(base, delta, scale = 1)`, `diff(after, base)` and `compose(d1, d2)`.

| Artifact | Delta form | Composition |
| --- | --- | --- |
| Soft block of length L | L residual vectors added in the base's positions; an optional appended extension block | Sum; extensions concatenate |
| Adapter | parameter delta (LoRA factors concatenate in rank; vector adapters add) | Sum, with learned merge coefficients |
| Crisp skill or instructions | a patch over the files | Merge; a conflict means the composition is rejected |

Why deltas:

- **Independent training composes.** Artifacts trained separately from a shared base (per family, per regime, per teacher) can be summed. Interference is measured (each delta alone versus the sum) and merge coefficients are learned on support cases, LoRAHub-style, as a cheap operator in its own right.
- **Ablation and attribution are free:** scale 0 removes an update, scale 0.5 halves it. Leave-one-update-out replaces leave-one-skill-out.
- **Learned operators predict small things.** An updater that writes a zero-initialised residual starts as the identity, the reason residual networks train well. Prediction errors are bounded by the delta's scale rather than by the artifact's whole content.
- **Safety and reversibility.** A program's context stores base plus deltas with provenance, so any learned change can be reverted without retraining.

The direct operators produce deltas too. A gradient run's result is `diff(tuned, init)`, which becomes a training target for the learned updater, the "paired soft target" of S6 §5.3 put in delta form.

**Who writes deltas (owner, 2026-10-04).** A model never writes a delta directly. Deltas exist in two places:

1. **Host arithmetic.** `diff`, `apply` and `compose` compute deltas from values (a gradient run's `diff(tuned, init)`, compositions, scaled ablations). They are stored in the `#delta` variant of the base's dialect so that they cannot be read as values or bound as adapters (`ts-host/src/neuralese/deltas.ts`). This is bookkeeping, not a language anyone speaks.
2. **Learned updaters write ordinary Neuralese.** An updater's output is a plain block in the model's dialect, written through the write port like any other value. A **trained delta projection** `D_A: Neuralese → Delta<A>`, one per artifact kind, turns it into the update, the same pattern as the adapter projection `P` (§6.4):
   - *Soft values:* `D` maps the written block to residual vectors in the base's positions. A learned per-position readout attends over the written block, conditioned on the base block, so the update can depend on what it changes. A zero-initialised output layer makes an untrained `D` the identity update.
   - *Adapters:* `D` is `P` applied to the written block, giving a coefficient delta. An adapter update and an adapter code are then the same kind of object.
   - *Crisp artifacts:* the updater writes text, and the patch is computed by `diffFiles`. No projection is needed.

   `D` belongs to the base model's port machinery and is versioned with the dialect, like `P`. It is trained in the same stages: fit to recorded deltas, then end to end on downstream loss through `apply(base, D(block))`, then fixed while updaters train. The written block stays an ordinary value, so it can be read, verbalised and combined, and the meaning of an update is inspectable.

## 6. Weights: tiny adapters shipped with programs

Decision 37 (owner, 2026-10-04): programs may carry small weight adapters as context artifacts, scoped to the program, a function or a skill, trained and shipped like `.nz` values. This amends S0 §9.7: `grad` and updaters may produce adapter values. The backbone itself is still trained only offline, and an adapter is bound by rebinding a context, never applied globally.

### 6.1 Rules

- **Base identity.** An `Adapter<Base, Kind>` names the exact base weights (content hash) it was trained on; the runtime refuses it on any other model. A model change means re-tuning or regenerating adapters, as for dialects (decision 24).
- **Scope.** An adapter applies only inside the calls of the functions whose context binds it, so its effect is local and attributable. Nested calls inherit it only if their own context binds it.
- **Ports are protected.** Adapters never touch the interface norm, the port heads or the control-token rows. **Default (§14):** adapters sit on the layers after the sketch cutoff, `[k, D)`. The writer's sketch recurrence then stays the base model's, and the dialect cannot drift through the write path.
- **Boundary check.** A block written inside an adapted call and read outside it must still mean the same thing. The gate and the conformance suite gain a *crossing test*: a block written under an adapter, read by the unadapted model, gives downstream behaviour within tolerance of reading it under the adapter. Where it fails, the adapter is trained with that crossing as a consistency term.

### 6.2 Parameterisations, smallest first

| Kind | Trainable parameters | Notes |
| --- | --- | --- |
| `tiny` (TinyLoRA) | 1 to a few hundred per model | A vector projected through fixed random tensors into the top singular subspace of frozen weights (on top of LoRA-XS). A few bytes. RL-first. |
| `svf` (Transformer²'s singular-value fine-tuning) | one scale per singular value of each adapted matrix | Compositional expert vectors trained with RL; mixed at inference by a dispatcher. |
| `xs` (LoRA-XS) | r×r per matrix between frozen SVD bases | Strong per-parameter efficiency; LoRA-compatible form. |
| `bank` (VB-LoRA, VeRA) | vectors selected from a shared global bank, or scaling vectors over shared random matrices | Many adapters share one bank, so a library of adapters is cheap. |
| `lora` (rank 1–8) | standard | The heaviest kind, for when the small ones are not enough. |

The frozen bases are stored once with the model (SVD of its weights, the random projections). An adapter file then carries only its coefficients, from bytes to kilobytes. **Default (§14):** restrict `tiny`, `xs` and `bank` to the top-r singular subspace so each converts exactly to a rank-r LoRA. Every server then serves every kind through its LoRA path (§6.5). `svf` over all singular values is kept for research only, because it is full-rank.

### 6.3 Mixtures

Several adapters can apply to one call (a program adapter, a function adapter, the adapters of loaded skills):

1. **Hard routing by selection.** The adapters of the skills the call actually loads apply; skill discovery already decides that, so routing comes for free. This is the default.
2. **Learned merge coefficients** per function (LoRAHub): a handful of scalars, trained by gradient or ES on support cases.
3. **Gated mixtures** (X-LoRA, MoLE): a gate on hidden states mixes adapters per token. This is a research item; it breaks batching and costs latency.

### 6.4 Writing adapters: the amortised path

Text-to-LoRA showed that a hypernetwork can produce useful adapters from a task description in one forward pass, compressing hundreds of trained adapters. Drag-and-Drop does the same from unlabeled task prompts, and Doc-to-LoRA amortises context distillation of a document into an adapter. The natlang analogue does not need a separate hypernetwork architecture:

**Adapters as Neuralese (default).** An `AdapterCode` is a Neuralese block. An adapter writer is an ordinary natlang function: `writeAdapter(description, evidence) -> Neuralese<AdapterCode>`. It is written through the same write port, trained with the same objectives (§4) and improved by the same operators. The writer is the model itself, so adapter writing benefits from everything the model knows about the task.

**The adapter projection.** Neuralese blocks live in the model's semantic space; adapter coefficients live in a different space (per adapted layer, the `tiny`/`xs` coefficients of §6.2). Between them sits a **trained projection** `P: Neuralese<AdapterCode> → Adapter<Base, Kind>`:

- **Form.** A block of n vectors is pooled into per-layer queries (one learned query per adapted layer and matrix, attending over the block's vectors), each mapped by a small linear or MLP head to that matrix's coefficients. A block of any length decodes to a fixed adapter shape, so the writer's stop head still decides length. A zero block decodes to the zero delta (§5).
- **Belongs to the base.** `P` is part of the model's port machinery, like the interface norm: one per base model and adapter kind, versioned with the dialect, shipped with the model and never with a program. Programs ship codes or decoded adapters; a decoded adapter is cached by the code's content ID.
- **Training in three stages.**
  1. **Fit to directly trained adapters:** from the adapter records (§8), each tuned adapter is paired with a code (its evidence written by the writer, or a free code optimised per adapter by gradient). `P` is trained to reconstruct the adapter coefficients, with a downstream KL from the tuned adapter's behaviour so that coefficient errors in directions that do not matter are not penalised.
  2. **End to end:** the downstream task loss of the decoded adapter is backpropagated through `P` into the code, and so into the writer and its context. The gradient session treats `P`'s parameters as trainable alongside the writer.
  3. **Codes as first-class adapters:** once `P` is good, adapters are trained directly as codes (a block is the leaf, `P` fixed), so a code is just a soft value that happens to act on weights. Codes can then be read, combined (`combine`, deltas) and verbalised like any other block.
- **Checks.** Round trip (adapter → best code → decoded adapter) keeps behaviour; codes of unrelated tasks decode to adapters that do not interfere when summed beyond what their direct-trained versions do; shuffled codes lose the gain (the channel-use check of S3).

### 6.5 Training and serving

- **Training.** An adapter is a differentiable context item: `valueAndGrad(f, adapter)` works as for blocks. The reference server's gradient session gains adapter leaves: coefficients become leaf tensors, and the adapted forward applies them functionally per call. The S3 trainer's LoRA code is the offline path for larger adapters. Jobs go through the memory ledger.
- **Serving.** The reference server applies adapters per request. llama.cpp's server already supports per-request LoRA with scales, but requests with different adapter sets are not batched together. The fork therefore needs grouping by adapter set and, later, batched multi-LoRA kernels (S-LoRA/Punica style) as an upstreamable patch. vLLM has multi-LoRA serving. The conformance suite gains adapter cases (§13).

## 7. Operators and the hybrid optimiser

### 7.1 Operator catalogue

All operators share one signature: `(state, evidenceView) -> Proposal`, where a proposal is a list of deltas with provenance.

| Operator | Order | Artifacts | Signal |
| --- | --- | --- | --- |
| Reflective rewrite (GEPA/crisp author) | discrete | crisp | any visible feedback |
| Resample / rewrite block (write port at τ > 0) | zeroth | soft | fitness |
| Evolution strategies on blocks or adapters | zeroth | soft, adapter | fitness |
| Score-function RL step | zeroth | soft, adapter, discrete choices | reward |
| Gradient step (Adam etc.) | first | soft, adapter | any differentiable objective |
| Second-order / unrolled step | second | soft | as above, short episodes |
| Merge (learned coefficients over deltas) | first or zeroth | any with deltas | fitness |
| Bridges: embed, write, verbalize, distil-to-weights | — | crisp ↔ soft ↔ adapter | — |
| Learned updater (§9) | amortised | soft, crisp, adapter | as trained |
| Adapter writer (§6.4) | amortised | adapter | as trained |
| Reward-blind improver (§10) | amortised | any | none at inference |

### 7.2 One optimiser for search and gradients

**Default (§14):** the S2 GEPA search becomes a **memetic optimiser**: population-based search whose individuals are whole program states (crisp text, soft values and adapters together). Its moves are drawn from the operator catalogue, including gradient descent.

- **Local refinement inside search.** After a discrete mutation (a crisp rewrite), a short gradient run refines that individual's soft parts and adapters before it is evaluated. Two variants are compared:
  - Lamarckian: keep the refined parameters.
  - Baldwinian: score after refinement but keep the unrefined genotype, so the search selects for refinable structure.
- **Gradient-guided mutation.** Per-item gradient norms and per-case gradient contributions locate where a program is failing. The reflective author is told which skill and which cases drove the gradient. A gradient digest rendered as text is a better-founded "textual gradient" than an LLM critique alone (TextGrad), because it is an actual derivative.
- **Bridging moves.** Verbalize a well-tuned soft skill into a crisp candidate the author can edit; embed or write an edited crisp skill back to soft and keep tuning. Knowledge can cross the crisp/soft boundary in both directions within one search.
- **Operator selection is a policy.** Which operator to apply to which individual starts as a bandit over operator families, credited by validation gain per unit compute. Its decisions are recorded and later trained (§9), so the search policy itself becomes learned.
- **Pareto selection per case** (as in GEPA) keeps individuals that win on different cases, which feeds the merge operator with complementary deltas.
- **Alternation as a special case.** Prompt optimisation and weight fine-tuning in sequence (DSPy BetterTogether) is one schedule this optimiser can run. Reported gains are compounding: in the BetterTogether paper, joint prompt and weight optimisation beat weights alone and prompts alone. In a Databricks case study on IE Bench, GEPA and fine-tuning combined gained more than either alone.

Matched-compute accounting (S6 §8) covers every operator, measured from traces.

## 8. The improvement record

Every operator application, direct or learned, writes one record, `natlang.improvement-step/1`. The record is the data spine of the meta-learning in §9 and §10.

| Field | Content |
| --- | --- |
| `episode`, `family`, `facets` | episode identity; facet tags (§9.2) |
| `before` | content IDs of the state's artifacts |
| `operator` | kind, version, hyperparameters, the operator's own context ID if learned |
| `view` | the evidence the operator saw, by content ID, with its **visibility class**: `full`, `reward-blind-observations`, `reward-blind-strict` (§10.1) |
| `proposal` | deltas with scales |
| `after` | content IDs |
| `outcome` | host-only: support, query and transfer gains, compute. Never part of any view. |
| `trajectory` | the episode's step index, so records chain into trajectories |

S2 authoring results, S6 tuning runs and the soft-skill arms are converted to this format, and new runs write it directly.

## 9. Learned improvers and faceted meta-skills

S6 §5 defines the learned updater. This plan fixes how it is trained from records and how its knowledge is organised.

### 9.1 Training sequence

1. **Imitation of improvement.** Advantage-weighted imitation of recorded steps with positive query gain: the updater learns to produce the delta the direct operator found, given the view that operator had. The updater writes an ordinary block, and the delta projection `D` (§5) turns it into the update. Training targets are the recorded deltas: the loss is the distance from `D(written)` to the recorded delta, in the downstream KL metric, plus the downstream loss of `apply(base, D(written))`. Gradients reach the writer through the write port and `D`, while the recorded delta itself is never a sequence the model has to emit. Crisp deltas are trained as text, and adapter deltas through `P`.
2. **Query-trained.** The outer objective is query gain after a bounded inner loop (S6 §5.4), first-order by default.
3. **Self-revision** (S6 §7) once the first two hold on held-out families.

Each step is compared against the direct operator at matched compute. The headline quantity is the **amortisation ratio**: the query gain of one learned step divided by the gain of a full direct optimisation run on the same episode.

### 9.2 Topic-faceted meta-learning skills

The updater's context holds **meta-skills**: soft skills about *how to improve* things, faceted by:

- artifact kind (crisp skill, soft instructions, adapter);
- signal regime (supervised, RL, distillation, blind);
- task family or topic (SQL schema reasoning, calibration of decisions, code repair, retrieval, formatting);
- failure type (missing knowledge, wrong procedure, miscalibration, format).

Facet tags come from episode metadata and from a classifier over the evidence view.

- Each meta-skill is trained only on records of its facet, as in `soft-skill-decision.mjs`'s *specific* arm: the generic meta-skill is the reference, and a facet meta-skill must beat it on its facet without changing behaviour off-facet (the hold term).
- The updater selects meta-skills through ordinary skill discovery, so applying the right improvement knowledge is the same problem as applying the right task knowledge, and is trained the same way.
- New facets are created by the soft-authoring operations of S6 §6 applied to the updater's own context.

## 10. The reward-blind improver: learning from experience

The owner's target: a general `improve(function)` operator that, from semantic background knowledge and natural observation of attempts, infers in which direction a function should change, **without seeing any reward, target or check result**.

### 10.1 What it may see

Two visibility classes, both enforced by the gate:

- `reward-blind-observations`: the function's source and context; its inputs; its full attempt trajectories (reasoning, tool calls, the environment's natural responses, such as a runtime error, an empty query result or a tool's output); background documents and skills. Excluded: expected outputs, scores, gates, pass/fail flags, checker and judge outputs, teacher comments, and anything derived from them.
- `reward-blind-strict`: as above, without environment responses other than the function's own outputs. The improver must then infer purely from what the function was asked to do and what it did.

The view builder strips the excluded fields by schema. The gate adds a fragment check (no expected-answer fragment or metric value appears in the view) and a structural check (no field derived from an `outcome` record).

### 10.2 How it is trained

Rewards exist only on the outer loop and never enter the improver's inputs:

1. **Privileged-to-blind distillation.** The teacher is the reward-aware updater (§9) or the direct operator's recorded delta. The student gets the blind view of the same step and must propose the same delta. This is conditioned distillation (§4.3) at the meta level: the privileged information is the feedback.
2. **Outcome-weighted.** Blind proposals are executed by the host, and their query gain weights `logLikelihood` of the proposal (RL with a reward the policy never observes, the bi-level structure of Meta-TTL and SEAL).
3. **Hindsight relabelling.** For failed attempts, the delta that fixed them in a later recorded step is a target for the earlier blind view, which multiplies the usable data.

### 10.3 Measuring it

The **blind ratio** is query gain of the blind improver divided by query gain of the reward-aware updater, on held-out families. The baselines are three:

- a null improver (no change);
- the generic meta-skill;
- a blind improver given shuffled trajectories from another function, which tests whether it reads the trajectories at all.

A blind improver that beats the null baseline on held-out families is the result to look for: it has learned what good functions look like from observation alone.

## 11. Data flows end to end

```text
episodes (gated, headroom-screened)
  → direct operators (search, gradient, RL, distillation; crisp, soft, adapter)
      → improvement records (§8) with host-only outcomes
          → learned updater + faceted meta-skills (§9)      → its own records
          → adapter writer (§6.4)                            → its own records
          → reward-blind improver (§10)                      → its own records
  → memetic optimiser (§7) uses all operators, learned ones included, and records operator choices
      → operator-selection policy (§7.2) trained on those records
```

All learned operators are contexts. Their improvement is another episode family ("improve this improver"), with inner episodes as its cases and the same gate, ticket evaluation and records (S6 §7).

## 12. Evaluation

The S6 comparison protocol applies throughout (same episodes, sealed query evaluation, matched compute, paired results, held-out families). This plan adds:

| Measure | Question |
| --- | --- |
| Gain per artifact kind and regime | Which combinations improve which families, at what cost? |
| Interference of composed deltas | Do independently trained updates add? |
| Amortisation ratio | How much of direct optimisation does one learned step deliver? |
| Blind ratio | How much of the reward-aware gain survives without reward? |
| Crossing test | Do adapted calls keep Neuralese meaning across their boundary? |
| Specificity | Does a facet meta-skill or family skill learn its facet rather than generic priming (`soft-skill-decision.mjs`)? |
| Shipping cost | Artifact bytes, serving latency with adapters, batching loss |

Reviews remain rubric-based (decision 30). Headroom screens (S2 §5.1b) are run per executor before any comparison, and shortcut checks (data/shortcuts.py, `audit_decision_shortcuts.py`) before any training.

## 13. Phases and work items

Phases follow dependencies, not dates. Each ends at a review.

**M0: records and arms (now).**
1. `natlang.improvement-step/1` schema, writer in the authoring queue, `iterateOn`-based tuning loops and soft-skill arms; converters for existing S2 and soft-skill results.
2. Generalise the soft-skill arms into a method-arm runner over episodes: operator, artifact kind, regime, compute accounting.
3. `objectives.conditionedDistill`, decision-readout form included; the gate's privilege check.

*Status 2026-10-04:* done. Item 1: `ts-host/src/improvement/step-record.ts` (content-addressed records; `stepView` drops the outcome); `convert-improvement-steps.mjs` (67 steps from today's authoring and soft-skill runs in `improvement-steps-20261004/converted-v1.jsonl`); `collect-episodes.mjs` and `soft-skill-decision.mjs` write steps directly. Item 2: `run-method-arms.mjs` runs none, soft-init, soft-gold, soft-teacher, adapter-gold, adapter-teacher and joint-gold arms, with compute per step. Item 3: `objectives.conditionedDistill` (an exact decision form with the teacher readout computed by the server; a KL form for replies; refuses unconditioned teachers and visible privileged text). Episode cases carry `privileged`, and the gate enforces `leak-privileged`.

**M1: direct training across regimes on soft artifacts.**
4. Supervised (exists), RL via advantage-weighted `logLikelihood` at τ > 0, and conditioned distillation for soft skills and instructions on decision, graded and SQL families.
5. Crisp/soft bridges as operators: embed and write (exist), verbalize (new), each evaluated as a proposal.

**M2: deltas.**
6. `Delta<A>`: `diff`, `apply`, `compose` for blocks, patches and adapters (host arithmetic); delta-form records; interference and learned-merge operator. The trained delta projection `D` (§5) comes with M5, next to `P`.

*Status 2026-10-04:* `ts-host/src/neuralese/deltas.ts` (diff, apply, compose for blocks and adapters; file patches). `learning.deltas.interference` and `learning.deltas.learnMerge` (first-order merge coefficients, one gradient session per step) are tested. Delta-form records come from the method-arm and memetic runners.

**M3: memetic optimiser.**
7. GEPA population with gradient refinement (Lamarckian and Baldwinian), gradient-guided mutation and gradient digests rendered for the author, bridging moves, and a bandit operator selector with recorded choices.

*Status 2026-10-04:* first version in `ts-host/scripts/skills/memetic-decision.mjs`, on decision families. An individual is guidance text plus a soft skill. Operators: reflective propose (Qwen; worst support cases with their gradient shares as the digest), embed bridge, Lamarckian and Baldwinian refinement, and learned merge of relatives. A UCB bandit credits operators by validation gain per second, selection keeps the per-case Pareto front, and every application is an improvement step. `natlang-memetic-decision-v1` (6 families) is queued. Adapters as individual parts, verbalize moves and crisp skills come next.

**M4: adapters.**
8. `Adapter<Base, Kind>` export in `.nz`, base identity, scoping rules, crossing test.
9. Adapter leaves in gradient sessions; `tiny`/`xs`/`bank`/`lora` parameterisations in the top-r singular subspace; RL-first training for `tiny`.
10. Serving: per-request adapters in the reference server and the fork (grouping by adapter set first, batched multi-LoRA kernels next), adapter cases in the conformance suite.
11. Context distillation into adapters (distil-to-weights) as a bridge.

*Status 2026-10-04:* first deliverable done. `xs` and `tiny` adapters in the top-r subspace (`model/tiny_adapters.py`). The spec is the block dialect, which states the base hash, and adapters are refused on another base. They are applied per row through forward hooks, so mixed-adapter batches work, and they are leaves in gradient sessions with Adam steps. Requests (`x_natlang_adapters`), `decide` and `grad` take adapters. TS: the `Adapter` type, `withAdapters`, `adapters.create`, and `valueAndGrad` over adapters. Adapters round-trip through `.nz`. The fork refuses adapter requests with 501. Not done yet: `bank`/`lora` kinds; binding through a function's context (`withAdapters` is a dynamic scope over the calls inside it, which the context binding of §6.1 will sit on); the crossing test; LoRA export to the fork and vLLM; RL-first training for `tiny`.

**M5: amortised operators.**
12. Learned updater trained from records (imitation in delta form, then query-trained), with faceted meta-skills and the hold-to-generic term.
13. Delta projection `D` (§5) and adapter projection `P` (per base and kind) fitted to directly trained adapters, then trained end to end; adapter writer producing `AdapterCode` blocks; codes as first-class adapters (§6.4).

*Status 2026-10-04:* `model/projections.py` has `BlockProjection` and `AdapterProjection` (P) plus `DeltaProjection` (D). In each, a zero block decodes to zero and an untrained projection is the identity update. The server takes adapters as `{code, projection}`, and gradients reach the code through P. Stage 1 (`train/projection.py`) is in: on 33 method-arm adapters, the held-out round trip has relative error 0.0023 against 0.277 through a random P (codes of 8 vectors; codes of 1–2 vectors cannot separate the output rows). Stage 2 (`train/projection_e2e.py`, end to end on decision loss; held-out families compare a code through P, a direct adapter, a random P and no adapter) is running as `projection-e2e-v1`. D's training waits for recorded soft deltas.

**M6: reward-blind improver.**
14. View builders and gate checks for both visibility classes.
15. Privileged-to-blind distillation, outcome-weighted training, hindsight relabelling; blind ratio on held-out families.

*Status 2026-10-04:* item 14 (view builders and gate checks) is in `ts-host/src/improvement/blind-view.ts`: allow-lists per trace event kind, environment fields only in the observations class, tool messages withheld in the strict class, and forbidden feedback keys stripped at any depth. `checkBlindView` is the structural and fragment gate. Real authoring traces pass it, and synthetic leaks are caught. Item 15 needs reward-aware updater records (M5) as teachers.

**M7: closing the loop.**
16. The operator-selection policy trained from memetic-optimiser records; learned operators as individuals in the search; meta-episodes improving the improvers (S6 §7).

M0–M2 can start immediately on the current port and server. M4 starts in parallel, with its first deliverable the gradient-session adapter leaves. M5 needs records from M1–M4. M6 needs M5's reward-aware updater as its teacher.

## 13a. Projections: Neuralese as a control language (direction, owner 2026-10-04)

`P` (§6.4) and `D` (§5) are instances of one pattern. The model writes in one semantic space, and **trained projections** carry a written block into whatever it should act on. Further projections can steer other machinery the same way: an image or audio model's conditioning (its text-encoder space, a style or LoRA space), steering vectors of another LLM, the parameters of a classical learner or a search procedure, a controller's setpoints.

**The port stays as it is.** The model's input and output port is ordinary Neuralese, the same write and read path every value uses. Projections are additional **projection adapters** attached outside the port, never changes to it:

- *Output projections* take a written block (a normal value in the model's dialect) and map it into a target's control space: `P`, `D`, an image model's conditioning.
- *Input projections* go the other way and map another system's representation (an image encoder's embeddings, another learner's state) into a normal Neuralese block, which the model then reads through its ordinary read port.

Adding or retraining a projection therefore never moves the dialect, and every projection can be swapped, versioned and conformance-tested on its own.

A projection is `Projection<Target>`, defined by:

- **Identity:** source dialect, target identity (the model or system and its version, e.g. a base hash), and the projection's own content hash. It is refused on any other source dialect or target, as adapters are refused on another base.
- **Form:** a learned pooling over the block's vectors (queries attending over it) into the target's fixed-shape control space. The zero block maps to the target's neutral control where one exists.
- **Training, in the three stages of §6.4:** fit to target controls known to work (reconstruction in a downstream metric of the target's behaviour), then end to end through the target when it is differentiable, or with the zero-order and RL operators of §7 when it is not, then fixed while writers train against it.
- **Checks:** round trip, shuffled blocks lose the effect, and non-interference of summed controls where the target composes.
- **Placement:** projections ship with the base model's port machinery (or with the target's adapter package), are versioned with the dialect, and are never part of a program. Programs ship blocks. A function that drives a target declares the projection it needs, and the runtime refuses a mismatch.

This keeps one language for everything the model produces. A block that steers an image model can still be read, verbalised, combined and improved by the operators of §7, and every projection trains with machinery that already exists. No projection beyond `P` and `D` is scheduled yet. The first concrete candidate should be a target whose control space is small and whose outcome is cheap to score.

## 14. Defaults (accepted 2026-10-04)

Accepted by the owner as defaults, not rules. Each names where work starts; an experiment, artifact or program may override it, and records the override and its reason with the run. Reviews revisit a default when the comparison data argue against it.

| # | Default | Typical reasons to override |
| --- | --- | --- |
| 1 | RL for `tiny` adapters; supervised or distillation for larger kinds. | A family with dense targets, where supervised signal is cheaper; RL comparison arms. |
| 2 | Adapters on layers `[k, D)`, protecting the sketch path. | Tasks whose writing behaviour should change; studies of full-depth adapters with the crossing test. |
| 3 | Top-r subspace parameterisations that convert to rank-r LoRA. | Research on full-rank `svf`; reference-server-only experiments. |
| 4 | Adapters as Neuralese codes through a trained projection (§6.4). | Direct adapter training as the baseline; codes too lossy for a family. |
| 5 | Deltas instead of replacements for learned updates. | Wholesale rewrites, such as resizing a block or restructuring a skill, recorded as a replacement delta. |
| 6 | The memetic optimiser as the search in S2/S6. | Plain GEPA as a comparison arm or when no differentiable part exists. |
| 7 | Both visibility classes for the reward-blind improver. | Families whose environment responses are themselves verdicts (a test runner's pass/fail) count those as feedback under both classes. |

## 15. Sources

- [Text-to-LoRA: Instant Transformer Adaption](https://sakana.ai/text-to-lora/) (ICML 2025, [paper](https://proceedings.mlr.press/v267/charakorn25a.html))
- [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://arxiv.org/pdf/2602.15902)
- [Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights](https://arxiv.org/abs/2506.16406)
- [TinyLoRA: Learning to Reason in 13 Parameters](https://www.alphaxiv.org/abs/2602.04118) ([PEFT docs](https://huggingface.co/docs/peft/en/package_reference/tinylora))
- [LoRA-XS](https://arxiv.org/html/2405.17604v3), [VB-LoRA](https://arxiv.org/pdf/2405.15179), [VeRA](https://alphaxiv.org/overview/2310.11454v2)
- [Transformer²: Self-adaptive LLMs](https://arxiv.org/pdf/2501.06252) (singular-value fine-tuning, expert vectors)
- [X-LoRA](https://arxiv.org/html/2402.07148v2); [hard-routed mixtures of reasoning LoRAs](https://arxiv.org/pdf/2606.31413) (LoRAHub, MoLE)
- [Self-Adapting Language Models (SEAL)](https://www.emergentmind.com/papers/2506.10943)
- [Learning to Learn-at-Test-Time: Meta-TTL](https://arxiv.org/abs/2604.00830)
- [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/pdf/2407.10930) ([DSPy BetterTogether](https://dspy.ai/api/optimizers/BetterTogether/))
- [TextGrad](https://arxiv.org/abs/2406.07496v1)
- [llama.cpp per-request LoRA](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/0da5d860266c6928b8c9408efbd264ae59fedda6) and its batching limits ([setting adapter scales](https://www.simplified.guide/llama-cpp/server-set-lora-adapter))
- [OP-LoRA](https://arxiv.org/html/2412.10362v2) and work on learned updates in LoRA weight space ([W2T](https://arxiv.org/html/2603.15990v1))
