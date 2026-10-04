# S6: Soft skills and meta-learning

Draft, 2026-10-03. Detailed plan for stage S6 of the [Neuralese programme](README.md). It builds on the learning surface of [S0](S0_SPEC.md) §9, the context model of S0 §7, the skills and episodes of [S2](S2_SKILL_AUTHORING.md), the trained port of [S3](S3_PORT.md), the runtime and gradient sessions of [S4](S4_RUNTIME_SERVERS.md) and the program-level training of S5. The design inputs are in [sources/semantics-and-training.md](sources/semantics-and-training.md) §7.

## 1. Goal

Programs improve themselves through Neuralese values in their contexts, and the procedure that improves them is itself learned.

Concretely, S6 delivers:

- **Soft skills.** Skills whose bodies, knowledge and scope data are Neuralese values in `.nz` files, initialised from S2's crisp skills and trained through downstream use.
- **Direct tuning.** Gradient tuning of skill, instruction and operator blocks through context items, as the baseline every other method is compared with.
- **Learned updaters.** Step functions, written in natlang, that read a program's current blocks and evidence and write the next blocks, trained on separate query performance.
- **Soft authoring.** The model writing and revising `.nz` skill content directly through the write port.
- **Self-revising updaters.** Updaters that revise their own instructions within bounded episodes.

Everything in S6 is a function from a context and evidence to a new context. The backbone is not changed; small adapters bound in contexts are allowed since decision 37 (S0 §9.7).

**Revision note (2026-10-04).** [LEARNING_CONTINUUM.md](LEARNING_CONTINUUM.md) extends this plan with conditioned distillation as a third regime, residual (delta) updates, adapters, the improvement record, a memetic optimiser joining search and gradients, faceted meta-skills and the reward-blind improver. Where the two differ, the continuum plan is newer.

## 2. Starting point

| From | What S6 uses |
| --- | --- |
| S0 §7 | Contexts as curried arguments; rebinding; executable nodes only from files; data entries open and editable. |
| S0 §8 | Termination rules: TypeScript predicates need `withLimit` or `withMeasure`; natural-language predicates run under the predicate prompt with the progress judge on. |
| S0 §9 | `grad`, `valueAndGrad`, `stopGradient`, nested `grad` (first-order default), `logLikelihood`, objectives, optimisers, `save`. |
| S2 | Standard `SKILL.md` skills with the `natlang:` block; `.nz` skills keeping the same fields in safetensors metadata; scope injection; per-program skills plus a global pool; `natlang.skill-episode/1` support/query/transfer episodes; the host-ticket evaluation protocol; crisp authoring trajectories. |
| S3 | `nd:natlang@1` and a writer that produces blocks through the single write procedure. |
| S4 | Gradient sessions on the Python server (replay with discrete choices fixed, gradients and optimiser states as store entries); `.nz` loading; context rebinding. |
| S5 | Trained core operators as functions with soft bodies; the replay trainer; crisp/soft agreement on complete programs. |

## 3. Soft skills

### 3.1 Form

A soft skill is a standard skill whose body or data is Neuralese (S2 §2.1):

- **A `.nz` skill file.** The safetensors metadata holds `name`, `description` and the `natlang:` block; the body is a `Neuralese<SkillBody>` export.
- **A mixed folder.** `SKILL.md` stays as the readable description and body; `.nz` files beside it hold soft forms of the procedure, knowledge or examples.
- **Purely soft.** A skill may drop its markdown body once tuning has moved it away from the text; its crisp ancestor stays in provenance as the fallback.
- **Scope-injected values.** The `natlang:` `scope` block names Neuralese exports to inject into the eval scope of bound functions, so a function receives soft knowledge without loading anything.

Helpers stay crisp executable nodes. A skill's soft parts are data entries, so they can be added, replaced and tuned freely (S0 §7.3).

### 3.2 Initialisation from crisp skills

Every soft skill begins from a crisp one, so that the starting point is already useful and comparisons are paired:

1. **Token-embedding initialisation.** The skill's markdown body (or a selected section) becomes a block whose vectors are the input embeddings of its tokens, in the model's space, tagged with the current dialect version. This is a runtime registration, not a write, so it does not depend on the operator being trained.
2. **Written initialisation.** The writer reads the crisp skill and writes a block through the single write procedure, from a function whose instructions say what the skill is for. Its length is learned.
3. **Scope data.** Typed knowledge (`knowledge.json`-style data referenced by `scope`) gets a soft view written by a function whose instructions name its consumer; the exact data stays alongside it.

Both initialisations are evaluated against the crisp skill on the same episodes before any tuning (§7). The crisp body is kept as provenance and as the fallback a reader can load.

### 3.3 Training through downstream use

A soft skill is trained on the tasks that use it, not on reproducing its text:

- The loss is computed on query-like held-in cases from the skill's family: `crossEntropy` of checked results and actions, `selfDistill` against the same program given the crisp skill text, and `logLikelihood` of accepted trajectories.
- Gradients reach the skill through its context item: `valueAndGrad(s => objective(program.in(ctx.with({ skill: s })), cases), ctx.skill)` (S0 §9.1).
- A skill used by several programs accumulates gradient from all of them in one replay.
- Representation monitors from S3 (norms, effective rank, collapse, correct-versus-shuffled substitution) run on every tuned skill.

## 4. Direct tuning: the gradient baseline

Gradient tuning is the reference method. Every learned method must be compared with it on the same episodes.

### 4.1 What can be tuned

| Target | Where it lives | Notes |
| --- | --- | --- |
| Skill bodies and scope values | `.nz` skill files in the program's context | The main target. |
| Instruction blocks | Soft bodies of a program's own functions, as data entries of the context | The function's signature, captures and interface do not change. |
| Operator blocks | The standard library's `.nz` file (`map`, `zip`, `combine`, `read`, `compose`, …) bound into the context | Tuned per program or per family; library-wide changes are published to the pool. |
| Data blocks | `.nz` data exports | Any `Neuralese<T>` a program reads. |

Several targets are tuned jointly by passing a record of context items as the argument to `valueAndGrad`.

### 4.2 Procedure

```ts
const opt = optimizers.adam({ lr });
const loss = (items: Items) => objectives.on(program.in(ctx.with(items)), support);
const step = async (s: { items: Items; opt: OptState }) => {
  const { grad: g } = await valueAndGrad(loss, s.items);
  return opt.step(s, g);
};
const tuned = await iterateOn(step, { items: selected(ctx), opt: opt.init(selected(ctx)) })
  .withLimit({ maxSteps })
  .until(s => converged(s));
```

- Support cases only. Query cases are evaluated by the host afterwards, through an evaluation ticket, exactly as in S2 §4.2.
- The loop uses a TypeScript predicate, so it carries a hard bound (S0 §8).
- Every step's values are content-addressed store entries; the trajectory of revisions is kept for analysis and for updater training data.
- Promotion of the result is rebinding: the program's context gets the tuned `.nz` entries, saved with `save`, and an improved shared skill can be published to the global pool.

### 4.3 Variants studied

Reparameterised gradients versus sampling-based estimators over distributional blocks (score-function or evolution-strategy style updates through `logLikelihood` at Neuralese temperature `τ > 0`), which do not need a differentiable path through discrete choices between the block and the outcome; learning rate and optimiser; number of support cases; joint versus single-item tuning; tuning from token-embedding versus written initialisation; with and without `selfDistill` against the crisp text; the effect of block length (resizing is a new value written by the writer, not a gradient operation).

## 5. Learned updaters

### 5.1 Shape

A learned updater is a step function with the same contract as the gradient step (S0 §9.6). It is a natural-language function, possibly with a soft body, bound to its own context:

```ts
type UpdaterState = { items: Items; notes: Neuralese<UpdateNotes> };
updater: (state: UpdaterState, evidence: Evidence) => Promise<UpdaterState>;
```

Because it has the gradient step's shape, it runs in the same `iterateOn` loop, on the same episodes, and can be swapped with gradient descent or combined with it (a learned step that receives a gradient digest and decides how to use it).

### 5.2 Inputs

| Input | Content | Gradient treatment |
| --- | --- | --- |
| Current blocks | The context items being revised, as literals | Differentiated through when training the updater end to end. |
| Trajectory digests | Support runs: attempted actions, observations, failed hypotheses, unresolved questions, outcomes | Soft digests are written by a digest function; recorded discrete choices fixed. |
| Feedback | Check results, failures, test outcomes, teacher comments where available | Exact data. |
| Gradient digest | Per item: identity, role, shape, direction, scale, optimiser state, and how the direction changed across steps | Observed feature by default: wrapped in `stopGradient`. Studied both ways (§5.5). |
| Updater notes | The updater's own carried state across steps | Part of the iteration state. |

A raw gradient magnitude is not a diagnosis; the digest gives structure (which item, which role, how consistent the direction is) and leaves interpretation to the updater.

### 5.3 Warm start from crisp authoring

S2's authoring episodes are trajectories of a crisp step function from context to context. They warm-start the learned updater:

- **Imitation.** `logLikelihood` of accepted crisp authoring trajectories, with the updater writing soft revisions where the crisp author wrote text revisions of the same skill.
- **Paired soft targets.** For each accepted crisp revision, the gradient-tuned soft form of the revised skill is a target the updater can learn to write directly.
- **Failures.** Rejected or regressing revisions from S2 are kept as labelled negatives; their reward-weighted log-likelihood is negative.

### 5.4 Training on separate query performance

The outer objective is query performance after the iteration, never support performance:

1. Run the updater in `iterateOn` on an episode's support cases for a bounded number of steps.
2. Evaluate the final context on the episode's query cases.
3. Differentiate the query loss with respect to the updater's own soft body and its operator blocks, through the continuous parts of the steps.
4. Train the updater's discrete choices (which diagnostic to run, which example to inspect, which item to revise, when to stop) with `logLikelihood` weighted by query gain.

Support and query cases are disjoint by source group and follow S2 §4.3. An updater that only fits support cases gains nothing in step 2.

### 5.5 First-order and second-order

- **Default: first-order.** Inner gradients and gradient digests are constants. The outer gradient flows through the updater's writes into the final items and on to the query loss.
- **Second-order on request.** `grad(…, { order: 2 })` differentiates through inner gradient computations, for short episodes where the cost is acceptable. S6 compares the two on matched episodes to decide where second-order pays.
- **Digest treatment.** The same comparison decides whether gradient digests stay behind `stopGradient`.

## 6. Soft authoring and revision

The learned updater revises existing items. Soft authoring also covers creating and retiring skills:

- **Create.** The model writes a new `.nz` skill: it writes the metadata (`name`, `description`, `natlang:` block) as exact data and the body as a Neuralese literal through the write port, from support evidence. The file is a data entry and can be added at will.
- **Revise.** Write a new body or scope value for an existing soft skill, from the old one and the evidence (`map` with revision instructions, or a direct write).
- **Select.** Rebind a function to a subset of available skills. Selection is a discrete choice, trained with `logLikelihood` weighted by query gain.
- **Repair.** Missing, irrelevant and incorrect soft skills, using S2's constructed-defect episodes with soft libraries.
- **Retire.** Remove a soft skill from the context.
- **Helpers.** A soft author cannot add executable nodes. When a revision needs a new helper, it falls back to crisp authoring: a directory reducer writes the helper into a staged tree, which is compiled and loaded (S0 §7.3).

Soft authoring is judged like crisp authoring: by query and transfer gain, not by reproduction of any particular skill.

## 7. Self-revising updaters

An updater's own instructions are a soft body in its context, so the same machinery can revise it:

- **Outer episodes.** A meta-episode runs the updater on several inner episodes, measures their query gains, and produces a revised updater body. The revision is either a gradient step (`valueAndGrad` of mean query loss with respect to the updater body) or an updater applied to itself as data.
- **No self-reference by construction.** Revising the updater is iteration over updater values: step `k` produces updater `k+1`, which is a new value. An updater is never called from inside its own body. Function-typed captures are by value (S0 §6), so an updater cannot reach a later revision of itself.
- **Bounds.** Inner episodes are bounded `iterateOn` loops. The outer loop uses a hard bound when its stopping predicate is TypeScript, or a natural-language predicate under the predicate prompt with the progress judge on (S0 §8, §13).
- **Separation.** The meta-episode's own evaluation uses inner episodes that none of its revisions trained on.
- **Authority.** An updater revision cannot widen capabilities, add executable nodes, or obtain services its binding did not have.

## 8. Comparison protocol

Crisp authoring (S2), gradient tuning (§4), learned updaters (§5), soft authoring (§6) and combinations are compared on the same episodes:

- **Same episodes, same evaluation.** Identical support, query and transfer cases; host-ticket query evaluation with identical seeds; fresh workers.
- **Matched compute.** Each method gets the same allowance of model calls, replay compute and wall time on support cases. Compute is measured from traces, not estimated.
- **Paired results.** Per episode: starting context, each method's final context, query and transfer gain, skill test results, leave-one-item-out attribution.
- **Reuse.** Revised skills are carried into later episodes (subject to the leakage rule in S2 §4.3) to measure lasting value.
- **Held-out families.** A share of families is never used to train updaters, to measure transfer of the improvement ability itself.

## 9. Checkpoint G3

G3 is a review point. Evaluation runs continuously, and the review decides what to improve next or when to hand the methods to S7.

| Dimension | Questions the review answers |
| --- | --- |
| Soft versus crisp skills | Do soft skills initialised from crisp ones match them, and does tuning improve on them? In which families? |
| Gradient baseline | How much does gradient tuning gain on query cases, and how does that vary with support size and initialisation? |
| Learned updaters | Do learned updates beat gradient tuning and crisp authoring at matched compute on held-out families? Where do they lose? |
| Transfer and reuse | Do improvements hold on transfer cases and in later episodes? |
| Channel use | Do tuned and authored skills beat shuffled and zeroed substitutes of matched length? |
| Stability | Representation monitors on tuned blocks; regressions on unaffected cases; drift of operator blocks. |
| Self-revision | Do revised updaters improve on later inner episodes they did not train on? |
| Failure reading | A sample of failed episodes, read and categorised, tracked over time. |

## 10. Handoffs

- **S7 (RL):** selection, creation, repair and stopping decisions of soft authoring become policy-learning tasks with query gain as reward; learned updaters run inside complete-program rollouts; gradient tuning remains the per-program baseline.
- **S8 (target model):** the trained library of soft operators and skills is published in `.nz` files tagged with its dialect version, and re-tuned or regenerated for the target model.
- **S2:** soft authoring failures that need new helpers feed back into crisp authoring episodes.

## 11. Risks

| Risk | Response |
| --- | --- |
| Tuned blocks overfit support cases | Query-only outer objective; disjoint source groups; transfer and reuse evaluation. |
| Tuned blocks collapse or stop carrying content | Representation monitors; correct-versus-shuffled checks on tuned skills; `selfDistill` against crisp text as a regulariser. |
| The reader ignores tuned skills | Shuffled and zeroed substitution on every evaluation. |
| Learned updater imitates crisp authoring without improving on it | Compare against the gradient baseline at matched compute; weight imitation down as query-trained objectives take over. |
| Second-order cost | First-order default; second-order only where the comparison shows benefit. |
| Updater gaming of the progress judge or predicates | Outer evaluation on query cases the updater never sees; hard bounds on TypeScript-predicate loops. |
| Shared operator changes harm other programs | Per-program or per-family tuning first; publishing to the pool only after evaluation across families. |

## 12. Work items

1. Soft-skill format support on top of S2: `.nz` skill metadata, mixed folders, scope injection of Neuralese values (with S4).
2. Token-embedding and written initialisation of soft skills from S2's crisp library; paired evaluation against the crisp skills.
3. Objective library wiring for skill training: `crossEntropy`, `selfDistill` against crisp text, `logLikelihood` of accepted trajectories.
4. Gradient-tuning baseline as a library step function, with optimiser variants and joint tuning of several items.
5. Episode runner for S6: S2's protocol with soft revisions, matched-compute accounting from traces, paired reporting.
6. Gradient digests: structured per-item summaries in the store, exposed to updaters.
7. Learned updater v0: warm start from S2 authoring trajectories and gradient-tuned targets.
8. Query-trained updater: outer objective on query cases, first-order through steps, discrete choices by weighted log-likelihood.
9. Soft authoring operations: create, revise, select, repair, retire; fallback to crisp authoring for helpers.
10. Second-order and digest-treatment comparisons.
11. Self-revising updaters: meta-episodes with bounded inner and outer loops.
12. Continuous G3 evaluation and review reports.

## 13. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Operator tuning scope:** tuned operator blocks are allowed per program, as values in its context, and promoted to the pool when they help broadly.
2. **Soft skills:** a skill may become purely soft; its crisp ancestor is kept in provenance.
3. **Teachers:** paid teachers write crisp revisions for S6 episodes as extra warm-start targets.
