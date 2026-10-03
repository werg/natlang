# S7: Task-level reinforcement learning

Draft, 2026-10-03. Detailed plan for stage S7 of the [Neuralese programme](README.md), consistent with [S0](S0_SPEC.md), [S1](S1_DATA.md), [S2](S2_SKILL_AUTHORING.md), [S3](S3_PORT.md) and [S4](S4_RUNTIME_SERVERS.md), and with §8 of the [semantics source](sources/semantics-and-training.md). S5 and S6 plans are being written in parallel; where this plan depends on them it names the dependency rather than its detail.

S7 improves complete programs through fresh rollouts in executable environments. It trains the decisions supervision cannot reach well: decomposition, helper and skill selection, skill authoring and revision, error recovery and block stopping, together with the continuous Neuralese parts those decisions depend on. It starts once supervised training (S3, S5, S2, S6) has established basic typed execution and authoring, and it is reviewed, like every stage, against a rubric rather than fixed thresholds.

## 1. Principles

- **Rewards come from checks, not from judgement.** Checked results, resulting state and permitted effects are the primary reward. Model-judged rewards are used only where a family has no mechanical check, at lower weight, recorded as such, and watched by the reward-hacking monitors (§6).
- **Rules are enforced by the host, not by the reward.** Type checks, capability boundaries, the termination rules (S0 §8) and the runtime hard maximum on blocks are hard. A rollout that violates them fails as a call; it is not a penalised success.
- **One gradient mechanism.** Policy learning is `grad` of reward-weighted `logLikelihood` over recorded trajectories (S0 §9.4). Continuous Neuralese parts receive gradients through the same replay (S0 §11.3). There is no separate RL engine with its own semantics.
- **Replay stays in.** Supervised examples (S1, S2, S5) and ordinary-language replay are mixed into every phase, and the pre-RL model remains the reference.

## 2. Environments and task mix

### 2.1 Environments

All environments are built (S1 §5.3) as host services with deterministic fixtures, and S7 uses them unchanged:

| Environment | Families (S1 §5.2) | Check |
| --- | --- | --- |
| SQLite | Text-to-SQL | Result sets compared |
| Function-call recorders | Function calling | Recorded arguments and effects |
| Unit-test runner | Code | Tests pass |
| Offline web pool and search | Offline web research | Answer against reference |
| Repository snapshots | Repository QA, localisation, overviews | Gold files and answers |
| Generator checkers | Reasoning with checkers | The generator's checker |
| Retired natlang checks | SCONE, SGD, FinQA, CLEVR | Original checks |
| World simulators | alfworld, scienceworld, webshop | Simulator outcome and state |
| SWE containers | SWE families | Test execution in the container |

Each environment exposes the same contract to the rollout driver: reset from a fixture, run a program under the runtime with recorded effects, return the checked outcome and the final state.

### 2.2 Task mix

The mix is built from S1's executable adapters and S2's skill episodes, split by source group as in S1. Families are weighted by the review (§7), not by row count: families where the model is near ceiling or near zero get less weight, families in the learnable middle get more. Every task carries its family, difficulty estimate and expected rollout length.

Two task shapes:

- **Execution tasks.** A program and its inputs; reward from the task's check.
- **Authoring tasks.** An S2 episode: the policy revises a context on support cases, and the reward comes from the revised context's performance on separate query cases (§3.2).

### 2.3 Curriculum

Difficulty and rollout length grow with measured performance:

1. Short single-call tasks with mechanical checks (SQL, function calling, reasoning).
2. Multi-call programs with soft intermediates, combinators and helper calls (S5's converted tasks).
3. Long-horizon environments: repositories, worlds, SWE, with `iterateOn` loops and directory reducers.
4. Authoring episodes with query evaluation, including repair of deliberately broken skill libraries (S2 §5.3).

Families advance independently. A family's difficulty rises when the review sees its success rate high and stable; it falls back when rollouts mostly fail.

## 3. Rewards

### 3.1 Execution reward

The reward of a rollout is its checked outcome: success of the check, partial credit only where a family's check defines it (for example, fraction of tests passed or of result rows matched). Final state and effects are compared against the task's expectations where the task declares them. A call that ends `blocked` or `failed` with a correct reason, where the task is infeasible, is a success for that task.

### 3.2 Skill contribution

Authoring and selection are judged by their contribution on separate query tasks, never on the support cases they were made from (S2 §4):

- **Authoring reward:** query success of the revised context minus query success of the starting context, on the same query cases.
- **Selection reward:** query success with the selected skills, compared against the same function with the program's default skills.
- **Reuse:** a skill authored in one episode is applied in later, disjoint episodes; its contribution there is credited back to the authoring rollout when that later evaluation completes.

### 3.3 Resource use

Token, tool-call, block-length and `iterateOn` step costs enter as a small, separately reported term, applied only among rollouts that pass their check. A failing rollout is never improved by being cheap. Cost terms are tuned through the review, and their effect on success is watched explicitly (§6).

### 3.4 What is not rewarded

- No reward for format compliance beyond what the check needs; invalid output fails as a call.
- No reward for matching teacher text.
- No reward from model judges where a mechanical check exists.

## 4. Algorithm

### 4.1 Group-relative policy optimisation

For each task, run a group of `G` rollouts from the current policy. Each rollout's advantage is its reward relative to the group's mean, scaled by the group's spread. The policy loss is the advantage-weighted log-likelihood of the rollout's discrete choices:

```text
loss = − Σ_rollouts A_i · logLikelihood(trajectory_i)  +  β · KL(policy ‖ reference)  +  λ · supervised/replay loss
```

Group-relative advantages need no value model and fit tasks with sparse, checked rewards. Groups with no spread (all succeed or all fail) contribute no policy gradient and are logged for the curriculum.

- **Reference.** The pre-RL model, frozen. The KL term is computed on the policy's own tokens.
- **Clipping.** Importance ratios between the rollout policy and the current policy are clipped, as in PPO-style objectives, to keep steps small when rollouts are slightly stale (§5.4).
- **Credit within a rollout.** The log-likelihood covers every discrete choice in the trajectory: sampled text and code, tool choices, helper and skill selection, `read` readouts, and the stop decisions of every block written. Stop decisions are scored by the stop head's probabilities.

### 4.2 Continuous parts

Neuralese payloads, soft bodies and port modules are continuous. The same replay that computes the log-likelihood of the recorded discrete choices also backpropagates into them:

- **Through the policy term.** A soft block changes the distribution of later discrete choices; the advantage-weighted log-likelihood therefore produces gradients into the block, the writer that produced it and the reader that consumed it.
- **Through differentiable objectives.** Where a rollout has a correct output, its cross-entropy and the self-distillation objective (S0 §9) are added for the continuous parts, as in S5.
- **Sampled payloads as actions.** Rollouts write at Neuralese temperature `τ > 0`, so each payload is a sampled continuous action with log-density `log N(z; μ, τ²σ²)`. The advantage-weighted policy term includes these log-densities, which explores the encoding space and trains writers and distributional stored blocks (spec/NEURALESE_FILES.md) even where no differentiable path reaches the reward.
- **Law consistency.** The S5 law terms stay in the mix at low weight so that compiler rewrites remain valid (S0 §4.3).

### 4.3 What is trained

| Component | Trained in S7 | Notes |
| --- | --- | --- |
| Backbone deltas (on top of the merged crisp base, S3 §13) | Yes | The main policy parameters. |
| Port modules (feedback projection, stop head, content projection, interface norm) | Yes, at lower learning rate | The stop head is trained by the policy term. |
| System operators (`map`, `zip`, `combine`, `read`, …) | Yes, as soft bodies | Through replay, as in S5. |
| Skill and instruction blocks in contexts | Per task, through `grad` | Values in contexts, not model weights (S0 §9). Promotion of an improved block follows S2/S6 rules. |
| Learned updaters (S6) | Yes, in authoring tasks | Their discrete and continuous outputs are part of the rollout. |

## 5. Rollout infrastructure

### 5.1 Serving

Rollouts run on the DGX through the vLLM fork with Neuralese support (S4 §7): prompt-side splicing for reads, the write procedure per sequence, the shared tensor store and block endpoints. The existing DGX setup already serves Qwen3.6 at 256 concurrent requests through the collector; the same orchestration pattern (assignment manifests, importers, durable logs) is reused for rollout campaigns.

The policy server is reloaded with new deltas at the end of each update round (§5.4). The reference model's log-probabilities for KL are computed during replay, not during rollout.

### 5.2 Environment sandboxes

Each rollout runs the native runtime against environment services in its own sandbox: a fresh fixture, its own folder overlays, its own container for SWE tasks. Effects are recorded in the trace and never repeated during replay (S0 §11.3). Wall-clock limits per rollout are host settings, separate from the termination rules.

### 5.3 Trace recording and replay

Every rollout produces a graph record (S0 §11.1) with its model turns, sampled tokens, block writes with stop decisions, reads, readouts, combinator calls, applied rewrites, effects and outcome. The trainer replays those records on the GPU: it restores the definitions and context revisions, supplies recorded effect results, holds discrete choices fixed, and computes log-likelihoods and gradients for the policy and the continuous parts in one pass.

### 5.4 Update rounds, staleness and off-policy correction

Training proceeds in rounds: generate a batch of groups with the current policy, update for a small number of steps, reload. Rollouts generated by an older policy are used with importance ratios (from recorded token log-probabilities at generation time) and clipping; rollouts older than a configured number of rounds are discarded. Block writes recorded under an older policy are replayed through the current writer for gradients, but the old stop decisions are scored with the current stop head and corrected by the same ratio.

If an update changes a discrete choice, an old observation following it is not evidence for the new choice (S0 §11.3). Such cases are left to fresh rollouts in the next round.

## 6. Stability monitors

Tracked continuously, with alerts reviewed by the owner or the running agent:

| Monitor | Signal |
| --- | --- |
| Reward hacking | Success rising while independent held-out checks do not; families where cost terms dominate. |
| Collapse | Representation norms, effective rank and common-direction share of written blocks (S3 §6.4); entropy of discrete choices per family. |
| Stop-head drift | Distribution of block lengths per family, truncation rate, degenerate blocks. |
| Channel use | Correct versus shuffled versus zeroed payloads at matched length on the S3 harness, re-run each round. |
| Law agreement | Agreement for each law; rewrite rules are re-compared when the model changes (S0 §4.3). |
| Ordinary capability | Natlang held-out loss, the protected execution evaluation, general perplexity. |
| KL to reference | Per family, with the β schedule adjusted from it. |

### 6.1 Dialect version

RL changes the backbone and port modules, so the model's space changes. Dialect stability is not a concern yet: after such a round the dialect version is bumped (S0 §12), stored values are regenerated or converted, and the rewrite comparisons are re-run on the new model. There is no conformance-preservation objective.

## 7. Evaluation rubric

S7 is reviewed against a rubric, comparing the post-RL model with the pre-RL model at matched inference budget (the same limits on tokens, model calls, blocks and steps) on held-out tasks split by source group:

| Dimension | Questions the review answers |
| --- | --- |
| Task success | Does complete-program correctness improve on held-out tasks, per family, at matched budget? Where does it regress? |
| Transfer | Do gains hold on held-out families and on longer or harder variants than those trained? |
| Authoring and reuse | Do authored and revised skills contribute more on separate query tasks than before RL? Do they keep helping on later reuse? |
| Neuralese use | Do correct payloads still beat shuffled and zeroed ones? Are blocks used where they help and avoided where they do not? |
| Stopping | Are block lengths sensible after RL, without drift toward truncation or degenerate blocks? |
| Resource use | Tokens, calls, blocks and steps per solved task, against the pre-RL model. |
| Exactness and authority | Exact values, effect arguments and reference resolution preserved; type and capability failures unchanged or fewer. |
| Ordinary capability | No meaningful loss on replayed and protected evaluations. |
| Failure reading | A sample of failed and suspiciously successful rollouts, read and categorised over time. |

The review decides the next round's task weights, curriculum moves, objective weights and whether to continue, adjust or roll back.

## 8. Risks and responses

| Risk | Response |
| --- | --- |
| Sparse rewards on long tasks | Curriculum by length; partial credit where checks define it; supervised seeding from S5 and teacher rollouts. |
| Reward hacking through check weaknesses | Strengthen checks and fixtures; held-out checks not used for reward; failure reading. |
| Policy collapses Neuralese into a side channel the reader ignores | Channel-use monitor each round; increase S5 objectives; roll back. |
| Stop head learns to always stop early or late | Length monitor; scored stop decisions in the policy term; length cost applied only to successes. |
| Stale rollouts dominate | Bounded staleness, importance ratios with clipping. |
| Environment non-determinism | Deterministic fixtures; flaky tasks quarantined. |
| Model changes invalidate stored values | Bump the dialect version and regenerate or convert stored values (§6.1). |
| Ordinary capability erodes | Replay and KL; capability monitors. |

## 9. Dependencies

- **S1:** executable adapters, environments and fixtures, split groups.
- **S2:** skill episodes, authoring operations, query evaluation via host tickets.
- **S3:** port modules, harness, the merged crisp base.
- **S4:** vLLM fork with Neuralese, block endpoints, graph-record traces, replay sessions, environment services.
- **S5:** soft-converted programs, trained operators, the replay trainer and its objectives.
- **S6:** soft skills, the gradient-tuning baseline and learned updaters, used in authoring tasks.

## 10. Work items

1. Define the rollout task record: task, fixture, environment, family, difficulty, expected length, check, query cases for authoring tasks.
2. Wrap each environment service in the common reset/run/check contract (§2.1), with determinism tests.
3. Build the rollout driver on the DGX: group sampling, sandboxes, trace collection, campaign manifests, reusing the existing collector orchestration.
4. Extend the replay trainer (S5) with advantage-weighted log-likelihood, importance ratios, clipping and KL to the frozen reference.
5. Score stop decisions and other non-text discrete choices in the log-likelihood.
6. Add authoring-task rewards with query evaluation and deferred reuse credit (§3.2).
7. Add the resource term applied among successful rollouts only.
8. Build the monitors (§6) and the per-round harness re-run (channel use, law agreement, capability).
9. Build the evaluation rubric runner for pre/post comparison at matched inference budget (§7).
10. Run a pilot on short single-call families, then extend through the curriculum (§2.3).
11. Add authoring tasks and long-horizon environments as the review allows.
12. Bump the dialect version after RL rounds that change the model's space, and regenerate or convert stored values (§6.1).

## 11. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Model judges:** allowed for families without a mechanical check, at lower weight, with reward-hacking monitors.
2. **Dialect:** RL does not hold the dialect; the version is bumped when the model changes (§6.1).
3. **Teacher rollouts:** allowed in groups for hard families early on, importance-weighted, and phased out.
