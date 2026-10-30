# Neuralese hindsight hints for self-distillation

Owner-approved design (2026-10-30), with the owner-confirmed two-phase structure and the owner refinement that the
writer is a natlang data-processing program over τ* as an input value. Helper strand `dgx-claude-hints`;
integration into the live JOLT+ loop belongs to `dgx-rl`.

## Purpose

Teacher text (the LFM3 SFT teacher corpus) is off-policy for the student, and on worlds where every student attempt
fails JOLT+ has no positive episode. A hindsight hint addresses both:

1. A **writer** episode, the policy model running the natlang function `writeHint`
   (`applications/guidance-writer`, `DECOMPOSITION.md` there), gets a known-good trajectory τ* as a proper input
   value. The value is structured (calls, each with its prompt, steps in the runtime's own transcript shape, and
   result) and listed truncated like any large value, with `<<cut off …>>` markers. The writer also gets a crisp
   structural overview and the facts `selectGuidanceFacts` selects. It explores τ* with eval and returns the hint as
   its `Neuralese<string>` result. The configured Neuralese writer re-encodes the returned text as the producer span.
2. A **self-teacher**, the same model running the target task, imports that hint (`mode: import`, the declared
   guidance turn after the leading system turn).
3. The **student** learns from the hint-conditioned self-teacher through the existing JOLT+ OPSD twin.

The self-teacher's runs come from the current policy's own distribution, shifted by a short hint. Verified runs
become student targets in place of the teacher's text, so the student learns from near-on-policy demonstrations.

What is new is a dense **reconstruction objective** for the writer, with a **rate penalty** as the bottleneck.
Everything else (writer function, facts, custody, visibility, rendering, graph edit, twin, outcome terms) reuses
the shared mechanism in `rl/privileged_context.py`, `rl/privileged_scoring.py` and `rl/jolt.py`.

## Natlang-native decomposition

| Part | Form | Where |
| --- | --- | --- |
| What the hint says, and which parts of τ* to read for it | policy: `writeHint.nl`, a multi-turn natlang function that explores τ* with eval | `applications/guidance-writer` |
| Which facts accompany τ* | policy: `selectGuidanceFacts` (existing pluggable, NL default), over the world's facts other than τ* itself | same |
| τ* as writer input | crisp data: `VerifiedTrajectory` (calls → `TrajectoryStep`, the runtime's `TranscriptEntry` shape), built from the τ* source | `rl/hindsight.py` `trajectory_value` |
| Cold-start overview | crisp helper `trajectoryOverview` (calls, roles, tools, statuses, sizes, errors) | `applications/guidance-writer` |
| Custody of τ* | crisp verifier: `common.custody.ProtectedWorlds.check` on the τ* world; only `split == 'train'` | `rl/hindsight.py` |
| Writer/consumer graph | crisp mechanism: `teacher_form` inserts the import of the writer state into τ*'s consumer turns | `rl/privileged_scoring.py` (existing) |
| Import channel (noise + KL) | crisp math | `model/import_channel.py`, hooked into `shared_state._forward_graph` |
| Reconstruction, rate and score-function terms | crisp math | `rl/hindsight.py` |
| Writer parameters | crisp mechanism: free-LoRA writer adapter bound to writer invocations only (`run_invocation_actors`) plus the producer adapter | `model/native_actor_graph.py`, `model/native_adapters.py` (existing) |
| Whether a hint goes to the self-teacher, and how many | policy: `planTeacherAttempts` (existing) | `applications/guidance-writer` |

New model-facing text: `writeHint.nl` and the overview's field names. These are debt in
`plans/MODEL_FACING_CHANGES.md` until they are sampled live; the pilot is their first live sample.

Rationale for the refinement: the writer's context stays bounded, because τ* appears in full only as the
self-teacher's teacher-forced target. The writer episode is itself a real data-processing RL task with a dense reward.

## Graph layout (one packed forward)

For one τ* and a group of K hints (i = 1..K):

```
writer_i:   writeHint episode turns: [listing: consumer, τ* (cut off), overview, facts] eval ... eval ... return_result(hint text)
hint_i:     <produce> hint text tokens </produce>      (the configured writer's re-encoding of the returned text)
consumer_i: [τ* prompt: leading system turn | guidance turn: header <consume> import(hint_i span) </consume> | rest of τ* prompt]
            τ* assistant actions, teacher-forced
```

The writer episode's sampled actions a_i (exploration code and the returned hint tokens) are ordinary captured
model turns. `hint_i` is the producer invocation the runtime's writer records for the `Neuralese<string>` result.
`consumer_i` is `teacher_form(replay(τ*), {turn: PrivilegedBlock(import of hint_i)})`: the same graph edit the JOLT+
student twin uses. For a multi-turn τ* (an RL episode), every consumer turn of τ* imports the same
writer state at its declared place, exactly as an in-run teacher attempt would. A teacher SFT row is one turn. All K
writer/consumer pairs of a group go into one packed graph.

Gradients flow from τ*'s action positions through the consumers' attention to the imported KV and convolution
state of the writer span, then into the writer parameters. That path is exact, with no detach (`ImportSpan.stop_gradient = False`).

## Call trees and nl sub-calls (owner rule, simple)

τ* is a call tree: the root call and the child calls it made (named functions and inline `nl` lambdas), each with its
own model turns. The rule is crisp, on one recorded parameter `child_hint_min_tokens` (own assistant-action tokens of a
call, excluding its children's):

1. **Root:** one writer pass over τ* as a value → hint → self-teacher (the root's guidance turn imports it).
2. **Short child** (fewer than `child_hint_min_tokens` own action tokens): no hint. Any useful hint would be comparable
   in size to the child's own content. It is trained by plain SFT on its own trajectory, with its actual inputs
   including Neuralese inputs. In phase 1 the backbone is frozen, so this SFT belongs to the student's training
   (the production recipe / phase 2), not to the writer.
3. **Long child:** an independent additional writer pass over just that child's subtrajectory as the value → a hint for
   that child call only, imported by that child's own guidance turn.

`rl/hindsight.py` `node_classes` applies the rule; `hinted_graph` takes a hint per hinted node. Reconstruction sums
over the hinted nodes' actions in the existing multi-invocation one-pass graph with exact occurrence bindings. For
phase 2, a τ* child hint applies to a self-teacher child call only when its function and arguments match: the same
function identity, and canonical-JSON-equal arguments with Neuralese arguments compared by state id. Otherwise that
child gets no hint. The default `child_hint_min_tokens = 256` is about four times the target hint length (≤ 64
tokens); the pilot reports the τ* length distribution and the class counts and revises it.

## Objectives (exact)

Notation: π is the policy. a_i are all sampled actions of writer episode i (exploration turns and the returned
hint tokens). L_i is the hint length in tokens, and E_i is the number of sampled exploration tokens. μ_i is the imported state of
writer_i's span as the consumer receives it. z_i is that state after the channel. T is the number of τ* action
tokens.

**Import channel (variational bottleneck).** For every imported slot s of a channelled import edge, and every
layer ℓ and field f (attention `raw_k`, `v`; convolution `gated_inputs`), let c_ℓf be a frozen reference scale (the
RMS of that field on ordinary tokens of the pinned base, measured once and recorded in the declaration):

    z = μ + σ c_ℓf ε,   ε ~ N(0, I)   (reparameterized; ε is fixed per (seed, edge, layer, field, span row))
    KL_i = Σ_{s,ℓ,f,d} [ (μ²/c_ℓf² + σ²) / (2 σ_p²) − ½ + log(σ_p / σ) ]      (KL(N(μ, σ²c²) || N(0, σ_p² c²)))

Keys are perturbed before rotary encoding, which is orthogonal, so the noise stays isotropic. The noise sample for
one imported span row is shared by every consumer slot that reads that row, and that row's KL is counted once.

**Reconstruction.**

    NLL_i = − Σ_{t ∈ τ* actions} log π(τ*_t | task prefix, z_i)      (teacher-forced, through the channel)

**Rate.** Two forms, as chosen by the owner: a per-token length cost λ_len·L_i, and the channel term λ_kl·KL_i.

**Writer reward and baseline.** Per hint:

    R_i = − NLL_i − λ_len L_i − λ_explore E_i − λ_kl KL_i + η · o_i
    A_i = R_i − (1/(K−1)) Σ_{j≠i} R_j          (leave-one-out baseline: the other hints for the same τ*)

o_i ∈ [0, 1] is the oracle outcome of a self-teacher run sampled with hint i, where an oracle exists (phase 2).
It is 0 with η = 0 otherwise.

**Row loss** (normalized per τ* token):

    loss = (1 / (K T)) Σ_i [ NLL_i + λ_kl KL_i − sg(A_i) Σ_{t ∈ a_i} log π(a_i,t | writer context_t) ]

The first two terms are the pathwise (reparameterized) gradient of the expected cost through the continuous
writer state. The third is the score-function gradient of the discrete choices (what to explore, what to write). Together they are
an unbiased estimator of ∇ E_{a∼π}[NLL + rate + exploration cost − η o]. λ_len L and λ_explore E have no pathwise
part and enter only through A_i. The hint state is the re-encoding of the returned text, so its information is
bounded by that text. The channel bounds how much the encoder can add on top of it.

The writer's sampling law is the actor's exact temperature/mask law (`token_action_logps`). Phase-1 writer samples
are drawn from the current weights each step (on-policy), so the plain log-probability is exact. When stale actors
appear (phase 2 inside JOLT+), this term uses the shared `score-centered/1` estimator with its importance correction,
as the existing teacher-outcome term does on writer turns.

Recipe declaration `hindsight-reconstruction/1`: `{k, lambda_len, lambda_explore, lambda_kl, eta, channel: {sigma,
sigma_prior, scales, seed}, trainable: 'writer'}`. Defaults: K = 4, λ_len = 0.05 nat/token, λ_explore = 0.002
nat/token, λ_kl = 1e-3, σ = 0.5, σ_p = 1, η = 0 (phase 1). The pilot measures these.

## Phases (owner-confirmed structure)

**Phase 1: supervised writer training.** The objective is loss_rec + rate, teacher-forced over τ*. **Only writer
parameters update**: the free-LoRA writer adapter, bound to writer invocations only through the invocation-actor
bank, and the producer adapter (`SharedStateModel.adapter`). The consumer/student backbone is frozen, so hints must
work for the current model and the backbone does not learn hint-dependence. Exit criterion: log p(τ*|hint) −
log p(τ*) is large at modest hint length, and self-teacher success rises.

**Phase 2: self-distillation; the writer keeps learning.** The self-teacher samples on-policy with the hint, and
only verified runs are kept. The student trains through the JOLT+ twin (reverse KL to the hint-conditioned
self-teacher) plus its outcome term. The writer continues on loss_rec + the self-teacher outcome reward (η > 0) +
rate. λ_len and λ_kl anneal upward so hints shrink. Hints are regenerated from current weights every round. Phases
may overlap per task.

Phase 2 mechanics in the live loop are the existing JOLT+ path. For a world with a τ*, the teacher attempt's
guidance comes from `writeHint` over τ* instead of `writeGuidance` over facts; `training-guidance.ts` selects it when
the run controls carry a trajectory. Teacher attempts with the hint are the self-teacher. Verified runs
become student targets through the existing OPSD twin. New in phase 2 is a reconstruction row per teacher-attempt
writer (its τ* consumer pair) packed beside the episode, plus η · o in the writer's advantage.

## τ* sources and custody

- **Successful RL episodes** on curriculum train worlds (`quality >= 1` in `jolt-r1`/`sd-r1` rollouts). These are
  on-policy τ* with exact captured turns and an oracle. They are long; the pilot uses short ones.
- **Admitted verified teacher trajectories** (the LFM3 SFT teacher corpus, `runs/lfm3-teacher-graphs/*/lfm3*/train`).
  Each row is one assistant turn with its context. τ* is the target suffix (`target_weights > 0`). There is no oracle
  for these; they are used for reconstruction only.
- **Curriculum reference solutions** (`curriculum.reference.root`) of the RL train worlds, rendered as
  `teacher_trajectory` facts.

Custody: every τ* world passes `ProtectedWorlds.check` (train split, unrelated to any protected world) before any
fact exists. A teacher SFT row is checked through its identity and source groups. The writer only sees facts visible
to the consuming invocation (`applicable`). Protected and held data are never τ*, facts or consumers. The pilot
loads `ProtectedWorlds.load()` and fails closed.

## Shared writer stack

The per-invocation hint writer is the same writer as program-level memetic guidance and the learned-optimizer
writer (`JOINT_LEARNED_OPTIMIZER.md` "Native states the writer can reach", "Writer identity"):

- the same `writeGuidance` function and consumer import (`PrivilegedRendering`);
- the same writer adapter kind (free-LoRA through `create_adapter`, trained as `leaves` of `run_native` /
  `run_invocation_actors`), so one writer adapter can serve memetic, metalearner and hindsight users, with recorded
  identity;
- reconstruction is the per-invocation form of the plan's behavior-first loss: the executor's own log-likelihood of
  verified actions under the written state. Its distillation KL against a target bundle is the same quantity with a
  soft target.

The import channel is a property of the import edge, not of the writer, so any consumer of written state
(memetic or metalearner) can use the same bottleneck.

## Reuse map

| Reused | From |
| --- | --- |
| `case_facts` (incl. `teacher_trajectory`), `applicable`, `PrivilegedRendering`, `guidance_plan`, `PrivilegedBlock` | `rl/privileged_context.py` |
| `teacher_form`, `writer_span` | `rl/privileged_scoring.py` |
| `compile_replay`, `token_action_logps` | `rl/native_replay.py` |
| `ProtectedWorlds` | `common/custody.py` |
| `create_adapter` (free-lora), `run_invocation_actors` | `model/native_adapters.py`, `model/native_actor_graph.py` |
| `Engine` (exact writer prompt rendering, template write, capture) | `serve/engine.py` |
| `writeGuidance`, `selectGuidanceFacts` | `applications/guidance-writer` |

New: `model/import_channel.py` (channel and KL in the packed forward), `rl/hindsight.py` (declaration, τ* facts,
reconstruction/rate/score-function loss), the pilot driver `scripts/neuralese_rl/hindsight_pilot.py`, and tests
(float64 gradient check on the tiny shared-state model, plus a channel-vs-explicit-cache equivalence test).

## Pilot (small, DGX, medium/low priority)

The model is the current LFM3 RL lineage parent, loaded read-only. Writer sampling uses the in-process native
`Engine` at the actor's temperature. Inputs are a few hundred τ* rows from train worlds: teacher SFT turns plus the
short successful RL episodes, with oracles where they exist.

Measurements (eval compute kept minimal):

1. log p(τ*) per token under no hint, a facts-only text hint (selected non-τ* facts, `mode: text`), the
   cold-start `writeHint` hint (eval exploration, its state imported with `mode: import`), and τ* verbatim as text (a leakage reference). Self-teacher success
   with vs without a hint on the oracle subset comes from existing JOLT+ teacher-attempt records (text facts) and,
   if phase 1 works, a small native run.
   Some τ* have a call tree of depth ≥ 2; Δ log p is reported per depth.
2. log p(τ*) against hint length under the rate penalty (K samples per τ*, binned by L, with and without channel noise).
3. A short phase-1 writer-training run: does loss_rec fall, and does the hint stay short? Then one phase-2 step if
   phase 1 works.

Go/no-go: go if the hinted gain Δ = log p(τ*|hint) − log p(τ*) is clearly positive after phase 1, at a mean hint
length of 64 tokens or fewer, with the channel on, and if it beats the facts-only text hint.

## Results

(filled in by the pilot)
