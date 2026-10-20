# Adaptive Neuralese reinforcement learning

Owner objective: improve both models throughout the production recipe, exploiting
new data and fixes immediately. RL is a stage of that process, not a short probe
that counts as completion. One shared harness, reward, replay and learner serves
LFM and Mellum; only backbone loading, chat templates and resource sizes differ.

## Natural-language functions

`chooseTrainingWork(taskOutcomes, pendingWork, modelProgress) -> TrainingWork[]`:
1. Read named train tasks and their exact verifier outcomes.
2. Continue mixed-success groups with policy-gradient updates. For groups with no
   successful samples, request more samples or successful teacher trajectories;
   train successes through the existing RFT stage. Keep easy tasks available for
   regression checks while allocating useful work to harder tasks.
3. Incorporate newly admitted train tasks promptly; never collect held tasks for
   training. Decide whether a concrete defect needs a restart or continuation.
4. Keep rollout queues bounded by actual learner consumption and free memory.
   Adjust worker placement from observed rollout and update throughput.
5. Preserve useful state when changing recipes. Base KL is telemetry only.

`acceptRollout(batch, policyIdentity) -> AcceptedRollout[]`:
1. Require training split, finite checked rewards, and actual policy turns.
2. Exclude infrastructure failures from both reward and group normalization.
3. Keep the sampled reasoning, raw tool-call arguments and IDs. Do not replace
   the sampled action with the harness's interpreted/normalized action.
4. Bind behavior scores to policy version and sampling settings. A later policy
   score must never masquerade as the behavior probability.
5. Reject unknown or unreplayable behavior; recollect or send independently
   verified successes to RFT rather than inventing missing probabilities.

## Crisp mechanisms and reuse

Owner clarification (2026-10-16): Neuralese use is optional for reward and policy
updates. A one-node ordinary execution can be a complete episode. Where the
execution contains a communication graph, retain its entire captured model-turn
inventory, sampled/forced writes and exact consumer occurrences. Reject a
missing internal producer rather than loading its archived state as a detached
constant. Explicit inputs originating outside this episode remain inputs.
This is exact graph-coverage verification, separate from task selection policy.
`episode_graph` authenticates the complete model-turn and captured state-event
inventories. Explicit outside origins stay outside even when an equal content
ID also has a local writer. Missing native graph events or unsupported decoded
state lowering are rejected before policy admission; archive bytes cannot
substitute for an omitted producer. Admitted objective rows retain the full
turn-name inventory and verify it against the learner graph.

- Native replacement: `native_replay` verifies actual sampled and forced input
  coordinates, resolves exact writer occurrences, and compiles the complete
  episode into one flattened invocation graph. State is the deterministic
  output of sampled producer tokens, with no separate stochastic latent action.
  Consumer gradients flow through the aliased producer states in that graph.
- The native objective uses the shared native trainer's supplied forward output,
  shared clipped-token arithmetic and shared optimizer/checkpoint lifecycle.
  It never starts a second backbone forward, recursively replays writers, or
  instantiates the old projection/GradSession path. Sampling capture /2 records
  actual forced template inputs separately from policy actions, including
  forced-only producers. Only actual sampled tokens receive behavior logps.
- Native actor attestation binds checkpoint, expanded tokenizer, embedding path
  and architecture. Native and legacy interfaces are explicit alternatives;
  availability of archived native bytes alone does not qualify an RL learner.
- The remaining mechanisms below describe the legacy diagnostic path while
  native stage/recipe/collector integration is completed and independently
  reviewed; they are not a fallback implementation of native state.
- `task_rollouts`: exact train-split enforcement, collector queues and the shared
  `classify_case` reward. These are host boundaries and exact verifiers.
- `grpo`: group statistics, clipped importance weights and existing HTTP gradient
  replay/optimizer. Arithmetic and transport are crisp; no duplicate reward.
- `stage`: existing whole-trajectory RFT uses the shared direct trainer, optimizer,
  continuation checkpoints, task evaluation and data cache.
- RL production coordinator: explicit collect/score/update/persist transitions,
  bounded parallel subprocesses and atomic state are durability/plumbing. The
  work-selection policy above remains explicit, separately configurable.
- Full-policy objective: differentiable replay reuses `GradSession` and shared
  training infrastructure rather than a per-backbone trainer or copied harness.

No new prompt or runtime error is introduced by replay fixes. Research and
sampling interventions are not production qualification or measured model lift.

## Exact sampling and value graph replay

- Optional server `x_natlang_rl_capture` records the already sampled prompt plan,
  emitted items, stopping token and token probabilities. It changes no sampling
  choice. Collector metadata retains it without adding model-facing text.
- `sampling` verifies the captured token coordinates and scores those tokens,
  including the actual stopping token. Guided rollback requires its full mask
  law; the initial production contract uses unconstrained temperature-1 samples
  and rejects guided/masked captures rather than guessing their probabilities.
- `graph_replay` persists exact safetensors block bytes and transitive producer
  metadata. Content IDs, SHA-256 and policy identity are exact host verifiers.
  Writer replay uses recorded values in the forward and a differentiable writer
  in the backward. Deterministic payloads have no invented sampling density.
- Full-policy replay opts into model-weight dependencies in `GradSession`;
  argument-only replay keeps its existing semantics by default.

Raw replay import compatibility: the existing four-field import denotes a live
producer-to-consumer gradient edge. Exact replay plumbing omits the optional
`stop_gradient` keyword for that default, allowing qualified older native ports
to execute the same graph. A source-declared true stop remains explicit and is
never removed for compatibility; an unsupported port rejects it.

## In-process owned trainer launch

A supervisor may load and qualify once, then call the existing trainer in the
same process. Record that actual live owner, the byte-bound wrapper, effective
trainer command, import environment and exact continuation before training.
The shared handoff verifier authenticates this explicit dispatch separately
from a supervisor with a child process; it never invents a trainer child.
Capture the normal live drain intent, then complete the existing handoff only
after that owner exits. This is exact process/transport plumbing, not training
readiness, completed exposure, or permission to invent historical owners.

Recovery can explicitly retain the entire mandatory supervised schedule when a
new external owner publishes genuine updates. Acknowledge only the stages
actually represented by its verified operation chain, restart schedule traversal
at its first stage, and let the existing per-row debt checker skip earned visits.
The runtime handoff accepts this exact unsealed boundary without interpreting
it as full-stage or whole-recipe completion.


## Native actor implementation deployment

Policy keeps the existing task curriculum, correct-execution reward and whole
episode update. Deployment is exact crisp plumbing: inventory the Python package
with the existing resident code-inventory mechanism; bind its digest in each new
native actor declaration and attest the actual actor package at load. Require
replica agreement and the declared digest before collection; retain the receipt
in the existing immutable collection binding. Legacy all-unattested packets stay
explicit legacy evidence. Never rewrite a training runtime or replace the global
provider environment just to advance an actor implementation. New actor snapshots
use explicit command/package paths and unchanged checkpoint/adapter identities.
No prompt, token, sampling distribution, imported state or replay graph changes.

An explicit actor runtime also owns the known native actor script and its import
environment inside an existing provider command wrapper. Rebind only those
fields on a copied argv; preserve learner commands and the original declaration.
Reject an unknown custom launcher instead of pretending its imports changed.
Role/coordinator adoption follows the existing verified later-runtime handoff at
a natural safe boundary; an old owner's source plan is never relabeled.

## Transported communication exposure

The recipe chooses a fixed admitted foundation and preserves later mandatory
cohorts. Source-host readiness already reconstructs every declared row from the
actual source. The peer must also verify the existing foundation receipt's
per-row binding/admission/quota/counter law and recompute its cohort hash and
rollups, without opening remote source paths. Share that exact receipt-summary
mechanism with its producer; aggregate visits or a `complete` flag cannot replace
missing row exposure. This is crisp evidence verification, not a new readiness
policy, another trainer or permission to relabel an old role authority.

## Join a runtime handoff under the existing direct owner's lease

The adaptive recipe decides when to refresh and which admitted sources to use.
A live direct owner can publish its genuinely loaded drained parent with the
existing no-credit import, then join normal runtime/recipe authority before SGD.
This is exact control plumbing: authenticate the actual inherited canonical
lock descriptor by device/inode, exclusive FLOCK and a fresh conflicting-open
probe; retain that lease continuously. Acquire every other existing production
and operation lock normally. Reuse one descriptor verifier across owner import,
retirement, deferred bridges and runtime joins. Preserve all cursor/case/task
and stage debt through the normal pending continuation law. A passed metadata
join grants no model qualification, extra exposure or completed recipe stage.
