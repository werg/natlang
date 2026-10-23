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

## Native whole-episode execution

The shared native path serves both LFM and Mellum. Task reward and policy
updates do not require Neuralese: an ordinary one-node turn remains eligible.
When an episode contains Neuralese, capture and replay retain every model turn,
sampled or forced producer, consumer occurrence, and downstream outcome. A
missing internal producer is rejected; an external state input stays explicit.

`episode_graph` checks whole-episode event coverage. `native_occurrences` binds
each captured state use to its exact writer or external origin. `native_replay`
verifies sampled token coordinates and compiles the episode into one shared
positional graph. Producer state and consumer imports execute in that graph, so
losses on later turns backpropagate through the actual producer state. Declared
stop-gradient imports remain explicit.

`shared_state` executes attention KV, finite convolution history, and LFM
gated-delta inputs at their logical consumer positions. Native state storage
retains named tensors and exact model/tokenizer provenance. In-episode state is
not replaced with a detached archive copy. External state gradients use named
state leaves when requested; in-episode producer and consumer dependencies
remain in the same autograd graph.

The native policy objective uses the shared trainer's forward output,
clipped-token arithmetic, optimizer, and checkpoint lifecycle. Actor bindings
are attached to actual graph invocations and their receipts preserve exact
checkpoint and adapter identities. Sampling capture records forced template
inputs separately from sampled actions; only sampled tokens receive behavior
log probabilities. Reject captures whose sampling law cannot be reconstructed.

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

## Keep pending actors paired with their runtime

Adaptive continuation preserves sealed case/task debt. During pending work,
retain the actor command, environment and Python runtime binding together; do
not apply the future actor override and then attempt to restore the old command.
At a genuine free boundary, derive all actor runtime metadata through the same
factory used by initial configuration, then use the normal provider verifier.
The complete future declaration remains durable in both cases. These are crisp
binding operations; they neither bypass command validation nor complete stages.
