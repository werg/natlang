# Main goal: keep moving forward (owner, 2026-10-11)

## Recovery and limited compute credit (owner, 2026-10-24)

Remaining CoreWeave access is opaque: $500 is an upper bound and actual remaining
credit may be lower. Do not treat it as a confirmed balance. Prepare a lean
placement with local LFM work, one cloud Mellum learner and burst generation
when recovered/admitted supply warrants it; use additional cloud GPUs only when
their measured useful throughput justifies the spend. Resolve actual account
rates, track cumulative estimated spend and bound allocations before launch.
Preparation does not itself authorize restarting the paused training services.

Filesystem snapshots are supplemental recovery. A quota cutoff can prevent
restoring or reading them without another sandbox. The primary durable backup
must remain independently downloadable from the workstation with zero running
sandboxes: private full-state checkpoint exports (optimizer/RNG/exposure), exact
recipe/control/runtime dependencies, admitted sources and generation journals.
Retain a verified local recovery copy too. A READY snapshot, model-only export,
upload acknowledgement or old readback alone does not cover these obligations.
Monitor earned progress versus independently exported progress and actual object
presence; preserve the last verified parent until its useful replacement is
durable. Use snapshot/restore checks where available, without replacing the
quota-independent export path or starting an otherwise idle recovery sandbox.

## Consolidated supervision (owner, 2026-10-20)

Keep supervision, implementation and operations in one primary-agent strand.
Close existing sub-agent strands after collecting their handoffs; preserve useful
training, generation and conversion processes. Use sub-agents sparingly and only
with the Luna model when delegation is needed. Luna is for engineering work,
never training-data generation. New training data must continue to come from our
self-hosted teacher in the CoreWeave generation sandbox. These instructions
supersede earlier requests for parallel non-Luna implementation agents.

## Core memetic learning and legacy removal (owner, 2026-10-23)

Memetic learning and soft skills are a core training and user capability. Learn
stored native Neuralese guidance from multiple specific invocations of the same
program, measure actual task outcomes on distinct development and confirmation
invocations, and keep iterating the guidance and training recipe for both models.
Keep related worlds on one side of the split. Do not claim capability lift from
stored artifacts, admission, source checks or gradient tests alone.

Use the current positional KV and hybrid-state architecture in training and
inference. Stored guidance must condition the real executing model. Training
exports retain the actual producer and every consumer import at their recorded
positions, with gradients through those dependencies. Retain text and tokenizer
provenance for re-encoding when the executing checkpoint changes; reuse compatible
state efficiently across repeated and concurrent calls. Generation and guidance
authoring remain owned by the self-hosted teacher, with exact actor provenance.

The system has no deployed-user backward-compatibility requirement. Remove legacy
memetic adapters and flat soft-block fallbacks, migrate their current callers and
supported functionality to the native path, and remove obsolete public entrypoints,
docs, tests and discarded-run registry entries. A legacy adapter hiding missing
functionality is work to implement, not a reason to retain the fallback. Preserve
usable source data and protected evaluation data. Historical hashes and manifests
alone do not justify retaining a current legacy entrypoint; actual frozen recovery
and historical-reader dependencies retain their own exact sources as needed.

Apply this default across the system: do not keep obsolete adapters or compatibility
shims merely because an older API, test or recipe mentions them. Inventory current
callers, implement their useful behavior in the native architecture, migrate those
callers, and remove the obsolete entrypoints. Delegate concrete architectural gaps
for implementation and independently review the integrated result. An explicit
unsupported error is useful during migration but is not completion of missing
functionality. Keep actual live recovery dependencies until their replacement is
verified.

The owner specifically requested a Sol subagent for this implementation and
allowed it to delegate missing functionality. That is a task-specific exception
to Luna-only engineering delegation; other helpers remain Luna, never generation.
The primary supervisor reviews, integrates and deploys the complete mechanisms.

## Execution and generation ownership (owner, 2026-10-15)

Pop is closed. The DGX supervisor now owns all remaining work and sandbox
operations; do not defer work to a Pop agent. Only Pop's generation was to
finish: DGX-supervised data generation must continue. Generate new training
data only with our own models through the CoreWeave sandboxes, not Luna or
other external teachers. Existing admitted third-party data remains usable.
Owner clarification (2026-10-15): "our own models" here means self-hosted
teachers in our sandboxes, not the student models we are trying to improve.
Use a substantially stronger teacher; Mellum and LFM are students. The selected
generation teacher is Qwen3.6-35B-A3B (owner selection, 2026-10-22) on the existing dedicated
generation sandbox, using the GPU-specific optimized serving engine. Preserve
verified full trajectories for both students and expand beyond a narrow task
pilot. Do not allocate an additional sandbox merely to change the teacher.
Coordinate sandbox inference with training so generation uses available
capacity without unnecessarily starving training. Preserve exact actor,
checkpoint, runtime and split provenance; base-model generations are not
on-policy samples from a trained checkpoint. This supersedes older machine
ownership and provider-generation restart instructions below and in plans.
The owner expressly authorized one extra sandbox for continuous own-model data
generation on 2026-10-15. Our current resource plan is two training sandboxes
plus that one generation sandbox; no further allocation is planned. Keep the
generation sandbox supplied with useful tasks and admit verified results into
the evolving training recipe. Measure actual CPU/RAM need on the allocated
machine rather than delaying useful generation to test smaller allocations.

Owner correction (2026-10-22): additional CoreWeave sandboxes are authorized,
including a dedicated evaluator shared by both model lines. The only required
resource specification for new GPU sandboxes is exactly one GPU. Omit CPU,
RAM, GPU-type and GPU-memory requirements; in particular, remove such filters
when capacity errors occur rather than repeatedly requesting the same constrained
shape. Inspect the actual assigned resources and adapt execution accordingly.
This supersedes the earlier three-sandbox allocation plan. Stop the current DGX
Mellum evaluation, preserve its partial evidence, investigate and repair concrete
problems, and launch the next evaluation on a sandbox.
Owner follow-up (2026-10-22): a minimal RAM requirement is also authorized when
useful. Keep CPU, GPU model and GPU memory unspecified; choose a RAM minimum
from measured workload needs. The Mellum evaluator completed on 16Gi without
OOM, but experienced substantial memory-limit pressure; its current replacement
uses 24Gi. These observations apply to evaluation, not training capacity.
Prefer an unconstrained one-GPU request when capacity is scarce and adapt to the
actual assigned resources.

Generation supply-chain supervision (engineering finding, 2026-10-23): monitor
the actual generator, exporter and admission worker separately. A busy teacher
GPU does not prove that new trajectories are reaching training. When changing
the guardian configuration, update each exact dependent publisher binding after
verifying the actual transition; a capacity-only change can still invalidate its
old configuration digest. Preserve historical actor/runtime identities, source
receipts and the durable closed-boundary/tranche journal. Recover the backlog
through the existing verifier; never relax an identity check to keep a publisher
alive. Distinguish a live restarted exporter from verified new exports and
actual learner exposure.

Owner instruction (2026-10-21): test `RecursiveSI/flint` as the generation
teacher and replace Qwen if its actual natlang task performance is decent.
Use the existing generation sandbox and preserve full trajectory, actor,
runtime and source provenance. The publisher's NVFP4 GGUF is a candidate for
our RTX PRO 6000 Blackwell; model-card scores alone do not establish our task
quality or useful generation throughput. This conditionally supersedes the
selected-teacher name above; it does not authorize external teacher generation.

Owner selection (2026-10-22): stick with `Qwen/Qwen3.6-35B-A3B` as the
self-hosted teacher. This supersedes the provisional Flint selection and the
earlier Qwen3.8 teacher. Use the existing generation sandbox, with an optimized
serving engine and an explicitly identified compatible quantization (initial
candidate: `nvidia/Qwen3.6-35B-A3B-NVFP4`). Preserve useful verified Flint and
earlier Qwen trajectories with their original identities. Give new Qwen3.6
batches their own exact checkpoint, tokenizer, runtime and actor provenance;
never relabel earlier generations. Keep both learners moving during preparation
and switch generation through explicit actor and guardian epoch bindings.

Historical comparison: Flint produced 32/48 successes versus Qwen's 39/48;
a later writer-occurrence runtime measurement produced 36/48 Flint successes.
These do not establish Qwen3.6 task performance. See
`plans/neuralese/FLINT_TEACHER_REPLACEMENT_20261021.md` and
`plans/neuralese/QWEN36_TEACHER_SELECTION_20261022.md`. Source admission and
learner adoption do not establish completed exposure or student capability lift.

Owner clarification (2026-10-16): standing authorization covers replacing any
stalled sandbox; do not ask again or wait indefinitely. Adjust the resource request if
capacity is exhausted. Restore the newest independently verified full state,
including optimizer/RNG/exposure, and preserve necessary data from the old box
until recovery dependencies are secured. This replacement is authorized despite
the earlier no-further-allocation plan. Other owner workloads, including
hindsight-kb training, have separate sandboxes: identify their ownership and
preserve them. A control-plane `Running` label alone does not establish a healthy
learner; monitor actual logs, progress and command responsiveness.

## Sandbox recovery coverage (engineering finding, 2026-10-23)

The quota incident stopped both learners, the teacher and both evaluators.
Full checkpoint exports and exported teacher archives survived, but those
sandboxes had no mounted snapshot storage. Both normal and authenticated gateway
file reads reject a completed sandbox; a stale container `Running` field does
not grant recovery access. Snapshot requests after shutdown failed because no
snapshot storage was configured. Unexported state requires provider recovery.

Prepare new production sandboxes with an explicit recovery workspace volume and
snapshot schedule alongside independent full-checkpoint and source-data exports.
The shared launcher accepts a one-GPU-only request, an optional measured RAM
minimum, and `scratch.restore_from_snapshot_id` for an explicit mounted-volume
restore. Do not silently discard a requested restore or mount it on ephemeral
root storage. A storage declaration or accepted snapshot request is not a backup:
record the actual READY snapshot and verify restored artifacts before relying
on it. Platform snapshot failures remain concrete operational work; do not claim
this incident's snapshot recovery is implemented or qualified from offline tests.
Retain the latest verified recovery state and its active dependencies; prune a
superseded snapshot/export only after its replacement is actually durable.

Every supervised entry, including zero-success RL seed fallback, must use the
shared exact-source resume and publication-owner mechanism. Otherwise useful
updates can remain invisible to the durable exporter. Preserve optimizer, RNG
and exposure; do not reinterpret an older exported parent as newer earned work.

## Neuralese architecture replacement (owner, 2026-10-15)

The destination is shared positional KV and hybrid-state spans, replacing the
projected-value channel in BOTH training and inference. These instructions
supersede older projection/recursive graph-replay foundation rules for new work.

- Neuralese tokens use ordinary token embeddings with a learned low-rank
  transformation in embedding space. Do not train or collect a projection into
  a separate communication latent space in this phase.
- Producers supply positional KV spans and other shareable hidden state;
  consumers splice that state with correct positions. Handle LFM convolution
  and linear state explicitly. Train teacher-forced graphs in one graph forward
  pass using masks and state handling, rather than recursive graph replay.
  Verify outputs and gradients against explicit cached execution; do not claim
  one-pass execution while secretly replaying the graph.
- Use different beginning AND end markers for producing versus consuming
  Neuralese. Producers learn independently useful exported state; consumers
  learn to use imported state. Stored Neuralese includes text and a sequence
  of token embeddings. Preserve tokenizer provenance and make cross-tokenizer
  matching explicitly best effort, not assumed exact state interchange.
- After several networks are trained, train a VAE through a shared joint
  semantic adaptation space to predict deltas reproducing source KV in the
  target. Follow with behavioral distillation/self-supervised SFT. Those
  per-token semantic encodings become the shared dialect. Do not claim that
  this later adapter is already trained or substitute the old projection.
  This requires completed training on genuine producer/consumer shared-state
  graphs in each participating model before VAE adaptation. Ordinary one-node
  text SFT alone does not qualify. Admit those communication graphs into the
  active shared trainer promptly, preserve completed per-row exposure and
  optimizer state, and record their complete exposure explicitly in readiness.
- Replace the old approach end to end: shared trainers, conversions, inference,
  block storage/transport, natlang harness, evaluators and recipes. Retain all
  usable source data and protected evaluation splits through conversion, and
  share mechanisms between LFM and Mellum.
- No prototyping shortcuts or half-solutions at architectural blockers. Work
  out both training and inference. After implementation, a separate agent must
  independently check the integrated work for gaps before cutover.
- Encode cross-model readiness and work directly in the training recipe and
  executable scheduling: begin shared-space VAE work as soon as at least two
  models are ready. Provide resumable, adjustable VAE, control write-out and
  additional joint training, so useful feedback changes the training rather
  than waiting for a clean run. A placeholder stage or manual note is not an
  implemented cross-model training provision.
- Current old-paradigm cloud runs continue as DIAGNOSTICS during implementation
  and local validation. Then restart BOTH models on the entire production
  recipe using the replacement. Continued legacy training is not replacement
  completion. Preserve useful diagnostic parents until cutover/dependency
  checks; then apply the standing discarded-work hygiene policy.

Track implementation and cutover in
`plans/neuralese/SHARED_STATE_MIGRATION_20261015.md`.

Publication dependencies (engineering finding, 2026-10-20): newly admitted
classification witnesses retain exact historical parser-policy dependencies.
Include their explicitly verified package roots in both provider and checkpoint
publication declarations. A shared provider fix does not update a frozen
trainer's publication helper. Check the actual privately verified step, not
service liveness or a saved checkpoint alone; keep useful training running with
the existing full-state fallback while repairing publication. Retain original
artifact hashes and exact path validation.

Exchange custody follows the effective shared contract (engineering finding,
2026-10-22): a runtime cutover can change role verification mechanism identities
and therefore the fixed exchange-cohort digest. Derive the complete effective
contract after all controller/publisher loaders and select the same fresh,
contract-addressed exchange directory on both models before the real launch.
Do not reuse or overwrite a differently bound cohort. Rebuild only the completed
adaptation and control declarations when necessary; preserve earned weights,
optimizer/RNG/exposure and the existing training lineage. A failed coordinator
before learner launch is not evidence that its saved model state is bad.

Retention follows active lineages (engineering finding, 2026-10-20): a larger
step number from an old lineage does not supersede a current model's artifacts.
At every owner/runtime cutover, bind cloud retention to the current full-backup
and inference states, current evaluation consumers and in-flight publications.
Protect required current parents explicitly. Check actual remote presence after
publication; a successful GET can finish after a concurrent retention deletion.
An old readback receipt alone does not prove that recovery still exists. If a
current export disappears, stop the exact retention service, repair its active
dependencies and restore useful exports before resuming verified cleanup.

Native token learnability (engineering qualification, 2026-10-15): existing
tokenizer entries are not evidence of usable frozen output rows. The QAD
thinking boundaries were almost identical, imposing a fixed output-probability
limit regardless of backbone training. Current native models use learned sparse
rows for the existing reasoning boundaries and FP32 masters for all learned
Neuralese/reasoning rows, with the backbone's activation casts. Upgrade legacy
parents explicitly, preserving exact numeric parameter/moment values, RNG and
exposure; never reset history or silently reinterpret an old checkpoint.
Check actual parameter changes as well as gradients: BF16 sparse rows lost
production-rate updates despite nonzero gradients. The shared trainer records
per-row gradient/update norms and changed coordinates. Diagnose an unexpected
zero update before attributing poor learning to the model. See
`plans/neuralese/shared-state/REASONING_TOKEN_LEARNABILITY_20261015.md`.

Native optimizer routing (engineering qualification, 2026-10-16): native
ordinary parameters, including sparse vocabulary rows and the low-rank adapter,
use AdamW; the backbone's declared MuonSR routing remains separate. Do not
reclassify ordinary matrices into Muon by shape. Existing affected checkpoints
require the explicit `native-ordinary-to-adamw-new-history-v1` migration:
retain original numeric Muon buffers, groups and named bindings as transition
provenance, preserve weights/RNG/exposure and unaffected optimizer history, and
start the newly selected Adam algorithm with honestly new zero moments/age.
Never fabricate historic Adam variance or silently reinterpret optimizer kinds.
Monitor actual gold-boundary support, gradient/update directions and held task
behavior; a routing bug alone does not establish the cause of every behavioral
failure or an optimal learning rate. See
`plans/neuralese/shared-state/NATIVE_OPTIMIZER_ROUTING_TRANSITION_20261016.md`.

Owner correction, 2026-10-15: the lodestar is comprehensive training of BOTH
models through the ENTIRE production recipe and all usable available data.
Stages are steps to progress through, not endpoints. Half measures, short
probes and partial stages do not count as production completion. Track whole
recipe progress and actual data exposure; integrate newly prepared sources
through explicit continuations. KL distance from the base is telemetry only:
improvement requires changing the model. Do not stop or deny qualification
solely for base-relative KL or agreement drift. Keep actual task-performance,
held-loss, numerical and runtime checks; diagnose and repair real regressions.
Runs may be stopped and tweaked at any time when an improvement warrants it
(owner clarification, 2026-10-15). The supervising agent decides whether the
problem requires a redo or continued training, preserving exact state and
recording the evidence. Comprehensive training is the objective, not a ban on
interruptions or sensible course corrections.

## Whole-graph RL and LFM curriculum (owner, 2026-10-16)

Reward correct task execution whether or not the policy uses Neuralese. Do not
require a communication edge or a minimum number of nodes for RL eligibility.
When an execution has a Neuralese graph, policy updates cover the entire actual
graph: preserve sampled and forced producer turns, consumer imports, exact
occurrence bindings and downstream outcomes, with gradients through the real
producer/consumer dependencies. Never turn an in-episode producer into a detached
external input or train a selected consumer/turn in place of the whole graph.
Missing or unsupported graph capture is repaired before that episode is used.

Prepare an executable LFM RL task curriculum, starting with tractable verified
natlang execution and growing into tool use, multi-step work and communication.
Use actual task outcomes to adapt difficulty; keep protected evaluation separate.
Share curriculum mechanisms and useful findings with Mellum. This does not relax
comprehensive training on the entire evolving production recipe.

## Adaptive production training (standing owner instructions, 2026-10-15)

- **Best possible models as soon as possible is the objective. We are not
  doing science.** Do not optimize operations for clean, frozen, reproducible
  runs or controlled comparisons at the expense of useful training. Existing
  runs are work in progress, not experiments that must remain untouched.
- **Constantly improve the recipe and keep going.** Monitor both models and
  their processes, investigate concrete impediments, and greedily exploit useful
  data, fixes and insights as soon as they are available. Do not let repeated
  probes, qualification windows, packaging or handoffs become substitutes for
  production training. Use focused checks that resolve a real correctness or
  operational concern, then return the GPUs to useful training promptly.
- **Comprehensive means the entire production recipe on both models and all
  usable available data.** Stages may differ, but progress through them; a short
  warmup, partial epoch, isolated stage or smoke run is not the goal. Track
  actual stage progress and data exposure, and resolve what prevents remaining
  stages or data from being used. Do not claim that downloaded or registered
  data has been trained on.
- **Adapt live.** Add newly usable data promptly rather than waiting for a
  distant clean run or an unrelated conversion batch. Stop/reload/resume at a
  safe update boundary when needed; retain optimizer/RNG state when continuing
  is appropriate. The supervisor may stop and tweak any run at any time for a
  potential improvement and judges whether to continue or redo it. Preserve
  useful checkpoints and record the practical reason; do not require a new
  lineage or a clean experimental restart for routine changes.
- **Checks serve production.** Keep checks that protect actual task quality,
  valid gradients/numerics, held-data separation, data quality, recoverable
  progress and functioning runtime/transport. Similarity to the untrained base
  is not the objective: base-relative KL/agreement are telemetry, never a stop
  or qualification barrier. Exact state and artifact identities support safe
  continuation and debugging; they are not a scientific-purity requirement.
- **Keep launch checks nonmutating and necessary** (engineering lesson, 2026-10-20).
  Run normal metadata/argument preflight once; let the real entry perform its
  required state, numerical and capacity checks before updates. Do not execute
  an entire administrative entry in the canonical production directory first:
  it can leave strict-write launch records that block the real learner. Put
  diagnostics in an isolated namespace, preserve concrete failure evidence,
  and avoid repeated whole-entry barriers while useful training is stopped.
- **Avoid redundant supervision barriers** (engineering practice, 2026-10-20).
  Routine data append payloads that change only counts, paths and authenticated
  generations may proceed through the already reviewed mechanism with an
  independent operator review and the normal schema, source/held, custody,
  geometry and compare-and-swap checks. They do not need another round trip to
  the root agent. Review mechanism changes separately. Record the actual
  reviewer and standing authorization; never claim an individual review that
  did not occur. Preserve exact exposure and continue useful training.
- **Bring in deferred improvements** (owner, 2026-10-16). Keep a current
  inventory of deferred data, performance, training and inference work; actively
  integrate each ready improvement. An old handover, incomplete source census,
  clean-run boundary or another model's unrelated qualification is not a reason
  to leave useful work disabled. Preserve real numerical, task-quality and
  cross-model foundation prerequisites, and resolve their concrete blockers.
  Independent model improvements may advance independently; propagate shared
  fixes and retain exact optimizer/RNG/exposure through adaptive continuations.
- **Share code as much as possible and distribute all generalizable insights
  across all models.** A useful fix, optimization, data improvement or recipe
  insight on one line must promptly be considered and propagated to the other
  lines and machines. Extend the shared trainers, loaders, objectives,
  evaluators and recipe inheritance instead of creating per-model forks.
  Keep only necessary architecture-specific behavior/settings separate, state
  why they differ, and communicate shared decisions through coordination.
- **Verify the effective control code after loaders.** A fix in the installed
  package can be overwritten by a sealed controller or publisher source closure.
  Rebase every control layer onto shared fixes, preserve required operational
  behavior, update mechanism identities, and test after the complete loader has
  run. A passing default-module test does not prove that the production
  controller executes the fix. This applies to both model lines.
- **Keep recovered work ahead of historical operations.** Restarting the full
  mandatory schedule after an external optimizer continuation must reserve an
  unused operation namespace in the same lineage. Never let a completed bridge
  result from an older attempt replace the newer earned parent. Verify the
  actual checkpoint and cursor, and observe the real optimizer child advancing;
  a live coordinator or a launched request alone is not resumed training.
- **Reuse actual controller custody across continuations** (engineering finding,
  2026-10-21). Pass the coordinator's continuously held canonical lease descriptor
  through worker threads into completed/runtime configuration; opening the same
  lock again can self-contend. A normal resident publication changes bootstrap
  metadata: verify and refresh the actual adopted configuration rather than
  discarding its data or weakening custody checks. Preserve concrete configuration
  tracebacks and diagnose a blocked retry signature. Exercise this path after the
  complete sealed controller/publisher loaders; a previously shared guard fix can
  still be missing from their frozen source closures.
- **Publish completed resident training before adopting its next stage**
  (engineering finding, 2026-10-22). A supervised stage can finish after several
  live data appends while the coordinator still holds its bootstrap recipe.
  Use the real completed request and result through the normal publication
  helper to bind the earned checkpoint, actual resident configuration and next
  cursor. Updating bootstrap metadata alone does not publish the adopted recipe.
  Preserve optimizer/RNG/exposure and verify the next real optimizer child.

Historical reader replay (engineering finding, 2026-10-22): recipe inputs can
include the exact classification reader implementation. A changed reader is not
a byte-identical relocation. Replay historical completed adaptations and drained
handoffs using the package that actually prepared their metadata, with complete
code inventory and bound recipe checks before and after isolated CPU replay.
Both old and future plans in a completed adaptation were enumerated by that
adaptation's implementation; its old numerical package is not automatically the
original metadata reader. Preserve strict source/held, lineage and exposure
checks. Include committed resident histories and both effective sealed control
closures in validation before cutover. A live scheduler holding the canonical
lease can still be blocked: monitor its actual configuration_errors and retry
signature alongside real optimizer progress. When a normal publication advances
the earned parent, rebind a completed invocation to that actual state through
the existing continuation mechanism rather than reusing an obsolete invocation.

These instructions supersede earlier requirements for frozen experimental
runs, base-drift hard stops or clean-run comparisons where they conflict.

The main goal in everything is to keep moving the work forward: move training
along and keep the machines doing useful work. Don't be overly cautious, and
don't wait for the owner to approve decisions. If something is sensible, do it,
record why, brief the other machine, and keep going. Raise with the owner only
what is truly theirs to decide, and keep working on everything else in the
meantime.

Don't reject an approach because our implementation of it failed (owner,
2026-10-11). A worse loss, degraded behaviour or an unexplained numerical gap is a
bug until proven otherwise: compare against an exact reference, localize the
divergence, fix it, and re-measure before judging the approach itself.

When something doesn't make sense, assume a bug and run a wide, varied dragnet
for it (owner, 2026-10-11). Usually there is a bug. Don't give up, and don't settle
for the naive explanation ("BF16 noise", "the approach doesn't work", "the model
can't learn this"). Check many hypotheses in parallel: data and masking, rendering
and templates, dtype and upcasts, caches and state carried between steps, kernels,
optimizer state and resume, gradient flow, loss weighting, evaluation code, and
differences from a known-good path. Record what was ruled out and how.

# Machine ownership and coordination

- Use `/home/werg/natlang` as the canonical development checkout on both Pop and
  DGX. The Pop agent owns execution/resource management on Pop; the DGX agent
  owns execution/resource management on DGX. Do not launch new jobs from retired
  `natlang-remote` snapshots or manage the other machine's jobs without a request.
- Agents talk through `scripts/coord.py` (see "Messages" in
  `plans/MACHINE_COORDINATION.md`). At session start, before changing shared
  resources, and at each monitoring cycle (at least every 50 minutes), run
  `python3 scripts/coord.py inbox --ack`. Answer requests with `coord.py reply`
  or `close`; send with `coord.py send --to pop|dgx|all --subject ...`. Keep
  your machine's state in `coord.py status --set`, not in a stream of notes.
- Coordination is status values plus exact verifiers (owner, 2026-10-12;
  protocol in `plans/MACHINE_COORDINATION.md`): no per-commit notes, one status
  page per machine, requests only for actions on the other machine, closure by
  the shared manifest verifier (`coord.py close-verified`) instead of
  acknowledgement chains, and one announcer per lineage.
- Give each concurrently running agent its own coordination reader identity:
  use `coord.py --as <machine>-<agent-role>` or `COORD_AS`. Codex helpers must
  not acknowledge the owner's reader cursor (inspect `coord.py whoami`). Forward
  decisions and requests outside your task to the owner; acknowledgement is
  not implementation or acceptance of a decision.
- Synchronize code through small commits pushed often to `origin/main`, using
  ordinary git: pull, merge and rebase as usual (owner, 2026-10-11), and keep the
  checkout current with main. Commit your own work promptly and stage only your
  paths; don't discard or stash away changes you didn't make.
  `scripts/git_publish.py` remains available as a helper. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.
  Reconcile coverage against every entry of the exact upstream inventory, with
  explicit member/index bindings; sequential directory labels are not a coverage
  proof. Keep conversion, admission, staging and actual learner exposure separate.
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

# Storage hygiene and discarded work (standing owner order, 2026-10-11)

Owner clarification (2026-10-16): delete pretty much everything except what is
necessary to carry on. This includes remote W&B transfer/export storage. Keep
the latest independently verified recovery state, explicitly active diagnostic
or handoff parents, useful source/training data and protected held data. Prune
superseded checkpoint exports continuously after replacement durability is
verified; do not accumulate every periodic full checkpoint. Write the exact
deletion list first and verify removal. Never delete an entire mixed data and
checkpoint run merely because most of its bytes are obsolete checkpoints.

Amendment (owner, 2026-10-13): make deletions less eager. With the external disk
available for space, lean towards moving data to `/mnt/external` rather than
deleting it when unsure.

- Delete discarded approaches and runs: their checkpoints, optimizer states,
  exports, frozen runtime copies, diagnostic outputs, obsolete recipes and run
  tracking. Do not archive them, move them to another disk or machine to avoid
  deleting them, or retain them because an old note or manifest references them.
  Remove obsolete registry entries, manifests and handover tracking as well.
- Keep the active lineage and its required parents, reusable training/source
  data, and protected/held evaluation data. Distinguish these from discarded run
  artifacts before deleting; check live process references and current recipe
  dependencies. A regression still needs diagnosis before rejecting an approach.
- Deduplicate exact copies and prune superseded intermediate conversions,
  rendered packets, review previews, replays and caches. Keep one current useful
  representation with its source provenance, rather than every attempted version.
  Use existing deduplication and retention tools; preserve content hashes and
  logical paths where a current consumer needs them. Never hardlink mutable files.
- Prune rejected outputs in general. Retain one only for a concrete current use:
  an assigned repair, an unresolved bug investigation, or an actual failure/DPO
  training input. Record that use and its consumer; "might be useful someday"
  is insufficient. After the use ends, remove it. Keep independently useful
  correct rows from a partly rejected batch; remove bad rows and redundant copies.
- Check storage during the normal monitoring cadence and after large runs or
  conversions. Clean autonomously under this order. Brief the other machine so
  discarded artifacts and obsolete registry entries are removed there by its
  owner too, rather than silently surviving in mirrors. Record the current
  decision and reclaimed totals, without building another archive of discarded
  approaches. This order supersedes older blanket preservation/PRUNE-ASK rules
  for explicitly discarded work.
- These deletions are expressly desired and authorized by the owner (2026-10-11):
  deleting discarded run directories, checkpoints, exports, caches and their
  registry/manifest entries on either machine is wanted. Do it without asking for
  confirmation; list the exact paths in a script first, then delete.

# Natlang-native work (owner rules)

These rules apply to every agent on every machine. Paste the short form into any
subagent brief. Reasons and history are in the linked documents.

**The main rule: think in natlang-native terms.** Before building anything (an
app, a runtime component, a compiler pass, a tool, a policy, a review step),
first ask how it would be a natlang program: typed natural-language functions,
callable folders, services for the outside world, and crisp code only where
exactness or the outside world requires it. Start from that design and justify
every piece of crisp code. Do not start from conventional code and add natlang
afterwards. The rules below are consequences of this one.

- **Policy is natural language; mechanism is crisp.** Any component that decides
  something (which tier serves, what to fuse, what to promote, which task runs,
  how to recover) is a natlang program. This includes natlang's own runtime and
  compiler, not only applications. Crisp code is for the outside world
  (services), the CLI, exact durability (atomic commit), exact verifiers of what
  the stages produce, and plumbing. When a policy is hot, give it one interface
  with a crisp and a natural-language implementation selected by a setting
  (`ts-host/src/runtime/pluggable.ts`); crisp may be the default.
- **Port end to end, unit by unit.** A system "in natlang" has its whole logic as
  natural-language stages decomposed like the real system. For every part,
  record whether it becomes its own function, one instruction, implicit (rare)
  or a crisp helper, and why. Spell algorithms out as numbered steps over named
  data for small models. Write DECOMPOSITION.md first, then implement and merge;
  owner review happens afterwards (queue: `plans/OWNER_REVIEW.md`).
- **Measure model-facing changes live.** Anything the executing model sees
  (prompts, rendering of values, tool text, error and feedback text, type text in
  signatures, removed or added instruction sentences) is sampled on the live
  executor (about 48 samples per variant) before it is adopted. A change that
  lands unmeasured is recorded in `plans/MODEL_FACING_CHANGES.md` as debt and
  measured in the next teacher window.
- **Teach in errors, not prompts.** Never spell out a wrong form in a prompt; it
  primes the model to produce it. Handle wrong forms in the runtime and teach the
  fix in a one-sentence error. Prefer making standard JS idioms work over new
  names; be very circumspect with new model-facing names. Every semantic change
  updates `skills/natlang-authoring` and `skills/natlang-integration` in the same
  push, with recovery advice for each new error.
- **No premature limits.** Termination is structural (no while loops; recursion
  on a smaller argument; `iterateOn` with a measure or a natural-language
  predicate), not budgets or default timeouts. Add a resource limit when a
  problem shows up.
- **Apps:** decide on a snapshot (model calls), then apply a pure bounded commit.
  Effects are data, timers are state, and derived values form a DAG. Stay close
  to JS semantics.
- **One implementation per concept.** Before adding a mechanism, look for the
  existing one (specializer, call store, pluggable, refinement shadow, shared
  hashing in `natlang_neuralese/common`) and extend it. Training lineages share
  trainers, objectives, evaluators and recipes (`extends`/`overrides`); only
  declared backbone-inherent differences are allowed.
- **Training data covers whole trajectories** (prompts, inputs, tool output at
  lower weight). Breadth comes from more natlang domains, not from anchors or
  broad mixing; anchors never cap learning.

# Working practice on shared machines

- DGX launch environments are explicit: project/data tests use
  `/home/werg/natlang/.venv-neuralese/bin/python`; CoreWeave SDK orchestration
  uses `/home/werg/.local/share/natlang/cw-venv/bin/python`. Pass
  `PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang` inside
  the ledger command for project/data entrypoints. Node builds use the Node24
  bin directory under `/home/werg/.local/share/natlang-dev/`. Do not assume a
  caller's shell environment is forwarded into a ledger unit. Verify these
  environments before launching substantial jobs; repeated import/launcher
  failures are a shared operational bug to fix, not useful qualification.
- A ledger unit's explicit relative executable path resolves in its admitted
  `--workdir`. Bare executable names retain PATH lookup. Use an absolute script
  path for an intended checkout when inspecting or recovering older launches:
  older `systemd-run` invocation paths resolved against the caller's checkout
  before changing directory. Verify the tested checkout's path and revision in
  the actual check log; a worktree claim alone is not evidence of its tests.
- For DGX administrative ownership guards requiring `os.pidfd_open`, use the
  host `/usr/bin/python3`; the CoreWeave SDK venv lacks that API. Invoke SDK
  operations through its venv as a subprocess when both capabilities are needed.
  Compile the assembled remote Python body before transport, not only the
  outer launcher. Honor each receipt's binding schema: compare path/SHA fields
  explicitly and check declared byte counts against the actual file; a legacy
  two-field binding is not unequal evidence merely because another includes bytes.
- Reuse authenticated live parser/index caches for CPU preparation and isolated
  consumer checks. Cache isolation is not an objective. Verify exact source,
  tokenizer, role, parser-policy and entry identities before reuse; preserve
  cold work only when no valid completed cache exists. Locate artifacts through
  registries and bound manifests; do not leave unbounded content searches or
  directory crawls running across the external disk after the answer is known.
- On DGX every build, test and probe runs through the memory ledger
  (`scripts/memory_ledger.py run --unit <family>-<id> --budget-gb N -- ...`;
  units run asynchronously, so wait for the unit to finish and read its log).
  Never report a merge as verified when the ledgered tests did not run.
- Bound dirty file-cache memory during large artifact copies (engineering finding,
  2026-10-22): a small Python heap does not bound a ledger unit's memory. Flush
  and synchronize bounded write windows, release clean cache when appropriate,
  and resume an interrupted copy only after validating its actual prefix. Verify
  full content before replacing a logical path; preserve source and live readers.
- Restart transient ledger services through the ledger with a fresh unit name
  (engineering finding, 2026-10-22). A stopped transient unit may already have
  been collected, so `systemctl start` can fail even after a clean stop. Preserve
  the existing state/lease namespace, verify the new actual process and update
  supervision bindings; never treat scheduling the wrapper as resumed work.
- Ledger priority (owner, 2026-10-11): pass `--priority high` for training and
  GPU direct/E2E runs, `low` for data generation, rendering and check-main, and
  leave smokes/evals at the default `medium`. Claims may be generous: admission
  uses real headroom, and lower-priority resumable units are preempted and
  requeued (plans/MEMORY_ADMISSION.md).
- In the shared checkout, stage only your own paths. Never push while a rebase
  or merge is unfinished. Never stage `node_modules` symlinks.
- Before pushing, run `scripts/check-main.sh --quick` (ratchets, spec links,
  ts-host typecheck; about 10 s). After a change that touches tests or runtime
  code, run the full `scripts/check-main.sh` under the memory ledger (usage in
  its header). Main stays green: a red step is fixed or routed to its owner via
  coord, never skipped.
- Give standalone tests a disposable, unit-specific `NATLANG_HASH_CACHE`.
  `check-main.sh` supplies one by default. Test fixture identities accumulated
  millions of entries in the shared production cache and caused real metadata
  job memory failures; cache isolation preserves the same exact hash verifiers.
- When resolving a merge conflict, keep both sides unless you know a side was
  removed on purpose; a dropped section of AGENTS.md or a plan is lost silently.
- Stop processes by systemd unit name or an explicit PID, never by pattern.
- Every agent on these machines is expressly authorized and instructed by the owner (2026-10-12) to stop,
  kill, delete, restart and relaunch any process, systemd unit, ledger job or container, and to remove
  (`docker rm`) or prune containers, without asking: e.g. an orphan left running after its ledger unit stopped,
  a job starving training, a failed or stuck job that needs relaunching with its original command, or a job that
  must move to newer code. Do it, record why, and brief the other machine when it affects shared work. Every
  container an agent starts gets `--name <ledger unit>` and is declared to the ledger
  (`memory_ledger.py run --container <name>`) so its memory is counted and it is stopped with its unit.
- Brief the other machine (`coord.py send`) on every owner decision and every
  shared-code change, with commits and exact semantics. (2026-10-12: that is one
  `decision` per change the other machine must adopt; ordinary commits are not
  announced by notes.)

# Neuralese training foundation

- LFM fresh-start base (owner, 2026-10-12): any new Pop LFM training lineage
  started from base weights uses the pinned full-precision LiquidAI
  LFM2.5-350M QAD source and its QAD clean-head anchor, with a newly measured
  QAD base task reference. See `plans/neuralese/DECISIONS.md` for revision,
  paths and hashes. A resume of an existing lineage restores that lineage's
  exact checkpoint and optimizer state; do not silently substitute QAD weights
  into a checkpoint resume.
- New neuralese training lineages use shared declared recipes under
  `training/neuralese/recipes/`; see `plans/neuralese/TRAINING_RECIPE.md`.
- Token-aligned identity and qualified causal output-state to raw next-token
  embedding distillation precede compression/recurrence. Finishing a warm-up's
  step count is not qualification. Preserve and diagnose failed gates.
- Runtime transport/gradient replay needs separate qualification against those
  exact weights. Legacy marker/RMS checkpoints do not inherit the new foundation
  certificate, and backbone changes require channel requalification.
- Exception, Neuralese direct v1 (owner decision 2026-10-11,
  `plans/neuralese/NEURALESE_DIRECT_V1.md`): a lineage made only of
  `neuralese_direct` stages starts from the base model with no warm-up and needs
  no identity or embedding-distillation stage. Its stage's step-0 identity check
  (NeuraleseProjection equals the soft-token mixture, TokenChoiceProjection
  absent or the identity, logits equal the base) is the identity qualification,
  declared as the recipe's `identity_qualification` and recorded in
  `direct-certificate.json`. Older lineages keep the rule above. Direct-stage
  qualification is the identity check, held CE below base and
  natlang task success (owner decision 2026-10-12): the in-run task-success
  evaluation (16 tasks, one per shape, of the registered held packet
  `natlang-held-task-packet-v3-20261012`, which is v2 with content-preserving
  edit_stream oracles, executed for real against the
  training model at every eval and at the end; `eval/task_cadence.py`) is at
  least the line's base and not below the previous evaluation minus the noise
  measured by repeat runs. The 48-probe
  behaviour gate is report-only: it runs at a drift warn and at the end and is
  recorded, never a stop or a requirement. Response KL warns at 0.5 (which runs
  the gate); it never stops a run or denies qualification merely for changing
  the base distribution (owner correction 2026-10-15). Actual task performance
  remains measured. The Neuralese value is report-only there: its
  requirement is the end-to-end stage's, consumers reading values against
  consumers reading tokens (owner decision 2026-10-11).
