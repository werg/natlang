# Shared-state Neuralese migration

Owner decision: 2026-10-15. Replace the projected-value approach throughout
training and inference. Existing cloud training continues as diagnostics while
the replacement is implemented and locally validated. Neither a design document
nor a standalone graph fixture completes the migration.

The production contract is ordinary token embeddings with a learned low-rank
embedding-space transform; exported positional KV and shareable hybrid state;
distinct producer/consumer beginning and end markers; stored text plus token
embeddings with tokenizer provenance. Teacher-forced graphs use masks and
architecture-specific state operations in one graph forward pass. Cross-model
state interchange requires explicit best-effort matching or learned adaptation;
equal tensor shapes alone do not establish semantic compatibility.

After several networks are trained, the next recipe stages train a VAE through
a shared joint semantic adaptation space to predict target-state deltas, then
behavioral distillation/self-supervised SFT. The resulting per-token semantic
encodings define the shared dialect. No trained adapter or downstream lift is
claimed at the start of this migration.

## Ownership and completion evidence

- Implementation agent `neuralese_shared_state`: graph decomposition/compiler,
  graph executor, LFM and Mellum state adapters, embedding transform, exact
  cached reference and forward/gradient checks. Architecture decomposition is
  written before implementation. Unsupported recurrent forms fail explicitly.
- Root supervisor: serving, durable block transport, natlang marker/harness
  integration, trainer/recipe/data wiring, model-facing measurement and cutover.
- Cloud supervisor: current diagnostic jobs, resource management, useful fixes,
  staged data, then replacement launches on both sandboxes.
- Data agent: existing source preparation/protections continue. Canonical native
  trajectories, text and full contexts remain reusable migration inputs.

Before switching, demonstrate real producer-to-consumer state transport through
the native harness on both actual backbones, positionally correct attention,
explicit hybrid-state treatment, and gradient agreement with exact cached
reference execution. Include fanout, multiple reads/writes, padding and resumed
durability. Confirm the training path actually executes one flattened graph
forward rather than recursively rerunning its producers. New markers require
live executor measurement or explicit recorded measurement debt.

Audit every old entry point: objectives and optimizer groups, data schemas and
conversions, rendering/token IDs, incremental serving, stored blocks and HTTP,
TypeScript transport, RL capture/rewards/replay, checkpoint/export, evaluation,
and production recipe inheritance. Do not silently route a replacement run
through the projected-value implementation.

## Current status

Raw native-state production training is deployed on BOTH sandboxes through
shared `native_state_sft` jobs under `/work/native-production`. Direct SDK
inspection on October 16 at 09:23 UTC found LFM at6269 (runtime2af640d3e5a3,
100% sampled GPU utilization) and Mellum at7010 (runtimea765bd1bea00,
83% sampled utilization). Both are performing real optimizer updates after
exact6181/6813 restores, retaining weights, optimizer/RNG and earned exposure.
Native interface rows use FP32 AdamW masters at.0003; LFM reasoning rows use
.0003, while Mellum reasoning/backbone/adapter rates remain3e-6. These are
current settings, not evidence of optimal rates or downstream task lift.

The immutable6181 LFM parent records25,755 graph visits/916,597 active rows,
including10,985/26,312 original foundation graphs. Mellum6813 records21,070
visits/949,801 rows, including8,237/26,312 foundation graphs. Live updates after
those parents are not included in these counts. Neither foundation is complete;
no model is ready for shared-dialect training. Additional admitted sources are
obligations, not earned exposure.

Current-capacity D12/D16/D18 v2 admissions retain exact full rows and protected
splits while making1,739 additional LFM training graphs fit the actual measured
92,606-physical/157,075-KV envelope. D15's complete source and fit/pending
partition are admitted; Qwen V17 adds8,309 train graphs per model and is staged
on both sandboxes. Sandbox adoption and actual visits remain separate checks.
The original pending cohorts and visit quotas must survive activation.

Resident transactions now support pending-only and combined active/pending
adoption through the existing atomic commit point. Independent review found
and fixed a withdrawal-history gap before commit. Integrated resident, offline
pairwise, continuation and memory tests passed109 cases. The new reference-free
offline preference objective binds both complete alternatives with zero ordinary
CE weights and scores their assistant suffixes through the shared optimizer.
Production source admission/capacity qualification and mandatory post-joint
interleave are still being finished; this is not a deployed preference-stage
completion claim. Native SFT, typed durable child handoffs, contextual writes,
empty values and hierarchical views remain integrated.

Full-model raw-state references and fixed-delta tests established the deployed
channel. Subsequent prefix-sharing investigation found and repaired a separate
sparse-attention traversal-order defect in newer code. LFM prefix reuse passes
local strict checks; Mellum's remaining adjoint differences are localized to
BF16 fan-in rounding and nonlinear FP8 input-adjoint quantization. Prefix reuse
remains off in current production. See
`shared-state/CAUSAL_PREFIX_QUALIFICATION_20261016.md`; do not transfer a
qualification between arithmetic modes. Mellum actual trained-parent3505
qualification passed with392 executedFP8projections/67,200routed rows,
BF16 input adjoints, zero forward/consumer-loss gap and adapter gradient
maximum4.66e-10 under unchanged gates; subsequent productionSGD reached3750.
This qualifies that target path, not prefix reuse or task-quality lift.
Standing fullcheckbfdc7e75 completed green, including CUDA and TypeScript.
The next standing fullcheck323680aa8 is running; current resident/preference
integration additionally passed109 focused tests,29 capacity-continuation
tests,17 provider-closure tests and the quick main check.

The shared semantic stage now implements VAE, joint producer/write-out/consumer
adaptation, behavioral distillation, deferred full-model optimizers, complete
per-row/per-model-pair exposure, checkpoint/RNG continuation, data append and
loss/LR/epoch changes. Two-model tiny tests prove exact interrupted continuation
and real joint gradients. The readiness controller checks actual production
stage evidence and starts with any two supported ready models; later model or
curriculum/data changes trigger continued work. Semantic adapter exports bind
the exact joint checkpoint, participant and small source tokenizers. Learned
decoding feeds the actual native cache/serving path, with explicit alignment.

The complete replacement remains OPEN. Real source-witnessed paired conversion
and both production recipe bindings are implemented and active. A full-size
both-model VAE/joint/control/distillation/controlled-GRPO update now passes on
the DGX, including actual counted FP8 projections, finite gradients and replay:
see `NATIVE_JOINT_CUDA_FIT_20261015.md`. Those tests used pinned base models,
not completed cloud-trained foundations, and do not establish task lift or
target-container qualification. Canonical readiness scheduling must still
progress from both completed foundations into trained shared-space adapters
and measured post-joint behavior; no model is ready yet.

Persistent semantic-code transport is implemented through the existing durable
artifact store, with text/tokenizer/embedding/adapter identities and decoded
target state; see `shared-state/SEMANTIC_WIRE_DECOMPOSITION.md`. Its existence
does not establish a trained shared dialect. First-order native graph gradients,
named immutable parameter/optimizer application, portable continuation and
guided generation are implemented. Functional native adapter application,
exact actor capture/replay and immutable coefficient optimizer continuation
are integrated (11fa82aff/e31371cdb). Optional FP32 coefficient cotangent
carriers preserve stored BF16 forward values and fix cached/flat adjoint gaps;
independent root actualCUDA integration passed26cases plus the repaired
CPU-reference diagnostic oracle. Forward values remain bitwise identical;
this does not enable production adaptedRL or claim task lift. Multi-update GRPO behavior/current actor
separation, heterogeneous per-call bindings, full-backbone adapted export and
higher-order derived graphs remain explicit unfinished paths in
`shared-state/NATIVE_LEARNING_SERVICE_DECOMPOSITION_20261015.md`. Capability
denial is not completion. The independent review and actual trained-weight
training/inference/transport qualification remain required.

Pop is closed; DGX supervises all remaining work. The dedicated generation
sandbox `dadd0462-0175-470d-b1ee-4d52edfe110e` runs the stronger self-hosted
Qwen3.8-Flash-Next NVFP4 teacher with PennyRoyal/SGLang on SM120. Mellum and
LFM are students, not generation teachers. Qwen V6 and the disjoint Big2
1,000-chain tranche are admitted; staging is not trained exposure. See the
current `HANDOVER.md`, immutable corpus manifests and live coordination status
for changing run/data frontiers. No Luna/API generation or extra allocation.

Owner clarification: no prototyping shortcuts or half-solutions at blockers.
Assign a separate agent to independently review the completed integrated
training and inference paths before cutover, and fix every material gap.
Cross-model readiness must be executable recipe logic: when at least two
models finish the required native training, start shared-space VAE adaptation
without waiting for every other model. Include configurable, resumable joint
training of the semantic space, write-out controls and downstream behavior.
Changes to losses, curricula and useful data must support adaptive continuation.
Do not count stage names or a readiness note as working training code.

After qualification, restart LFM and Mellum on their complete updated production
recipes, including all admitted usable sources and every remaining stage.
Record actual exposure and stage progress. Remove discarded legacy mechanisms
and artifacts only after live consumers and required diagnostic parents are
accounted for. Do not discard reusable source or protected evaluation data.
