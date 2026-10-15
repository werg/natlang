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

Replacement not yet deployed; both cloud lineages remain diagnostic. Full
Mellum fixed-delta sparse attention now matches forward, consumer CE, all 28
layer backward adjoints, source-adapter gradients and incremental inference
bitwise. Actual QAD LFM and the isolated H100 destination pass the unchanged
forward/gradient tolerances. Native SFT and typed/durable child handoffs,
contextual writes, empty values and hierarchical views are integrated. The
standing full check of 606f1ec3c5 is green.

The shared semantic stage now implements VAE, joint producer/write-out/consumer
adaptation, behavioral distillation, deferred full-model optimizers, complete
per-row/per-model-pair exposure, checkpoint/RNG continuation, data append and
loss/LR/epoch changes. Two-model tiny tests prove exact interrupted continuation
and real joint gradients. The readiness controller checks actual production
stage evidence and starts with any two supported ready models; later model or
curriculum/data changes trigger continued work. Semantic adapter exports bind
the exact joint checkpoint, participant and small source tokenizers. Learned
decoding feeds the actual native cache/serving path, with explicit alignment.

Cutover remains OPEN: admit the real source-witnessed paired graph conversion,
bind both complete production recipes, verify an actual full-model joint update
and placement, measure post-joint natlang behavior, finish advanced native
learner/gradient/optimizer/adapter/guidance consumers, and complete persistent
semantic-code transport. A functioning raw-state-to-semantic-to-target adapter
is not yet a compressed shared-code wire dialect. Independent review reported
these gaps; none is waived by capability denial or tiny-model tests.

Pop is closed; DGX supervises all remaining work. Generation continues ONLY
through our own sandbox models. The dedicated generation sandbox is
`dadd0462-0175-470d-b1ee-4d52edfe110e`; exact Mellum3600 assets and the harness
are being prepared there. The stopped local Mellum teacher's four verified
successes remain assigned for admission. Only Pop generation was to finish;
the prior DGX generation-closeout interpretation was corrected by the owner.

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
