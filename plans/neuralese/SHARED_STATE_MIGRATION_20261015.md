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

October 18, 08:55 UTC operational position: both replacement models are
training through the production recipe. LFM continues ordinary SFT from the
exact step-13228 parent with optimizer, RNG and exposure history intact. At
step 13529 it adopted a verified 3,492-graph stronger-teacher continuation;
adoption is not completed row exposure. Its communication exposure was
20,167/107,766 at step 13527, with foundation exposure still incomplete.
Mellum completed another genuine whole-graph policy update at step 21227
and resumed preference training, reaching step 21228. Its separately verified
3,486-graph teacher addition is pending the next ordinary SFT stage. Earlier
whole-graph RL, 250-update preference stages and ordinary SFT updates remain
part of each saved lineage; none completes the whole recipe.

Both teacher views passed the actual resident training consumer and remote
staging. The first proposal failed before publication because its merged
receipt lacked flat geometry coverage. The shared coverage helper now
validates the original component receipts and emits authenticated flat
coverage without reparsing or changing the original graphs. Both replacement
CPU preparations passed; the separate generation-checked submission preserves
the existing row prefix and completed visits. The supervisor is checking actual
new row visits after adoption, rather than crediting prepared data as trained.

All 48 D32 source indices now have conversion receipts, including the recovered
source-index09. Conversion alone does not establish admission, capacity,
resident adoption or exposure. Remaining admissions and staged continuations
continue independently of student SGD. The newly prepared 14-source D32
geometry proofs are complete for both models and are being integrated against
the current resident generations, retaining all whole pending rows. LFM's
24,968 auxiliary bridge rows have a streaming consumer implementation; its
full staged-data consumer proof is still running. No bridge exposure is claimed.

Ordinary training is independent of unrelated semantic-relay polling or
cross-model readiness. Its own exact checkpoint/configuration, optimizer history
and live-owner checks remain mandatory. Both new plan bindings are installed
in the current relay, which reports repair-required; the completed-adaptation
positive-release verification and shared CPU control bindings still need repair
before semantic exchange. This grants no VAE readiness. Metadata services must
not hold healthy independent SFT idle while those separate checks are repaired.

Task quality remains a real impediment: the same protected16-task packet yields
LFM11134=0/16, LFM11502=0/16, LFM12184=1/16, LFM12488=0/16 and
Mellum16364=9/16, Mellum17786=10/16. LF12488 has nine repetition failures,
two budget exhaustions and five semantic mismatches, with no infrastructure or
unresolved cases. LF12742=1/16; its step-12964 evaluation is executing.
Mellum20513=12/16 (three semantic mismatches and one repetition failure,
no infrastructure failures); its sealed step-20515 request awaits scoring.
These are historical actor measurements, not claims about current cloud weights. Generic
recovery training, source exposure and the paired history-policy investigation
are being checked without reusing held inputs or targets for training.
Complete-gradient arithmetic qualification also remains unresolved; current
reference-only results do not qualify production arithmetic or cross-model
readiness. Shared-space VAE training must still wait for genuine completed
producer/consumer graph training on both participants and its required evidence.

The dated October16 deployment and qualification history below records the
earlier transitions; its counters are not the current live counters.

Raw native-state production training is deployed on BOTH sandboxes through
shared `native_state_sft` jobs under `/work/native-production`. On October16,
LFM reached8451 and Mellum11368 before the current controlled handoff. LFM's
original owner and all five captured descendants exited successfully; its exact
full parent is independently pinned and actual-parent normal comparison is
running. Mellum has written its new full48.28GB checkpoint and is still draining
at18:47UTC. These are actual update/save counters, not whole-recipe completion
or a claim that the replacement RL controller has begun policy updates.

Mellum earlier lost250 completed updates when a held-source metadata check
failed before a save. Current held bytes match the original source identity.
The shared source repair revalidates identical bytes under stable fullSHA and
strict initial parsing; its cloud runtime/index adoption remains pending.
`cc34edb23` also preserves a completed optimizer update on held-evaluation
failure, with exact optimizer/RNG/exposure, then rethrows the original error.
Its actual interruption/resume and failed-save checks pass. Partial optimizer
failures never enter this checkpoint recovery path.

Native interface and reasoning rows have FP32 masters with the backbone's
activation casts. Exact row learning rates belong to the bound production
configuration and progress telemetry; neither token presence nor nonzero
gradients establish useful parameter updates or task quality. Actual LF7750
reasoning-row separation rules out the original frozen-QAD pairwise output
ceiling as the explanation for current failures; behavioral diagnosis continues.

Neither foundation is complete; no model is ready for shared-dialect training.
Additional admitted sources are obligations, not earned exposure. Current
resident generations and subsequent progress belong in the live coordination
status and exact checkpoint/data receipts, rather than inferred from downloads.

Whole-episode semantic RL is integrated in e517466a7, with historical optimized
message provenance in14482bfe1. Independent root review passed46 CPU cases and
19 actual CUDA/archive cases, including two-model/codec optimizer, RNG and
exposure continuation. Real source/codec/consumer adjoints stay connected.
Task correctness earns reward without requiring Neuralese; successful native
flat-reward batches use positive whole-episode RFT (62c369df5). Raw rounds no
longer require a shared VAE, and their existing provider bridge preserves exact
full-checkpoint continuation (db53d2f95). Production bootstrap, ownership-safe
SFT/RL interleave, full-size runtime/geometry qualification and actual cloud RL
updates remain operational work; these component checks do not claim task lift
or completed migration. Track their mechanisms in
`shared-state/EARLY_RAW_RL_DECOMPOSITION_20261016.md` and
`shared-state/NATIVE_OCCURRENCE_CODEC_RL_DECOMPOSITION_20261016.md`.

Current-capacity D12/D16/D18 v2 admissions retain exact full rows and protected
splits while making1,739 additional LFM training graphs fit the actual measured
92,606-physical/157,075-KV envelope. D15's complete source and fit/pending
partition are admitted; Qwen V17 adds8,309 train graphs per model and is staged
on both sandboxes. Sandbox adoption and actual visits remain separate checks.
The original pending cohorts and visit quotas must survive activation.
D12 feature-status now adds the complete1,993 factual views per model:
LFM1,488 fit/505 pending and Mellum1,613 fit/380 pending. Both are staged;
no pending row is truncated or counted as visited. Search completed-shard
admission continuously verifies source-bound receipts for both models.
Resident adoption and sequential mmap advice packages are now deployed on
both models (LFM3042af96a386; Mellum4bda74b8c99a, including the exact combined
active/pending continuation fix60ecb1505). Mellum has adopted26,907 new train
rows and169 held rows; adoption is not exposure. LFM's new94,236-physical/
157,075-total-KV certificate01386db6c736592e6d69584a39b48d3414a126d65e5949a6522ee8e8e4e07bdd
covers the longer complete case. Its75,700-train/156-held append still needs a
corrected geometry-report row offset and safe adoption after recovery.

Resident transactions now support pending-only and combined active/pending
adoption through the existing atomic commit point. Independent review found
and fixed a withdrawal-history gap before commit. Integrated resident, offline
pairwise, continuation and memory tests passed109 cases. The new reference-free
offline preference objective binds both complete alternatives with zero ordinary
CE weights and scores their assistant suffixes through the shared optimizer.
Mandatory post-joint interleave, objective revisions with preserved visit
credits, and resumable both-model source conversion/admission are integrated
(23c15cb69/3faedf374/2b7f68c60). Every admitted pair is independently re-rendered
from its original source and tokenizer; all-pair token/source proof tests pass.
The complete D24 source contains147,985 SFT sessions and12,413 alternative
pairs. Exact ZIP/JSONL content-multiset coverage is verified (62e918569).
Resumable per-export LFM→Mellum conversion and member-level incremental
admission are underway. Actual D24 admission, target pair-capacity qualification
and production-stage adoption remain unfinished; no deployed preference exposure
is claimed. Native SFT, typed durable child handoffs, contextual writes,
empty values and hierarchical views remain integrated.

Full-model raw-state references and fixed-delta tests established the deployed
channel. Subsequent prefix-sharing investigation found and repaired a separate
sparse-attention traversal-order defect in newer code. LFM prefix reuse passes
local strict checks and the actual H100 trained 6620-parent check (candidate
8fc03eb2c250):13,431→8,188 physical queries, bitwise hidden/logits/loss,
input-adjoint relativeL2 4.14e-6 under unchanged gates. Concurrent timing is not
isolated throughput evidence. Source-index continuation 399162917 was followed
by actual LF production adoption: all three source-bound MAX cases passed,
TRUE-mode certificate ff47a021…5daa binds 94,236 physical/157,075 KV, and the
retained live proof records training 7058. The October 17 LF ordinary recipe
and live command still enable prefix reuse; current certificate 596978b3…ed474
also binds that mode. The packer and GraphPlan visibility/workspace methods
remain unchanged through deployed 4ea and staged f8; f8 retains the existing
provision under its separate source-v1 arithmetic witness. No duplicate LF
flag or prefix qualification is pending. Mellum's remaining adjoint differences
are localized to BF16 fan-in rounding and nonlinear FP8 input-adjoint
quantization; its prefix reuse remains disabled. See
`shared-state/CAUSAL_PREFIX_QUALIFICATION_20261016.md`; do not transfer a
qualification between arithmetic modes. Mellum actual trained-parent3505
qualification passed with392 executedFP8projections/67,200routed rows,
BF16 input adjoints, zero forward/consumer-loss gap and adapter gradient
maximum4.66e-10 under unchanged gates; subsequent productionSGD reached3750.
This qualifies that target path, not prefix reuse or task-quality lift.
Standing fullcheck0369e9110 completed green across all11 steps, including CUDA,
TypeScript and conformance. The earlier323 check's sole stale capability fixture
was repaired, then the full check rerun. Current resident/preference integration
also passed109 focused tests,29 capacity-continuation tests,17 provider-closure
tests and the quick main check. Shared source reading and per-shard numerical
context74e47f187 passed27 focused tests plus quick; linked compressed inputs
now use the common checked reader instead of being silently skipped.

The shared semantic stage now implements VAE, joint producer/write-out/consumer
adaptation, behavioral distillation, deferred full-model optimizers, complete
per-row/per-model-pair exposure, checkpoint/RNG continuation, data append and
loss/LR/epoch changes. Two-model tiny tests prove exact interrupted continuation
and real joint gradients. The readiness controller checks actual production
stage evidence and starts with any two supported ready models; later model or
curriculum/data changes trigger continued work. Semantic adapter exports bind
the exact joint checkpoint, participant and small source tokenizers. Learned
decoding feeds the actual native cache/serving path, with explicit alignment.

File-backed checkpoint inputs (a7f7bc66b/72957ded9) preserve exact strided
layouts through repeated unpack and higher-order derivatives, with leased
stores and strict stale cleanup. Independent actual native CUDA derivative
checks passed43 cases. Actual cloud long-graph capacity and throughput remain
unqualified; the implementation does not imply pending rows have been used.

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
this does not enable production adaptedRL or claim task lift. Native adapter
application, immutable restart and heterogeneous actor sampling/replay passed
40 integrated actual CUDA/typed-HTTP tests in the independent root review on
October16; supported adapter modes are now advertised. Remaining full-size
cloud deployment, actual trained-weight multi-update adapted-RL qualification
and higher-order derived graphs stay explicit in
`shared-state/NATIVE_LEARNING_SERVICE_DECOMPOSITION_20261015.md`. Capability
denial is not completion. Independent review and actual trained-weight
training/inference/transport qualification remain required.

Pop is closed; DGX supervises all remaining work. The dedicated generation
sandbox `78530f71-08fc-4f57-ace9-7fbad64aeeb8` runs the stronger self-hosted
Qwen3.8-Flash-Next NVFP4 teacher on SM120. Current batches bind the actual NVIDIA
actor and serving runtime in their collection receipts. Mellum and
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

October 17 qualification correction: ordinary and joint publication now share
the complete communication-gradient verifier (`58ebd220f`), independently
reviewed and tested. Historical adapter-only checks remain useful channel
evidence, but cannot certify backbone or learned-row derivatives. The new check
binds the actual optimizer inventory, connected input, producer/consumer case,
arithmetic and live backend; it preserves optimizer, gradients, RNG and exposure.
Missing or failed proof prevents readiness without discarding useful training.

The actual trained LFM step-9478 checkpoint was checked on the DGX: forward and
incremental differences were below 2e-6, but 100 of 151 derivative comparisons
failed the unchanged elementwise gates. An explicit diagnostic IEEE run also
failed, so the accelerated GEMM alone is not an established cause. Controlling
the readout input removed the loss difference and reduced failing derivatives
to 37; that controlled diagnostic cannot qualify the production path. Full
reports are `lfm9478-*-diagnostic-root*-20261017.json` under the DGX operations
directory. Continue localization and repair before claiming complete-gradient
qualification or cross-model readiness; do not widen gates or infer task
regression from this diagnostic. Both models still owe complete recipe exposure.

### Source-host scheduling and current readiness (2026-10-17)

The shared source-host role mechanism now explicitly partitions raw optimizer ownership, validates each local full parent and complete communication/runtime/task evidence, and exchanges immutable bound eligibility/final-release proofs. The declared semantic consumer is Mellum for the current two-model deployment: its large parent stays local; the remote LFM checkpoint and exact consumed metadata are transferred only after actual joint-stage consumption is requested. A future third model is not a barrier to two genuinely ready participants. A single designated consumer prevents duplicate joint optimizers. Original raw cursors, optimizer/RNG/source credits, inherited recipes and mandatory ordinary deficits remain authoritative.

This is implemented and locally tested scheduling/transport, **not a cloud deployment or VAE-readiness claim**. Both actual fixed 26,312-graph communication foundations remain incomplete. Complete full-parameter communication-gradient qualification is also unresolved: the actual LFM 9478 diagnostic still fails the strict checks despite small relative errors. Legacy channel-only evidence cannot grant full-backbone readiness. No shared-space VAE training or qualification is earned by this change. Useful student SGD continues while those concrete prerequisites are repaired.

The implementation and exact mechanism/policy boundary are documented in [SOURCE_HOST_ROLE_RENDEZVOUS_DECOMPOSITION_20261017.md](shared-state/SOURCE_HOST_ROLE_RENDEZVOUS_DECOMPOSITION_20261017.md). Focused checks include the unchanged real both-port RL→SFT→RL restore/update path, full readiness rejection cases, independent portable checkpoint copying and actual semantic loading through original legacy recipe/launch/exit guards. Deployment remains with the cloud owner after independent composition review.


### October 18 production supervision

Both shared-state students continue useful work; the complete recipe and both
26,312-graph communication foundations remain unfinished. LFM resumed exact
step 13,228 after its standalone hourly refresh caused an unattended stop. Only
the automatic refresh-exit flag was removed; periodic saves and supervisor stop,
tweak and exact continuation remain available. Actual step 13,246 / GPU 95% was
observed at 07:55 UTC. Mellum completed 250 DPO updates and then 461 ordinary updates
to 21,226, published a verified full parent and inference view, and is collecting
fresh whole-graph RL trajectories on that checkpoint. These counters are not
whole-recipe completion or current-weight task qualification.

The stable R19 execution repair and actor-view restoration are implemented and
have focused checks. Current-parent complete communication-gradient, actor/cache
parity and whole-policy memory checks still precede cutover. Historical LFM 9478
FFN-boundary evidence does not replace those checks. Source-only deployment is
not successful training; no VAE readiness is earned. New ordinary data is being
prepared independently of this qualification so it need not wait for long RL
backward checks. The staged source03 bridge remains an explicit communication
consumer obligation. Whole over-limit rows remain available for measured
capacity expansion rather than clipping or permanent exclusion.

The teacher guardian now releases completed lookahead capacity while retaining
ordered source custody (fe9c4a924). Both the guardian and private publisher were
restored, with closed batches 33/34 independently fetched and verified. Completed
48-task callback measurement produced 47 strict-positive trajectories; the other
original result is rejected, not relabeled. A shared evaluator fix
(3b73b3764) resumes disk-paused workers independently of busy workers; LFM's same
protected request is retrying with all eight workers active. Captured-context
omissions now require the renderer's exact original-capture proof (794f52ab5),
allowing the independently usable 265/model batch33 decisions without waiving
its five omitted actions.

Exact D32 inventory reconciliation found a missing normalized member, index 9;
early directory labels were one-based before switching to index labels. Its
distinct source-index09 corpus is converted and has passed source protection
and sample qualification; capacity partition and registration remain pending. The 48-entry inventory map, stage receipts and
learner exposure remain separate. Live changing frontiers are in the DGX coord
status and operations receipts, not inferred from historical stage counts.


### Production follow-through (2026-10-18 13:33 UTC)

LFM has cut over to current-parent-qualified stable R19 shared execution and is
training; Mellum's exact 21,930 parent remains in retained whole-graph RL
collection/recovery before its R19 transition. LFM's communication exposure was
21,067 / 107,766 at step 14,251: neither comprehensive recipe completion nor
cross-model VAE readiness is established. Current live frontiers remain in the
DGX coordination status.

All 48 D32 sources are registered and staged on both training sandboxes,
superseding the earlier index-09 registration obligation. Registration and
staging do not establish consumer adoption or training exposure. The source03
bridge still owes actual consumer closure/adoption. New static D31 reference QA
has 4,012 ordinary whole-trajectory graphs per model, staged with explicit
geometry and unknown execution outcome; these do not earn communication credit.
Its old-consumer check exposed a real full-entry JSON-hash memory allocation
failure. Shared fix `c5cca7373` preserves the exact canonical array digest with
bounded temporary memory; explicit metadata-only runtime overlays and the
prepared consumer request are being used to finish admission.

A bounded manifest-DAG inventory of LFM's active generation
`bc070eca8347495118120c8ff761ca9bdf85f1ce993c45399eea6b85342750b9`
verified all four Search 86–89 candidate leaves already present at the exact
candidate paths and SHA-256 values. Do not append those 6,741 rows again on LFM.
This is physical source inclusion, not a new exposure claim. A separate
Mellum step-21,930 resident snapshot audit at generation
`8ce08d536928bcdbf2f09e7631f8914b1f3b79fa2d15a52bd58d507083a558b7`
also verifies all four exact Search 86–89 leaves. Neither model needs another
6,741-row append. Exact Mellum evidence is
`/home/werg/data/dgxops/root-mellum-search86-89-leaf-inventory-result-20261018.json`.
The subsequent source comparison found an actual Mellum teacher backlog:
153 + 57 + 618 + 319 graphs (1,147 total) were excluded from the newer unions
and never appended by the recorded Mellum preparer. Include these with the
other ready teacher data in the next coherent continuation, with final exact
ID/content deduplication against the current prefix; absence of matching
compressed-file hashes alone does not establish row absence. Local exact evidence is
`/home/werg/data/dgxops/root-search86-89-leaf-inventory-result-20261018.json`.

Teacher additions 1,000, 1,052 and 465 rows have actually been adopted on LFM
without resetting optimizer or previous exposure. Batch38/tranche0 adds another
824 LFM / 823 Mellum registered graphs after exact per-model deduplication;
consumer admission and measured visits remain separate obligations. Protected
quality evidence remains weak: the actual step-13,951 LFM partial oracle score
has zero successes among 13 terminal cases, with seven repetitions, two invalid
actions and four semantic mismatches; three cases remain unresolved. This does
not qualify the current model or establish improvement. Feedback is present in
the inspected requests. Preserve held separation and use distinct nonheld tasks
to improve tool-error recovery, API indexing and typed returns while useful
production training continues.
