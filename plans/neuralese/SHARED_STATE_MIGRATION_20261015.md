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

October 19, 17:50 UTC: LFM completed update16909 from16397. Exact full
state was pinned and drained; the whole-recipe controller entered at17:43:46.
Two launch defects were repaired: its handoff API needed the original bootstrap
receipt rather than the raw plan, and its child needed the exact R9 PYTHONPATH.
The same immutable corrected controller now runs as3890261/birth43084252;
the guarded provider is genuinely ready. The next request is queued, but a new
optimizer update after16909 has not yet been observed. No optimizer/RNG or
exposure reset was introduced. At16673 the saved exposure map proves395 of3275
new teacher graphs had actually been visited; admission is not full exposure.

Mellum preserves full parent24508 (4775b9f3…). The R9 failure at161972 tokens
was localized to signed32 expert-address overflow. R10 widens only that kernel's
address arithmetic, retaining the remaining frozen R9 module bytes. Independent
source review, actual above-2^31 GPU access and historical bitwise gradients
passed. Its new owner2446465 passed fresh full-versus-compact actor parity and
fresh complete315-gradient qualification at17:36:27. All ten capacity cases are
now running; no new SGD or inherited R9 capacity credit is claimed.

The shared current-owner checkpoint publication follower is implemented in
05e28b8a8, with41 focused tests and the quick check passing. It binds each
actual continuation command, parent, runtime and output to an immutable owner
generation, and permits a new generation's private backup while an old upload
retries. Integration of controller pointer emission, successor exporters and
current-generation quality ingestion is underway. The original optional LFM
publisher failure was a missing pair of precise readonly dependency roots,
not checkpoint corruption. Both necessary roots were verified remotely without
broadening access to their ancestor directory.

The recovered16673 full checkpoint passed private upload and independent full
readback, then local decode/restore checks. Its actual16-case held evaluation is
running; no score is claimed before accounting completes. Root-space pressure
had blocked some collectors. Pruning12 finished, integrated worktrees reclaimed
5,027,831,808 bytes and allowed the remaining cases to start. Mellum24508's
qualified compact d6bdca44… is sealed locally and being verified for evaluation.
Two closed immutable source pairs were fully verified and deduplicated, retaining
both logical paths and reclaiming3,752,579,072 external bytes. Active artifacts,
source data and protected evaluations were preserved.

The readiness gap from unset task_eval has an implemented shared boundary-only
recipe helper, awaiting the explicit live recipe continuation. Accounted task
failure is telemetry rather than a fabricated success gate. The passive outer
readiness mechanism now supports externally owned raw training and has an
actual role auto-seal integration test. On the current Mellum sandbox the latest
57,144 training/11,278 held semantic pairs, qualified LFM QAD base and built
harness runtime are staged and verified. The outer service still needs its
current role-plan binding. Genuine shared-graph foundation completion in both
models and actual joint-stage numerical/capacity checks remain prerequisites;
no VAE readiness or trained shared dialect is claimed.

D39 now admits99,997 source-replayed complete static code examples per model.
Current fit partitions contain99,979 LFM and99,975 Mellum examples, with18/22
whole oversized graphs and three invalid upstream rows/model explicitly retained
for follow-up. Both fit indexes and geometry checks passed; cloud staging and
classification are underway. This is ordinary coding SFT, without execution or
communication qualification. D20's first complete transcript yields seven
source-audited contexts/model, already registered, classified and staged. New
teacher51/tr0 adds196/model; other prepared teacher cohorts still require actual
recipe adoption. Own-sandbox Qwen NVFP4 generation continues with no OOM.

Shared optimizations beyond the address repair remain outside live R9/R10 and
need their actual runtime checks before adoption. Full main67ac289320 passed;
the later tmpfs observer assertion was repaired to check logical bytes without
requiring physical disk writes. Full81cd12c0fb has passed its Python CPU/GPU
stages and is running TypeScript checks; complete green status is not yet
claimed. The five-minute monitor is being updated to the genuine handoff owner
and exact SDK interpreter.

October 19, 15:20 UTC: replacement continuation is still in startup; neither
recovered entry has earned a new optimizer update. LFM's exact 16397 parent is
now loaded by R9 owner 3782488, with completed current-owner import and fresh
actor parity. Its preceding entry passed the complete gradient check, then
ran short of host memory during capacity preparation. We retired that paused
process only after testing two concrete fixes: release decoded predecessor
metadata immediately after optimizer/RNG restoration, and reuse source trees
already loaded by the trainer during capacity verification. Both preserve the
actual numerical checks and all prior exposure. The original sandbox remains
reachable; the unexplained guest-wide shared-memory accounting is not evidence
that reusable source files can be deleted.

Mellum owner 2379860 exited before SGD after its FFN 0-to-4096 transition failed.
Both individual settings passed all 315 cached-versus-flat derivatives, and
cross-setting forward outputs/CE were exact. Only cross-setting expert weight
derivatives failed: 34 tensors, 80 coordinates, maximum difference 0.00048828125.
A tiny actual Mellum FFN/carrier reproduction localized changed expert-row
reduction boundaries across a physical chunk seam. Repair whole-invocation
reduction provenance while keeping the FFN memory bound; do not dismiss the
failure as BF16 noise or relax its tolerance.

The supervisor authorized returning to the genuinely qualified FFN0 setting
while that optimization is repaired. A typed selector must authenticate the
original complete baseline certificate, exact parent/configuration/cohort and
reviewed unchanged numerical code; it must retain the rejected transition as
failed. Fresh actor and actual whole-source maximum capacity checks remain
required before SGD, without cutting sources or quotas. No new standalone
precision PASS may be fabricated from the failed wrapper. Approximately 1.1 TB
of writes over 3738 seconds belonged to this qualification's several backward
passes, not production updates. The shared observer now brackets actual update
I/O and bounded host/GPU memory telemetry for the next legitimate continuation.

Both sandboxes' stale source-host RAW rounds were retired under their exact
live controller leases with zero training/readiness credit. Whole-recipe work
and RAW debt remain owed. Shared joint data/classification append now admits
verified new leaves together with their evidence, preserving prior proof and
held-data prefixes, optimizer/RNG and earned row visits. R9 deliberately stays
at the reviewed `df15aa423` package (code `68060d44`); later unused handoff or
registry changes do not justify interrupting it again.

Teacher generation remains continuous on its dedicated sandbox. Its selector
already alternates qualified broad tasks (the active broad packet covers 56
families) with a recovery priority packet, rather than only the small typed
copy/extract/join curriculum. Newly admitted views remain zero exposure until
the learners actually consume them; source registration and staging are not
whole-recipe progress.

The earlier exits motivating this recovery reached their finite initial
foundation boundary and exited instead of handing back to the whole-recipe
controller. The original sandboxes remain reachable on the same boots; neither
exit is a new OOM. LFM ended at global 16397 / ordinary 15641, Mellum at
29861 / 28845. Each completed its initial 26,312-graph foundation, but actual
communication exposure is only 26,314/107,766 and 26,313/107,766 respectively.
The remaining production recipe is not complete. Source classification and all
18 LFM / 7 Mellum geometry successors passed their existing verifiers. Require
durable whole-recipe handback after entry, not another unattended finite stop.
LFM's adopted source remains `bf102d0c` (1,313,869 rows); Mellum remains
`3b7085c0` (1,234,855 rows). Its further 158 verified rows passed the actual
continuation consumer but were not adopted before its old owner exited.

Protected historical task scores are LFM16397=0/16, Mellum22016=11/16,
Mellum24464=10/16 and Mellum29861=1/16. All sixteen latest cases have classified
outcomes without infrastructure errors; these historical evaluations do not
qualify newly restored weights. Mellum's later checkpoint skips execution and
repeats invalid typed completions under the same task packet, tool interface,
sampler and model configuration. The symptom is real; its training cause is
still under investigation. Do not train on these protected traces or answers.
Resume Mellum from its protected full 24508 parent, restoring that parent's
optimizer/RNG/exposure rather than the regressed 29861 history. The full
48,739,398,749-byte parent was independently uploaded and fully read back
with SHA-256 `4775b9f3485f510be613bbc80c1bc526cb27e772697d4034d22e57d38979e03b`.
Its proximity to 24464 does not inherit 24464's task proof. Preserve 29861 for
the assigned regression investigation.

The previous Mellum successor passed exact full/view/cache forward comparison,
then failed its complete 314-parameter gradient check before any SGD. A real
FP32 reduction/BF16 midpoint reproducer and an independent actor/microbatch
counterexample localized concrete mechanisms. Shared repair `0bc25389f`
preserves invocation-local reductions, cached reverse-token convolution fan-in,
interleaved actor ordering and checkpoint execution context while still running
one layer-major graph traversal. Independent full review is clear and focused
tests pass. Earlier full-check failures have fixes for decoded options lifetime
`6be74d124`, test filesystem selection `8893c64c2`, and stale continuation/held
cadence assertions `882ab24cb`. That full check and its successor exposed test
fixture import problems, now fixed by `aa2db3741`; its full suite is running.
The early metadata release passed 76 focused tests and independent review;
deferred-owner handoff/configuration retries passed 123 focused tests after
review fixes. Integrated quick checks passed. No full-green claim yet.
Actual-parent cloud gradient/capacity qualification remains required.
See [gradient partition implementation](shared-state/GRAPH_GRADIENT_PARTITIONS_20261019.md)
and [ordered convolution](shared-state/FINITE_CONVOLUTION_COTANGENT_ORDER_20261019.md).
LFM's successor removes checkpoint-input file offload and may use the explicit
device FP32 accumulator after actual restored-model capacity proof. Complete
common runtime composition is underway for both models; import preflight
caught a missing explicitly supplied profiler observer before any learner stop.
No runtime package earns production credit merely for being staged.

A proposed 2343-row LFM continuation contained 1739 already represented rows.
The learner rejected it before commit; the pending request was withdrawn and
received zero training credit. Its replacement contains 604 genuinely new
whole rows, preserving all pending obligations and excluding over-frontier rows
without truncation. Consumer closure, publication and actual adoption are now
verified; adoption alone does not establish that all 604 rows were visited.
Shared safeguards are integrated (`236e4b130`, `bbb481a21`, `a47ca02c9`).

Native graph communication classification and explicit earned-credit migration
are integrated in `6400f4241`, independently reviewed, with 46 root continuation/
classification tests and 218 cloud integration tests passing. Native consumers
need nonempty imports and positive supervised continuation; genuine in-model
production is distinguished from external-only consumption. Only ONE of the
two admitted 797-family long graphs actually communicates. Exact historical
row visits, quotas, optimizer and RNG remain preserved; old caches require
an explicit authenticated index migration. The raw classification proofs are complete and privately staged for
1,004,737 LFM rows and 974,931 Mellum rows. The guarded combined operator is
ready; cache installation and the actual full consumer remain cloud-owned.
Neither private staging nor schema verification establishes training exposure.

The self-hosted Qwen teacher remains busy. Actual hardware is RTX PRO 6000
Blackwell with 96 GB, correcting older B200 descriptions. Its serving engine
uses the SM120 B12X linear, MoE and GDN paths with mixed ModelOpt FP4, FP8 KV
and MTP. The Node runtime passed strict physical verification and fresh 3/3
typed plus 24/24 ordinary recovery and 3/3 private recovery references.
The 198-packet source/runtime epoch is actually adopted; the publisher
recovered from a transient gateway file-read error and completed its first
matching healthy poll without a restart. Existing collectors retain their
original runtime custody. The first typed window actually completed 48/48 captured and host-accepted
trajectories, including five plaintext private producers. Conversion/admission
remains explicit; a captured success is not an admitted or trained row. At
13:02 UTC the teacher GPU was 88% busy with 14 running and 18 waiting
requests. The reviewed source-authority successor expands 195 to 198 packets
with unchanged quality and protected-split checks; typed/recovery and ordinary
backlog admission is in progress for both students. The shared guardian now verifies naturally archived
completed slots against exact original owner/start and boundary custody
(`74d374d21`); the future operator derivative is prepared, not executed merely
to change a working guardian.

Both initial foundation quotas are now complete, but neither model has
completed the whole production recipe or earned cross-model VAE readiness. Continue genuine shared-state graphs, ordinary data,
full-graph policy updates and executable existing coordinator stage handback
as their real dependencies become ready. Exact live owners, save/adoption
receipts and counters are maintained in the DGX coordination status.

## Earlier October 18 operational record

October 18, 18:15 UTC operational position: both models are training. LFM has
adopted its source03 continuation; Mellum continues useful R18 ordinary
diagnostic SGD while the shared cotangent repair awaits its actual R19 check.
Neither foundation is complete, and neither model has earned cross-model VAE
readiness. Current custody and counters are in the DGX coordination status;
the dated history below is not a live process report.

LFM reached global update 14420 / ordinary update 13664, with foundation
exposure 21,287/26,312 and communication exposure 21,287/107,766. Its latest
sampled update took 6.55 seconds. Verified teacher continuations of 1,000,
1,052, 465, 824 and 1,187 rows were actually adopted, preserving optimizer,
RNG and earlier row exposure. The resident recipe contains 1,230,694 rows.
A further 1,046 rows are prepared and staged for prompt continuation. Adoption
establishes availability to training, not completed visits to every new row.

Mellum's R18 RAW diagnostic was stopped at 14:13 UTC with zero new RAW update
credit. It had achieved six valid case closures after repairs to its inherited
wall-time limit and missing native-state archives, but still lacked actual
full-policy learner backward/capacity qualification. Completing that diagnostic
collection alone would not promptly produce SGD. All four TRAIN sessions and
the separate protected monitoring session were drained by exact ownership;
partial trajectories, authenticated evidence and recursive native artifacts
were durably retained. Unfinished RAW obligations remain explicit, and protected
monitoring data must never enter training.

The R19 ordinary process started at 14:31:43 UTC, owning the canonical lease
and one standard CPU/mmap decode of the exact 48,584,411,037-byte full parent
(SHA-256 `7b02c76f60cd30b32fc0bbba9027139d3e0a9b6e1ce9b18305a85b9ec28b32f0`).
It preserves the genuinely completed ordinary frame 20914, full optimizer/RNG,
and per-row history. The reviewed runtime combines shared rotary optimizations,
bounded-memory resident digests and an explicit one-use preloaded-checkpoint
argument to the standard restore path. Actual two-model preload tests passed;
this is not a claim that production SGD has resumed. The ordinary entry must
still pass current-parent full/view/cache comparison, the 4099-token transition
with all named gradients, and whole-row backward capacity for the complete
ordinary recipe, including its 161,172-token long row. Ordinary qualification
does not grant policy-training qualification. Fresh R19 full-graph RL must prove
its actual whole-policy backward path before its policy updates.

Mellum's coherent 8,574-row stronger-teacher union passed exact ID/content
exclusion audits, the unchanged resident consumer and remote staging. It includes
1,147 older usable teacher rows found missing from Mellum, plus the pending
2,906 rows and subsequent verified additions. Replace the pending component
once with this union; do not append its components again. A separate 1,044-row
continuation is ready. Both models already contain all four Search86–89 leaves
at exact recorded paths and hashes; do not append those 6,741 rows a second time.

All 48 D32 source indices have conversion receipts and staged artifacts. Remaining
admission, whole-row capacity and actual exposure remain separate obligations.
Source03's 24,968 LFM auxiliary bridge rows have strict geometry but still await
the actual full-prefix consumer closure. Passive profiling attributes the long
consumer scan to actual source/index parsing; missing prior-leaf cache reuse is
being localized without repeatedly restarting the owned scan. Pending long rows
retain their complete source and are not truncated to claim admission.

The static D31 reference-QA addition is prepared as 4,012 whole-trajectory graphs
per model, with complete geometry and protected-split provenance. An actual
private LFM resident consumer committed all 4,012 rows on the full prefix. Its
durable reconciliation records the duplicate-operator race explicitly rather
than inventing a successful host terminal result. No production exposure or
verified execution is credited to these unknown-outcome source trajectories.
The bounded-memory digest fix preserves the old canonical digest exactly; the
existing metadata-only runtime continuation admits the loader helper without
changing numerical training. Production request preparation continues while
Mellum restarts; it must not delay useful training on already-ready data.

Task quality remains a real impediment. Historical protected-packet scores
include LFM12742=1/16 and Mellum20513=12/16. LFM13951's newer partial oracle
had zero successes among 13 terminal cases, with seven repetitions, two invalid
actions and four semantic mismatches; three cases remained unresolved. These
are historical actor measurements, not assertions about current cloud weights.
Independent TRAIN-only stronger-teacher recovery tasks now cover explicit page
ranges, reserved bindings and typed-return repair. All 24 service reference
cases passed; the 193-packet generation sequence reached its new guardian
configuration. Publication reconciliation and 48 future self-hosted teacher
trajectories remain to be verified; no held task or answer is reused for training.

Ordinary training is independent of unrelated semantic-relay polling and
cross-model readiness. Actual parent/configuration, numerical correctness,
recoverable state and task quality checks protect production; packaging and
unrelated unfinished probes must not hold useful SGD idle. The objective remains
both models through the entire evolving recipe, with prompt data continuations
and later full-graph RL and cross-model work as their real dependencies become
ready. Shared-space VAE training still requires genuine completed producer/
consumer graph training on both participating models; ordinary text SFT alone
never grants that readiness.

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

### Production continuation (current at 2026-10-18 18:15 UTC)

Mellum's retained R18 diagnostic collection stopped at 14:13 UTC with zero RAW
updates. Its completed TRAIN and protected HELD partials, evidence and native
state artifacts were captured before stopping the owned processes. Unfinished
RAW work remains unfinished. R19 startup restored the exact full 21,930
parent, optimizer/RNG and completed ordinary frame 20,914; it included the
deduplicated 8,574-row teacher continuation, including the missing 1,147 rows.
Newly loaded data is not evidence of completed row visits.

The complete cached-gradient comparison exposed a genuine GPU OOM while
retaining all flat reference derivatives beside cached backward. Shared fix
`7f17d5546` moves exact reference derivatives and comparison pairs to CPU,
without changing returned gradients, model execution or acceptance gates.
Actual 67 focused tests passed, including both models, CUDA/BF16 reference
lifetime and whole optimizer/RNG preservation. Independent review cleared the
change. The same-parent v4 owner restored state, but its actual complete
comparison then failed numerically before SGD: 166 / 315 derivative comparisons
failed, all observed failures in BF16 backbone weights, despite bitwise-equal
forward outputs and input-embedding derivatives. The shared repair is now published through `a17a6ca3b`: FP32 parameter
cotangents accumulate across graph branches and cast once to the original
parameter dtype. Expert-bank views and requested VJP subsets retain exact
ancestry; qualification and capacity checks use the production scopes with
failure cleanup. Independent integrated CPU/CUDA gradient tests passed 88/88;
capacity/source-reuse checks passed 45/45 and quick checks passed. The actual
current-parent 315-gradient comparison, ordinary maximum-row backward and R19
SGD remain pending; focused tests do not waive those gates.
Completed actor/cache parity was reused only under the verified comparator-only
delta. Useful R18 ordinary diagnostic training has resumed from that exact
parent while the complete cotangent repair is implemented and independently
reviewed. Actual step22,965 / ordinary21,949 at 17:51 represents 1,035 earned updates;
these do not qualify R19 or finish the communication foundation. Full step22,016
is privately backed up with independent GET SHA-256
`a0c02d4ad4e57e7facc8e41b99931acae3fe827dab37ce4e5e6886d284d2b1ec`.
Its optional actor export exposed an old/new package-origin collision. A tiny
real bitwise writer fixture qualified old R18 dependencies with the explicit
`b3cf1af2` sidecar, and the cursor configuration was corrected. The actual
22,016 view was then extracted and bitwise verified without model construction;
its exact receipt was attached under the cursor observation lease, preserving
the full private receipt and the existing publisher. The 24,308,806,125-byte
view subsequently passed private independent GET verification at 16:53 UTC,
SHA-256 `094dbfe76f945e579044bc6df8d97dad347b625626a7526d371d9d075c8ba5b7`.
The 4,100-row and 4,285-row continuations were actually adopted. The resident
recipe reached 1,190,990 rows. A further 4,012 QA rows passed incremental CPU
preparation; final submission/adoption is owned by the cloud supervisor. Newly
loaded data does not establish completed per-row exposure.

LFM continues R19 training. The 384-row teacher continuation was adopted at
14,588 and followed by actual optimizer updates. Communication
exposure remains incomplete (21,500 / 107,766 at the latest recorded boundary).
The loader-only update restored the earned full 14,841 parent and passed all
three actual whole-graph backward and exact numerical-reference checks: the
94,236-token ordinary frontier, 86,285 forward / 157,075 KV communication
frontier, and authentic 137,052-token ordinary row. All 150 named gradients
were present in each check, with zero qualification optimizer updates. The
owned candidate actually adopted all 4,012 QA rows at the saved 14,841 boundary
and subsequently reached 14,969 / ordinary 14,213 at 18:15. Genuine communication
exposure reached 21,991 / 107,766, including 21,991 / 26,312 foundation rows.
QA rows are ordinary examples and do not earn communication credit. The
resident capacity authority still names the previous 94,236 / 157,075 frontier;
long-row live admission needs an explicit continuation despite the real new
backward PASS. Exact
source-bound maximum rows inspected for both students and source16 repairs are
single-node graphs with no edges; the earlier 500k policy case alone does not
qualify their ordinary capacity admission. Complete source rows remain pending
rather than clipped or silently lost. The first USR1 save request did not reach the installed Python handler,
although kernel and CPython handler mappings were correct. One exact-owner
cooperative TERM reached that same handler; full 14,841 state was saved and
the old learner exited normally before its replacement launched. Signal
delivery remains an implementation investigation, not grounds to repeat
interruptions of the new learner. The protected step14,123
task evaluation actually completed at 0/16: twelve repetitions, one invalid
action, one budget exhaustion, two semantic mismatches and no infrastructure
errors. Thirteen retained partial traces contain 179 reasoning-bearing turns
out of231, so the older absence-of-reasoning symptom cannot explain this result.
This is unresolved task-quality debt, not a behavioral improvement claim.

Source03's original private consumer check has now produced its completed
proof. The proposed scanner retirement correctly stopped before sending any
signal when it found that proof; no replacement scan was launched. The exact
24,968 bridge graphs passed current-prefix incremental preparation, then actual
learner adoption was independently confirmed at 18:15:05. Current generation
`02b0d807326e73526b368366c843d3eb8fac1dc371354ee1bca82881b4d2c4bb`
contains 1,262,856 rows, followed by actual SGD. The first pending submission
failed before pointer mutation on a preparer/learner cache-root mismatch; exact
index relocation to the learner's `/root/.cache/natlang/row-index` corrected it
without changing source rows, geometry or completed exposure. All bridge rows
are ordinary auxiliary examples, with zero communication credit. Actual
per-row visits remain tracked by the learner.
All 185 TRAIN and 14 HELD row indexes were verified under the selected cache
root; the earlier nine supposedly missing entries were semantic pairs queried
with incorrect cache keys. Its optional CPU pause guard
now persists STOP intent before signaling and lets either peer complete the
intent and resume safely. Seven actual process/fault tests and independent
review preceded deployment; both replacement peers have the verified new
implementation while preserving the original scanner, pause and thresholds.
Exact closure is recorded in
`/home/werg/data/dgxops/source03-pause-guard-upgrade-v2-result-20261018.json`.

Teacher generation continues on the dedicated self-hosted Qwen sandbox. Actual
batch42 completed the 24 new recovery cases with two trajectories each, with
48/48 source-verified exact successes and1,699 actions. The bound live48 receipt
in `plans/MODEL_FACING_CHANGES.md` closes teacher sampling debt; actual student
admission and per-row exposure remain separate. Shared
bounded SDK reads now cover both tranche payloads and worker queue/journal
prefixes (`3f5cadf6f`), with original partition, provenance, incomplete-append
and private durability checks retained. Actual 80 exporter tests and independent
review passed. Publisher-only handoff leaves the guardian, serving engine and
collectors running. Live counters and subsequent adoption evidence remain in
the DGX coordination status and bound operations receipts.

Teacher sampling controls are published in `d55d1e2b9` with 227 independently
verified focused checks. Future source declarations select their root mode and
group size; existing slots retain their exact original sampling authority.
The current live windows require three private-child authored tasks ×16 and
all twelve student-root tasks ×4. Nine literal authored roots make no model
calls and cannot count toward live48. Both windows remain pending. Qwen text
emulation cannot qualify native student state transport.
Verified teacher continuations of 308, 32 and 343 graphs per model are registered;
standard consumer/staging, actual learner adoption and exposure remain separate
receipts. The source-level and decision protections remain intact.

### Operational continuation, 2026-10-18 19:15 UTC

LFM actually adopted all683 teacher rows at global15,067, followed by genuine
SGD at15,070. Its recipe contains1,263,539 rows. These are ordinary examples;
their admission grants no completed per-row exposure or communication credit.
Another15 verified teacher rows per model from batch43/tranche3 are registered
in `34f80c47d`, passed both standard consumers and reached both sandboxes.
Their learner adoption remains separate.

Mellum saved global23,485 / ordinary22,469 and exited cleanly. The immutable
48,668,097,885-byte full parent, SHA-256
`94765f281ffcc7ccd182769badae68e83a5e97a782a82815b623d8e2efa34bc0`,
is retained under the owned cotangent boundary's v1 directory. Two attempts
failed before model decode or SGD: the preload guard counted reclaimable file
cache as unavailable, and the diagnostic fallback inspected a newer parser
attribute. The independently reviewed v2 fixes both, reuses that exact saved
parent, and performs no new save, checkpoint copy or old-learner signal.
Candidate1436959 completed its one standard preload at19:10:07 and CPU metadata
preparation at19:10:12, then failed before model construction, gates or SGD:
the helper importer assumed a `bytes` field in a valid path/SHA-only reference.
The same-parent R18 fallback1440157 is restoring while the entry is repaired.
All administrative imports, parser and geometry checks are being moved before
the expensive parent decode, with an explicit two-phase metadata contract and
no invented decode receipt. Actual315 named gradients, fresh actor parity,
capacity and subsequent SGD remain required. The cursor exporter is paused
during this critical path and must be rebound to the actual new owner and
specification; any next attempt must preserve newly earned fallback updates.

The full `a20f51abb8` check completed with six neural failures and all other ten
sections passing. Failures cover BF16 sparse-row inventory, functional model
VJP, an adapted-RL gradient observer, and optimizer/reference parity. Shared
repairs and exact-reference diagnosis are active; tolerances are not relaxed.
Production uses FP32 learned rows, but the counterfactual/public APIs still
require repair. The earlier full `c23968eeba` pass is not a pass for this code.

The all48-part D32 audit found25 admitted fit components absent as exact leaves
from both current recipes. They contain38,526 LFM /39,155 Mellum candidate rows
before complete-prefix duplicate checks. Fit-only continuation preparation is
active; the remaining11,021 /9,807 whole long rows stay explicitly tracked.
Missing manifest identity alone does not prove missing row IDs or exposure.

The source-host role relay's repeated strict rejection was localized to stale
pending-proof `native_raw_rounds` hashes. Both current helpers have identical
mechanisms and contracts. Existing `withdraw` reissued only pending LF revision8
and Mellum revision14 under exact pointer checks and unchanged unreleased raw
state. Its natural poll returned `waiting-for-semantic-consumption` at19:10:49.
No eligible/released certificate was reinterpreted; neither model gains readiness.
Exact audit and renewal receipts are in
`/home/werg/data/dgxops/native-role-relay-authority-root-audit-20261018.json` and
`/home/werg/data/dgxops/native-role-pending-authority-renewal-root-20261018.json`.

### Operational continuation, 2026-10-18 19:46 UTC

Both students are again doing actual SGD. Mellum's healthy R18 diagnostic
fallback owns the saved continuation after the R19 helper-reference failure;
no R19 or RAW qualification is claimed. Administrative imports and parser checks
now complete before any expensive parent decode. The cursor exporter is bound
to the actual fallback owner and output, preserving the existing writer profile.
Any next candidate must use the fallback's newly earned checkpoint.

LFM suffered a global OOM at19:09:45. Its last valid full checkpoint was actually
written at18:07:19, correcting an earlier timestamp conversion to17:27.
The hourly save appears to have become due immediately before failure; a missed
checkpoint timer is not established. Save assembly/serialization memory is under
investigation. The recovered learner restored global14,952 with exact optimizer,
RNG and saved exposure, and resumed actual SGD at14,955. The180 observed unsaved
updates are lost and receive no training credit. The existing complete137,052
forward/157,075 KV capacity proof was applied explicitly without changing the
numerical runtime or claiming new long-row exposure.

All50 exact standard-parser indexes for the25 missing fit components are built
and staged. LFM's standard private preparation accepted38,526 rows at19:45:02,
proposing1,302,065 total rows while leaving current/pending/held state unchanged.
Learner adoption remains a separate event. Mellum's coherent39,838-row successor
combines39,155 fit rows with683 verified teacher rows. Whole-prefix row identity
and binding checks remain active; cross-ID content disjointness is not inferred
from absent manifest identities. The all48-part long-row audit retains11,021 LFM
and9,807 Mellum rows, with maxima138,300 and161,972. Existing LFM capacity covers
11,019 geometrically; its two larger whole rows remain explicit pending work.

The completed protected LFM14,841 task score is0/16: ten repetitions, three
budget exhaustions, one invalid output and two semantic failures, with no
infrastructure failure. LFM14,952 evaluation is running. Mellum22,016's exact
private actor transfer completed and entered the evaluation queue at19:31:57.
None of these held trajectories enter teacher generation or fit data.

The public cotangent fixes are integrated locally and the independent full-file
focused suite is running. It has exposed a remaining failure; publication and
the next full check wait for diagnosis rather than reusing an earlier pass.

### Operational continuation, 2026-10-18 20:15 UTC

Both fit continuations are actually admitted and followed by SGD: LFM38,526
at global14,982 (1,302,065 total rows) and Mellum39,838 (1,234,840 total rows,
including683 teacher examples). No per-new-row visit is inferred from admission.
Both cursor exporters now follow their actual recovered owners. Mellum remains
on its useful diagnostic numerical profile until the R19 gates really pass.

The public model-VJP defect is repaired through an original-parameter hook that
verifies the exact shared FP32 reference cotangent before authorizing the VJP.
Independent scope/derivative checks passed47 with two architecture skips; the
separate optimizer/sparse/carrier/higher-order remainder passed105. Quick22 and
TypeScript passed. The next full suite must assess this integrated code.

A real LFM14,952 metadata-only save measurement reached15.50GiB high water,
excluding models/source indexes, and produced a1,135,003,485-byte pickle buffer.
The old sandbox has less than5GiB global headroom. Real resumed-training
weak-reference tests exposed the retention cause: `locals()` created a cached
frame dictionary that retained setup objects after fast-local deletion.
The shared trainer now uses explicitly initialized checkpoint references and
releases predecessor/restored-frame aliases after setup. Both rolling and
objective-continuation lifetime regressions pass; the complete continuation/SFT
focused suite passed66. The fix preserves owned snapshots, exact history,
optimizer/RNG, numerical operations and checkpoint format. Its deployment uses
an explicitly reviewed metadata-only entry composition; no live code injection
or successful final save is assumed. Recovery must record any unsaved loss.
See [the decomposition](shared-state/checkpoint-memory/DECOMPOSITION.md).
