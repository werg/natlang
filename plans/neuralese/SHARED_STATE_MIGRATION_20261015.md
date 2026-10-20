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

- Root supervisor owns implementation, integration, training, inference, data,
  evaluation and all sandbox operations in one primary-agent strand. The former
  implementation/review/cloud/data agents have handed off and are completed.
- Future delegation is sparse and Luna-only, for engineering work. Luna never
  generates training data. The dedicated self-hosted Qwen3.8 Flash Next NVFP4
  generation sandbox continues supplying verified trajectories for both students.

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

October 20, 15:24 UTC: supervision and all implementation are consolidated in
root. Mellum R6v5 owner3247273 remains active on c8a1, with observed updates
past25955; genuine communication exposure remains incomplete. LFM's preference
interval completed at20619; its normal production controller launched the next
full-SFT continuation from that exact trained checkpoint (learner861211, observed
past20749). LFM still uses R4. Its genuine R6 runtime/controller migration and
admission of the ready additions remain outstanding; neither full recipe nor
communication exposure is complete.

Two normal live Mellum adoptions retained the complete prior prefix, held rows,
optimizer and exposure:38639 new rows committed at25780 (generation593029e7…)
and28207 more at25832 (f4d39062…). Active TRAIN now has1442983 rows, with25968
held unchanged. Both normal preparations, complete CE contracts, exact source
classifications, capacity and full learner union checks passed; no rejections.
All20 data components are physically prepared on BOTH sandboxes, with66850
LFM/66846 Mellum candidate additions. These availability counts are not LFM
admission or a claim that every added row has already received an optimizer visit.
Readiness inventory: post-r6-pending-source-additions-v6-20261020.json under
/mnt/external/natlang-development-data/teacher198-current-admission-20261019.

The actual frozen publication helper rejects valid parent-relative D24 paths.
The current shared provider helper fixes that, then exposes a second omission:
new classification witnesses consume the historical R10 parser policy, missing
from Mellum's declared readonly roots. Root verified its exact seven-file policy
(e6fce2c3…) and activated a source-only SDK successor with that explicit package
root; learner/queue/numerical package remained untouched. Current readiness:
4249eeb91eb305d3b0784defbf5c83884912621dc066b37ca6ce6b6b4487527c,
provider-mellum-q3-r6v5-policy-roots-root-20261020. The future LFM R6 provider
also has the corresponding R9 package root prepared, not activated.

Full checkpoints are saved, but failed optional publication controls are not
successful private backups. A normal fallback exporter now pins and privately
exports actual Mellum full state. Independent full readback initially passed for25803
(48.229GB,2081f108…) and25853 (48.235GB,3d6e37d8…). Actual CPU derivation
completed its bitwise-verified25803 inference view and corrected controls using
the current shared dependency producer and explicit R10 root. The existing
actor-view publisher completed independent private readback of that24.309GB
view (6fffed5b…) and updated the current quality service's followed state to25803.
Subsequent live retrieval found25803 inference missing (W&B404,size0), and the
retention journal proves the applying old29861-lineage frontier selected it
and full25853 for deletion while their independent readbacks were in flight.
Those earlier readbacks no longer establish present private recovery. Root
stopped `root-shared-wandb-retention-restored-20261017`, changed its Mellum
full/inference states to the actual current R6 pointers, added the current quality
consumer and explicit current identities, and started25803 inference re-export.
The existing full fallback is uploading25949. Keep retention stopped until
present-object custody, completed new backup and a corrected dry-run pass are
verified. Sandbox checkpoints remain available. Evidence is in
`retention-r6-current-lineage-frontier-correction-root-20261020.json` and
`wandb-storage-audit-20261016/mellum-rolling-intent-{1792508503376082882,1792509106746049741}.json`
under `/home/werg/data/dgxops`.

Local quality ingestion was also waiting for disk headroom. Verified relocation of
the4.498GB BIRD codebase_comments source database to the external disk completed,
preserving its original path with a symlink and raising root free space to5.2GB;
the existing quality service has started its next ingestion attempt. Continuous corrected in-process publication and
LFM R6 handoff are still required.

Mellum25692's16 protected tasks are fully accounted:12 successes,2 invalid
actions,1 semantic mismatch,1 repetition,0 infrastructure errors. The report is
eligible but complete=false because the repetition collector exited2. This is a
historical actor measurement, not qualification of newer weights. The redundant
24.31GB local inference payload was deleted only after the actual task unit exited,
no live references remained and private recovery was verified. Reports and full
training parent remain; external free space rose from39GB to62GB.

The full a082a8cd0 suite passed, including native GPU and TypeScript conformance.
The real R6 update profile measured29.34/75.52 seconds in pageable CPU/GPU
copies. Root found that file storage ignored the wrapper's pin_memory flag and
implemented bounded reusable pinned staging in the shared file store.55 focused
checks passed, including both native graph gradients, next optimizer updates,
RNG, higher derivatives and stride/lifetime cases. The small target-device copy
measurement in the actual offload directory had a variable1.0192x median; this
is not an end-to-end training speed claim. The composed R6v6 runtime60a26ec7…
is now physically staged and verified on BOTH sandboxes,396 transport members.
Its isolated141-check run passed: all112 loaded Neuralese modules came from
the composed runtime, including both models' complete-gradient tests. Earlier
tests whose conftest prepended main are not used as composed-runtime evidence.
Current cloud learners still use their original frozen storage code. See
[transfer decomposition](shared-state/CHECKPOINT_FILE_TRANSFER_DECOMPOSITION_20261020.md).

LFM's current-source R6v6 continuation preview passed against actual controller
602333 and learner861211: all1461160 TRAIN/25955 held rows,407+14 source indexes
and every recipe stage remain intact; no source graph reparse or checkpoint
decode. Receipt457e798c… is at
`/work/native-production/control/lfm-r4-r6-live-source-policy-preview-20261020-v7/prepared-request.json`.
The normal prepared provider successor for Q6/H9 passed the actual sandbox
dependency/guard preflight (2904 unique dependencies), remains inactive, and
preserves the R9 historical policy root. Capacity metadata reuse also passed
(28 actual case requests plus1 pending-only request,aaea9c8e…); this grants no
fresh numerical qualification or source admission. The trainer's existing SIGUSR2
normal refresh and completed-supervised transition support an early useful
quantum, so a cooperative migration need not fabricate a failed operation.
Next work is actual qualified handoff and prompt admission of66850 prepared LFM
additions, retaining optimizer/RNG/exposure. Preparation is not adoption.

Full-epoch VAE snapshot training and bound external scheduler producer fixes are
integrated. The57144-train/11278-held semantic curriculum and reviewed scheduling
assets are physically staged on BOTH sandboxes. No VAE training, completed
participant communication cohort or shared-dialect lift is claimed. Finish the
normal LFM runtime/controller handoff and automatic scheduler wiring while both
students continue through their entire recipes. VAE needs genuine completed
communication exposure on both participants.

October20, 03:45 UTC: actual observed updates reached18,861 LFM and24,889
Mellum. Mellum's historical24,575 checkpoint finished the protected16-family
task evaluation:8 successes,5 semantic mismatches and3 repetitions, with no
infrastructure errors. This is not a qualification of newer weights or a
same-runtime comparison with the older24,508 result. LFM18,619 evaluation
continues. Full recipes and communication-cohort exposure remain unfinished.

The coherent122c runtime passed120 focused composed-package checks and both
role-authority checks; its full physical mirror passed actual shipped CLI
imports. Independent source review also cleared the sealed dependency
replacement and exact CE-only index continuation, published as ade08e496 and
66528d2ee. These preserve graph entries and exposure while explicitly changing
the source-policy key. A newly found semantic held-evaluation seam must remove
the training-only CE policy before deployment; its focused regression is
passing. No new cloud-runtime adoption is claimed yet.

FP32 prefix CE readout reuse is published in f73607865 after34 CPU and4 CUDA
checks plus the integrated quick check. BF16 folding changed real derivatives,
so BF16 retains the original path with bitwise derivative agreement. This
optimization is excluded from the immediate122c bundle because its source
context depends on earlier readout changes; that dependency is not silently
imported. Useful CE/data/runtime adoption proceeds independently.

October20, 03:29 UTC: both students are taking useful updates. LFM shared4395
reached18,792 with26,312/26,312 foundation and30,574/114,794 communication
visits; Mellum R14 reached24,844 in the next offline-preference stage. Neither
has completed its entire recipe or qualified a trained shared dialect. The
Mellum handoff's missing publication-owner fields are repaired and covered by
the shared all-stage provider fix. Its full24,831 state remains independently
private PUT/GET verified. Both backup services now follow actual owner manifests;
the previous fixed Mellum export journal is retained as recovery evidence.

The shared source-bound CE mass correction is independently reviewed and
published in254da0c4b/f7154bf87, with integrated13 focused tests and quick checks
passing. It preserves raw source bindings, completed exposure and optimizer/RNG,
retains node mass, and leaves policy likelihoods and already segmented sources
unchanged. A coherent successor for BOTH runtime inventories, role authorities,
on-request foundation qualification and semantic preparation is being reviewed;
useful current training continues during preparation. New LFM KEEP contexts
have7,721 additional fully validated, staged graphs. Ready ordinary additions
are14,029 LFM/1,428 Mellum, separately from Mellum's unadopted1,761-row pending
generation. Prepared data is not trained exposure.

The local shared checksum cache contained8,526,772 disposable test identities.
Removing only those entries reduced2.286GB to113.85MB, preserving every other
complete record verbatim and in original order, with independent replay and16
actual shared-helper/digest checks. Source data and hash algorithms are unchanged.
The actual subsequent data pipeline peaked at604MiB instead of several GiB.
check-main now defaults to a disposable private cache, as recorded in AGENTS.
The full-check timer remains active; the last complete PASS is7edc9078c.

### Earlier operational record

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

### Operational continuation, 2026-10-19 21:22 UTC

Comprehensive training remains unfinished. LFM has completed its fixed26,312
communication foundation, but still owes complete exposure to its evolving
whole communication corpus. Mellum's latest observed foundation exposure is
20,731/26,312; its complete communication inventory is114,793. Neither ordinary
SFT admission nor a metadata continuation grants cross-model VAE readiness.

LFM's protected16-family checkpoint17,214 task evaluation scored0/16. Exact
checkpoint/native-row restoration is independently verified; repeated tool
actions and wrong terminal answers remain concrete behavioral defects under
investigation. Protected evaluation trajectories stay outside teacher data.
The operational task boundary has been corrected to the16-family specification.
The saved17,619 parent preserves optimizer, RNG, exposure and the genuinely
completed250-update DPO stage. Its same-stage recovery is committed, and the
new controller and provider have entered retry2. A fresh worker request is
staging; resumed SGD is not yet observed. The earlier failed request remains
failed and receives no completion or qualification credit.

Mellum is doing genuine SGD again on the qualified R10 profile, observed at
24,550. The R11 candidate passed exact actor/cache parity but ran out of GPU
memory during complete-gradient comparison before any SGD. It earns no capacity
qualification. The shared checker repair streams completed leaf gradients to
CPU and compares the cached derivatives individually; it remains under
independent review and caller validation. Keep useful R10 training running
until an explicitly deployed successor passes the actual full capacity checks.
Preserve newly earned updates when changing profiles.

The active admitted source generations contain1,423,372 LFM training rows and
1,344,558 Mellum training rows. Additional ready, verified teacher additions
total4,400 LFM and5,913 Mellum rows; preparation/adoption and actual visits are
separate events. The existing Qwen generation sandbox continues with preserved
in-flight work and the expanded state/effects/recovery curriculum. No additional
sandbox or external teacher is required. A checksum-verified relocation of
reusable source data to the external disk restored local root headroom to
25.22GiB while retaining its original logical path.


### Adaptive continuation, 2026-10-19 22:49 UTC

Both learners have genuine resumed SGD: LFM was observed at18,036 on the
corrected16-family operational task specification; Mellum at24,565 on the
qualified R10 CPU-file profile. LFM has adopted1,433,779 training rows and25,955
held rows; Mellum1,350,471 and25,968. Newly admitted sources remain distinct
from adopted sources and completed visits. Foundation and whole-communication
exposure debt, optimizer moments and RNG remain attached to their earned
parents. Neither model has earned the pair's cross-model release readiness.

R12 restored the exact Mellum24,564 parent and passed actor parity, then failed
before SGD during the cached-reference backward: token-interleaved expert
contributions kept multiple FP32 bank accumulators live until their final
shared leaf use. The flat candidate backward had already completed. The
restored R10 continuation preserves every earned update; the failed R12
initial exporter is retired and actual R10-v3 monitoring/publication is bound
to the observed owner and original qualified runtime. The24,557 historical
private full export was independently downloaded and verified.

The shared checker now places the explicit cached reference on the existing
ordered CPU-file FP32 carrier while leaving the candidate and production
backend unchanged. Both accumulator identities are recorded in complete
gradient proofs. Every declared parameter and explicit input is still compared
at the existing gates; cross-mode cached attention/FFN comparisons share this
mechanism. Local tests passed89, plus4 CPU/CUDA cross-mode backend assertions
across both architectures. Independent source review cleared the fix. A new
cloud composition and actual capacity qualification remain required; these
local checks grant no cloud qualification.

The shared task runtime now binds genuine native writer and reader ports.
A corrected public48-session measurement returned37 exact expected values and
8/8 actual state-record writes; this is not held-task improvement evidence.
Operational boundary continuations can bind the same exact built runtime to
both task evaluation and the RAW RL collector, preserving the separate Python
publication runtime. The complete runtime snapshot is prepared for deployment;
existing learners have not silently adopted it. Continuous self-hosted Qwen
generation and verified whole-trajectory admission continue.

### Production continuation, 2026-10-20 01:27 UTC

Mellum's R14 complete-gradient and capacity checks passed against the actual
24,575 parent. Production SGD resumed and reached24,708. The active continuation
has adopted1,372,949 training rows and25,968 protected held rows, including13,756
additional prepared training graphs. Its latest recorded genuine communication
exposure is20,835/114,793. Adoption does not count as exposure. The complete
communication recipe and cross-model readiness remain unfinished.

LFM earned18,619 before a synchronous end-of-quantum task evaluation delayed
handoff. Its full checkpoint preserves optimizer, RNG, per-row exposure and the
actual1,000-update quantum. The evaluation was operationally interrupted; its
request remains failed and unqualified. Exact typed same-stage recovery retains
the earned updates. The R8 successor's first start failed before checkpoint
preload because immutable capacity requests named the old row-index cache
directory. Recovery is restricted to byte-verified canonical cache aliases and
explicit preservation/retirement of that failed startup before relaunch. No
successor numerical qualification or resumed SGD is claimed yet. The prepared
source generation contains1,447,132 training rows and25,955 held rows.

Shared task collectors now honor cooperative training stops, drain their exact
children and retain the pending frontier without scoring a cancellation as a
task failure. The update-boundary acknowledgement accepts an operational stop
only after the captured trainer's declared semantic-refresh quantum was really
earned. Shared classifier validation authenticates the original historical
producer bytes across runtime paths without rewriting historical proofs. The
focused checks for these changes passed; the integrated full check is running.

The corrected-runtime LFM18,006 protected evaluation completed0/16, with14
repetition and2 budget outcomes and no unresolved infrastructure cases. This is
historical checkpoint evidence, not a score for18,619. Its exact local payload
and reports are retained for the assigned behavioral bug investigation. Cached
reference output/gradient tests and prompt-prefix checks have not explained the
failure; learned reasoning rows are finite FP32 and have actually changed.
Repeated full readiness evaluation at every refresh quantum is also being
replaced with explicit shared readiness requests; candidates must remain
unqualified until their actual current-weight checks pass.

The existing strong Qwen teacher remains busy. Its latest48-session curriculum
measurement accepted48/48 complete trajectories with exact return oracles. The
first946 converted graphs per model are admitted; the final494 per model await
the strict source-closure verifier. The next verify-return curriculum has
12/12 exact reference executions and48 queued teacher requests. Its guardian
configuration is adopted with existing batches preserved, but those future
requests have not yet been measured. No additional sandbox is allocated.

### DGX recovery and production follow-through, 2026-10-20 02:09 UTC

The Q5 LFM attempt restored the exact18,619 optimizer/RNG/exposure parent and
passed actual actor parity and the complete150-parameter/151-derivative
FFN4096 gradient check. It subsequently failed before capacity measurement or
SGD because historical geometry reports bound cache witnesses to the old
row-index directory. All388 prequalified source indexes are now authenticated
in the canonical cache. Six immutable successor geometry reports preserve the
original report contents except for byte-identical cache locations. The Q6
entry is staged; its frozen4395-runtime metadata-only gradient carry preflight
passed against the original actual proofs, preserving the genuine null
precision-transition report and requiring every capacity check afresh. The
new SDK owner is waiting behind the authenticated Q6 controller guard. No Q6
SGD or capacity completion is claimed at this point. Detailed stage, carry,
SDK, and passive-monitor receipts are under `/home/werg/data/dgxops/`.

Mellum completed its genuine256-update continuation at24,831, with an actual
terminal receipt at02:08:16. The full-recipe controller handoff is being
inspected so the next useful training stage follows promptly. The complete24,739 checkpoint was independently
private PUT/full-GET verified, preserving48,220,799,353 bytes of recovery state.
The next1,761-row source generation is pending through the sole supervisor's
CAS; pointer installation is not exposure. Both current18,619 LFM and24,575
Mellum historical held-task actors are running locally. The retained18,006
behavioral diagnostic measured bitwise-equal training-versus-cached inference
logits on the tested real sequences; this rules out that arithmetic mismatch
there, not the remaining success-target/supervision investigation.

Teacher199's full48 verified trajectories are now admitted as1,440 graphs per
student. Teacher200's first48 trajectories passed the original execution and
required-file oracles; their normal per-decision conversion and admission are
still pending. The existing generation sandbox remains supplied. Low-priority
CPU queue ordering was adjusted so completed-source conversions can run while
the larger SearchOWL acquisition waits for sufficient memory.

The on-request foundation qualification and frozen-snapshot VAE preparation
work passed135 focused integration tests, with one existing skip, plus quick
checks. Independent review of complete source-host dependency transport is
still required before adopting it in both frozen production runtimes. VAE
preparation does not grant readiness, release live optimizers, or permit joint
backbone updates before exact descendant adoption. The regular full-check timer
has been restored after yielding memory to urgent recovery and these focused
checks; the latest complete full-check PASS remains7edc9078c.

Storage cleanup removed exactly11 superseded canonical private checkpoints,
with the candidate list checked before deletion and absence verified afterward,
reclaiming210,568,386,261 bytes. Current recovery/quality/assigned-repair parents,
modern owned publications, reusable source data and held data were retained.
The rolling retention service was relaunched through the memory ledger after
its transient unit unloaded. Modern owned-publication retention still needs
extension; historical canonical step frontiers are not current training progress.

### DGX useful training and shared repair, 2026-10-20 03:03 UTC

LFM Q6 passed all28 unique fresh capacity cases (29 logical references) and
resumed genuine SGD from18,619, reaching18,661 in the current observation.
Actual learned FP32 rows changed; foundation exposure is26,312/26,312 and
communication exposure30,204/114,794. These are progress counters, not complete
recipe or task-quality qualification. The corrected waiting-controller/SDK
cache handoff is metadata-only and leaves its canonical learner untouched.

Mellum24,831 completed its256-update recovery continuation. The full
48,220,799,097-byte checkpoint passed private PUT and independent complete GET.
The controller reached the next declared offline-preference stage, but a worker
rejected its request before model allocation: the shared provider configuration
copied the publication selector while dropping four required owner fields.
Repair is preserving the same checkpoint, recipe, optimizer/RNG and exposure.
No new preference update or completed recipe is claimed. Both models still need
completed genuine communication exposure before cross-model readiness. Shared
on-request foundation checks, frozen trained-snapshot VAE preparation, drained
continuation and exact child-runtime plumbing have been reviewed and published;
adoption into the frozen cloud runtimes remains explicit work.

The loss audit found a concrete supervision inconsistency: ordinary and paired
raw context weights let long prompts dominate each normalized graph. Sampled
ordinary full sources devote only8.87%/9.44% median loss mass to the LF/Mellum
answer, versus96.2–96.3% for already segmented whole trajectories. The shared
source-bound correction is being implemented and checked against exact
full-graph losses and derivatives. It preserves node mass, immutable rows and
completed exposure; policy-RL likelihoods are outside this CE correction.

An additional7,721 LFM graphs now pass strict static validation with explicit
KEEP reasoning history. Every emitted answer token and suffix weight matches
its original admitted target; current captured-action approval, holds, global
source closure and normal deduplication pass. Existing LF/Mellum rows remain,
Mellum needs no duplicate KEEP view, and one newly rejected action is excluded.
This is admission evidence, not trained exposure or a causal task-success claim.
Strong self-hosted Qwen generation remains supplied; SearchOWL has253/256
shard/model receipts and a resource-only external-disk SQLite successor running.

Modern owned-publication retention is implemented and independently reviewed,
including reciprocal protection across old canonical and new owned aliases.
Exactly11 old canonical and19 explicitly discarded owned publications were
removed and their absence verified, reclaiming350,413,993,062 bytes altogether.
Current/recovery/assigned parents, source and held data remain protected. The
regular full-check timer is active; the last complete PASS is still7edc9078c.

### DGX source continuation and runtime closure, 2026-10-20 04:59 UTC

Both sandboxes now contain the independently reviewed shared runtime4655bb2f
and its complete on-request foundation/publication/recipe dependency closure.
Installation is not learner adoption: LFM's current4395 entry reached19,198;
Mellum's R14 preference entry published25,081. Both retain their actual earned
optimizer/RNG/exposure state. The new source continuation passed cache-only
classification and CE-policy preparation for LFM407 train/14 held leaves
(1,461,160/25,955 rows) and Mellum361/14 (1,376,137/25,968). No exposure is
credited by preparation. One genuinely unexecuted V5 action was withdrawn;
all other usable rows and protected held data remain. Later651 newly admitted
teacher graphs per model are separate additions for the next continuation.

Actual normal recipe handoff/dependency preview and current-source capacity
mapping are closing before the learner change. A historical D32 capacity row
was absent from the proposed source; its exact old declaration/source is being
localized rather than silently dropped or granted old qualification credit.
The LF SDK is genuinely running the new default worker behind a future-controller
guard; Mellum's active old SDK remains until its authenticated idle boundary.
Neither guard readiness nor copied runtime bytes mean new SGD has occurred.

The LF owned-publication follower was repaired to follow the actual Q6 writer.
Its full19,188 checkpoint has independent private PUT/full-GET verification;
Mellum's follower verified24,987. The current19,006 LF held-task evaluation is
running locally; historical18,619 measured1/16 and Mellum24,575 measured8/16
with the same runtime02b. These remain poor task scores requiring improvement,
not an assertion that declining training loss is sufficient. Source-bound CE
weighting correction is installed for the impending continuation, not yet active.

Shared source-builder eligibility now agrees with runtime retention of direct
native graphs, and checkpoint setup releases a retained CE-options object before
updates. Twenty combined focused tests passed. The full suite is running against
6e707cf8b2; the earlier0482 suite failed precisely the two setup-lifetime tests
subsequently fixed. The last complete full-suite PASS remains7edc9078c.

SearchOWL has all256 aggregate shard/model receipts; ten historical top-level
markers are being authenticated using their exact historical text dependency.
Only missing metadata is regenerated. Final merge remains subject to measured
storage, including the next Mellum compact actor and20GiB free floor. Completed
local LF16,397/18,619 evaluation transfers were removed after locked ingestion,
complete results and process-reference checks, reclaiming3,980,906,496 allocated
bytes; verified private recovery, all scores and assigned repair parents remain.
Strong self-hosted Qwen generation continues with batches71/72 supplied.


### DGX supervision update — 2026-10-20 05:46 UTC

LFM's requested normal SIGUSR2 refresh completed at **global19,368**, preserving
749 earned optimizer updates since18,619. The saved checkpoint is
`f61c34af2ca27102195e9f56d984d1231310c02a0ec1d4b95b6ed126784da9d6`.
The final task evaluation reported missing Node on PATH; that infrastructure
failure supplies no task-quality result. The replacement launch explicitly
propagates the verified Node directory through direct and future learner commands.
Earned-parent pinning and full-recipe objective/source continuation are in progress;
installation and metadata readiness are not new SGD or completed stage exposure.

The shared runtime handoff now accepts an authenticated inherited canonical
controller FD, retaining the continuous exclusive lease while normal production
locks and runtime configuration are joined before SGD. Exact FD/inode/flock and
separate-open contention checks protect the borrowed authority. This preserves
RAW cycle4, stage1, retry3 and pending task/case debt rather than resetting the
schedule. Commit47145cc90 passed55 focused tests and quick checks.

Source-bound failed-assistant-history correction is implemented through the
existing CE policy continuation for BOTH students (b9148bbf7/073428530). It checks
original TRAIN row/capture custody, exact tokenizer assets and independently
rederived failed spans, then zeros only those positive positions before balanced
CE. Original rows, source IDs, held data, completed exposure, optimizer and RNG
remain intact. Root integrated36 focused tests and quick checks passed. The
selected LFM KEEP cohort has7,721 rows,6,519 affected and1,187,434 failed-token
positions; selected V5 cohorts have9,535 raw rows each,6,504 affected and
1,058,860 LF /1,267,307 Mellum positions. These are prepared correction counts,
not active-row admission, training exposure or measured behavioral improvement.
The known one-row V5 withdrawal remains in force. Remaining source families are
being inventoried; a selected cohort does not complete whole-recipe repair.

The6e707cf8b2 full-suite check actually finished with exit0, superseding the old
7ed complete result. Later shared changes have focused/quick validation and await
the next timer-managed full check. All ten historical SearchOWL metadata
hydrations completed with original groups and zero rerender. The final merge is
running under measured storage protection with the next24.3GB Mellum compact
actor reservation plus20GiB free floor. Later source-ready teacher additions
remain explicit future continuation work, not silently credited exposure.

### DGX supervision update — 2026-10-20 06:52 UTC

LFM still has **global step 19,368** as its earned parent. Replacement Q1
failed before model allocation because pending runtime configuration paired a
new actor command with the retained factory environment. Shared fix 736da07e4
keeps the old actor command/environment/inventory together while pending, and
derives the activated triple through the normal factory. Root's 77 focused
checks passed. The external metadata helper carries that fix without changing
the frozen numerical runtime `4655bb2f…cab4`.

Q2 launched at 06:30:42 and passed actual parent restoration, normal runtime
join, bitwise actor parity and complete parameter-gradient checks. Its RAW
cursor remained unchanged and the join decoded no additional parent. It then
failed before SGD because a capacity case index did not identify the declared
row in the loaded source. All evidence is retained. The current repair compares
every case and numerical reference against the actual cache-backed source DAG,
retaining exact row ID, binding and geometry while investigating index order.
There is no capacity pass, new optimizer update or new data exposure to claim.
The same metadata check must protect both students before expensive restores.

Mellum's existing learner continues useful work: the actual 06:39 observation
was **global step 25,280**, loss 0.1071 and 1,302.6 supervised tokens/second.
Its next recipe has 1,376,137 TRAIN rows: the actual current 1,372,949 plus
3,189 additions minus the single authenticated V5 withdrawal. Historical
exposure remains recorded. The pending 1,761-row declaration is not credited
as adopted. Capacity preparation is resolving an exact pending-long source
alias before the normal refresh; the old learner has not been interrupted.
The handoff reuses the genuine completed request's shared publisher and
resident-adoption logic, preserving optimizer/RNG state and remaining recipe
obligations rather than manufacturing a completed stage.

Failed-history correction now includes prepared 21-teacher cohorts containing
104,298 rows per student, with 19,133 affected rows each. The exact active V5
cohort is 9,534 rows after withdrawal; earlier 9,535-row receipts remain only
historical authorities. The broad LF teacher mask overlaps the earlier KEEP
mask and must replace it rather than duplicate it. Shared fix ea9a4bbf0 permits
literal chat markers inside a legitimate current answer only with a clean
actual structured prefix/tools guard and exact reconstruction of the entire
original captured answer. Target tokens and weights remain untouched. The
retained real failing row qualified with 1,507 failed prefix positions; existing
manifest compatibility and root's 32 focused checks passed. Broader source
preparation and separate native-record custody work continue; these counts
remain preparation, not active training exposure or behavioral lift.

The old-runtime 19,006 held evaluation finished with 1/16 successes, 12
repetition outcomes, three budget exhaustions and zero infrastructure errors.
Its scores were ingested. It does not qualify current weights. The new
no-action recovery runtime has actual physical workspace tests and CLI checks;
its immutable inventory is `7f4e8f04…be86`. It is staged on the existing GEN
sandbox for a future 12 × 4 TRAIN teacher window. All 12 references passed.
Current generation slots keep their original runtime and first-finished sample
bindings until the explicit future sequence handoff. Held answers are excluded
from training. D09 still returned 403 at 06:45; scheduled checks continue.

The b2e4eb016 full suite actually passed; the f52ddee16 full suite is running.
Later mask and index changes have focused and quick/TypeScript validation.
Search's final merge is advancing through exact source authentication without
rerendering, under the unchanged 46.052 GB reservation. Completed 19,006 local
evaluation weights were removed after ingestion and live-reference checks,
reclaiming 2,565,996,544 allocated bytes. Its scores, assigned repair weights
from other steps, and the independently verified private 19,368 full recovery
checkpoint remain.

### DGX continuation, 2026-10-20 07:38 UTC

LFM Q3 launched at 07:10:55 from the same earned 19,368 parent (`f61c34af…`).
The source-order error was corrected by resolving exact IDs and bindings against
actual source-DAG order before parent restoration: 22 case indices changed,
without changing rows or exposure. Parent, optimizer and runtime import completed.
Five fresh capacity requests had passed by 07:32:32; the remaining cohort was
still advancing, with no new SGD credit or failure. GPU pauses between checks
were verified as ordinary transitions, not a stalled owner. The supervisor is
investigating repeated whole-index authentication as a shared startup cost.
Mellum continued R14 training, reaching actual global 25,410 at 07:30. Its normal
214-row pending-source classification passed. The actual-source-order capacity
projector is fully reviewed; cloud staging and execution remain supervisor-owned.

The next shared runtime will combine the verified failure-history objective
correction with the completed-parent transition API. Mellum will move directly
from R14 to this runtime, preserving its earned optimizer, RNG and exposure,
rather than paying for an intermediate restoration. The API must cover the
real resident publication, inherited owner lease, one preloaded checkpoint,
explicit old/new runtime identities, complete dependency closure and new actor
command/environment binding. A synthetic owned-drain receipt is not acceptable.
The default package and both role loaders must execute the same new API.

The mask implementation now authenticates original approved native-record
renders as well as teacher captures. Resumable candidate derivation is explicitly
non-admitted and independently rechecks capture custody. Recipe plans carry
all mask and decoder dependencies. Root reviewed and integrated these changes:
49 focused tests and the quick ratchets/TypeScript checks passed. The prepared
policy derivatives cover 117,035 existing TRAIN rows per model, including 3,203
native rows with 216 affected decisions on each port. These are objective
corrections, not new row exposure. Broader preparation retains its single known
literal-context custody blocker explicitly; verified source data stays usable.

GEN201 was adopted at 06:57 UTC with API/core unchanged. Matching publisher polls
completed normally, including the 550-second poll at 07:27. Reserved slots 72–74
retain their original runtime; future slot 75 starts the new runtime's 48-TRAIN
measurement. That window has not yet started and earns no measurement credit.
Search authenticated all 256 groups and is merging final outputs under the
unchanged 46.052 GB storage reservation. A further 1,287 LFM / 1,286 Mellum rows
are ready for a subsequent explicit continuation, separately tracked from the
current cutover. The 19,368 held evaluation remains in flight. Full-suite f52
failed only an outdated actor-command fixture; the fixture was corrected and
57 focused checks passed. The replacement full check at 78e7 is still running;
b2e4 remains the latest completed full-suite pass.

### 2026-10-20 08:29 UTC — mixed-source recovery and shared R4

LFM Q3 failed at 08:02 after reporting global step 19,380 because source
`78e645145c95c460ffe3f7a494cf50a64bfcc73ecc3041f43ac4d7c175ffe058`
contains segmented trajectories followed by ordinary reasoning rows. The CE
builder had inferred the whole source's objective from its first row. This was
a producer bug, not evidence against the model or the shared-state approach.
The genuine saved Q3 checkpoint is preserved read-only:
`62ec820183b8689aaaaf7a631f60dd08a5d355667aa6d1e54c6b55604be38614`
(2,569,897,967 bytes). Its saved global step will be read during the next owned
restore; reported updates after its last save do not earn saved-progress credit.

The shared consumer now dispatches each declared source row through its existing
ordinary, witnessed-pair or explicit native-mask objective. Complete policies
cover all 407 LFM and 361 Mellum active source leaves while retaining prior
coefficients, held declarations and failed-history masks. Both policies were
physically staged and hash verified at 08:16. The canonical producer now creates
complete contracts by default; a first-row observation cannot exempt later rows.
Existing frozen cloud producer scripts must not be used to create future source
continuations: use the current canonical host producer and stage its bound policy.

R4 changes exactly two runtime modules from the validated R3 package: the shared
CE consumer and the composition of the existing CE/classification continuation
verifiers. The classifier and seven parser-policy files remain unchanged. The
package code is `ea24e5ca8471e790739748e5b8818195635024fd06105fa0d731fdf96632ff1f`;
its construction is `d6f2fcea182eff34856fc4f0410f88909afe32ca1875526f54c6c6a27e518c29`.
Independent full review, both physical role loaders, 93 actual composed-runtime
checks, and quick ratchets/TypeScript passed. An initial test correctly rejected
an undeclared relocated runtime root; the fixture now explicitly declares the
actual producer checkout, with the production guard unchanged.

The sole cloud actuator is staging R4. LFM will restore its actual preserved Q3
checkpoint with fresh index-optimized capacity checks. A proposed descendant
geometry certificate was timeboxed and deferred because old optimizer-layout
witnesses were insufficient; no unverified capacity credit is carried. Its old
RAW runtime proof remains historical until a genuine current-child drain permits
the normal whole-recipe transition. Mellum continues useful SGD until the final
R4 objective continuation and completed-parent operators are ready. Neither a
finite worker publication nor staging is whole-recipe completion. No R4 SGD or
cross-model readiness is claimed yet.

Main includes the shared mixed-source fix, exact continuation composition, and
complete-by-default producer (`072d05bc4`, `68dd6b709`, `46f0d81e9`). The last
completed full-suite check is now `78e7b2d4b3` (pass); the subsequent `7d06810379`
full check is still running. Additional verified teacher data remains tracked
separately from actual training exposure, and the stronger self-hosted teacher
continues generating.

### 2026-10-20 09:23 UTC — genuine LFM recovery and Mellum cutover preparation

LFM Q4 restored the preserved Q3 checkpoint exactly once at 08:56 UTC. Its
actual saved state is global 19,369 / local 18,363; the eleven subsequently
reported but unsaved Q3 updates are explicitly lost. Optimizer, RNG and earned
exposure resume from that saved state. Actor prefill, incremental execution and
cache parity passed with zero maximum error. The complete parameter-gradient
transition passed at 09:01, followed by ordinary capacity checks (20 of 29
passed at 09:20, H100 fully occupied). These checks grant no new SGD exposure;
the supervisor has not yet confirmed an R4 optimizer update. The old RAW runtime
proof remains historical until genuine child publication enables its transition.

Both physical R4 runtimes and complete source-bound CE policies are installed.
The Mellum source-only preflight covered all 361 train and 14 held source leaves
without model allocation or graph replay. Mellum continues useful R14 SGD while
the normal completed-worker publication and R4 restore are finalized; it reached
global 25,668 at 09:18. The old finite worker must publish its genuine saved
state before provider succession and the completed-parent API can restore it.
The current frozen API requires canonical path/SHA recipe references; optional
byte annotations must not be mistaken for a different recipe identity.

Additional verified teacher data is ready for explicit resident continuation:
2,838 LFM / 2,837 Mellum rows at 09:14. Readiness and transport do not claim
training exposure. New policies must use the canonical complete-contract host
builder. The computed-result curriculum adds twelve new TRAIN tasks, each with
four self-hosted teacher samples, to the future source sequence. Reference
execution passed all twelve tasks; live measurement and admission remain pending.
The future-only handoff preserves each active batch's actual runtime, including
older batches that legitimately span the previous runtime epoch.

Main's full check at `7d06810379` passed all steps. A shared metadata optimization
now verifies each identical seven-file objective policy map once per operation
and rechecks it before granting continuation, removing repeated filesystem work
without changing intrinsic parser/classifier identities or cached row classes.
Independent review, 41 focused tests and quick checks passed. The optimization
is for future runtime continuations and does not delay or alter frozen R4.


### 2026-10-20 10:12 UTC — actual R4 SGD, data-admission repair and semantic scheduling

LFM completed all 29 capacity cases and began genuine R4 optimizer updates at
09:31 UTC. At 10:11 it reached global 19,595, with the H100 fully occupied.
Communication exposure is 33,034 / 114,794 graph visits and remains incomplete;
the earlier fixed foundation's completion is not completion of this larger
communication cohort or of the production recipe. Learned sparse rows have
measurable parameter changes, not only nonzero gradients.

Mellum's old worker finished at global 25,692 / local 24,426. Its genuine
48,219,206,457-byte saved checkpoint is
`60358645736ef502ff3cf68174db2ce44d0a5898ed041ce13b53bfa271093af2`.
Normal shared publication passed at 09:55:49; the SDK successor passed at
09:56:51. The old worker's long post-save delay was an actual legacy full
runtime-gradient qualification, followed by publication hashing, rather than
an incomplete checkpoint write. R4 avoids repeating that routine qualification
when readiness is on shared request. An explicit official Node v24.0.1
installation and the normal binder baseline passed at 10:03. The completed-parent
handover is still progressing; no Mellum R4 restore or optimizer update is
claimed in this entry. Guard failures before mutation are not cutover success.

A further 3,211 LFM / 3,210 Mellum verified teacher rows are prepared across
14 components. Attempted resident admission stopped before any pending-pointer
commit: the frozen resident rejected an additive complete CE policy, and the
trainer would otherwise have continued using its old closed-over policy. The
shared repair must validate unchanged prior coefficients/masks/held declarations,
complete appended source coverage, install the preloaded new policy with the
source transaction before another batch, and recover that policy on restart.
Actual tiny LFM and Mellum live-adoption tests match complete-source reference
parameters, optimizer, RNG and exposure exactly. Independent review identified
optional-byte and source-alias gaps, corrected and independently cleared before
deployment. The final shared fix passed 161 focused tests, including both real
model architectures; quick checks also passed. Existing
R4 SGD continues while a minimal, explicitly reviewed runtime successor is
prepared; no live function patch or new-row exposure is claimed.

The Search-Owl renderer completed at 09:24:23: PH 658,396 and WT 215,009 documents
per model, with zero merge omissions. Final rendering is not training admission.
The strong self-hosted teacher remains active; batch 76 is measuring the actual
runtime repair, and the next computed-result curriculum has not yet earned live
measurement or admission credit.

The prospective automatic role scheduler now connects frozen VAE preparation,
genuine descendant owner handoff, joint adaptation and existing full-graph
policy training. Root's independent full review found a recovery gap: an expired
handoff lease followed by useful SGD could permanently suppress requalification
for the same request. The author is implementing explicit renewed rendezvous
attempts before approval. The scheduler is not deployed, and no cross-model
readiness is inferred from code availability or an aggregate visit count.

Main `e0703038ce` is undergoing the standing full suite. The prior `2e0fac697a`
check's sole duplicate-readonly-root fixture failure was fixed in `880f331aa`
and focused checks passed. Storage cleanup removed 1,003,253,760 bytes of exact
inactive synthetic browser fixtures; future fixture cleanup is committed.
Private storage audit measured LFM 52.5 GB and Mellum 1.597 TB. Refresh discarded
checkpoint cleanup after actual Mellum cutover and dependency checks, preserving
current parents, pending exports, reusable sources and protected evaluation.


### 2026-10-20 10:54 UTC — shared contract repairs, recoverable checkpoints and storage

LFM continues genuine R4 SGD: global 19,774 at 10:48, H100 fully occupied,
communication exposure 33,795 / 114,794 visits. Its normal publication hook
failed because an explicitly declared historical row-index alias resolves into
the canonical production cache. The verifier required one identical root for
both paths. `353b56d08` preserves the declared lexical path and exact bytes/hash,
allows the resolved target only within the explicit production/read-only union,
and retains the private helper's old contract without an explicit namespace.
Independent full review, 62 dependency tests and quick checks passed.

The supervisor used the existing bound-output full-checkpoint ferry for LFM
quality telemetry while the immutable R4 hook remains unchanged. Saved global
19,639, full SHA `953c0680c4eec37275564b6511923eb27e2200c71a8c32f41e2dd98639993923`,
2,571,957,999 bytes, passed independent private readback. Its task evaluation is
pending; this is not a normal owner-publication certificate or readiness credit.

Mellum's first R4 entry stopped before decoding because `_recipe_plan` deliberately
omits the caller-supplied code inventory. The corrected ordinary preparer and
its complete metadata preview passed. A genuine subsequent single parent decode
and compact actor export passed at 10:30:58, preserving saved global 25,692 and
all optimizer/RNG/exposure. The entry then stopped before model allocation/SGD:
`_qualified_refresh_updates` expected signal-name strings, while actual
`StopSignal` receipts contain `signal`/`unix` records. `57fbaf646` validates the
actual producer format, including finite numeric times, without rewriting the
historical receipt or replaying its 611 earned updates. Actual OS-signal tests,
68 completed-API tests, independent review and quick checks passed.

The Mellum compact actor payload (24,308,806,189 bytes,
SHA `55e48d6c5c0ac6a07e74c02c258e0d0c353e280e6fbf259f0452c000cec7b8bd`)
is independently privately readback verified. Local transfer/task evaluation is
in progress; no new result or R4 optimizer update is claimed. A reviewed entry
successor can reuse this verified payload after a genuine restore, preserving
the full source's five-field stat including ctime and writing no duplicate view.

R5v3 combines the shared resident CE addition fix (`300c33f30`, `969444054`),
provider alias correction, and both reviewed completed-API corrections
(optional recipe byte annotations and real signal records) in the default and
both role bases. Package code is
`a5982611de3c3877854f99c32d7943c4a7925906f754b3a073e6a53603330a81`;
construction SHA is
`db6b160b8d34bc63eabff1740a85ee714a761d90e30214c6f9a485622573b2c2`.
All 291 composed CPU checks and four role-loader checks passed. Physical staging,
actual SDK/preflight and safe owner succession remain separate from training.
Mellum's next target is direct-owner v2 / whole-recipe controller v5. Prepared
3,211 LFM / 3,210 Mellum teacher rows remain uncredited: preserve classification
continuation laws and append promptly after the appropriate actual R5 context.

Storage dependency review cleared thirteen superseded private Mellum full
checkpoint copies. Strict dry-run equality, current verified recovery custody,
all pending exports, and required parent protections preceded deletion.
Actual reclaimed bytes: **626,761,117,093**. No source/held data, sandbox payload,
active parent or compact view was selected. Receipt:
`/home/werg/data/dgxops/mellum-owned-checkpoint-cleanup-result-successor-root-20261020.json`.
The standing retention service is active with current global-25,692 recovery.

The prospective semantic scheduler's independent review also required actual
allocation ownership: the consumer must genuinely release RAW before loading
frozen VAE while its peer keeps SGD. Further lifecycle repairs cover expired
rendezvous, operational cancellation with useful SGD, fixed saved-VAE parents,
and successful preparation consumption. Final review and coherent deployment
remain pending; code and protocol fixtures do not grant production readiness.
The full main suite for `e0703038ce` passed; `353b56d084` is running.
