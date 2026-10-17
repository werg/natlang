# CoreWeave external-service mechanism

## Continuous teacher batch closure (2026-10-16)

`finishTeacherBatch(queue, journal, workerReturncodes, score)` returns a collection
continuation receipt. Its declared policy is: keep the self-hosted stronger
teacher supplied after accounted model failures; stop for missing collection
results, broken transport or unbound errors. Preserve true outcomes and never
turn collection completion into training admission.

1. Match every distinct queue key to the collector's latest finish accounting.
2. Accept the existing exact-result verifier's successful closure, regardless
   of whether that model result passed its task.
3. For model-request exhaustion, verify the actual typed error, saved error SHA,
   job identity and source-bound manifest. Explicitly report legacy errors that
   lack per-error source digests. They remain failures and are not admitted.
4. Require score cardinality and recognized worker exits. Publish repair keys
   for every unresolved case; advance only when every key is accounted for.

The natural-language policy above is the operator's continuous-generation
decision. `teacher_batch_completion.py` is its crisp verifier: SHA checks,
identity joins and accounting arithmetic. It changes no model-facing wording,
collector outcome, reward, training data or teacher allocation.

Placement policy is the existing `applications/ops-placement/placeJob.nl`
program, with its declared crisp alternative. Job declarations contain the
policy's resource and storage decisions. This directory implements the external
CoreWeave service and exact transfer, rather than a second placement policy.

| Unit | Representation | Reason |
| --- | --- | --- |
| Placement | Existing typed natlang function | Decide where useful work runs. |
| Job validation and SDK translation | Crisp `job.py` | Verify declared resources and encode the vendor API exactly. |
| Sandbox lifecycle | Crisp `cwctl.py` | External API calls and durable identifiers. |
| Inputs, checkpoints and receipts | Existing crisp bundle/bootstrap | Hash verification and exact durability. |
| Root filesystem versus scratch volume | Job declaration `scratch.storage` | Explicit external-service option, encoded by `job.py`; no implicit policy fallback. |

2026-10-14: GPU scratch-volume requests remain creating in controlled retests;
the otherwise identical request without a volume runs CUDA training and
checkpoint reload. A declaration may select `scratch.storage: "root"` to use
the existing work path on the sandbox's ephemeral root filesystem. `volume`
remains the default for existing jobs. Root storage has no reserved size or
snapshot guarantee; the declared size is a planning estimate, and existing
verified output collection remains necessary before stopping the sandbox.

Whole-chain CPU placement uses an explicit declared assignment of original
trajectory IDs. The existing native pair builder verifies that excluded chains
belong to the pinned source, have not already completed locally, and cannot be
silently reclaimed on resume. Its receipt keeps these external assignments
separate from converted rows and omissions; finishing a partition is not whole
source completion. Exact raw-chain extraction and the existing source archive
provide the remote worker's bytes and parent-membership evidence.

## Native participant bridge (2026-10-15)

The existing native recipe and `rl/native_rounds.py` decide which participant
trains and when the joint semantic refresh runs. They publish exact requests;
this service does not decide training policy or allocate resources.

| Unit | Representation | Reason |
| --- | --- | --- |
| Participant/refresh scheduling | Existing native recipe and rounds controller | Training policy remains in the declared shared pipeline. |
| Queue envelope and request/result binding | Existing `rl/native_bridge.py` | One exact file protocol, including bound stop requests. |
| Provider execution | Crisp CoreWeave provider bridge | External SDK, detached process ownership and exact-PID signal forwarding. |
| Cross-sandbox artifacts | Existing private scoped artifact transport | Credentials remain on DGX; source and target bytes must match the declared SHA/size, with results exposed last. |
| Recovery | Durable per-request provider receipts | Reattach to the same owned process; never duplicate a model worker after a controller interruption. |

Authenticated request stops are exact provider plumbing: check the request-bound
stop before resolving dependencies, between staged files, and before launch.
An already stopped request must not load an actor merely to terminate it. For
an existing actor, retain exact process ownership and drain its closed artifacts.

The provider validates the declared resource against an explicit existing-ID
allowlist, stages bound inputs at their unchanged absolute namespaces, starts
one detached worker, and records its PID plus process start time. A stop request
for that exact request sends TERM to that exact worker and continues collecting
its checkpoint/result. The bridge fetches all inventoried artifacts and verifies
their hashes before exposing the final result to the coordinator. No account
credential, allocation decision or model-generation policy is sent to a sandbox.

Consumed input closure is a crisp verifier (`provider_dependencies.py`). It
reuses the native recipe source-manifest resolver and native checkpoint bundle
resolver, including optimizer and RNG sidecars. It follows the actual complete
mandatory-SFT seed manifest and its raw pair inputs. Public base inventories and
built task-runtime inventories are compared on the destination before launch;
existence alone is insufficient. A portable resolved production recipe avoids
silently missing inherited configuration files. Failure leaves the request
unlaunched for repair, rather than falling back to top-level inputs only.

The same closure also follows incomplete carried CE-stage source receipts,
foundation manifests and real task packets. The old qualified semantic cohort
and every bound external native-state block survive a pending new cohort.
Built task runtimes bind their dist plus npm manifests and run the actual CLI
import preflight. A provider never replaces a published runtime file through
its alias: matching code must already be deployed through the Git archive path.
The same inventory binds the external `prelude.js` and frozen three-file source
review registry consumed during module import. These dependencies were exposed
by the real two-sandbox fixture; an existing dist directory alone is insufficient.

Large metadata programs use an owned temporary file with an exact SHA check
before execution, then remove it. This is SDK argument-size plumbing: a full
compiled-runtime inventory can exceed Linux's single-argument limit even when
the metadata itself is small. Remote failures retain bounded stdout/stderr in
the private host receipt directory, never in public logs or source bundles.

Small immutable artifact inventories are transported as one existing TAR+zstd
source archive. The real SDK fixture exposed roughly forty minutes of per-file
RPC latency for 1.4 MB. This is crisp transport plumbing, not a new scheduling
policy: large reusable checkpoint objects retain the existing private ferry.
The source archive verifier binds every member; atomic restore rejects escaped
paths and links and verifies each destination byte before the result commit.

## Resumable private evaluation exports

The receipt decides which already-verified private artifact is fetched. This
fetcher only transports those declared bytes; it does not choose or qualify an
evaluation checkpoint.

| Unit | Representation | Reason |
| --- | --- | --- |
| Export selection and expected identity | Existing owner-side receipt | The exact size and SHA are already independently verified upstream. |
| Private artifact lookup and authenticated HTTP stream | Crisp `fetch_native_eval_export.py` | W&B API/credentials and HTTP Range are external-service plumbing. |
| Partial progress | `<destination>.pending` plus recomputed prefix SHA | A retry can continue at the exact durable byte offset without a second state format. |
| Resume acceptance and publication | Crisp exact verifier | Check HTTP status, exact `Content-Range`, total length and final SHA before atomic rename. |

An existing destination is reused only when its complete receipt-bound SHA and
size match. A complete valid pending file is published without another API
request. A partial pending file is hashed from byte zero before requesting the
remaining range. A server that ignores Range with HTTP 200 causes a safe
truncate-and-rewrite from the beginning; a malformed 206 is rejected without
changing the pending prefix. A complete artifact is exposed only after its
receipt-bound byte count and SHA match, followed by parent-directory fsync.

## Participant-sharded shared checkpoint storage

The existing recipe still chooses and serializes participant updates, shared
adaptation and publication. One joint optimizer history remains authoritative
on Mellum. The following is storage and exact-consumption plumbing, not a second
training protocol.

1. Extend `save_shared` to commit immutable participant payloads containing each
   model, its complete optimizer/FP8 state, global step and mandatory CE frames;
   commit separate VAE/specification and global RNG/training payloads. Publish
   one small shared manifest last, binding every payload by SHA and size.
2. Use that actual manifest's SHA as the existing global checkpoint identity
   for both participants and their semantic adapter. Do not rewrite an identity
   to pretend a smaller selected checkpoint has the full parent's bytes.
3. Extend the existing shared loader to restore only the requested participant
   plus global training/RNG metadata. Extend `SemanticChannel` to read the VAE
   payload from the same manifest. The other participant's tensors are not a
   consumed inference dependency.
4. Extend existing native bundle/composite dependency enumeration: complete
   recovery exports contain all payloads; an explicitly selected worker closure
   contains its participant, global continuation state, VAE and tokenizers.
   Composite parents already carry a participant selector. Keep all saved base
   inventory, optimizer, row-credit and semantic identity checks.
5. Preserve `native_refresh`'s exact global-parent continuation check. Return
   updated participant artifacts to the joint owner before any next publication;
   drain workers before another stage can update their optimizer stream.
6. Qualify with actual two-port restored weights, all optimizer tensors, RNG,
   exposure frames, semantic outputs and a subsequent update. Delete an unused
   peer payload on the selected worker to prove that it is not opened; complete
   recovery on the coordinator must still require every payload.

Current blockers to simply filtering transfers: `native_composite` recursively
requires the monolithic parent, `SemanticChannel.load` opens that same full file
for VAE tensors, and adapter/refresh checks bind both models to its SHA. All
three must consume the same new storage form together. Loader changes are
coordinated with the reasoning-row repair owner before implementation.


Shared storage pins use the existing native bundle inventory and verifier. A
file lock serializes atomic shared commits with full-closure inode pins; only
committed global manifests and immutable SHA-named payloads are hardlinked.
The rolling/best/final manifests retain their payloads, while explicit live
pins retain every consumed payload inode in their own portable object root.
After a successful save, superseded object paths are pruned; active pins remain
recoverable even after those original paths disappear. Releasing a pin belongs
to the existing export/ferry owner after verified consumption. Failure before
the global commit cleans only newly written, uncommitted objects; failure after
the atomic commit preserves its complete recoverable closure. A manifest-only
inode export is insufficient and must not be used for this storage form.

Shared periodic-export preparation uses `shared_export.py`: pin the COMPLETE
committed global closure through `pin_shared_checkpoint`, seal that pin with the
existing source archive/roundtrip verifier, and pass the archive binding to the
existing scoped private PUT/full-GET exporter. Release only that assigned pin
after the private receipt matches archive SHA and size in both upload identity
and independent GET. This exact durability plumbing grants no task qualification,
selects no data and schedules no training. Current monolithic student exporters
remain separate; a future shared schedule must invoke this preparation rather
than upload its small global manifest inode.
Each independent durability/ferry consumer uses its own named closure pin scope,
so releasing a completed private export cannot remove a simultaneous full-state
ferry's pin. The scope changes storage ownership only; the global checkpoint SHA
and participant/VAE/optimizer/RNG payload identities are unchanged.
Packing and release additionally share a dedicated archive-lifecycle lock, separate
from the training checkpoint lock. A same-consumer release waits until packing
has finished; removing the verified pin is serialized with commit/pin/prune.
The actual race test holds packing open, proves release blocks, and independently
acquires the training commit lock so archive compression cannot stall a save.

## Superseded inference-view retention (2026-10-17)

The standing owner policy retains current recovery and active consumers, then
removes superseded artifacts. Extend the existing checkpoint retention unit:

1. Verify the current full backup and independently fetched inference view.
2. Protect the current view, explicit dependencies, active native-async requests
   and incoming export receipts (including a transfer not yet queued).
3. Protect those requests' full training parents as well as their physical views.
4. Select only older canonical inference filenames in the declared same run.
5. Write the existing exact deletion intent, delete, and verify absence.

Artifact identity, queue/request custody, strict name matching and deletion are
crisp durability/external-service mechanisms. No model policy or prompt changes.
An invalid dependency fails the run closed; absent optional view state prunes
no inference objects. Existing full-checkpoint retention remains the default.

## Cohort exposure during existing checkpoint custody (2026-10-17)

The owner declares immutable source cohorts. During the cursor exporter's existing safe full checkpoint decode, a read-only observer joins those exact row bindings against the matching ordinary stage's saved credits and quotas. It distinguishes unadmitted rows from admitted rows with zero visits, and records completed/required/remaining visits without constructing a model or decoding the checkpoint again. Source loading reuses the qualified runtime's unchanged graph/index verifier; only the small declared cohorts are loaded. Observation failures retain the full backup and an assigned diagnostic failure. This is exact reporting plumbing, not training/readiness policy.

### Cursor observation RPC recovery (2026-10-17)

A completed source-host checkpoint observation must survive loss of its RPC
response. The crisp durability layer saves the exact completed result under a
request-source digest, with the immutable pin's dev/inode/size/mtime/ctime identity. Empty observations are never cached; later checkpoints must still be discovered. A retry reads that
result and verifies the unchanged pin before any checkpoint copy or model-state
decode. Full private upload and independent GET remain the existing separate
acknowledgement gate. Optional actor-view work gets a longer asynchronous RPC
window; this change adds no tensor/model mutation or alternate reader.
