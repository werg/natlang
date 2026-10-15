# CoreWeave external-service mechanism

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
