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
