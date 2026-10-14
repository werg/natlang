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
