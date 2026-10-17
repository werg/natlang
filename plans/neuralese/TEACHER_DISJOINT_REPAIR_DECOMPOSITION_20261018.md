# Disjoint teacher repair batches

The natural-language policy is: keep the teacher supplied with useful admitted
TRAIN repair work, preserve each valid running trajectory, and start work from a
different sealed repair batch when an explicitly configured slot is available.
Never steal unfinished keys from an active worker's snapshotted queue. Original
failed trajectories stay unchanged; repairs retain independent attempt provenance.

1. Authenticate every sealed repair queue and recover each current attempt's
   process token and birth identity. This is exact process/custody plumbing.
2. Preserve all live owners, including the legacy single-owner state. A smaller
   configured parallelism drains naturally and does not cancel work.
3. For available slots, select the existing ordered disjoint batch queues and
   use the existing single-attempt launcher with one batch-specific lock.
4. Publish a recoverable owner map while preserving the legacy owner field for
   existing readers. Recover token/receipt crash windows before spawning again.

The existing serial policy remains the default. `repair_parallel_batches` is an
explicit operational capacity setting, initially two on the generation sandbox.
Primary generation, source-sequence selection, oracle/split checks, teacher
identity, runtime and original queues are unchanged. No executing model sees a
new prompt or error. Focused tests cover live legacy preservation, disjoint
launching, recovery, decreased capacity, and immutable queue rejection. Deployment
requires independent review and authenticated guardian-only replacement; no model
server or primary feeder restart.
