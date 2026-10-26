---
description: Reconcile saved episode identity with the current authoritative world before interpreting a new action.
args: { world: PersistentWorldState; checkpoint: GameCheckpoint; sourceRecord: GameWorld; sourceEvidence: EvidenceItem[]; readback: WorldSimulatorReadback | null }
returns: CheckpointReconciliation
---
Require checkpoint.worldId and sourceRecord.worldId to match world.id, and the ruleSetId in the source record to match the world's declared rule version. A checkpoint ahead of the supplied world blocks. If its revision is behind, advance only when readback names the exact world, revision, ruleVersion and full snapshot supplied here; preserve pending/unknown action IDs until their own receipt is reconciled. A rule version change requires current source evidence naming the approved migration and a checkpoint boundary with no unknown receipt; otherwise hold and preserve the old checkpoint. Return ready only when the checkpoint can safely resume this same episode without discarding an outstanding effect.
