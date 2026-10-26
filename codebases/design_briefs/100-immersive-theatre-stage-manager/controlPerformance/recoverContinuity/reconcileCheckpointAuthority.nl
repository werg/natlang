---
args: { production: ImmersiveProduction; checkpoint: TheatreCheckpoint; incident: ShowIncident; showEvidence: EvidenceItem[] }
returns: CheckpointAuthority
---
1. Match the checkpoint's production and version to this run's current authoritative sources. An existing approved checkpoint must be traceable to its actual acceptance record. For first initialization, permit an empty executed-cue and knowledge ledger only with the current production's explicit start authorization and established initial knowledge; a candidate source record is not that authorization.
2. Collect active safety-stop signals from the checkpoint and this incident. Preserve every unresolved signal. A signal is cleared in this proposed interpretation only by an actual external safety-system all-clear observation naming the authorized lead, this production/run, the exact hold signal and current conditions. A written request to clear, scene approval or model judgment is insufficient.
3. Return CheckpointAuthority with sourceReady, priorCheckpointId, safetyHold, holdSignalIds, authorityReceiptIds and blockers. sourceReady is true only when initial or resumed source authority is established. safetyHold stays true while any signal is unresolved or source authority for a prior hold is uncertain. Report each absent binding precisely and keep external safety execution distinct from this semantic review.
