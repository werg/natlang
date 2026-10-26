---
description: Replan a live event under a disruption while preserving promises, access and staff revision control.
args:{ sourceRecord: LiveEvent; sourceEvidence: EvidenceItem[];event:EventProductionState;newReports:Disruption[];checkpoint:LiveCheckpoint}
returns:ChangePacket
---
Read `eventId`, `agendaIds`, `venueId`, `producerId` and `commitmentIds` from the top level of the LiveEvent record. Base schedule and disruption decisions on these exact promises and responsible parties, not on an inferred event template. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

If `checkpoint.emergencyActive` or a new report identifies an emergency, return the approved emergency-command contact and suspend all generated replanning. Otherwise bind commitments = `recoverEventPromises(event, sourceRecord, sourceEvidence)`, preserving public, contractual, accessibility and internal authority classes. Bind failure = `interpretCapabilityLoss(event, newReports)`; confirmed loss and estimated delay must remain separate.

Bind alternatives = `constructRecoveryFormats(event, commitments, failure)`. Keep hard access/safety requirements and essential outcomes in every option; disclose consent and contract impacts. Bind runPlan = `rebuildEventRunSheet(event, commitments, alternatives, checkpoint)`. Require exact venue close, setup order, resource capacity and the last feasible switch deadline. A failed resource proof blocks announcement.

Bind acknowledgements = `coordinateRoleUpdates(event, runPlan, checkpoint)`. Only announce after producer approval, reservations and role-specific acknowledgements for the same revision. A conflicting or pending critical-role acknowledgement holds the change. During execution, apply only actual supplied reports and preserve prior planned revisions in the checkpoint.
