---
args: { book: MigrationCutoverBook, proposal: DomainMigrationProposal, receipt: MigrationCutoverReceipt }
returns: MigrationCutoverAssessment
---
1. Require receipt.migrationId to equal proposal.migrationId, receipt.frozenMappingVersion to equal proposal.cutover.frozenMappingVersion, receipt.priorTargetVersion to equal book.currentTargetVersion, and receipt.evidenceRefs to contain actual cutover evidence.
2. A "switched" receipt may advance only when proposal.cutover.switchAllowed is true, the rehearsal passed, receipt.unaccountedSourceIds is empty, receipt.appliedSourceIds cover every source disposition accepted for migration, receipt.replayedDeltaRefs cover the proposal's replayed deltas, and receipt.preservedTargetWriteIds cover every target write the reconciliation plan requires to preserve.
3. A "rolled-back" receipt is accepted only when rollbackEvidenceRefs establish restoration of receipt.priorTargetVersion. Partial or unknown writes never authorize switch or retry against a presumed prior state.
4. Return every uncovered source ID, delta, target write, or missing evidence as a blocker. A checked rehearsal is not evidence that production cutover happened.
