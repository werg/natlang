---
args: { book: MigrationCutoverBook, proposal: DomainMigrationProposal, receipt: MigrationCutoverReceipt }
returns: MigrationCutoverReview
---
1. A consumed receiptId replays the old outcome unchanged only when every field of the new receipt matches the stored receipt. A conflicting use of an ID is held.
2. Bind assessment = verifyCutoverReceipt(book, proposal, receipt). A stale migration ID, mapping version, target version, or missing target-write protection remains held; preserve the actual event but do not move currentTargetVersion.
3. For a switched receipt, set currentTargetVersion to observedTargetVersion only when assessment.safeToAdvance is true. For partial or unknown work, set currentTargetVersion to null until the migration host supplies an authoritative observed version; block further writes while it is null.
4. For rolled-back, restore currentTargetVersion to the prior target only when assessment confirms the rollback evidence. Never mark a migration complete from a command response without the applied record and write reconciliation evidence.
5. Append every newly observed receipt once, increment the ledger revision once, and retain source/target conflicts and unresolved IDs. Return the actual outcome and focused record, mapping, or target-write follow-up.
