---
args: { ledger: FloorCopyLedger, adaptation: InstructionAdaptationCase, release: InstructionReleasePacket, receipt: FloorCopyReceipt }
returns: FloorCopyReconciliation
---
1. If receiptId is consumed, compare every field to stored receipt. Exact replay returns { ledger, disposition: "replayed", activeOldCopyIds: old copies still active in ledger, followUp: stored search or posting tasks }. Conflicting or unbound identity returns { ledger, disposition: "held", activeOldCopyIds: old copies still active in ledger, followUp: ["reconcile receipt identity and floor copy location"] }; neither branch mutates the ledger.
2. Bind assessment = assessCopyDisposition(adaptation, release, ledger, receipt). Record the actual search/collection event even when it does not authorize a state change.
3. Mark the old copy withdrawn only when assessment.oldCopyWithdrawn is true. Add replacementCopy only when assessment.replacementMayBeAdded is true. Preserve any unavailable or unknown copy as active until physically withdrawn.
4. Append receipt once, increment revision once, bind activeOldCopyIds to old controlled copies not witnessed withdrawn, and bind followUp to assessment.reasons plus remaining search/posting tasks. Bind result = { ledger: updatedLedger, disposition: assessment.valid ? "recorded" : "held", activeOldCopyIds, followUp }; return result. A released document does not replace the controlled floor copy until receipt evidence confirms it.
