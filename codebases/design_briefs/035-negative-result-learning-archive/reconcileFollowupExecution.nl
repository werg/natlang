---
args: { ledger: FollowupLearningLedger, prediction: FollowupPrediction, predictionRef: Ref, receipt: FollowupExecutionReceipt }
returns: FollowupReconciliation
---
1. If receipt.receiptId was consumed, return the unchanged ledger and stored outcome as replayed only when the complete receipt matches. Reused IDs with changed data are quarantined as an identity conflict.
2. Require receipt.predictionRef to equal predictionRef, the explanation identity to match, and receipt.changedVariable to match the one declared change. A mismatch is retained as uncertain evidence and cannot update any rival lesson.
3. Bind outcome = classifyObservedFollowup(prediction, predictionRef, receipt). Preserve invalid, incomplete, unknown-measurement, valid, and uncertain results separately; only a valid result that favors one predeclared prediction may open a scoped lesson-review task.
4. Append the actual receipt and its classified outcome once. Increment the ledger revision once, preserve prior attempts and contradictory evidence, and return lessonUpdateAllowed only for a valid directional result. This flag authorizes review, not automatic lesson publication or deletion of earlier versions.
