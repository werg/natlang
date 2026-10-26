---
args: {"portfolio": "HandoverPortfolio", "reply": "SuccessorAcknowledgment"}
returns: "TransferReplyReview"
---
Verify the reply identity, current portfolio revision, exact slice, signer and demonstrated successor authority from actual reply evidence. A stale reply returns the changed slice and leaves the portfolio untouched. For an established partial acceptance bind partitions = partitionAcknowledgedScope(slice, reply), where slice is the matching current OwnershipSlice. Replace that slice with its accepted and unowned/declined partitions, preserving original promise attribution. Confirm a checkpoint only when the reply actually confirms it. Identical repeated reply IDs are no-ops; conflicting reuse remains unresolved.
