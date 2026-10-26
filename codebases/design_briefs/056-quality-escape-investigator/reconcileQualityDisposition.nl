---
args: { book: QualityDispositionBook, proposal: QualityEscapeProposal, receipt: QualityLotDispositionReceipt }
returns: QualityLotReconciliation
---
1. For a consumed receiptId compare every field with stored evidence. Exact replay returns { book, outcome: stored physical outcome, containmentTasks: stored unresolved containment tasks }. Conflicting or unbound identity returns { book, outcome: "unknown", containmentTasks: ["reconcile receipt identity and physical unit status"] }; neither branch mutates the book.
2. Bind review = qualifyLotRelease(proposal, receipt). A hold may be recorded from actual physical control evidence even when causal analysis is incomplete. A release requires every containment and effectiveness condition in the review.
3. Record observed disposition, never requested disposition. A matching evidenced hold sets only the supplied units to held. A properly qualified release sets only covered units to released. Partial and unknown physical outcomes remain unresolved.
4. If the system reports release while review.safeToRelease is false, record released-unverified and open urgent containment tasks for the exact units; do not hide the physical release or claim it was authorized.
5. Append each new actual receipt once, update only its lot and units, increment revision once, and preserve prior lot and escape evidence. Bind containmentTasks to review.reasons and any released-unverified urgent containment. Bind result = { book: updatedBook, outcome: actual evidenced disposition, containmentTasks }; return result. Release of containment does not by itself close the corrective action or quality escape.
