---
args: { ledger: ResearchReservationLedger, plan: ResearchRescuePlan, receipt: ResearchReservationReceipt }
returns: ReservationReceiptReview
---
1. If receipt.receiptId is already consumed, return the unchanged ledger as replayed only when every field matches the stored receipt. A reused ID with different contents is held as a receipt-identity conflict.
2. Require receipt.caseId to equal plan.caseId and require actual evidenceRefs. For a reserved, rejected, or unknown response, bind match = verifyReceiptAgainstAssignment(receipt, plan.slotProposal.assignments). If match is false, preserve the receipt as held evidence without counting it against a different experiment or slot.
3. For a cancellation response, require a prior reserved receipt with the same reservationId, experimentId, instrumentId, and interval. A cancellation with no matching active reservation, or after a terminal cancellation, remains held for operator reconciliation.
4. Append the exact new external receipt once. A rejection leaves the experiment unscheduled; an unknown response leaves reservation status unknown and blocks duplicate booking; a cancellation releases only the evidenced reservation. None of these outcomes changes readiness or proves experiment execution.
5. Increment ledger.revision once for each newly recorded receipt, append receipt.receiptId once to consumedReceiptIds, preserve prior receipts, and return the resulting ledger. A reserved receipt is a confirmed booking only; readiness must still be rechecked immediately before the experiment.
