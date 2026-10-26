---
args: { book: HandoverAcceptanceBook, context: HandoverAcceptanceContext, receipt: HandoverAcceptanceReceipt }
returns: HandoverAcceptanceReview
---
1. If receipt.receiptId is already in book.consumedReceiptIds, return the unchanged book as "replayed" only if the stored receipt matches every field. A reused ID with different contents is held as an identity conflict; never replace the original receipt.
2. Bind assessment = assessReceiptCoverage(book, context, receipt). Use its exact deliverable coverage, same-edition history, exception, and authority findings; do not reconstruct those judgments from an overall readiness label.
3. Append every new actual receipt once, including held receipts, and consume its receiptId. Add the ID to validReceiptIds only when assessment.disposition is not "held". Increment book.revision once and preserve all other cases and editions.
4. Return "accepted" only for assessment.disposition "complete"; return "partial" for a valid incomplete handover and "held" for a failed binding or authority check. Carry assessment.missingDeliverableIds and exceptionItems into nextTasks. Acceptance records dossier receipt only; it does not certify asset operation or close temporary arrangements.
