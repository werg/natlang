---
args: { book: HandoverAcceptanceBook, context: HandoverAcceptanceContext, receipt: HandoverAcceptanceReceipt }
returns: HandoverAcceptanceAssessment
---
1. Treat the receipt as valid coverage only for the exact caseId and currentDossierVersion. Require the receivingRole to be permitted and signedEvidence to support a real receiving party; unknown or mismatched authority holds the receipt.
2. Every acceptedDeliverableId must belong to requiredDeliverableIds. An unknown deliverable ID is not coverage. A listed exception remains a handover gap even when the receiver signed for other artifacts.
3. Combine this receipt with prior receipts only when their IDs appear in book.validReceiptIds and their caseId and dossierVersion match this context. Held receipts and receipts for older editions do not cover current requirements.
4. Return missingDeliverableIds from the exact required set, unexpected IDs, declared exceptions, and reasons. Mark complete only when context.readyForAcceptance is true, no required deliverable is missing, no exception remains, and the receipt authority is valid; mark partial for a valid but incomplete handover; mark held for any identity, version, scope, or authority mismatch.
