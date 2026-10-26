---
args: { ledger: FacilityVisitLedger, caseId: ID, receipt: FacilityVisitReceipt, plannedLocationId: ID, plannedIssueIds: ID[], requiredCheckIds: ID[], authorizedVisit: VisitAccessPermission | null, consentStatusAtAttempt: "valid" | "expired" | "unknown" }
returns: FacilityVisitReconciliation
---
1. If receipt.receiptId already appears in ledger.consumedReceiptIds, return the unchanged ledger as "replayed" only when the stored receipt matches every field. Reuse with different contents is an identity conflict; hold it and do not overwrite history.
2. Bind assessment = interpretVisitEvidence(caseId, receipt, plannedLocationId, plannedIssueIds, requiredCheckIds, authorizedVisit, consentStatusAtAttempt). Let its binding and resolution consume the exact work, access, verification, and residual-symptom findings.
3. Append every newly observed receipt exactly once, including held, partial, and unknown outcomes; add its ID once to consumedReceiptIds and increment the ledger revision once. Preserve all earlier attempts.
4. Return the assessment's resolution, missingCheckIds, reasons, and nextActions. A duplicate receipt never adds another visit; a fresh consented window is required after failed access; recurring work remains open until the technical and occupant or objective evidence supports closure.
