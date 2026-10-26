---
args: {"submission": "ExpenseSubmission", "prior": "ExpenseCaseFile"}
returns: "AppealRevision"
---
Compare receipt identity, claimant, transaction date, merchant reference and prior reimbursement identity. Call identifyNewAppealEvidence(submission.receiptFolders, prior) to distinguish new facts from a repeat upload or changed explanation. Reconsider only the components those new facts could change, preserving unrelated prior decisions. A duplicate with no new evidence remains linked to the existing operation; do not create a new payable identity.
