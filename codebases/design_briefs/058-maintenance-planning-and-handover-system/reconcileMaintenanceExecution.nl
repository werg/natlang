---
args: { book: MaintenanceExecutionBook, caseId: ID, plan: ShiftHandoverProposal, tasks: MaintenanceTask[], partChanges: PartRevisionChange[], receipt: MaintenanceTaskReceipt }
returns: MaintenanceExecutionReconciliation
---
1. If receipt.receiptId is already consumed, return the unchanged book as replayed only if the stored receipt matches every field. Reused IDs with different contents are held as an identity conflict.
2. Bind assessment = assessTaskReceipt(receipt, plan, tasks, partChanges, book.completedTaskIds). Preserve every actual receipt event, including partial, not-done, and unknown outcomes. A receipt from another case or a nonmatching task is held and cannot complete work.
3. Add receipt.taskId to completedTaskIds only when receipt.outcome is "completed", assessment.matchesPlannedTask and assessment.verificationComplete and assessment.partCompatible are all true. All other outcomes remain unresolved; never infer completion from an end-of-shift report alone.
4. If plan.isolationState is null, report isolationDisposition "not-applicable". Otherwise keep it "retained" unless receipt.isolationStateAtHandover matches the live state, every named handover check is present in receipt.isolationHandoverChecks, and isolationEvidenceRefs are present. This records a verified shift handoff only; it never de-isolates equipment.
5. Compute remainingTaskIds from plan.incompleteTaskIds plus planned tasks not yet completed in the book. Append the receipt exactly once, add its ID once to consumedReceiptIds, increment book.revision once, and preserve all prior task and lock history. Return follow-up for each missing required check, rejected part, incomplete task, or isolation handover.
