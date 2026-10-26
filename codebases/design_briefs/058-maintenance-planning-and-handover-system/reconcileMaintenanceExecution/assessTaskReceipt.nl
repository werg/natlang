---
args: { receipt: MaintenanceTaskReceipt, plan: ShiftHandoverProposal, tasks: MaintenanceTask[], partChanges: PartRevisionChange[], completedTaskIds: ID[] }
returns: MaintenanceReceiptAssessment
---
1. Bind sequence = verifyTaskPrerequisites(receipt, plan, tasks, completedTaskIds). A task is eligible for completion evidence only when it is in the ordered plan, its official asset matches, and every prerequisite is already evidenced complete.
2. Bind part = qualifyInstalledPartRevision(receipt, plan, partChanges). A proposed substitution is accepted only when the installed part and revision match a reviewed compatible change and the substitution is not rejected.
3. Compare receipt.verifiedChecks with every required check of its unique planned task and require verificationRefs. Return each missing prerequisite, required verification, and part defect. This assessment never creates a completion receipt.
