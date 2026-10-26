---
args: { receipt: MaintenanceTaskReceipt, plan: ShiftHandoverProposal, partChanges: PartRevisionChange[] }
returns: InstalledPartRevisionCheck
---
1. If no part change applies to receipt.taskId, accept the part condition only when the receipt reports no unapproved installed substitution.
2. For an applicable part change, require receipt.installedPartId and installedRevision to match that task's offeredPart and offeredRevision. Require compatibilityEvidence and ensure the task is not in plan.rejectedPartChanges.
3. A missing or mismatched revision is not compatible because it fits the equipment family; preserve the installed-part evidence and return the exact correction or qualified review needed.
