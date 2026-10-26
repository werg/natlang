---
args: { receipt: MaintenanceTaskReceipt, plan: ShiftHandoverProposal, tasks: MaintenanceTask[], completedTaskIds: ID[] }
returns: PlannedTaskSequenceCheck
---
Find exactly one task with receipt.taskId. It must appear in plan.orderedTasks and its assetId and receipt.officialAssetId must both equal plan.officialAssetId. Every task prerequisiteId must be in completedTaskIds. A task listed as incomplete remains eligible only if its prerequisites are complete; do not infer predecessor completion from schedule order. Return the precise missing prerequisite IDs and mismatched identity reasons.
