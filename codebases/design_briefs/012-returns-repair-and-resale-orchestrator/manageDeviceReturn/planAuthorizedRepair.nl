---
description: Plan repair only from approved procedure, exact compatibility and workshop capacity receipts.
args:{scope:GoalContract;policy:PolicySelection;disposition:StageResultReturnDisposition;compatibility:PartCompatibilityReceipt[];capacity:WorkshopCapacityReceipt[]}
returns:StageResultRepairPlan
---
Bind feasibility = `checkRepairPrerequisites(disposition, compatibility, capacity)`. Match part ID and revision to the exact device model. Require an approved procedure within handling restrictions and a current accepted slot covering its tasks. Keep unverified substitutions unresolved and costs beyond delegated authority as alternate plans for approval. Never provide improvised hazardous repair steps.
