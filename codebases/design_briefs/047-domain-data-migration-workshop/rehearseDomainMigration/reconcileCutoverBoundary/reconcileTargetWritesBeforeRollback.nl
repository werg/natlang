---
args: { targetWrites: Evidence[], policy: string[] }
returns: TargetWriteReconciliationPlan
---
For each target-side write after cutover compare its source identity, mapping version, and rollback visibility. Preserve both the current target value and its source lineage. Require an explicit merge, forward-fix, or manual resolution for conflicts; never overwrite new target work during rollback.
