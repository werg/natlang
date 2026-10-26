---
description: Measure onboarding progress against task-level completion contracts.
args:{goal:GoalContract;plan:OnboardingTaskPlan;evidence:EvidencePage[]}
returns:TaskProgressLedger
---
Match system events and participant updates to exact task IDs and completion evidence. Separate attempt, partial result, reconciled migration, attendance and demonstrated capability. Request only the missing verification artifact. A free-text percentage cannot close a task. Preserve residual work and the exact evidence revision.
