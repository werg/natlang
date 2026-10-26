---
args:{goal:GoalContract;task:OnboardingTask;evidence:EvidencePage[]}
returns:ProgressFinding
---
Check this task against its own outcome IDs, prerequisite status and explicit completion contract. Accept completion only when the supplied evidence demonstrates the requested observable behavior in the intended environment. Preserve attempted-but-unverified work and residual steps; do not infer operator capability from attendance or migration intent. Return the exact evidence refs and remaining work for this task.
