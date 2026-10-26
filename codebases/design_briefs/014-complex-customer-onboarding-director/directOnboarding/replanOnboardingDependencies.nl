---
description: Revise only affected task descendants when dependencies or customer scope change.
args:{goal:GoalContract;plan:OnboardingTaskPlan;progress:TaskProgressLedger;evidence:EvidencePage[]}
returns:OnboardingChangeProposal
---
Identify changed assumptions and affected descendants from explicit task dependencies. Preserve completed accepted tasks and their evidence. Offer feasible scope/timing alternatives with effort consequences; do not rewrite an outcome or contract date without customer authority. Require exact approval for a material change and replay end-to-end acceptance after the revised path.
