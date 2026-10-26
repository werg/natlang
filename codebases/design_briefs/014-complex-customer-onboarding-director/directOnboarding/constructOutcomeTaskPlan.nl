---
description: Build an acyclic task graph whose paths deliver each authorized outcome.
args:{goal:GoalContract;policy:PolicySelection;outcomes:OnboardingOutcome[];environment:EnvironmentAssessment;evidence:EvidencePage[]}
returns:OnboardingTaskPlan
---
Decompose accepted outcomes into configuration, data, training and acceptance tasks. Name one owner and completion contract for each task. Merge shared prerequisites once but retain every dependent outcome edge. Put privacy-approved representative validation and reversible pilot before bulk import. Keep tasks blocked when interface, security, owner or authority evidence is absent.
