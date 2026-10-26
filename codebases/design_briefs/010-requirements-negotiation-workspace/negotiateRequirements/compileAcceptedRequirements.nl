---
description: Turn confirmed negotiated outcomes into atomic, phase-bound requirement records.
args:{scope:GoalContract;positions:StageResultStakeholderPosition;compromises:StageResultCompromiseOption;agreements:EvidencePage[]}
returns:StageResultRequirement
---
Bind clauses = `atomizeRequirementBehavior(compromises.accepted, agreements)`. Include original positions only where their current authority and confirmation permit. Each requirement states observable condition, behavior, failure behavior, owner, release phase and traceable source. Keep disputed or unconfirmed clauses outside the committed baseline; reconcile dependencies and return all omissions as unresolved.
