---
description: Build discriminating acceptance cases from atomic requirement behavior.
args:{scope:GoalContract;requirements:Requirement[]}
returns:AcceptanceScenario[]
---
For each requirement derive ordinary, boundary and adverse initial state/action/outcome scenarios. Ensure the condition is testable and traces to the requirement ID; omit scenarios that only encode a particular implementation preference.
