---
args: { request: PolicyScenarioRequest, simulations: PolicySimulation[] }
returns: GoalContract
---
Bind exactly one baseline policy, candidate policy, case collection snapshot, owner, and analysis purpose. Preserve the simulation record version and case-set identity. Results from different versions or cohorts cannot be merged into this comparison.
