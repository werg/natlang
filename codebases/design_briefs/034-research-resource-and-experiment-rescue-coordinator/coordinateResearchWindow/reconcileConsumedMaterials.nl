---
args: { snapshot: ResearchOperationsSnapshot, assignments: AssignmentSearch }
returns: ConsumptionRecord[]
---
1. Bind trace = tracePreparedMaterialUse(snapshot, assignments), joining preparation and inventory by experiment plus material identity.
2. Bind impact = identifyDependentExperiments(trace, snapshot). Carry a downstream ID only when an explicit prerequisite edge connects it to the cancelled experiment.
3. Build one ConsumptionRecord per evidenced material amount. Mark reuse only when its handling conditions establish it; append the cancellation event without deleting the prior booking.
