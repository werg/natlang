---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHarvestPlan
---
Construct a HarvestPlan draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Use exact scheduling and capacity services to coordinate labor, equipment, harvest windows, storage, and collection. Construct contingency plans for uncertain readiness and avoid assigning the same resource to overlapping lots or ignoring transport bottlenecks.
