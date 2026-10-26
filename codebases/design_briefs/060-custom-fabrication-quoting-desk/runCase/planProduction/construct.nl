---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFabricationRoute
---
Construct a FabricationRoute draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Translate each concept into a production route with material preparation, operations, finishing, assembly, and inspection. Identify outsourced steps and dependencies, reserve realistic capacity, and retain specialist approval requirements for consequential structural or safety questions.
