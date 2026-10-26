---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFabricationRequirement
---
Construct a FabricationRequirement draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret sketches, descriptions, and intended use to distinguish functional requirements from aesthetic preferences and unstated assumptions. Identify consequential missing dimensions or loads and request clarification rather than inventing specifications that determine manufacturability or safety.
