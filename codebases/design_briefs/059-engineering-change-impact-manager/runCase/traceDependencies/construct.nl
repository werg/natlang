---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftEngineeringImpact
---
Construct a EngineeringImpact draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Traverse bills of material, interfaces, process instructions, and supplier commitments through exact dependency graphs. Judge whether each indirect dependency is materially affected and retain uncertain impacts for engineering review instead of broad automatic replacement.
