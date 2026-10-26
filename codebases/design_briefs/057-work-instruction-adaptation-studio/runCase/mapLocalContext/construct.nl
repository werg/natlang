---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftLocalProcedureConstraint
---
Construct a LocalProcedureConstraint draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare local equipment, materials, staffing, and terminology with the master procedure. Separate harmless naming differences from capability changes and identify where local conditions make a step impossible or require engineering review.
