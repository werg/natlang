---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRepairDiagnostic
---
Construct a RepairDiagnostic draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Choose a bounded diagnostic task from supplied safe procedures that distinguishes plausible issues. Specify required qualifications, tools, and stop conditions; route work beyond cooperative capability to an appropriate specialist rather than improvising.
