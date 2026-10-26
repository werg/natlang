---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMaintenanceProcedureFit
---
Construct a MaintenanceProcedureFit draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Match the asset variant and observed condition to approved procedures, checking revision, qualification, and required isolation or access conditions. Escalate unsupported cases rather than inventing maintenance instructions from general knowledge.
