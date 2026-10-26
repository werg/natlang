---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMaintenanceHandover
---
Construct a MaintenanceHandover draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Prepare a shift handover describing actual asset state, work in progress, unresolved findings, and required next checks. Require acknowledgment from the receiving owner and preserve the distinction between maintenance completion and authorized return to service.
