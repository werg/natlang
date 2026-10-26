---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMaintenanceProgress
---
Construct a MaintenanceProgress draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare technician notes and system events with task completion criteria. Distinguish work attempted, work completed, and verification passed; preserve changed conditions and unexpected findings that alter the remaining plan.
