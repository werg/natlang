---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMaintenanceWorkOrder
---
Construct a MaintenanceWorkOrder draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Decompose the approved procedure into accountable tasks, resource reservations, and verification steps. Sequence shared access and parts availability, preserve stop conditions, and distinguish planned work from authorized readiness to begin.
