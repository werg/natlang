---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIncidentImpact
---
Construct a IncidentImpact draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Combine alerts, user reports, and service telemetry to describe observed impact and uncertainty. Separate symptoms from causes and distinguish missing telemetry from healthy behavior; identify the service outcomes that recovery must actually restore.
