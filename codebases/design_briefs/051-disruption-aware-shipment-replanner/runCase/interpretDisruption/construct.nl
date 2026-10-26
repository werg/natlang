---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDisruptionImpact
---
Construct a DisruptionImpact draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read carrier notices and operational reports to determine what is unavailable, for how long, and with what certainty. Distinguish a delayed service from a closed route and preserve exceptions that may allow some shipment classes to continue.
