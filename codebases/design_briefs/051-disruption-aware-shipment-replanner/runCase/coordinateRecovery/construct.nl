---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftShippingRecovery
---
Construct a ShippingRecovery draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Sequence approved bookings and communications, reconcile carrier acknowledgments, and track actual shipment progress against revised promises. Replan only affected legs when new events arrive and explain residual uncertainty rather than claiming recovery after booking alone.
