---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReuseReceipt
---
Construct a ReuseReceipt draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Coordinate approved transfers and verify quantity, condition, ownership, and recipient acceptance at handoff. Record rejected or partially accepted material and update future matching evidence without falsely reporting a completed reuse benefit.
