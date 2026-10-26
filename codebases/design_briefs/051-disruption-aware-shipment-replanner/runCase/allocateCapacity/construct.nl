---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftShippingAllocation
---
Construct a ShippingAllocation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Submit candidate plans to exact capacity and timing services, then compare feasible allocations under supplied priorities. Reserve selected capacity before promising it, retain unfulfilled demand explicitly, and detect conflicts across shipments sharing scarce resources.
