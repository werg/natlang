---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReuseSiteFact
---
Construct a ReuseSiteFact draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile inventories, surveys, historical plans, and stakeholder descriptions into verified facts and unknowns. Distinguish current observations from outdated drawings and preserve uncertainties that could materially change feasibility or cost.
