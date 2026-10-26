---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDonationOffer
---
Construct a DonationOffer draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret offers and condition evidence, distinguishing available items from tentative promises. Record quantities, collection constraints, and any permitted-use restrictions; request missing information that changes suitability rather than accepting vague offers into inventory.
