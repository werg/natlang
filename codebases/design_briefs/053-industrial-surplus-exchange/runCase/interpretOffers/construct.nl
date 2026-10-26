---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftSurplusOffer
---
Construct a SurplusOffer draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Translate surplus descriptions into functional properties, quantities, condition, and known limitations. Distinguish verified specifications from seller assumptions, preserve ownership and handling restrictions, and identify information needed before proposing a consequential reuse.
