---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHarvestMatch
---
Construct a HarvestMatch draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare lot characteristics and readiness with customer requirements, identifying compatible combinations and gaps. Preserve uncertain quality findings and request targeted verification rather than allocating a lot solely because its crop name matches.
