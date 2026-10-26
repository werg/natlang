---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHarvestCommitment
---
Construct a HarvestCommitment draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile delivery agreements and customer preferences to identify quantity, quality, and timing requirements. Separate permitted substitutions from assumptions and retain dependencies such as processing or storage windows that constrain otherwise feasible harvest choices.
