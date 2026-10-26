---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRestorationObjective
---
Construct a RestorationObjective draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Convert landholder and ecological goals into observable objectives under supplied guidance. Separate desired long-term outcomes from near-term actions, identify conflicting uses, and preserve uncertainty about causal pathways rather than promising guaranteed ecological results.
