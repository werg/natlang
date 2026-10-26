---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftBehaviorObservation
---
Construct a BehaviorObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Form bounded sandbox probes using approved capabilities and representative inputs. Record environment and release identity, compare observed outcomes with the documented promise, and avoid treating one failed environment as proof of a product defect.
