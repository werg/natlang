---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftBehavioralRequirement
---
Construct a BehavioralRequirement draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret prose requirements, bug reports, and relevant discussion to identify observable behavior, preconditions, and exclusions. Separate implementation suggestions from requirements and preserve ambiguous expectations for owner resolution before encoding them as tests.
