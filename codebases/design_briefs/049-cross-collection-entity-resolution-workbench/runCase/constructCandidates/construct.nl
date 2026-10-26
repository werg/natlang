---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIdentityCandidate
---
Construct a IdentityCandidate draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Generate candidate pairs using exact normalized keys, aliases, and bounded semantic retrieval. Preserve blocking provenance and include audit samples outside candidate blocks so recall failures are measurable rather than hidden by the candidate generator.
