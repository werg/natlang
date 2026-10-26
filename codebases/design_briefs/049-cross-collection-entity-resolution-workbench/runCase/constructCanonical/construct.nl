---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCanonicalEntity
---
Construct a CanonicalEntity draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Assemble a canonical view without overwriting source records or selecting one source globally for every field. Resolve attributes using field-specific authority and effective time, preserve aliases and conflicting claims, and separate historical from current values.
