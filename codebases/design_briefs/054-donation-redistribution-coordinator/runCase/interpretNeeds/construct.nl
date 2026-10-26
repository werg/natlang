---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRecipientNeed
---
Construct a RecipientNeed draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Recover recipient needs from descriptions and prior confirmations without inferring sensitive circumstances. Distinguish necessary features from preferences, record accessibility and timing constraints explicitly supplied by recipients, and preserve consent for contact or delivery.
