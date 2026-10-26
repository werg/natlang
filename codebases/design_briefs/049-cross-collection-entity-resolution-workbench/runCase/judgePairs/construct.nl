---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftIdentityJudgment
---
Construct a IdentityJudgment draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare names, attributes, chronology, and context while distinguishing identity from affiliation or succession. Treat missing attributes as unknown rather than matching, and record both positive support and hard cannot-link evidence.
