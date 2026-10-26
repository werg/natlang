---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPortfolioEvidence
---
Construct a PortfolioEvidence draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read artifacts, reflections, and supervisor comments to identify observable work and its context. Separate the learner’s own contribution from group output, distinguish direct evidence from self-report, and preserve uncertainty about authorship or assistance.
