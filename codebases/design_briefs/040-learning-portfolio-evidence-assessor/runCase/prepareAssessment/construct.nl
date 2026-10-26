---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPortfolioDisposition
---
Construct a PortfolioDisposition draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Assemble a reviewable recommendation with evidence, gaps, and proposed next work. Allow the learner to correct factual attribution, preserve assessor disagreements, and record the authorized decision separately from the program’s recommendation.
