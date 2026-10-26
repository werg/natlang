---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCompetencyFinding
---
Construct a CompetencyFinding draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Evaluate breadth, recency, independence, and consistency under the supplied assessment standard. Prevent double-counting one artifact as multiple independent demonstrations and preserve contradictory evidence rather than averaging it away.
