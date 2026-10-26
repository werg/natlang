---
args: { impact: ObservedIncidentImpact, evidence: Evidence[] }
returns: HypothesisEvidenceMatrix
---
For each symptom, deployment, and dependency observation record which candidate mechanisms it supports and which it contradicts. Keep evidence with ambiguous causal relevance in a separate column rather than forcing it to favor the most recent change.
