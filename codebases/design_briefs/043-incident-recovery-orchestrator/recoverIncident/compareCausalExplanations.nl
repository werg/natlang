---
args: { impact: ObservedIncidentImpact, incident: IncidentRecoveryRequest }
returns: IncidentCauseSet
---
Construct competing mechanisms from deployment, dependency, and symptom evidence. For each hypothesis record evidence for, evidence against, an observation that distinguishes it, and the result that would change the response. Keep every cause provisional until a discriminating check returns.
Bind candidateEvidence = separateEvidenceByHypothesis(impact, incident.incidentEvidence). Exclude checks that would produce the same result under every candidate cause.
