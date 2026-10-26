---
args: { evidence: Evidence[], prior: IncidentImpact }
returns: TelemetryCoverageBounds
---
Compare the observed service and cohort references across current evidence. For every affected function lacking an observation, retain unknown coverage and name the exact missing check. Do not infer health from an absent alert.
