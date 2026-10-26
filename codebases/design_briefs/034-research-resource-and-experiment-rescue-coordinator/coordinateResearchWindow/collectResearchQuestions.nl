---
args: { mandates: ProtocolMandate[], resources: ResourceAssessment[], assignments: AssignmentSearch, readiness: ReadinessDecision[] }
returns: Question[]
---
Ask the protocol owner only for missing authority or an interval change; ask the resource owner for absent qualification/calibration evidence; ask the investigator for a measured-behavior deviation or go decision. Deduplicate only questions requesting the same fact from the same authorized role. Keep any question whose answer changes a scientific condition separate from routine calendar clarification.
