---
args: { incident: IncidentRecoveryRequest }
returns: ObservedIncidentImpact
---
Join alerts, user reports, and telemetry by observed time and affected function. Separate a symptom from any claimed cause. Mark functions and cohorts without telemetry as unknown rather than healthy. State the exact observation needed to establish user-visible recovery.
Bind bounds = identifyTelemetryBlindSpots(incident.incidentEvidence, incident.priorImpact). Attach each blind spot to the cohorts and functions the current records do not observe.
