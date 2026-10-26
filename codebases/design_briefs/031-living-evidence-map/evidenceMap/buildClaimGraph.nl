---
args: { request: EvidenceMapRequest, observations: ObservationLedger, measures: MeasurementAlignment, lineage: StudyLineage }
returns: EvidenceClaimGraph
---
For each review proposition, trace supporting and opposing study IDs through report versions to source spans. Distinguish consistency, applicability and confidence rationale; do not turn study count into certainty or omit negative findings because they are inconvenient.
