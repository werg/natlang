---
args: { design: EffectiveDesignTimeline[], schemas: EvidencePage[], missingness: EvidencePage[] }
returns: MetricMeasurementAudit[]
---
For each metric that spans an effective plan version or observation window, compare source-backed event definitions, timestamp semantics and schema versions. For each affected window call compareMeasurementWindow(metric, window, schemas, missingness). Use each actual missingness review separately from event-definition comparison. Mark a window invalid when treatment changes the measurement process or assignment/exposure denominator; mark limited when the supplied evidence bounds but does not resolve the shift. Return per-metric audits with affected windows, child window reviews, evidence IDs and exact diagnostics needed. Matching event names alone never establish comparability.
