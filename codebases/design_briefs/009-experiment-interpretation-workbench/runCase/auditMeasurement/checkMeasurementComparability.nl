---
args: { design: EffectiveDesignTimeline[], schemas: EvidencePage[], missingness: EvidencePage[] }
returns: MetricMeasurementAudit[]
---
Compare the event definition and timestamp semantics across each effective plan version and observation window. Determine whether treatment could change the measurement process itself. Missingness or exposure drift is a separate measurement issue even when event definitions match.
