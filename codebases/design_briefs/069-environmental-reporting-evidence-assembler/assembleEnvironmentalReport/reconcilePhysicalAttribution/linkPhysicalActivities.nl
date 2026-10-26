---
description: Match internal, contractor, invoice and field records to physical event identities.
args: { dossier: ReportingDossier; measures: EnvironmentalMeasure[] }
returns: ActivityIdentity[]
---
Compare source references, operation IDs, time period, location, contractor relationship and event descriptions. Link records as the same physical activity only when evidence supports that identity. Preserve responsibility transfer as an attribution relationship separate from physical event identity. Return unresolved candidates when no discriminating key is available.
