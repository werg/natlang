---
args: { prior: ObservatoryCheckpoint | null, request: CustomerObservatoryRequest, incidents: IncidentDedupLedger, concepts: StableProblemMap, trends: ComparableTrendSet | null, invalidations: ObservatoryInvalidation }
returns: TrendLineageReview
---
Compare trend receipts by exact population, observation window, channel coverage, exposure denominator, query hash and stable concept edition. A changed identity link or incident grouping invalidates only windows that consume it; concept renames preserve a trend only when the concept lineage is explicitly continuous. Keep the old receipt edition for audit, identify windows that cannot be compared, and never calculate a replacement aggregate without a matching exact receipt.
