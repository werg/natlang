---
args: { request: CustomerObservatoryRequest, concepts: StableProblemMap, coverage: ComparableTrendSet, receipts: TrendAggregateReceipt[] }
returns: ComparableTrendSet
---
Use only supplied exact aggregation receipts bound to concept, cohort query, time window, channel version and exposure version. Check each numerator is a subset of its denominator and preserve its query hash. Do not compute counts or rates from prose or claim a trend when no exact receipt is supplied.
