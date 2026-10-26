---
args: { request: CustomerObservatoryRequest, sampling: EvidencePage[], identities: IdentityVersionLedger, receipts: TrendAggregateReceipt[] }
returns: ComparableTrendSet
---
Before comparing periods, require the same eligible population definition, exposure basis, collection channel semantics and complete searched windows. If a channel disappears or identity mapping is unresolved, mark comparability broken and withhold improvement claims.
