---
args: { request: SupplierIncidentRequest, incidents: SupplierIncident[], evidence: EvidencePage[] }
returns: StageResultIncidentCluster
---
Group reports only when product identity, lot or authenticated alias, time window, and failure mechanism have evidence of a shared factor. Preserve contradictory reports and separate problems from the same supplier when no shared mechanism is supported. Retain uncertain links as unresolved rather than forcing one cause.
