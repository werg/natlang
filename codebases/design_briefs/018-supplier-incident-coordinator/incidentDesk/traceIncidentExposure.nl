---
args: { request: SupplierIncidentRequest, clusters: StageResultIncidentCluster, evidence: EvidencePage[], aliases: IncidentGuardrail, splitTrace: EvidencePage[] }
returns: StageResultExposureFinding
---
Trace the supported lot and alias identities through receiving, split shipment, allocation, and downstream order evidence. Keep confirmed exposure separate from precautionary scope. If a lot or quantity cannot be traced, widen the precautionary boundary and do not mark untraced stock unaffected. Bind partition = partitionSplitStock(request, splitTrace). Use the resulting exact quantity partitions when identifying affected stock and preserve the unaffected portion of a split order.
