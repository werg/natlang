---
args: { request: SupplierIncidentRequest, evidence: EvidencePage[] }
returns: IncidentGuardrail
---
Partition an order only from exact receiving and allocation evidence by lot and quantity. Keep untraced remainder held; release only a portion that is specifically evidenced as unaffected.
