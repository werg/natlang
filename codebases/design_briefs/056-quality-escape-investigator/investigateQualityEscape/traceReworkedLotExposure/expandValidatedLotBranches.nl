---
args: { lineage: LotLineage[], histories: ProcessHistory[], audit: GenealogyEdgeAudit }
returns: EscapeTrace
---
Expand only across verified edges. Keep a separate branch for every reworked child lot and preserve its old and new process route. Attach contradicted and missing edges as explicit uncertainty boundaries; never let them erase verified exposure on another branch.
