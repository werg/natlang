---
args: { lineage: LotLineage[], histories: ProcessHistory[] }
returns: EscapeTrace
---
Bind audit = validateGenealogyEdges(lineage, histories). Start only from observed defect lots. Trace parents and rework descendants across verified edges, preserving every process route version. If an edge is invalid or a parent is missing, widen only that boundary and keep the implicated branch precautionary.
Bind branches = expandValidatedLotBranches(lineage, histories, audit). Return the exact affected and reworked branch IDs, retained history IDs, unresolved edges, and exposure boundary.
