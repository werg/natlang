---
args: { lineage: LotLineage[], histories: ProcessHistory[] }
returns: EscapeTrace
---
Start from the observed defect lots. Traverse parentLotId and reworkRef links in both directions only where the supplied lot genealogy supports the relation. Retain each process route version and expose unknown lineage boundaries. Identify possible affected descendants without merging separately reworked histories.
Bind trail = validateGenealogyEdges(lineage, histories). Use invalid or missing edges to widen the uncertainty boundary, never to assert that a lot was unaffected.
