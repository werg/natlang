---
args: { lineage: LotLineage[], histories: ProcessHistory[] }
returns: GenealogyEdgeAudit
---
Check each parent and rework edge against both lot records and process history. A route version must agree with the history for that lot. Return verified, contradicted, and missing edges separately; do not manufacture an ancestry relation from naming similarity.
