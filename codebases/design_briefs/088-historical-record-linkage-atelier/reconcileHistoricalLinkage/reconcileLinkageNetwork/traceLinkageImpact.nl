---
description: Trace corrected records and entity splits to dependent links, networks and research-edition statements.
args: { dossier: LinkageDossier; change: LinkageChange; findings: HistoricalIdentityFinding[] }
returns: LinkEdgeImpact[]
---
Follow source references from corrected records and split requests into candidate edges, identity findings, network components and edition statements. Mark only those dependent artifacts for review or reversal. Preserve the prior network version and all unaffected links.
