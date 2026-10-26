---
description: Apply identity findings to the candidate graph and support reversible splits of premature merges.
args: { dossier: LinkageDossier; change: LinkageChange; findings: HistoricalIdentityFinding[] }
returns: NetworkReconciliation
---
Bind impacts = `traceLinkageImpact(dossier, change, findings)`. for every corrected source and requested split. Preserve supported edges; reverse only links dependent on contradictory or corrected evidence. Recompute affected downstream identity groups and edition statements without invalidating unrelated components. Keep uncertain edges unresolved. Return the revised network and impacts together.
