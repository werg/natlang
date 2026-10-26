---
args: { procedure: MethodProcedure, evidence: EvidencePage[] }
returns: ReplicationGraph
---
Represent each method step as a node with exact prerequisites, resources, source/version and artifact dependencies. Detect cycles and unavailable inputs; retain unresolved alternatives until source evidence or an investigator resolves them.
