---
args: {"records": "DatasetCandidate[]", "events": "Evidence[]", "prior": "SourceFamily[]"}
returns: "SourceFamily[]"
---
Build candidate identity links from source lineage, shared underlying world and possible copies. For each proposed link call distinguishIndependentMechanism(left, right, events). Union only established same-source/world links; keep uncertain links quarantined from split certification. Preserve family identity across revisions. A paraphrase or translated copy does not become an independent source or a new world merely because its text differs.
