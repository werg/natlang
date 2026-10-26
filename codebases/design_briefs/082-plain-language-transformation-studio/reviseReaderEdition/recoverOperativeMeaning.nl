---
args: {"sources": "EvidenceItem[]", "changes": "EvidenceItem[]", "prior": "OperativeUnit[]"}
returns: "OperativeUnit[]"
---
For each current source call recoverClauseGraph(source, prior). Use stable source/locator-based unit IDs and explicit UTF-16 start/end spans. Preserve duties, permissions, negations, conditions, exceptions and uncertainty separately. Changed source units retain their history and invalidate dependent paragraphs/reviews only; unchanged units keep their identity. Never remove a consequential qualification because the original is dense.
