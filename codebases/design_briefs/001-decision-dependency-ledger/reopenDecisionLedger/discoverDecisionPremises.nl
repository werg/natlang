---
args: {"decisions": "LedgerDecision[]", "sources": "EvidencePage[]", "previous": "PremiseLink[]"}
returns: "PremiseLink[]"
---
For each decision identify candidate premises from the passage explaining its condition. Bind links = testCounterfactualDependence(decision, sources). Each link needs both a source passage and an answer to what would change in the decision if that premise became false. Distinguish necessary, supporting and background links. Preserve prior reviewed identities and confirmation; do not upgrade a newly inferred link to confirmed. Return unresolved candidates as unreviewed links with the missing source/owner requirement.
