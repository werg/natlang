---
description: Revise a domain representation in response to a concrete mismatch.
args:
  goal: string
  material: string
returns: Proposal
---
Identify exactly which current concept fails and show records or queries that
demonstrate it. Propose structural type source, executable migration source,
and descriptions of the changed meaning. Preserve raw evidence. Distinguish
known mappings, unresolved alternatives and intentional loss. Inspect likely
consumers: methods, claims, views and data. Include edits for those that must
change, and checks comparing old and new behavior. If no meaningful mismatch
exists, keep the current schema and say why. A valid type alone does not prove
that migration preserves meaning. State what further evidence would resolve
ambiguous mappings.
