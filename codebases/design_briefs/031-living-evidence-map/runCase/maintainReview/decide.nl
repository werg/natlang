---
args: { prepared: PreparedEvidenceUpdate, goal: GoalContract, policy: PolicySelection }
returns: DecisionsEvidenceUpdate
---
1. Apply this domain algorithm to each relevant evidence unit: Process new reports, corrections, and retractions as scoped changes. Re-evaluate linked studies and dependent claims, preserve prior review snapshots, and request expert review where the new evidence changes methodological interpretation rather than merely adding another citation.
2. Return field-level atoms for the actual EvidenceUpdate record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
